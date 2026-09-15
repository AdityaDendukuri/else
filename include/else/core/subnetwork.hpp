// Finite state set with its restricted generator and factorization.
#pragma once

#include "else/core/types.hpp"
#include "linear/factorization/block_tridiagonal.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu_no_pivot.hpp"
#include "linear/sparse/sparse.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace else_sim {

namespace detail {

// H M H^-1 with H = diag(weights), preserving the sparsity pattern.
[[nodiscard]] inline num::spmat similarity_scaled(const num::spmat &matrix,
                                                  view<const real> weights) {
    array<real> values(matrix.nnz());
    array<idx> columns(matrix.col_idx(), matrix.col_idx() + matrix.nnz());
    array<idx> rows(matrix.row_ptr(), matrix.row_ptr() + matrix.n_rows() + 1);
    for (idx i = 0; i < matrix.n_rows(); ++i)
        for (idx k = matrix.row_ptr()[i]; k < matrix.row_ptr()[i + 1]; ++k)
            values[k] = matrix.values()[k] * weights[i] / weights[matrix.col_idx()[k]];
    return num::spmat(matrix.n_rows(), matrix.n_cols(), std::move(values), std::move(columns),
                      std::move(rows));
}

// sqrt(pi_S), the similarity weights for H M H^-1.
[[nodiscard]] inline num::vec stationary_weights(view<const real> stationary) {
    num::vec weights(stationary.size());
    for (idx i = 0; i < stationary.size(); ++i) {
        if (!(stationary[i] > 0.0))
            throw std::invalid_argument("stationary weights must be positive");
        weights[i] = std::sqrt(stationary[i]);
    }
    return weights;
}

} // namespace detail

struct NormalizedCholesky {
    num::vec weights;
    std::variant<num::cholesky_result, num::block_cholesky_factor> factor;
};

using FactorVariant = std::variant<num::no_pivot_lu, num::block_lu_factor, NormalizedCholesky>;

// Explicit level structure is worthwhile well below the sizes at which a
// completely unstructured dense factorization becomes expensive.  Keep this
// threshold aligned with the original block solver so, in particular, the
// 60-state reversible benchmark actually exercises block Cholesky.
inline constexpr idx block_threshold = 32;

namespace detail {

// Symmetrized H M H^-1: block Cholesky if large enough, else dense Cholesky.
[[nodiscard]] inline FactorVariant factor_reversible(const num::spmat &operator_matrix,
                                                     const num::vec &stationary,
                                                     view<const idx> levels) {
    num::vec weights = stationary_weights(stationary.span());
    const num::spmat normalized = similarity_scaled(operator_matrix, weights.span());
    if (!levels.empty() && operator_matrix.n_rows() > block_threshold) {
        return FactorVariant(
            NormalizedCholesky{std::move(weights), num::factor_block_cholesky(normalized, levels)});
    }

    auto factor = num::cholesky(num::assume_spd(num::dense(normalized)));
    if (!factor.success)
        throw std::runtime_error("normalized subnetwork is not positive definite");
    return FactorVariant(NormalizedCholesky{std::move(weights), std::move(factor)});
}

// M directly. A transient CME restriction is a nonsingular M-matrix, so its
// elimination pivots are positive in every fixed ordering: use no-pivot LU for
// the dense and block-tridiagonal paths rather than a generic pivoting solver.
[[nodiscard]] inline FactorVariant factor_irreversible(const num::spmat &operator_matrix,
                                                       view<const idx> levels) {
    const idx n = operator_matrix.n_rows();
    if (!levels.empty() && n > block_threshold) {
        if (levels.size() != n)
            throw std::invalid_argument("one block level is required per state");
        return FactorVariant(num::factor_block_lu(operator_matrix, levels));
    }
    num::mat dense = num::dense(operator_matrix);
    auto factor = num::factor_no_pivot(num::assume_square(dense));
    if (factor.singular)
        throw std::runtime_error("subnetwork operator is singular");
    return FactorVariant(std::move(factor));
}

} // namespace detail

// Reversible subnetworks take the symmetrized Cholesky path, else general LU/solver.
[[nodiscard]] inline std::optional<FactorVariant>
choose_factor(bool factorize, const num::vec &stationary, const num::spmat &operator_matrix,
              view<const idx> levels, bool reversible) {
    if (!factorize)
        return std::nullopt;
    return reversible ? detail::factor_reversible(operator_matrix, stationary, levels)
                      : detail::factor_irreversible(operator_matrix, levels);
}

// A finite state set S: restricted generator R = R_bar(S,S) (row convention)
// and a factorization of M = -R. Z = M^-1 gives expected time in j before
// escape, starting from i.
template <typename State = num::multi_index>
struct Subnetwork {
    array<State> states;
    table<State, idx> index;
    num::spmat generator;
    array<BoundaryTransition<State>> boundary;
    num::vec stationary; // pi_S, empty for an irreversible restriction
    bool reversible = false;
    num::spmat operator_matrix;
    array<idx> levels;
    std::optional<FactorVariant> factor;
    num::vec escape_rates;
    array<idx> escape_states;
};

namespace detail {

// w_j = sum_c r_jc from the boundary transitions; states with w_j > 0 escape.
// Uses the boundary rates directly rather than the row sum of `operator_matrix`,
// since that row sum cancels to round-off rather than exact zero.
template <typename State>
void escape_rates_from_boundary(const array<BoundaryTransition<State>> &boundary, idx n,
                                num::vec &escape_rates, array<idx> &escape_states) {
    for (const auto &transition : boundary) {
        if (transition.source >= n) {
            throw std::out_of_range("boundary transition source is outside the subnetwork");
        }
        escape_rates[transition.source] += transition.rate;
    }
    for (idx j = 0; j < n; ++j) {
        if (escape_rates[j] > 0.0) {
            escape_states.push_back(j);
        }
    }
}

} // namespace detail

template <typename State = num::multi_index>
[[nodiscard]] Subnetwork<State>
make_subnetwork(array<State> states, num::spmat generator,
                array<BoundaryTransition<State>> boundary, num::vec stationary = {},
                view<const idx> levels = {}, bool factorize = true, bool reversible = true) {
    if (stationary.size() != 0 && stationary.size() != states.size()) {
        throw std::invalid_argument("stationary weights must have one entry per state");
    }
    if (generator.n_rows() != states.size() || generator.n_cols() != states.size()) {
        throw std::invalid_argument("generator must be square with one row per state");
    }

    // Locals, not default-constructed fields: num::spmat has no default constructor.
    num::spmat operator_matrix = num::scaled(generator, -1.0);
    reversible = reversible && stationary.size() != 0;
    std::optional<FactorVariant> factor =
        choose_factor(factorize, stationary, operator_matrix, levels, reversible);

    table<State, idx> index;
    index.reserve(states.size());
    for (idx i = 0; i < states.size(); ++i) {
        index.emplace(states[i], i);
    }

    num::vec escape_rates(states.size(), 0.0);
    array<idx> escape_states;
    detail::escape_rates_from_boundary(boundary, states.size(), escape_rates, escape_states);

    return Subnetwork<State>{std::move(states),          std::move(index),
                             std::move(generator),       std::move(boundary),
                             std::move(stationary),      reversible,
                             std::move(operator_matrix), array<idx>(levels.begin(), levels.end()),
                             std::move(factor),
                             std::move(escape_rates),    std::move(escape_states)};
}

template <typename State>
[[nodiscard]] idx size(const Subnetwork<State> &subnetwork) {
    return subnetwork.states.size();
}

// `size(subnetwork)` when `state` is outside the subnetwork.
template <typename State>
[[nodiscard]] idx find(const Subnetwork<State> &subnetwork, const State &state) {
    const auto found = subnetwork.index.find(state);
    return found == subnetwork.index.end() ? size(subnetwork) : found->second;
}

template <typename State>
[[nodiscard]] bool is_reversible(const Subnetwork<State> &subnetwork) {
    return subnetwork.reversible;
}

// Whether solves use a block-tridiagonal LU or Cholesky factorization.
template <typename State>
[[nodiscard]] bool uses_block_factor(const Subnetwork<State> &subnetwork) {
    if (!subnetwork.factor)
        return false;
    return std::visit(
        [](const auto &factor) {
            using Factor = std::decay_t<decltype(factor)>;
            if constexpr (std::is_same_v<Factor, num::block_lu_factor>) {
                return true;
            } else if constexpr (std::is_same_v<Factor, NormalizedCholesky>) {
                return std::holds_alternative<num::block_cholesky_factor>(factor.factor);
            } else {
                return false;
            }
        },
        *subnetwork.factor);
}

namespace detail {

// Scale `value` row-wise by `weights[i]` (or `1/weights[i]` when `invert`).
template <typename RightHandSide>
void scale_rows(RightHandSide &value, const num::vec &weights, bool invert) {
    if constexpr (std::is_same_v<RightHandSide, num::vec>) {
        for (idx i = 0; i < value.size(); ++i)
            value[i] *= invert ? 1.0 / weights[i] : weights[i];
    } else {
        for (idx i = 0; i < value.rows(); ++i)
            for (idx j = 0; j < value.cols(); ++j)
                value(i, j) *= invert ? 1.0 / weights[i] : weights[i];
    }
}

// Move `rhs` into H-coordinates, solve the stored Cholesky factor, move back.
template <typename RightHandSide>
void solve_normalized(const NormalizedCholesky &normalized, const RightHandSide &rhs,
                      RightHandSide &solution, bool transpose) {
    RightHandSide scaled = rhs;
    scale_rows(scaled, normalized.weights, /*invert=*/transpose);
    if constexpr (std::is_same_v<RightHandSide, num::vec>) {
        solution = num::vec(scaled.size(), 0.0);
    }
    std::visit(
        [&](const auto &factor) {
            if constexpr (std::is_same_v<std::decay_t<decltype(factor)>, num::cholesky_result>)
                num::cholesky_solve(factor, scaled, solution);
            else
                num::solve(factor, scaled, solution);
        },
        normalized.factor);
    scale_rows(solution, normalized.weights, /*invert=*/!transpose);
}

template <typename State, typename RightHandSide>
[[nodiscard]] RightHandSide dispatch_solve(const Subnetwork<State> &subnetwork,
                                           const RightHandSide &rhs, bool transpose) {
    if (!subnetwork.factor)
        throw std::logic_error("subnetwork has no factorization");
    RightHandSide result;
    std::visit(
        [&](const auto &factor) {
            using Factor = std::decay_t<decltype(factor)>;
            if constexpr (std::is_same_v<Factor, NormalizedCholesky>) {
                solve_normalized(factor, rhs, result, transpose);
            } else {
                if (transpose)
                    num::solve_transpose(factor, rhs, result);
                else
                    num::solve(factor, rhs, result);
            }
        },
        *subnetwork.factor);
    return result;
}

} // namespace detail

template <typename State>
[[nodiscard]] num::mat solve_transpose(const Subnetwork<State> &subnetwork, const num::mat &rhs) {
    return detail::dispatch_solve(subnetwork, rhs, true);
}
template <typename State>
[[nodiscard]] num::vec solve_transpose(const Subnetwork<State> &subnetwork, const num::vec &rhs) {
    return detail::dispatch_solve(subnetwork, rhs, true);
}
template <typename State>
[[nodiscard]] num::vec solve(const Subnetwork<State> &subnetwork, const num::vec &rhs) {
    return detail::dispatch_solve(subnetwork, rhs, false);
}
template <typename State>
[[nodiscard]] num::mat solve(const Subnetwork<State> &subnetwork, const num::mat &rhs) {
    return detail::dispatch_solve(subnetwork, rhs, false);
}

template <typename State>
[[nodiscard]] num::vec inverse_diagonal(const Subnetwork<State> &subnetwork) {
    const idx n = size(subnetwork);
    num::vec diagonal(n, 0.0);
    constexpr idx columns_per_solve = 64;
    for (idx first = 0; first < n; first += columns_per_solve) {
        const idx count = std::min(columns_per_solve, n - first);
        num::mat basis(n, count, 0.0);
        for (idx column = 0; column < count; ++column)
            basis(first + column, column) = 1.0;
        const num::mat columns = solve(subnetwork, basis);
        for (idx column = 0; column < count; ++column)
            diagonal[first + column] = columns(first + column, column);
    }
    return diagonal;
}

} // namespace else_sim
