// The three state-shedding rules.
#pragma once

#include "container/matrix_expr.hpp"
#include "else/core/subnetwork.hpp"
#include "linear/eigen/lanczos.hpp"
#include "linear/factorization/inverse_diagonal.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/graph/randommat/preconditioner.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/sparse/sparse_op.hpp"
#include "operator/properties.hpp"
#include "stochastic/probe.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace else_sim {

// rho: entrance mixture. u = Z^T*rho: expected occupation. q = Z*1: expected
// exit time. Shared by all three shedding rules.
struct SheddingState {
    num::vec entrance_mixture; // rho
    num::vec occupation;       // u = Z^T * rho
    num::vec exit_time;        // q = Z*1
};

// Solve M^T u = rho and M q = 1 against the stored factorization.
template <typename State>
[[nodiscard]] SheddingState shedding_state(const Subnetwork<State> &subnetwork,
                                           num::vec entrance_mixture) {
    if (entrance_mixture.size() != size(subnetwork)) {
        throw std::invalid_argument("entrance mixture size must match the subnetwork");
    }
    num::vec occupation = solve_transpose(subnetwork, entrance_mixture);
    num::vec exit_time = solve(subnetwork, num::vec(size(subnetwork), 1.0));
    return {std::move(entrance_mixture), std::move(occupation), std::move(exit_time)};
}

// Exact cut-time loss: l_j = u_j * q_j / Z_jj (exact, not a heuristic).
template <typename State>
[[nodiscard]] num::vec exact_cut_time_losses(const Subnetwork<State> &subnetwork,
                                             const SheddingState &state) {
    const num::vec diagonal_of_inverse = inverse_diagonal(subnetwork);
    num::vec losses(size(subnetwork), 0.0);
    for (idx j = 0; j < size(subnetwork); ++j) {
        losses[j] = diagonal_of_inverse[j] > 0.0
                        ? state.occupation[j] * state.exit_time[j] / diagonal_of_inverse[j]
                        : 0.0;
    }
    return losses;
}

// Expected number of entries: s_j = (-R_jj) * u_j - rho_j. Cheapest of the
// three -- no inverse diagonal needed.
template <typename State>
[[nodiscard]] num::vec expected_entries(const Subnetwork<State> &subnetwork,
                                        const SheddingState &state) {
    const num::vec operator_diagonal = num::diagonal(subnetwork.operator_matrix);
    num::vec entries(size(subnetwork), 0.0);
    for (idx j = 0; j < size(subnetwork); ++j) {
        entries[j] = operator_diagonal[j] * state.occupation[j] - state.entrance_mixture[j];
    }
    return entries;
}

// Joint cut-time loss for removing a whole set at once, with its backward residual.
struct JointCutTime {
    real loss = 0.0;              // delta_T = q_J^T Z_JJ^-1 u_J
    real backward_residual = 0.0; // max|Z_JJ*c - u_J| / max|u_J|
};

// Cut time lost by removing `removed` all together: delta_T = u_J^T Z_JJ^-1
// q_J (order matters -- Z is generally nonsymmetric). `backward_residual` is
// the actual error of the block solve, not an epsilon bound.
template <typename State>
[[nodiscard]] JointCutTime joint_cut_time_loss(const Subnetwork<State> &subnetwork,
                                               const SheddingState &state,
                                               view<const idx> removed) {
    const idx count = removed.size();
    if (count == 0) {
        return {};
    }
    for (idx j : removed) {
        if (j >= size(subnetwork)) {
            throw std::out_of_range("removed state is outside the subnetwork");
        }
    }

    num::mat basis(size(subnetwork), count, 0.0);
    for (idx column = 0; column < count; ++column)
        basis(removed[column], column) = 1.0;
    const num::mat inverse_columns = solve(subnetwork, basis);
    num::mat block(count, count, 0.0);
    for (idx row = 0; row < count; ++row)
        for (idx column = 0; column < count; ++column)
            block(row, column) = inverse_columns(removed[row], column);
    num::vec occupation(count, 0.0);
    num::vec exit_time(count, 0.0);
    for (idx i = 0; i < count; ++i) {
        occupation[i] = state.occupation[removed[i]];
        exit_time[i] = state.exit_time[removed[i]];
    }

    // c = Z_JJ^-1 q_J, then the loss is u_J . c.
    const auto factor = num::lu(num::make_square(block));
    num::vec corrected(count, 0.0);
    num::lu_solve(factor, exit_time, corrected);

    const real loss = num::dot(occupation, corrected);

    const num::vec reconstructed = num::matvec(block, corrected);
    real residual = 0.0;
    real scale = 0.0;
    for (idx i = 0; i < count; ++i) {
        residual = std::max(residual, std::abs(reconstructed[i] - exit_time[i]));
        scale = std::max(scale, std::abs(exit_time[i]));
    }
    return {loss, residual / std::max(scale, 1.0)};
}

namespace detail {

// (A + A^T)/2 in sparse storage. Duplicate triplets are summed by spmat.
[[nodiscard]] inline num::spmat symmetric_part(const num::spmat &matrix) {
    array<idx> rows;
    array<idx> columns;
    array<real> values;
    rows.reserve(2 * matrix.nnz());
    columns.reserve(2 * matrix.nnz());
    values.reserve(2 * matrix.nnz());
    for (idx row = 0; row < matrix.n_rows(); ++row) {
        for (idx entry = matrix.row_ptr()[row]; entry < matrix.row_ptr()[row + 1]; ++entry) {
            const idx column = matrix.col_idx()[entry];
            const real half = 0.5 * matrix.values()[entry];
            rows.push_back(row);
            columns.push_back(column);
            values.push_back(half);
            rows.push_back(column);
            columns.push_back(row);
            values.push_back(half);
        }
    }
    return num::spmat::from_triplets(matrix.n_rows(), matrix.n_cols(), rows, columns, values);
}

// H*S*H, the grounded graph-Laplacian block associated with the stationary
// symmetrization.  Unlike S itself, this congruence is diagonally dominant.
[[nodiscard]] inline num::spmat grounded_symmetric_part(const num::spmat &symmetric,
                                                        view<const real> weights) {
    array<idx> rows;
    array<idx> columns;
    array<real> values;
    rows.reserve(symmetric.nnz());
    columns.reserve(symmetric.nnz());
    values.reserve(symmetric.nnz());
    for (idx row = 0; row < symmetric.n_rows(); ++row) {
        for (idx entry = symmetric.row_ptr()[row]; entry < symmetric.row_ptr()[row + 1]; ++entry) {
            const idx column = symmetric.col_idx()[entry];
            rows.push_back(row);
            columns.push_back(column);
            values.push_back(weights[row] * symmetric.values()[entry] * weights[column]);
        }
    }
    return num::spmat::from_triplets(symmetric.n_rows(), symmetric.n_cols(), rows, columns, values);
}

// B = C^-1 L C^-T, where C is the grounded ApproxChol factor of L.
class approxchol_scaled_operator final {
  public:
    using math_laws = num::math::type_list<num::law::linear_map>;
    using domain_type = num::vec;
    using codomain_type = num::vec;

    approxchol_scaled_operator(const num::spmat &grounded,
                               const num::grounded_approx_chol_factor &factor)
        : grounded_(grounded), factor_(factor) {}

    void apply(const num::vec &input, num::vec &output) const {
        const num::vec upper = factor_.solve_upper(input);
        num::vec product(rows(), 0.0);
        num::sparse_matvec(grounded_, upper, product);
        output = factor_.solve_lower(product);
    }

    [[nodiscard]] idx rows() const noexcept { return grounded_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return grounded_.n_cols(); }

  private:
    const num::spmat &grounded_;
    const num::grounded_approx_chol_factor &factor_;
};

} // namespace detail

// Positive probe estimate of diag(Z) after stationary normalization. ApproxChol
// factors the grounded stationary symmetrization, and Lanczos applies the square
// root correction of the resulting well-conditioned operator.
template <typename State>
[[nodiscard]] num::vec
probed_inverse_diagonal(const Subnetwork<State> &subnetwork, view<const real> stationary,
                        idx probes, unsigned seed = 42, idx krylov_steps = 64,
                        real krylov_tolerance = 1e-8, bool reversible = true) {
    const idx n = size(subnetwork);
    if (stationary.size() != n) {
        throw std::invalid_argument("stationary weights must have one entry per state");
    }
    if (probes == 0) {
        throw std::invalid_argument("probe estimate requires at least one probe");
    }

    const num::vec weights = detail::stationary_weights(stationary);
    const num::spmat normalized =
        detail::similarity_scaled(subnetwork.operator_matrix, weights.span());
    const num::mat probe = num::rademacher_probe(n, probes, seed);
    num::mat probed(n, probes, 0.0);
    const num::spmat symmetric = reversible ? normalized : detail::symmetric_part(normalized);
    const num::spmat grounded = detail::grounded_symmetric_part(symmetric, weights.span());
    const num::grounded_approx_chol_factor approximate_factor =
        num::grounded_approxchol_factor(grounded, 2, seed ^ 0x9e3779b9U);
    const auto scaled_operator = num::operators::assume_spd(
        detail::approxchol_scaled_operator(grounded, approximate_factor));
    std::optional<num::auto_linear_solver> normalized_factor;
    if (!reversible)
        normalized_factor.emplace(normalized);
    for (idx column = 0; column < probes; ++column) {
        num::vec direction(n, 0.0);
        for (idx row = 0; row < n; ++row)
            direction[row] = probe(row, column);
        const auto action = reversible ? num::inverse_sqrt_lanczos(scaled_operator, direction,
                                                                   krylov_tolerance, krylov_steps)
                                       : num::sqrt_lanczos(scaled_operator, direction,
                                                           krylov_tolerance, krylov_steps);
        num::vec value(n, 0.0);
        if (reversible) {
            const num::vec unscaled = approximate_factor.solve_upper(action.value);
            for (idx row = 0; row < n; ++row)
                value[row] = weights[row] * unscaled[row];
        } else {
            const num::vec factored = approximate_factor.apply_lower(action.value);
            num::vec right_hand_side(n, 0.0);
            for (idx row = 0; row < n; ++row)
                right_hand_side[row] = factored[row] / weights[row];
            value = num::solve(*normalized_factor, right_hand_side);
        }
        for (idx row = 0; row < n; ++row)
            probed(row, column) = value[row];
    }

    return num::hutchinson_row_mean_square(probed);
}

// Cut-time loss using the probed inverse diagonal in place of the exact one.
template <typename State>
[[nodiscard]] num::vec
probed_cut_time_losses(const Subnetwork<State> &subnetwork, const SheddingState &state,
                       view<const real> stationary, idx probes, unsigned seed = 42,
                       idx krylov_steps = 64, real krylov_tolerance = 1e-8,
                       bool reversible = true) {
    const num::vec estimate = probed_inverse_diagonal(subnetwork, stationary, probes, seed,
                                                      krylov_steps, krylov_tolerance, reversible);
    num::vec losses(size(subnetwork), 0.0);
    for (idx j = 0; j < size(subnetwork); ++j) {
        losses[j] =
            estimate[j] > 0.0 ? state.occupation[j] * state.exit_time[j] / estimate[j] : 0.0;
    }
    return losses;
}

// Score every state under `options.rule`, using whichever of the three
// shedding computations above the rule selects. Shared by `else_ensemble`
// and `shed_to_capacity` so the three branches are defined once.
template <typename State>
[[nodiscard]] num::vec score_states(const Subnetwork<State> &subnetwork, const SheddingState &state,
                                    const EnsembleOptions &options) {
    switch (options.rule) {
    case SheddingRule::ExpectedEntries:
        return expected_entries(subnetwork, state);
    case SheddingRule::ExactCutTime:
        return exact_cut_time_losses(subnetwork, state);
    case SheddingRule::ProbedCutTime:
        if (subnetwork.stationary.size() == 0)
            throw std::invalid_argument(
                "probed cut time requires stationary weights for the subnetwork");
        return probed_cut_time_losses(subnetwork, state, subnetwork.stationary.span(),
                                      options.shedding_probes, options.shedding_seed,
                                      options.shedding_krylov_steps,
                                      options.shedding_krylov_tolerance, is_reversible(subnetwork));
    }
    throw std::invalid_argument("unrecognized shedding rule");
}

// The `count` eligible states with the smallest score. Protected states are
// never returned; ties break by index so the selection is reproducible.
[[nodiscard]] inline array<idx> lowest_scores(view<const real> scores,
                                              view<const idx> protected_states, idx count) {
    array<bool> ineligible(scores.size(), false);
    for (idx j : protected_states) {
        if (j < scores.size()) {
            ineligible[j] = true;
        }
    }

    array<idx> eligible;
    eligible.reserve(scores.size());
    for (idx j = 0; j < scores.size(); ++j) {
        if (!ineligible[j]) {
            eligible.push_back(j);
        }
    }

    count = std::min(count, static_cast<idx>(eligible.size()));
    std::partial_sort(
        eligible.begin(), eligible.begin() + count, eligible.end(),
        [scores](idx a, idx b) { return scores[a] != scores[b] ? scores[a] < scores[b] : a < b; });
    eligible.resize(count);
    return eligible;
}

} // namespace else_sim
