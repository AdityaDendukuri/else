// Woodbury reuse: replacing k states changes at most 2k rows and columns.
#pragma once

#include "container/matrix_expr.hpp"
#include "else/core/subnetwork.hpp"
#include "else/core/types.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace else_sim {

// Low-rank factors U and V for A = A0 + U V^T.
struct LowRankDelta {
    num::mat left;  // U
    num::mat right; // V
};

namespace detail {

// Numerics exposes value-returning solves for solver objects and output-argument
// solves for lightweight factors such as block LU and block Cholesky.  Normalize
// both interfaces here so the Woodbury layer can reuse either representation.
template <typename Solver, typename RightHandSide>
[[nodiscard]] RightHandSide apply_transpose_solve(const Solver &solver,
                                                  const RightHandSide &right_hand_side) {
    using num::solve_transpose;
    if constexpr (requires { solve_transpose(solver, right_hand_side); }) {
        return solve_transpose(solver, right_hand_side);
    } else {
        RightHandSide solution;
        solve_transpose(solver, right_hand_side, solution);
        return solution;
    }
}

template <typename BlockFactor>
[[nodiscard]] inline idx first_changed_block(const BlockFactor &base,
                                             const num::detail::block_layout &current,
                                             view<const idx> changed) {
    array<idx> base_position(base.size);
    for (idx position = 0; position < base.size; ++position)
        base_position[base.order[position]] = position;

    idx first = current.offsets.size() - 1;
    for (idx state : changed) {
        first = std::min(first, current.block_of[state]);
        const idx position = base_position[state];
        const auto boundary = std::upper_bound(base.offsets.begin(), base.offsets.end(), position);
        first = std::min(first, static_cast<idx>(boundary - base.offsets.begin() - 1));
    }
    return first;
}

template <typename BlockFactor>
[[nodiscard]] inline bool same_prefix_layout(const BlockFactor &base,
                                             const num::detail::block_layout &current,
                                             idx first) {
    if (first > base.blocks() || first + 1 > current.offsets.size())
        return false;
    for (idx k = 0; k <= first; ++k)
        if (base.offsets[k] != current.offsets[k])
            return false;
    for (idx position = 0; position < current.offsets[first]; ++position)
        if (base.order[position] != current.order[position])
            return false;
    return true;
}

// Assemble only blocks touched by a suffix refactorization. The diagonal starts
// at `first`; the incoming coupling starts at `first - 1`. Earlier rows need not
// be visited because their factors and couplings are copied from the base.
[[nodiscard]] inline num::detail::assembled_blocks
gather_suffix_blocks(const num::spmat &matrix, const num::detail::block_layout &layout,
                     idx first) {
    const idx count = layout.offsets.size() - 1;
    num::detail::assembled_blocks blocks;
    blocks.diagonal.resize(count);
    blocks.upper.resize(count > 0 ? count - 1 : 0);
    blocks.lower.resize(count > 0 ? count - 1 : 0);
    for (idx k = first; k < count; ++k) {
        const idx width = layout.offsets[k + 1] - layout.offsets[k];
        blocks.diagonal[k] = num::mat(width, width, 0.0);
    }
    const idx first_edge = first > 0 ? first - 1 : 0;
    for (idx k = first_edge; k + 1 < count; ++k) {
        const idx left = layout.offsets[k + 1] - layout.offsets[k];
        const idx right = layout.offsets[k + 2] - layout.offsets[k + 1];
        blocks.upper[k] = num::mat(left, right, 0.0);
        blocks.lower[k] = num::mat(right, left, 0.0);
    }

    for (idx row = 0; row < matrix.n_rows(); ++row) {
        const idx block_row = layout.block_of[row];
        if (first > 0 && block_row + 1 < first)
            continue;
        const idx local_row = layout.position[row] - layout.offsets[block_row];
        for (idx entry = matrix.row_ptr()[row]; entry < matrix.row_ptr()[row + 1]; ++entry) {
            const real value = matrix.values()[entry];
            if (value == 0.0)
                continue;
            const idx column = matrix.col_idx()[entry];
            const idx block_column = layout.block_of[column];
            const idx local_column = layout.position[column] - layout.offsets[block_column];
            if (block_row == block_column && block_row >= first) {
                blocks.diagonal[block_row](local_row, local_column) += value;
            } else if (block_column == block_row + 1 && block_row >= first_edge) {
                blocks.upper[block_row](local_row, local_column) += value;
            } else if (block_row == block_column + 1 && block_column >= first_edge) {
                blocks.lower[block_column](local_row, local_column) += value;
            }
        }
    }
    return blocks;
}

// Slot within `changed`, or `n` if `state` isn't in it.
[[nodiscard]] inline array<idx> changed_slot_lookup(idx n, view<const idx> changed) {
    array<idx> slot(n, n);
    for (idx k = 0; k < changed.size(); ++k) {
        if (changed[k] >= n) {
            throw std::out_of_range("changed slot is outside the operator");
        }
        slot[changed[k]] = k;
    }
    return slot;
}

// Add sign * matrix's changed rows/columns into `left`/`right`. A changed row
// contributes a full row-diff via `right`'s row-indicator column; a changed
// column not already covered by a changed row contributes via `left`'s
// column-indicator column. Called once for `current` (sign=+1) and once for
// `base` (sign=-1), so together the two calls accumulate current - base.
inline void accumulate_row_column_diff(const num::spmat &matrix, real sign, view<const idx> slot,
                                       idx count, num::mat &left, num::mat &right) {
    const idx n = matrix.n_rows();
    for (idx i = 0; i < n; ++i) {
        const idx row_slot = slot[i];
        for (auto p = matrix.row_ptr()[i]; p < matrix.row_ptr()[i + 1]; ++p) {
            const idx j = static_cast<idx>(matrix.col_idx()[p]);
            const real value = sign * matrix.values()[p];
            if (row_slot != n) {
                right(j, row_slot) += value;
            } else if (slot[j] != n) {
                left(i, count + slot[j]) += value;
            }
        }
    }
}

} // namespace detail

struct BlockSuffixInfo {
    idx block_count = 0;
    idx reused_prefix_blocks = 0;
    idx reused_prefix_states = 0;
};

// Recompute only the block-Thomas suffix affected by the changed state slots.
// Diagonal factors strictly before the first changed block, and the scaled
// lower couplings before its incoming edge, are copied from `base`.
[[nodiscard]] inline std::optional<num::block_lu_factor>
refactor_block_suffix(const num::block_lu_factor &base, const num::spmat &current,
                      view<const idx> levels, view<const idx> changed,
                      BlockSuffixInfo *info = nullptr) {
    if (changed.empty())
        return base;
    const num::detail::block_layout layout = num::detail::build_block_order(levels);
    const idx first = detail::first_changed_block(base, layout, changed);
    if (!detail::same_prefix_layout(base, layout, first))
        return std::nullopt;
    num::detail::validate_block_structure(current, layout);

    num::detail::assembled_blocks blocks = detail::gather_suffix_blocks(current, layout, first);
    const idx count = layout.offsets.size() - 1;
    if (info)
        *info = {.block_count = count,
                 .reused_prefix_blocks = first,
                 .reused_prefix_states = layout.offsets[first]};

    num::block_lu_factor factor;
    factor.size = current.n_rows();
    factor.offsets = layout.offsets;
    factor.order = layout.order;
    factor.upper = std::move(blocks.upper);
    factor.lower = std::move(blocks.lower);
    factor.diagonal.reserve(count);
    for (idx k = 0; k < first; ++k)
        factor.diagonal.push_back(base.diagonal[k]);
    for (idx k = 0; k + 1 < first; ++k) {
        factor.upper[k] = base.upper[k];
        factor.lower[k] = base.lower[k];
    }

    for (idx k = first; k < count; ++k) {
        if (k > 0) {
            num::mat transposed = num::transpose(factor.lower[k - 1]);
            num::mat scaled_transposed;
            num::solve_transpose(factor.diagonal[k - 1], transposed, scaled_transposed);
            factor.lower[k - 1] = num::transpose(scaled_transposed);
            num::detail::subtract_from(
                blocks.diagonal[k],
                num::detail::product(factor.lower[k - 1], factor.upper[k - 1]));
        }
        factor.diagonal.push_back(
            num::factor_no_pivot(num::assume_square(blocks.diagonal[k])));
        if (factor.diagonal.back().singular)
            throw std::runtime_error("block suffix update encountered a zero pivot");
    }
    return factor;
}

[[nodiscard]] inline std::optional<num::block_cholesky_factor>
refactor_block_suffix(const num::block_cholesky_factor &base, const num::spmat &current,
                      view<const idx> levels, view<const idx> changed,
                      BlockSuffixInfo *info = nullptr) {
    if (changed.empty())
        return base;
    const num::detail::block_layout layout = num::detail::build_block_order(levels);
    const idx first = detail::first_changed_block(base, layout, changed);
    if (!detail::same_prefix_layout(base, layout, first))
        return std::nullopt;
    num::detail::validate_block_structure(current, layout);

    num::detail::assembled_blocks blocks = detail::gather_suffix_blocks(current, layout, first);
    const idx count = layout.offsets.size() - 1;
    if (info)
        *info = {.block_count = count,
                 .reused_prefix_blocks = first,
                 .reused_prefix_states = layout.offsets[first]};

    num::block_cholesky_factor factor;
    factor.size = current.n_rows();
    factor.offsets = layout.offsets;
    factor.order = layout.order;
    factor.lower = std::move(blocks.lower);
    factor.diagonal.reserve(count);
    for (idx k = 0; k < first; ++k)
        factor.diagonal.push_back(base.diagonal[k]);
    for (idx k = 0; k + 1 < first; ++k)
        factor.lower[k] = base.lower[k];

    for (idx k = first; k < count; ++k) {
        if (k > 0) {
            const num::mat &previous = factor.diagonal[k - 1].L;
            factor.lower[k - 1] = num::transpose(num::detail::forward_substitute(
                previous, num::transpose(factor.lower[k - 1])));
            num::detail::subtract_from(
                blocks.diagonal[k],
                num::detail::product(factor.lower[k - 1], num::transpose(factor.lower[k - 1])));
        }
        auto diagonal = num::cholesky(num::assume_spd(blocks.diagonal[k]));
        if (!diagonal.success)
            throw std::runtime_error("block suffix update is not positive definite");
        factor.diagonal.push_back(std::move(diagonal));
    }
    return factor;
}

// Install a suffix-updated block factor in `current`.  Reversible subnetworks
// are first moved into their symmetric coordinates, matching initial factorization.
template <typename State>
[[nodiscard]] bool refactor_block_suffix(const Subnetwork<State> &base,
                                         Subnetwork<State> &current,
                                         view<const idx> changed,
                                         BlockSuffixInfo *info = nullptr) {
    if (!base.factor || current.levels.empty())
        return false;
    try {
        if (const auto *factor = std::get_if<num::block_lu_factor>(&*base.factor)) {
            auto updated = refactor_block_suffix(*factor, current.operator_matrix,
                                                 view<const idx>(current.levels), changed, info);
            if (!updated)
                return false;
            current.factor = FactorVariant(std::move(*updated));
            return true;
        }
        const auto *normalized = std::get_if<NormalizedCholesky>(&*base.factor);
        if (!normalized)
            return false;
        const auto *factor = std::get_if<num::block_cholesky_factor>(&normalized->factor);
        if (!factor)
            return false;
        num::vec weights = detail::stationary_weights(current.stationary.span());
        const num::spmat symmetric =
            detail::similarity_scaled(current.operator_matrix, weights.span());
        auto updated = refactor_block_suffix(*factor, symmetric, view<const idx>(current.levels),
                                             changed, info);
        if (!updated)
            return false;
        current.factor = FactorVariant(
            NormalizedCholesky{std::move(weights), std::move(*updated)});
        return true;
    } catch (const std::runtime_error &) {
        return false;
    }
}

// Build Delta = U V^T from changed rows and columns. Rank is at most 2|S|.
[[nodiscard]] inline LowRankDelta
row_column_delta(const num::spmat &base, const num::spmat &current, view<const idx> changed) {
    if (base.n_rows() != base.n_cols() || current.n_rows() != base.n_rows() ||
        current.n_cols() != base.n_cols()) {
        throw std::invalid_argument("factor update requires equal square operators");
    }

    const idx n = base.n_rows();
    const idx count = changed.size();
    const idx rank = 2 * count;
    const array<idx> slot = detail::changed_slot_lookup(n, changed);

    num::mat left(n, rank, 0.0);
    num::mat right(n, rank, 0.0);
    for (idx k = 0; k < count; ++k) {
        left(changed[k], k) = 1.0;          // U's row-indicator column
        right(changed[k], count + k) = 1.0; // V's column-indicator column
    }

    detail::accumulate_row_column_diff(current, 1.0, slot, count, left, right);
    detail::accumulate_row_column_diff(base, -1.0, slot, count, left, right);

    return {std::move(left), std::move(right)};
}

// ELSE only needs transpose solves. For A=A0+UV^T, store A0^-T V and factor
// I+U^T A0^-T V.
template <typename Solver>
struct FactorUpdate {
    const Solver &base;
    num::mat left;
    num::mat transformed;
    num::lu_result reduced;
};

template <typename Solver>
[[nodiscard]] FactorUpdate<Solver> make_factor_update(const Solver &base, const num::mat &left,
                                                      const num::mat &right) {
    using namespace num::ops;
    num::mat transformed = detail::apply_transpose_solve(base, right);
    num::lu_result reduced =
        num::lu(num::make_square(num::identity(left.cols()) + num::transpose(left) * transformed));
    if (reduced.singular)
        throw std::runtime_error("Woodbury factor is singular");
    return FactorUpdate<Solver>{base, left, std::move(transformed), std::move(reduced)};
}

template <typename Solver>
[[nodiscard]] idx rank(const FactorUpdate<Solver> &update) {
    return update.left.cols();
}

template <typename Solver>
[[nodiscard]] num::vec solve_transpose(const FactorUpdate<Solver> &update, const num::vec &rhs) {
    using namespace num::ops;
    const num::vec base_solution = detail::apply_transpose_solve(update.base, rhs);
    num::vec coefficients(update.left.cols(), 0.0);
    num::lu_solve(update.reduced, num::transpose(update.left) * base_solution, coefficients);
    return base_solution - update.transformed * coefficients;
}

template <typename Solver>
[[nodiscard]] num::mat solve_transpose(const FactorUpdate<Solver> &update, const num::mat &rhs) {
    using namespace num::ops;
    const num::mat base_solution = detail::apply_transpose_solve(update.base, rhs);
    num::mat coefficients;
    num::lu_solve(update.reduced, num::transpose(update.left) * base_solution, coefficients);
    return base_solution - update.transformed * coefficients;
}

// Relative backward residual; use it to decide when to refactor.
[[nodiscard]] inline real relative_residual(const num::spmat &matrix, const num::vec &solution,
                                            const num::vec &right_hand_side) {
    num::vec product(matrix.n_rows(), 0.0);
    num::sparse_matvec(matrix, solution, product);
    real residual = 0.0;
    real scale = 0.0;
    for (idx i = 0; i < matrix.n_rows(); ++i) {
        residual = std::max(residual, std::abs(product[i] - right_hand_side[i]));
        scale = std::max(scale, std::abs(right_hand_side[i]));
    }
    return residual / std::max(scale, 1.0);
}

} // namespace else_sim
