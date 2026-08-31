// Woodbury reuse: replacing k states changes at most 2k rows and columns.
#pragma once

#include "container/matrix_expr.hpp"
#include "elsex/types.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace elsex {

// Low-rank factors U and V for A = A0 + U V^T.
struct LowRankDelta {
    num::Matrix left;  ///< U
    num::Matrix right; ///< V
};

// Build Delta = U V^T from changed rows and columns. Rank is at most 2|S|.
[[nodiscard]] inline LowRankDelta row_column_delta(const num::SparseMatrix &base,
                                                   const num::SparseMatrix &current,
                                                   std::span<const idx> changed) {
    if (base.n_rows() != base.n_cols() || current.n_rows() != base.n_rows() ||
        current.n_cols() != base.n_cols()) {
        throw std::invalid_argument("factor update requires equal square operators");
    }

    const idx n = base.n_rows();
    const idx count = changed.size();
    const idx rank = 2 * count;

    std::vector<idx> slot(n, n);
    for (idx k = 0; k < count; ++k) {
        if (changed[k] >= n) {
            throw std::out_of_range("changed slot is outside the operator");
        }
        slot[changed[k]] = k;
    }

    num::Matrix left(n, rank, 0.0);
    num::Matrix right(n, rank, 0.0);
    for (idx k = 0; k < count; ++k) {
        left(changed[k], k) = 1.0;          // E
        right(changed[k], count + k) = 1.0; // E
    }

    const auto accumulate = [&](const num::SparseMatrix &matrix, real sign) {
        for (idx i = 0; i < n; ++i) {
            const idx row_slot = slot[i];
            for (auto p = matrix.row_ptr()[i]; p < matrix.row_ptr()[i + 1]; ++p) {
                const idx j = static_cast<idx>(matrix.col_idx()[p]);
                const real value = sign * matrix.values()[p];
                if (row_slot != n) {
                    right(j, row_slot) += value; // D_r(row_slot, j)
                } else if (slot[j] != n) {
                    left(i, count + slot[j]) += value; // D_c(i, slot[j])
                }
            }
        }
    };
    accumulate(current, 1.0);
    accumulate(base, -1.0);

    return {std::move(left), std::move(right)};
}

// ELSE only needs transpose solves. For A=A0+UV^T, store A0^-T V and
// factor I+U^T A0^-T V.
template <typename Solver>
class FactorUpdate {
  public:
    FactorUpdate(const Solver &base, const num::Matrix &left, const num::Matrix &right)
        : base_(base), left_(left), transformed_(apply_transpose(base, right)) {
        using namespace num::ops;
        reduced_ = num::lu(
            num::make_square(num::identity(left.cols()) + num::transpose(left) * transformed_));
        if (reduced_.singular)
            throw std::runtime_error("Woodbury factor is singular");
    }

    [[nodiscard]] idx rank() const { return left_.cols(); }

    [[nodiscard]] num::Vector solve_transpose(const num::Vector &rhs) const {
        using namespace num::ops;
        const num::Vector base_solution = apply_transpose(base_, rhs);
        num::Vector coefficients(left_.cols(), 0.0);
        num::lu_solve(reduced_, num::transpose(left_) * base_solution, coefficients);
        return base_solution - transformed_ * coefficients;
    }

    [[nodiscard]] num::Matrix solve_transpose(const num::Matrix &rhs) const {
        using namespace num::ops;
        const num::Matrix base_solution = apply_transpose(base_, rhs);
        num::Matrix coefficients;
        num::lu_solve(reduced_, num::transpose(left_) * base_solution, coefficients);
        return base_solution - transformed_ * coefficients;
    }

  private:
    template <typename RightHandSide>
    [[nodiscard]] static auto apply_transpose(const Solver &solver, const RightHandSide &rhs) {
        if constexpr (requires { solver.solve_transpose(rhs); })
            return solver.solve_transpose(rhs);
        else
            return num::solve_transpose(solver, rhs);
    }

    const Solver &base_;
    num::Matrix left_;
    num::Matrix transformed_;
    num::LUResult reduced_;
};

/// Relative backward residual; use it to decide when to refactor.
[[nodiscard]] inline real relative_residual(const num::SparseMatrix &matrix,
                                            const num::Vector &solution,
                                            const num::Vector &right_hand_side) {
    real residual = 0.0;
    real scale = 0.0;
    for (idx i = 0; i < matrix.n_rows(); ++i) {
        real value = -right_hand_side[i];
        for (auto p = matrix.row_ptr()[i]; p < matrix.row_ptr()[i + 1]; ++p) {
            value += matrix.values()[p] * solution[static_cast<idx>(matrix.col_idx()[p])];
        }
        residual = std::max(residual, std::abs(value));
        scale = std::max(scale, std::abs(right_hand_side[i]));
    }
    return residual / std::max(scale, 1.0);
}

} // namespace elsex
