// The Woodbury correction: replacing the states in k slots changes at most
// 2k rows and columns of R_bar, so solves with the retained factor of the
// old matrix are corrected through a rank-2k system, and cached
// current-state rows are updated by the same identity.
#pragma once

#include "subsweep/linear/factorization.hpp"
#include "blas/matrix_ops.hpp"
#include "container/matrix_expr.hpp"
#include "kernel/factor.hpp"
#include "linear/factorization/lu.hpp"
#include <optional>
#include <stdexcept>
#include <utility>

namespace subsweep {

// The Woodbury correction. Replacing the states in k slots changes at most 2k
// rows and columns of R_bar, so R_bar_r = R_bar_b + P Q^T with rank p <= 2k.

struct low_rank_delta {
    num::mat left;  // P
    num::mat right; // Q
};

[[nodiscard]] inline low_rank_delta
row_column_delta(const num::spmat &base, const num::spmat &current, view<const idx> changed) {
    const idx n = base.n_rows();
    if (base.n_cols() != n || current.n_rows() != n || current.n_cols() != n)
        throw std::invalid_argument("a low-rank delta needs equal square matrices");
    const idx count = changed.size();
    array<idx> slot(n, n);
    for (idx k = 0; k < count; ++k) {
        if (changed[k] >= n)
            throw std::out_of_range("changed slot is outside the matrix");
        slot[changed[k]] = k;
    }
    num::mat left(n, 2 * count, 0.0);
    num::mat right(n, 2 * count, 0.0);
    for (idx k = 0; k < count; ++k) {
        left(changed[k], k) = 1.0;
        right(changed[k], count + k) = 1.0;
    }
    // A changed row i enters Q's column for its slot; a changed column j
    // whose row is unchanged enters P's column for its slot.
    const auto accumulate = [&](const num::spmat &matrix, real sign) {
        for (idx i = 0; i < n; ++i)
            for (auto p = matrix.row_ptr()[i]; p < matrix.row_ptr()[i + 1]; ++p) {
                const idx j = static_cast<idx>(matrix.col_idx()[p]);
                const real value = sign * matrix.values()[p];
                if (slot[i] != n)
                    right(j, slot[i]) += value;
                else if (slot[j] != n)
                    left(i, count + slot[j]) += value;
            }
    };
    accumulate(current, 1.0);
    accumulate(base, -1.0);
    return {std::move(left), std::move(right)};
}

// Solves with R_bar_b + P Q^T through the retained factor of R_bar_b:
//   x = y - W G^-1 P^T y,  y = Z_b^T b,  W = Z_b^T Q,  G = I + P^T W,
// and the mirror identity with Z_b P for forward solves. Setup is one
// rank-p transpose solve; Z_b P and Z_b^T W are computed on demand.
class woodbury {
  public:
    woodbury(const factorization &base, low_rank_delta delta)
        : base_(&base), left_(std::move(delta.left)), right_(std::move(delta.right)),
          transpose_right_(base.solve_transpose(right_)) {
        using namespace num::ops;
        num::mat reduced_transpose = num::identity(rank()) + num::transpose(left_) * transpose_right_;
        reduced_transpose_ = num::lu(num::assume_square(reduced_transpose));
        reduced_ = num::lu(num::assume_square(num::transpose(reduced_transpose)));
        if (reduced_transpose_.singular || reduced_.singular)
            throw std::runtime_error("the Woodbury correction is singular");
    }

    [[nodiscard]] idx rank() const { return left_.cols(); }
    [[nodiscard]] idx size() const { return left_.rows(); }
    [[nodiscard]] const num::mat &left() const { return left_; }
    [[nodiscard]] const num::mat &transpose_right() const { return transpose_right_; }
    [[nodiscard]] const num::mat &inverse_left() const {
        if (!inverse_left_)
            inverse_left_ = base_->solve(left_);
        return *inverse_left_;
    }
    [[nodiscard]] const num::mat &transpose_right_squared() const {
        if (!transpose_right_squared_)
            transpose_right_squared_ = base_->solve_transpose(transpose_right_);
        return *transpose_right_squared_;
    }

    // Y <- Y G^-1 for a block of rows Y: each row solves G^T x = y in place.
    void right_solve(num::mat &rows, num::vec &scratch) const {
        const idx p = rank();
        if (scratch.size() != p)
            scratch = num::vec(p, 0.0);
        for (idx i = 0; i < rows.rows(); ++i) {
            real *row = rows.data() + i * p;
            for (idx j = 0; j < p; ++j)
                scratch[j] = row[j];
            num::kernel::lu_solve(row, reduced_transpose_.LU.data(), reduced_transpose_.piv.data(),
                                  scratch.data(), p);
        }
    }

    [[nodiscard]] num::mat right_solve(const num::mat &rows) const {
        num::mat solution = rows;
        num::vec scratch;
        right_solve(solution, scratch);
        return solution;
    }

    template <typename RightHandSide>
    [[nodiscard]] RightHandSide solve_transpose(const RightHandSide &rhs) const {
        using namespace num::ops;
        const RightHandSide y = base_->solve_transpose(rhs);
        RightHandSide coefficients;
        num::lu_solve(reduced_transpose_, num::transpose(left_) * y, coefficients);
        return y - transpose_right_ * coefficients;
    }

    template <typename RightHandSide>
    [[nodiscard]] RightHandSide solve(const RightHandSide &rhs) const {
        using namespace num::ops;
        const RightHandSide y = base_->solve(rhs);
        RightHandSide coefficients;
        num::lu_solve(reduced_, num::transpose(right_) * y, coefficients);
        return y - inverse_left() * coefficients;
    }

    // diag(Z_r) = diag(Z_b) - diag(Z_b P G^-1 Q^T Z_b).
    [[nodiscard]] num::vec diagonal(view<const real> base_diagonal) const {
        if (base_diagonal.size() != size())
            throw std::invalid_argument("the base diagonal has the wrong size");
        num::mat coefficients;
        num::lu_solve(reduced_, num::transpose(transpose_right_), coefficients);
        const num::mat &zp = inverse_left();
        num::vec diagonal(size(), 0.0);
        for (idx i = 0; i < size(); ++i) {
            real correction = 0.0;
            for (idx column = 0; column < rank(); ++column)
                correction += zp(i, column) * coefficients(column, i);
            diagonal[i] = base_diagonal[i] - correction;
        }
        return diagonal;
    }

  private:
    const factorization *base_;
    num::mat left_, right_, transpose_right_;
    num::lu_result reduced_, reduced_transpose_;
    mutable std::optional<num::mat> inverse_left_, transpose_right_squared_;
};

// Scratch for the cached-row update: two d x p blocks and a rank-p vector,
// resized on demand and reused across sweeps.
struct row_workspace {
    num::mat c, e;
    num::vec scratch;
};

// The cached update of current-state rows U_b = E Z_b, V_b = E Z_b^2 to the
// corrected matrix, in place: with B = Z_b^T Q, D = Z_b^T Z_b^T Q, G = I + P^T B,
//   C = U_b P G^-1,  U = U_b - C B^T,  V = V_b - C D^T,  V -= (V P G^-1) B^T.
// `u` and `v` hold U_b and V_b on entry and U and V on exit.
inline void update_rows(const woodbury &wb, num::mat &u, num::mat &v, row_workspace &work) {
    const idx d = u.rows(), p = wb.rank();
    if (work.c.rows() != d || work.c.cols() != p) {
        work.c = num::mat(d, p, 0.0);
        work.e = num::mat(d, p, 0.0);
    }
    num::blas::gemm(1.0, u, false, wb.left(), false, 0.0, work.c);
    wb.right_solve(work.c, work.scratch);
    num::blas::gemm(-1.0, work.c, false, wb.transpose_right(), true, 1.0, u);
    num::blas::gemm(-1.0, work.c, false, wb.transpose_right_squared(), true, 1.0, v);
    num::blas::gemm(1.0, v, false, wb.left(), false, 0.0, work.e);
    wb.right_solve(work.e, work.scratch);
    num::blas::gemm(-1.0, work.e, false, wb.transpose_right(), true, 1.0, v);
}

// max_i |(M x - b)_i| / max_i |b_i|.
[[nodiscard]] inline real relative_residual(const num::spmat &matrix, const num::vec &solution,
                                            const num::vec &rhs) {
    num::vec product(matrix.n_rows(), 0.0);
    num::sparse_matvec(matrix, solution, product);
    real residual = 0.0, scale = 0.0;
    for (idx i = 0; i < matrix.n_rows(); ++i) {
        residual = std::max(residual, std::abs(product[i] - rhs[i]));
        scale = std::max(scale, std::abs(rhs[i]));
    }
    return residual / std::max(scale, 1.0);
}

} // namespace subsweep
