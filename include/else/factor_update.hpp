#pragma once

#include "else/linalg.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace else_sim {

template <typename Float = double>
struct FactorCorrection {
    // Woodbury data for A = A0 + U V^T.
    Matrix<Float> transformed_columns; // W = A0^-1 U
    Matrix<Float> right_factor;        // V^T
    LUFactor<Float> woodbury_factor;   // K = I + V^T W
};

namespace detail {

template <typename Float, typename SolveBase>
[[nodiscard]] FactorCorrection<Float> factor_correction_from_uv(Matrix<Float> U, Matrix<Float> Vt,
                                                                SolveBase solve_base) {
    const std::size_t n = U.rows();
    const std::size_t rank = U.cols();
    Matrix<Float> W(n, rank);
    std::vector<Float> rhs(n), solution;

    for (std::size_t column = 0; column < rank; ++column) {
        for (std::size_t i = 0; i < n; ++i)
            rhs[i] = U(i, column);
        solve_base(rhs, solution);
        for (std::size_t i = 0; i < n; ++i)
            W(i, column) = solution[i];
    }

    //  factorize K = I + V^T A0^-1 U
    Matrix<Float> K(rank, rank, static_cast<Float>(0));
    for (std::size_t i = 0; i < rank; ++i)
        K(i, i) = static_cast<Float>(1);

    kernel::raw::gemm_add(K.data(), Vt.data(), W.data(), rank, rank, n);

    auto woodbury_factor = factorize_lu<Float>(std::move(K));
    if (woodbury_factor.singular)
        throw std::runtime_error("singular Woodbury factor");
    return {std::move(W), std::move(Vt), std::move(woodbury_factor)};
}

} // namespace detail

template <typename Float = double, typename Index = std::size_t, typename SolveBase>
[[nodiscard]] FactorCorrection<Float>
factor_correction(const SparseMatrix<Float, Index> &base, const SparseMatrix<Float, Index> &current,
                  std::span<const Index> changed, SolveBase solve_base,
                  Float scale = static_cast<Float>(1)) {
    if (base.rows != base.cols || current.rows != base.rows || current.cols != base.cols)
        throw std::invalid_argument("factor correction requires equal square matrices");

    const Index n = base.rows;
    const Index q = static_cast<Index>(changed.size());

    // Changes confined to q rows and columns have rank at most 2q.
    const Index rank = 2 * q;
    Matrix<Float> U(n, rank, static_cast<Float>(0));
    Matrix<Float> Vt(rank, n, static_cast<Float>(0));
    std::vector<Index> changed_index(n, n);

    // E selects changed slots. Delta = E Delta[S,:] + Delta[:,S] E^T,
    // omitting changed rows from the second term to avoid counting their entries twice.
    for (Index k = 0; k < q; ++k) {
        if (changed[k] >= n)
            throw std::out_of_range("changed matrix slot is out of range");
        changed_index[changed[k]] = k;
        U(changed[k], k) = static_cast<Float>(1);
        Vt(q + k, changed[k]) = static_cast<Float>(1);
    }

    const auto add = [&](const SparseMatrix<Float, Index> &matrix, Float sign) {
        for (Index i = 0; i < n; ++i) {
            const Index row = changed_index[i];
            for (Index p = matrix.row_ptr[i]; p < matrix.row_ptr[i + 1]; ++p) {
                const Index j = matrix.col_idx[p];
                const Float value = sign * scale * matrix.values[p];
                if (row != n)
                    Vt(row, j) += value;
                else if (changed_index[j] != n)
                    U(i, q + changed_index[j]) += value;
            }
        }
    };
    add(current, static_cast<Float>(1));
    add(base, static_cast<Float>(-1));

    return detail::factor_correction_from_uv<Float>(std::move(U), std::move(Vt), solve_base);
}

// Factor A = A0 + U V^T without modifying the stored factors of A0.
template <typename Float = double, typename Index = std::size_t, typename SolveBase>
[[nodiscard]] FactorCorrection<Float>
factor_correction(const Matrix<Float> &base, const Matrix<Float> &current,
                  std::span<const Index> changed, SolveBase solve_base) {
    if (base.rows() != base.cols() || current.rows() != base.rows() ||
        current.cols() != base.cols())
        throw std::invalid_argument("factor correction requires equal square matrices");

    const Index n = static_cast<Index>(base.rows());
    const Index q = static_cast<Index>(changed.size());
    const Index rank = 2 * q;
    Matrix<Float> U(n, rank, static_cast<Float>(0));
    Matrix<Float> Vt(rank, n, static_cast<Float>(0));
    std::vector<bool> changed_row(n, false);
    for (Index slot : changed) {
        if (slot >= n)
            throw std::out_of_range("changed matrix slot is out of range");
        changed_row[slot] = true;
    }

    // Delta = E Delta[S,:] + Delta[:,S] E^T. Rows S are omitted
    // from the second term because the first term already contains them.
    for (Index k = 0; k < q; ++k) {
        const Index slot = changed[k];
        U(slot, k) = static_cast<Float>(1);
        Vt(q + k, slot) = static_cast<Float>(1);
        for (Index i = 0; i < n; ++i) {
            Vt(k, i) = current(slot, i) - base(slot, i);
            if (!changed_row[i])
                U(i, q + k) = current(i, slot) - base(i, slot);
        }
    }

    return detail::factor_correction_from_uv<Float>(std::move(U), std::move(Vt), solve_base);
}

template <typename Float = double, typename SolveBase>
void solve_correction(const FactorCorrection<Float> &correction,
                      const std::vector<Float> &right_hand_side, std::vector<Float> &solution,
                      SolveBase solve_base) {
    // Woodbury: x = x0 - W K^-1 V^T x0, where x0 = A0^-1 b.
    std::vector<Float> base_solution;
    solve_base(right_hand_side, base_solution);
    const std::size_t rank = correction.right_factor.rows();
    const std::size_t n = correction.right_factor.cols();
    std::vector<Float> reduced(rank), coefficients;
    kernel::raw::gemv(reduced.data(), correction.right_factor.data(), base_solution.data(), rank,
                      n);
    lu_solve(correction.woodbury_factor, reduced, coefficients);

    solution = std::move(base_solution);
    kernel::raw::gemv_subtract(solution.data(), correction.transformed_columns.data(),
                               coefficients.data(), n, rank);
}

template <typename Float = double, typename SolveBase>
void solve_correction(const FactorCorrection<Float> &correction,
                      const Matrix<Float> &right_hand_side, Matrix<Float> &solution,
                      SolveBase solve_base) {
    const std::size_t n = right_hand_side.rows();
    const std::size_t columns = right_hand_side.cols();
    const std::size_t rank = correction.right_factor.rows();

    // Apply the stored base factors to each column before the shared low-rank update.
    Matrix<Float> base_solution(n, columns);
    std::vector<Float> rhs(n), x;
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t i = 0; i < n; ++i)
            rhs[i] = right_hand_side(i, column);
        solve_base(rhs, x);
        for (std::size_t i = 0; i < n; ++i)
            base_solution(i, column) = x[i];
    }

    // Woodbury: X = X0 - W K^-1 V^T X0, with one compact solve for all columns.
    Matrix<Float> coefficients(rank, columns, static_cast<Float>(0));
    kernel::raw::gemm_add(coefficients.data(), correction.right_factor.data(), base_solution.data(),
                          rank, columns, n);

    Matrix<Float> woodbury_coefficients;
    lu_solve(correction.woodbury_factor, coefficients, woodbury_coefficients);

    solution = std::move(base_solution);
    kernel::raw::gemm_subtract(solution.data(), correction.transformed_columns.data(),
                               woodbury_coefficients.data(), n, columns, rank);
}

template <typename Float = double, typename Index = std::size_t>
[[nodiscard]] Float
relative_residual(const SparseMatrix<Float, Index> &matrix, const std::vector<Float> &solution,
                  const std::vector<Float> &right_hand_side, Float scale = static_cast<Float>(1)) {
    // Infinity-norm backward check, scaled so a zero right-hand side remains well defined.
    Float residual = static_cast<Float>(0);
    Float rhs_scale = static_cast<Float>(0);
    for (Index i = 0; i < matrix.rows; ++i) {
        Float value = -right_hand_side[i];
        for (Index p = matrix.row_ptr[i]; p < matrix.row_ptr[i + 1]; ++p)
            value += scale * matrix.values[p] * solution[matrix.col_idx[p]];
        residual = std::max(residual, std::abs(value));
        rhs_scale = std::max(rhs_scale, std::abs(right_hand_side[i]));
    }
    return residual / std::max(rhs_scale, static_cast<Float>(1));
}

} // namespace else_sim
