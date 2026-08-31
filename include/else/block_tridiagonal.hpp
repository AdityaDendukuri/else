#pragma once

#include "else/linalg.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace else_sim {

template <typename Index = std::size_t, typename State, typename LevelFunction>
[[nodiscard]] std::vector<Index> make_block_levels(const std::vector<State> &states,
                                                   const LevelFunction &level) {
    if (states.empty())
        return {};

    std::vector<long long> labels(states.size());
    for (std::size_t i = 0; i < states.size(); ++i)
        labels[i] = static_cast<long long>(level(states[i]));

    // Compress arbitrary level labels into consecutive block indices.
    std::vector<long long> unique = labels;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    if (unique.size() < 2)
        return {};

    std::vector<Index> levels(states.size());
    for (std::size_t i = 0; i < states.size(); ++i)
        levels[i] = static_cast<Index>(std::lower_bound(unique.begin(), unique.end(), labels[i]) -
                                       unique.begin());
    return levels;
}

template <typename Float = double, typename Index = std::size_t>
struct BlockLUFactor {
    // Diagonal entries are Schur factors; lower entries are block elimination multipliers.
    Index size = 0;
    std::vector<Index> offsets;
    std::vector<Index> order;
    std::vector<LUFactor<Float>> diagonal;
    std::vector<Matrix<Float>> upper;
    std::vector<Matrix<Float>> lower;
};

template <typename Float = double, typename Index = std::size_t>
struct BlockCholeskyFactor {
    // A block bidiagonal Cholesky factor stored as diagonal and lower blocks.
    Index size = 0;
    std::vector<Index> offsets;
    std::vector<Index> order;
    std::vector<CholeskyFactor<Float>> diagonal;
    std::vector<Matrix<Float>> lower;
};

template <typename Factor>
[[nodiscard]] auto block_size(const Factor &factor, std::size_t block) {
    return factor.offsets[block + 1] - factor.offsets[block];
}

template <typename Float, typename Factor>
[[nodiscard]] std::vector<Matrix<Float>> gather_block_rows(const Factor &factor,
                                                           const Matrix<Float> &B) {
    std::vector<Matrix<Float>> blocks(factor.diagonal.size());
    for (std::size_t block = 0; block < factor.diagonal.size(); ++block) {
        blocks[block] = Matrix<Float>(block_size(factor, block), B.cols());
        for (std::size_t row = 0; row < block_size(factor, block); ++row)
            for (std::size_t column = 0; column < B.cols(); ++column)
                blocks[block](row, column) = B(factor.order[factor.offsets[block] + row], column);
    }
    return blocks;
}

template <typename Float, typename Factor>
void scatter_block_rows(const Factor &factor, const std::vector<Matrix<Float>> &blocks,
                        Matrix<Float> &X) {
    X = Matrix<Float>(factor.size, blocks.front().cols());
    for (std::size_t block = 0; block < factor.diagonal.size(); ++block)
        for (std::size_t row = 0; row < block_size(factor, block); ++row)
            for (std::size_t column = 0; column < X.cols(); ++column)
                X(factor.order[factor.offsets[block] + row], column) = blocks[block](row, column);
}

template <typename Index>
Index build_block_order(const std::vector<Index> &levels, std::vector<Index> &offsets,
                        std::vector<Index> &order, std::vector<Index> &inverse_order) {
    // Group states by level while retaining maps to and from the original order.
    const Index blocks = *std::max_element(levels.begin(), levels.end()) + 1;
    offsets.assign(blocks + 1, 0);
    for (Index level : levels)
        ++offsets[level + 1];
    for (Index block = 0; block < blocks; ++block)
        offsets[block + 1] += offsets[block];
    for (Index block = 0; block < blocks; ++block)
        if (offsets[block] == offsets[block + 1])
            throw std::invalid_argument("empty block level");

    order.resize(levels.size());
    inverse_order.resize(levels.size());
    std::vector<Index> next = offsets;
    for (Index old = 0; old < levels.size(); ++old) {
        const Index reordered = next[levels[old]]++;
        order[reordered] = old;
        inverse_order[old] = reordered;
    }
    return blocks;
}

template <typename Float, typename Index>
void validate_block_matrix(const SparseMatrix<Float, Index> &A, const std::vector<Index> &levels) {
    if (A.rows != A.cols || levels.size() != A.rows)
        throw std::invalid_argument(
            "block factorization requires a square matrix and one level per state");
}

template <typename Float, typename Index>
void allocate_lu_blocks(BlockLUFactor<Float, Index> &factor, std::vector<Matrix<Float>> &diagonal) {
    const Index blocks = diagonal.size();
    factor.upper.resize(blocks - 1);
    factor.lower.resize(blocks - 1);
    for (Index block = 0; block < blocks; ++block) {
        diagonal[block] = Matrix<Float>(block_size(factor, block), block_size(factor, block));
        if (block + 1 < blocks) {
            factor.upper[block] =
                Matrix<Float>(block_size(factor, block), block_size(factor, block + 1));
            factor.lower[block] =
                Matrix<Float>(block_size(factor, block + 1), block_size(factor, block));
        }
    }
}

template <typename Float, typename Index>
void scatter_lu_blocks(const SparseMatrix<Float, Index> &A, const std::vector<Index> &levels,
                       const std::vector<Index> &inverse_order, Float scale,
                       BlockLUFactor<Float, Index> &factor, std::vector<Matrix<Float>> &diagonal) {
    // Copy sparse entries into the diagonal and two neighboring block bands.
    for (Index old_row = 0; old_row < factor.size; ++old_row) {
        const Index row_level = levels[old_row];
        const Index row = inverse_order[old_row] - factor.offsets[row_level];
        for (Index p = A.row_ptr[old_row]; p < A.row_ptr[old_row + 1]; ++p) {
            const Index old_col = A.col_idx[p];
            const Index col_level = levels[old_col];
            const Index col = inverse_order[old_col] - factor.offsets[col_level];
            const Float value = scale * A.values[p];

            if (row_level == col_level)
                diagonal[row_level](row, col) += value;
            else if (col_level == row_level + 1)
                factor.upper[row_level](row, col) += value;
            else if (row_level == col_level + 1)
                factor.lower[col_level](row, col) += value;
            else
                throw std::invalid_argument("supplied ordering is not block tridiagonal");
        }
    }
}

template <typename Float, typename Index>
void factor_lu_blocks(BlockLUFactor<Float, Index> &factor, std::vector<Matrix<Float>> diagonal) {
    factor.diagonal.clear();
    factor.diagonal.reserve(diagonal.size());
    for (Index block = 0; block < diagonal.size(); ++block) {
        // Eliminate the previous block, then factor the resulting Schur complement.
        Matrix<Float> schur = std::move(diagonal[block]);
        if (block > 0) {
            Matrix<Float> &lower = factor.lower[block - 1];
            // L_k <- C_k S_{k-1}^{-1}; each stored row is a right-hand side.
            kernel::raw::lu_solve_right_multiple(lower.data(), factor.diagonal[block - 1].LU.data(),
                                                 lower.rows(), lower.cols());

            const Matrix<Float> &upper = factor.upper[block - 1];
            kernel::raw::gemm_subtract(schur.data(), lower.data(), upper.data(), schur.rows(),
                                       schur.cols(), lower.cols());
        }

        auto diagonal_factor = factorize_lu<Float>(std::move(schur));
        if (diagonal_factor.singular)
            throw std::runtime_error("block Thomas Schur complement is singular");
        factor.diagonal.push_back(std::move(diagonal_factor));
    }
}

template <typename Float = double, typename Index = std::size_t>
[[nodiscard]] BlockLUFactor<Float, Index> factorize_block_lu(const SparseMatrix<Float, Index> &A,
                                                             const std::vector<Index> &levels,
                                                             Float scale = static_cast<Float>(1)) {
    validate_block_matrix(A, levels);
    BlockLUFactor<Float, Index> factor;
    factor.size = A.rows;
    if (factor.size == 0)
        return factor;

    std::vector<Index> inverse_order;
    const Index blocks = build_block_order(levels, factor.offsets, factor.order, inverse_order);
    std::vector<Matrix<Float>> diagonal(blocks);
    allocate_lu_blocks(factor, diagonal);
    scatter_lu_blocks(A, levels, inverse_order, scale, factor, diagonal);
    factor_lu_blocks(factor, std::move(diagonal));
    return factor;
}

template <typename Float>
void subtract_product(const Matrix<Float> &A, const Matrix<Float> &B, Matrix<Float> &C) {
    kernel::raw::gemm_subtract(C.data(), A.data(), B.data(), A.rows(), B.cols(), A.cols());
}

template <typename Float, typename Index>
void block_solve(const BlockLUFactor<Float, Index> &factor, const Matrix<Float> &B,
                 Matrix<Float> &X) {
    if (B.rows() != factor.size)
        throw std::invalid_argument("block Thomas right-hand side size mismatch");

    // Forward recurrence y_k = b_k - L_k y_{k-1}, then solve the Schur blocks backward.
    std::vector<Matrix<Float>> y = gather_block_rows(factor, B);
    std::vector<Matrix<Float>> solution(factor.diagonal.size());
    for (Index block = 0; block < factor.diagonal.size(); ++block) {
        if (block > 0)
            subtract_product(factor.lower[block - 1], y[block - 1], y[block]);
    }

    for (Index block = static_cast<Index>(factor.diagonal.size()); block-- > 0;) {
        if (block + 1 < factor.diagonal.size())
            subtract_product(factor.upper[block], solution[block + 1], y[block]);
        lu_solve(factor.diagonal[block], y[block], solution[block]);
    }

    scatter_block_rows(factor, solution, X);
}

template <typename Float, typename Index>
void block_solve(const BlockLUFactor<Float, Index> &factor, const std::vector<Float> &b,
                 std::vector<Float> &x) {
    if (b.size() != factor.size)
        throw std::invalid_argument("block Thomas right-hand side size mismatch");
    Matrix<Float> B(factor.size, 1), X;
    for (Index row = 0; row < factor.size; ++row)
        B(row, 0) = b[row];
    block_solve(factor, B, X);
    x.resize(factor.size);
    for (Index row = 0; row < factor.size; ++row)
        x[row] = X(row, 0);
}

template <typename Float, typename Index>
void allocate_cholesky_blocks(BlockCholeskyFactor<Float, Index> &factor,
                              std::vector<Matrix<Float>> &diagonal) {
    const Index blocks = diagonal.size();
    factor.lower.resize(blocks - 1);
    for (Index block = 0; block < blocks; ++block) {
        diagonal[block] = Matrix<Float>(block_size(factor, block), block_size(factor, block));
        if (block + 1 < blocks)
            factor.lower[block] =
                Matrix<Float>(block_size(factor, block + 1), block_size(factor, block));
    }
}

template <typename Float, typename Index>
void scatter_cholesky_blocks(const SparseMatrix<Float, Index> &A, const std::vector<Index> &levels,
                             const std::vector<Index> &inverse_order,
                             BlockCholeskyFactor<Float, Index> &factor,
                             std::vector<Matrix<Float>> &diagonal) {
    // Store one triangle of the diagonal and lower block bands.
    for (Index old_row = 0; old_row < factor.size; ++old_row) {
        const Index row_level = levels[old_row];
        const Index row = inverse_order[old_row] - factor.offsets[row_level];
        for (Index p = A.row_ptr[old_row]; p < A.row_ptr[old_row + 1]; ++p) {
            const Index old_col = A.col_idx[p];
            const Index col_level = levels[old_col];
            const Index col = inverse_order[old_col] - factor.offsets[col_level];

            if (row_level == col_level)
                diagonal[row_level](row, col) += A.values[p];
            else if (row_level == col_level + 1)
                factor.lower[col_level](row, col) += A.values[p];
            else if (col_level != row_level + 1)
                throw std::invalid_argument("supplied ordering is not block tridiagonal");
        }
    }
}

template <typename Float, typename Index>
void factor_cholesky_blocks(BlockCholeskyFactor<Float, Index> &factor,
                            std::vector<Matrix<Float>> diagonal) {
    factor.diagonal.reserve(diagonal.size());
    for (Index block = 0; block < diagonal.size(); ++block) {
        // Apply the previous block update, then factor the next diagonal block.
        Matrix<Float> schur = std::move(diagonal[block]);
        if (block > 0) {
            Matrix<Float> &lower = factor.lower[block - 1];
            const Matrix<Float> &previous = factor.diagonal[block - 1].L;
            // L_k <- C_k L_{k-1}^{-T} before the symmetric Schur update.
            kernel::raw::solve_right_lower_transpose_multiple(lower.data(), previous.data(),
                                                              lower.rows(), lower.cols());
            kernel::raw::syrk_lower_subtract(schur.data(), lower.data(), lower.rows(),
                                             lower.cols());
        }

        auto diagonal_factor = factorize_cholesky<Float>(schur);
        if (!diagonal_factor.success)
            throw std::runtime_error("block Cholesky Schur complement is not positive definite");
        factor.diagonal.push_back(std::move(diagonal_factor));
    }
}

template <typename Float = double, typename Index = std::size_t>
[[nodiscard]] BlockCholeskyFactor<Float, Index>
factorize_block_cholesky(const SparseMatrix<Float, Index> &A, const std::vector<Index> &levels) {
    validate_block_matrix(A, levels);
    BlockCholeskyFactor<Float, Index> factor;
    factor.size = A.rows;
    if (factor.size == 0)
        return factor;

    std::vector<Index> inverse_order;
    const Index blocks = build_block_order(levels, factor.offsets, factor.order, inverse_order);
    std::vector<Matrix<Float>> diagonal(blocks);
    allocate_cholesky_blocks(factor, diagonal);
    scatter_cholesky_blocks(A, levels, inverse_order, factor, diagonal);
    factor_cholesky_blocks(factor, std::move(diagonal));
    return factor;
}

template <typename Float>
void subtract_transpose_product(const Matrix<Float> &A, const Matrix<Float> &B, Matrix<Float> &C) {
    kernel::raw::gemm_transpose_left_subtract(C.data(), A.data(), B.data(), A.rows(), A.cols(),
                                              B.cols());
}

template <typename Float, typename Index>
void block_solve(const BlockCholeskyFactor<Float, Index> &factor, const Matrix<Float> &B,
                 Matrix<Float> &X) {
    if (B.rows() != factor.size)
        throw std::invalid_argument("block Cholesky right-hand side size mismatch");

    // Forward substitution through the block bidiagonal factor, then through its transpose.
    std::vector<Matrix<Float>> y = gather_block_rows(factor, B);
    std::vector<Matrix<Float>> x(factor.diagonal.size());
    for (Index block = 0; block < factor.diagonal.size(); ++block) {
        if (block > 0)
            subtract_product(factor.lower[block - 1], y[block - 1], y[block]);
        kernel::raw::solve_lower_multiple(y[block].data(), factor.diagonal[block].L.data(),
                                          y[block].rows(), y[block].cols());
    }

    for (Index block = static_cast<Index>(factor.diagonal.size()); block-- > 0;) {
        x[block] = std::move(y[block]);
        if (block + 1 < factor.diagonal.size())
            subtract_transpose_product(factor.lower[block], x[block + 1], x[block]);
        kernel::raw::solve_lower_transpose_multiple(
            x[block].data(), factor.diagonal[block].L.data(), x[block].rows(), x[block].cols());
    }

    scatter_block_rows(factor, x, X);
}

template <typename Float, typename Index>
void block_solve(const BlockCholeskyFactor<Float, Index> &factor, const std::vector<Float> &b,
                 std::vector<Float> &x) {
    if (b.size() != factor.size)
        throw std::invalid_argument("block Cholesky right-hand side size mismatch");
    Matrix<Float> B(factor.size, 1), X;
    for (Index row = 0; row < factor.size; ++row)
        B(row, 0) = b[row];
    block_solve(factor, B, X);
    x.resize(factor.size);
    for (Index row = 0; row < factor.size; ++row)
        x[row] = X(row, 0);
}

} // namespace else_sim
