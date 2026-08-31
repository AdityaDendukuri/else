#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace else_sim {

namespace kernel::raw {

// Accumulate C <- C + alpha A B for row-major dense matrices.
template <typename T>
void gemm_update(T *C, const T *A, const T *B, std::size_t rows, std::size_t columns,
                 std::size_t inner, T alpha) {
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t k = 0; k < inner; ++k)
            for (std::size_t j = 0; j < columns; ++j)
                C[i * columns + j] += alpha * A[i * inner + k] * B[k * columns + j];
}

template <typename T>
void gemm_add(T *C, const T *A, const T *B, std::size_t rows, std::size_t columns,
              std::size_t inner) {
    gemm_update(C, A, B, rows, columns, inner, T(1));
}

template <typename T>
void gemm_subtract(T *C, const T *A, const T *B, std::size_t rows, std::size_t columns,
                   std::size_t inner) {
    gemm_update(C, A, B, rows, columns, inner, T(-1));
}

// Accumulate y <- y + alpha A x for a row-major dense matrix.
template <typename T>
void gemv_update(T *y, const T *A, const T *x, std::size_t rows, std::size_t columns, T alpha) {
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < columns; ++j)
            y[i] += alpha * A[i * columns + j] * x[j];
}

template <typename T>
void gemv(T *y, const T *A, const T *x, std::size_t rows, std::size_t columns) {
    // Overwrite is zero followed by the canonical accumulation kernel.
    std::fill_n(y, rows, T(0));
    gemv_update(y, A, x, rows, columns, T(1));
}

template <typename T>
void gemv_subtract(T *y, const T *A, const T *x, std::size_t rows, std::size_t columns) {
    gemv_update(y, A, x, rows, columns, T(-1));
}

// Scale every row of X by d_i, or by 1/d_i when inverse is true.
template <typename T>
void scale_rows(T *X, const T *d, std::size_t rows, std::size_t columns, bool inverse) {
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < columns; ++j)
            if (inverse)
                X[i * columns + j] /= d[i];
            else
                X[i * columns + j] *= d[i];
}

// C <- C - A^T B for row-major dense matrices.
template <typename T>
void gemm_transpose_left_subtract(T *C, const T *A, const T *B, std::size_t inner, std::size_t rows,
                                  std::size_t columns) {
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t k = 0; k < inner; ++k)
            for (std::size_t j = 0; j < columns; ++j)
                C[i * columns + j] -= A[k * rows + i] * B[k * columns + j];
}

// Update the lower triangle of C <- C - A A^T.
template <typename T>
void syrk_lower_subtract(T *C, const T *A, std::size_t n, std::size_t inner) {
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            for (std::size_t k = 0; k < inner; ++k)
                C[i * n + j] -= A[i * inner + k] * A[j * inner + k];
}

// Factor A = L U in place without pivoting.
template <typename T>
bool lu_factor(T *A, std::size_t n) {
    using Real = decltype(std::abs(std::declval<T>()));
    constexpr Real tolerance = static_cast<Real>(1e-15);
    bool nonsingular = true;
    for (std::size_t k = 0; k < n; ++k) {
        if (std::abs(A[k * n + k]) < tolerance) {
            nonsingular = false;
            continue;
        }
        const T inverse_pivot = static_cast<T>(1) / A[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            A[i * n + k] *= inverse_pivot;
            const T multiplier = A[i * n + k];
            for (std::size_t j = k + 1; j < n; ++j)
                A[i * n + j] -= multiplier * A[k * n + j];
        }
    }
    return nonsingular;
}

// Forward substitution with the unit lower triangle packed below LU's diagonal.
template <typename T>
void solve_unit_lower_multiple(T *X, const T *LU, std::size_t n, std::size_t columns) {
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            for (std::size_t c = 0; c < columns; ++c)
                X[i * columns + c] -= LU[i * n + j] * X[j * columns + c];
}

// Back substitution with the upper triangle packed in LU.
template <typename T>
void solve_upper_multiple(T *X, const T *U, std::size_t n, std::size_t columns) {
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + 1; j < n; ++j)
            for (std::size_t c = 0; c < columns; ++c)
                X[i * columns + c] -= U[i * n + j] * X[j * columns + c];
        for (std::size_t c = 0; c < columns; ++c)
            X[i * columns + c] /= U[i * n + i];
    }
}

// Solve L U X = B in place from packed LU factors.
template <typename T>
void lu_solve_multiple(T *X, const T *LU, std::size_t n, std::size_t columns) {
    solve_unit_lower_multiple(X, LU, n, columns);
    solve_upper_multiple(X, LU, n, columns);
}

template <typename T>
void lu_solve(T *x, const T *LU, std::size_t n) {
    // A vector is a one-column block right-hand side.
    lu_solve_multiple(x, LU, n, 1);
}

// Solve U^T L^T x = b in place.
template <typename T>
void lu_solve_transpose(T *x, const T *LU, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            x[i] -= LU[j * n + i] * x[j];
        x[i] /= LU[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;)
        for (std::size_t j = i + 1; j < n; ++j)
            x[i] -= LU[j * n + i] * x[j];
}

// Solve X A = B in place for rows of B, where A is stored as packed LU.
template <typename T>
void lu_solve_right_multiple(T *X, const T *LU, std::size_t rows, std::size_t n) {
    for (std::size_t row = 0; row < rows; ++row)
        lu_solve_transpose(X + row * n, LU, n);
}

// Factor A = L L^T into the lower triangle of L.
template <typename T>
bool cholesky(T *L, const T *A, std::size_t n) {
    std::fill_n(L, n * n, T(0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            T sum = A[i * n + j];
            for (std::size_t k = 0; k < j; ++k)
                sum -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (!(sum > T(0)))
                    return false;
                L[i * n + j] = std::sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    return true;
}

// Solve L X = B in place.
template <typename T>
void solve_lower_multiple(T *X, const T *L, std::size_t n, std::size_t columns) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            for (std::size_t c = 0; c < columns; ++c)
                X[i * columns + c] -= L[i * n + j] * X[j * columns + c];
        for (std::size_t c = 0; c < columns; ++c)
            X[i * columns + c] /= L[i * n + i];
    }
}

// Solve L^T X = B in place.
template <typename T>
void solve_lower_transpose_multiple(T *X, const T *L, std::size_t n, std::size_t columns) {
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + 1; j < n; ++j)
            for (std::size_t c = 0; c < columns; ++c)
                X[i * columns + c] -= L[j * n + i] * X[j * columns + c];
        for (std::size_t c = 0; c < columns; ++c)
            X[i * columns + c] /= L[i * n + i];
    }
}

// Solve X L^T = B in place for several rows of B.
template <typename T>
void solve_right_lower_transpose_multiple(T *X, const T *L, std::size_t rows, std::size_t n) {
    for (std::size_t row = 0; row < rows; ++row)
        solve_lower_multiple(X + row * n, L, n, 1);
}

// Solve L L^T X = B in place.
template <typename T>
void cholesky_solve_multiple(T *X, const T *L, std::size_t n, std::size_t columns) {
    solve_lower_multiple(X, L, n, columns);
    solve_lower_transpose_multiple(X, L, n, columns);
}

template <typename T>
void cholesky_solve(T *x, const T *L, std::size_t n) {
    // Keep the scalar path identical to the block solve.
    cholesky_solve_multiple(x, L, n, 1);
}

} // namespace kernel::raw

/// Owning row-major storage used only by the small dense kernels in this library.
template <typename Float = double>
class Matrix {
  public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(std::size_t rows, std::size_t cols, Float fill = static_cast<Float>(0))
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] Float *data() noexcept { return data_.data(); }
    [[nodiscard]] const Float *data() const noexcept { return data_.data(); }

    [[nodiscard]] Float &operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }
    [[nodiscard]] const Float &operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

  private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<Float> data_;
};

/// Owning compressed-row storage for restricted generators.
template <typename Float = double, typename Index = std::size_t>
struct SparseMatrix {
    Index rows = 0;
    Index cols = 0;
    std::vector<Index> row_ptr;
    std::vector<Index> col_idx;
    std::vector<Float> values;

    [[nodiscard]] Float operator()(Index r, Index c) const noexcept {
        if (r >= rows || row_ptr.empty())
            return static_cast<Float>(0);
        for (Index k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            if (col_idx[k] == c)
                return values[k];
        }
        return static_cast<Float>(0);
    }

    static SparseMatrix from_triplets(Index n_rows, Index n_cols, const std::vector<Index> &r,
                                      const std::vector<Index> &c, const std::vector<Float> &v) {
        // Count entries per row, prefix-sum the offsets, then scatter the triplets.
        SparseMatrix mat;
        mat.rows = n_rows;
        mat.cols = n_cols;
        mat.row_ptr.assign(n_rows + 1, 0);

        for (Index row : r) {
            if (row < n_rows)
                mat.row_ptr[row + 1]++;
        }
        for (Index i = 0; i < n_rows; ++i) {
            mat.row_ptr[i + 1] += mat.row_ptr[i];
        }

        mat.col_idx.resize(r.size());
        mat.values.resize(v.size());
        std::vector<Index> current_ptr = mat.row_ptr;

        for (std::size_t k = 0; k < r.size(); ++k) {
            Index row = r[k];
            Index dest = current_ptr[row]++;
            mat.col_idx[dest] = c[k];
            mat.values[dest] = v[k];
        }
        return mat;
    }
};

/// @brief Packed no-pivot dense LU factorization result.
template <typename Float = double>
struct LUFactor {
    Matrix<Float> LU;
    bool singular = false;
};

/// @brief Compute a no-pivot dense LU factorization.
template <typename Float = double>
inline LUFactor<Float> factorize_lu(Matrix<Float> A) {
    LUFactor<Float> result;
    result.LU = std::move(A);
    result.singular = !kernel::raw::lu_factor(result.LU.data(), result.LU.rows());
    return result;
}

/// @brief Solve A x = b from a precomputed LU factorization.
template <typename Float = double>
inline void lu_solve(const LUFactor<Float> &f, const std::vector<Float> &b, std::vector<Float> &x) {
    x = b;
    kernel::raw::lu_solve(x.data(), f.LU.data(), f.LU.rows());
}

template <typename Float = double>
inline void lu_solve(const LUFactor<Float> &f, const Matrix<Float> &B, Matrix<Float> &X) {
    if (B.rows() != f.LU.rows())
        throw std::invalid_argument("LU block right-hand side size mismatch");
    X = B;
    kernel::raw::lu_solve_multiple(X.data(), f.LU.data(), X.rows(), X.cols());
}

/// @brief Solve A^T x = b from a precomputed LU factorization.
template <typename Float = double>
inline void lu_solve_transpose(const LUFactor<Float> &f, const std::vector<Float> &b,
                               std::vector<Float> &x) {
    x = b;
    kernel::raw::lu_solve_transpose(x.data(), f.LU.data(), f.LU.rows());
}

/// @brief Cholesky factorization result for symmetric positive definite matrices.
template <typename Float = double>
struct CholeskyFactor {
    Matrix<Float> L;
    bool success = true;
};

/// @brief Compute LL^T Cholesky factorization.
template <typename Float = double>
inline CholeskyFactor<Float> factorize_cholesky(const Matrix<Float> &A) {
    CholeskyFactor<Float> result;
    result.L = Matrix<Float>(A.rows(), A.cols());
    result.success = kernel::raw::cholesky(result.L.data(), A.data(), A.rows());
    return result;
}

/// @brief Solve LL^T x = b.
template <typename Float = double>
inline void cholesky_solve(const CholeskyFactor<Float> &f, const std::vector<Float> &b,
                           std::vector<Float> &x) {
    x = b;
    kernel::raw::cholesky_solve(x.data(), f.L.data(), f.L.rows());
}

template <typename Float = double>
inline void cholesky_solve(const CholeskyFactor<Float> &f, const Matrix<Float> &B,
                           Matrix<Float> &X) {
    if (B.rows() != f.L.rows())
        throw std::invalid_argument("Cholesky block right-hand side size mismatch");
    X = B;
    kernel::raw::cholesky_solve_multiple(X.data(), f.L.data(), X.rows(), X.cols());
}

/// @brief Complex Talbot contour node.
template <typename Float = double>
struct TalbotNode {
    std::complex<Float> shift;
    std::complex<Float> weight;
};

/// @brief Generate modified Talbot contour nodes and weights for numerical Laplace inversion.
template <typename Float = double>
inline std::vector<TalbotNode<Float>> talbot_contour(Float t, std::size_t modes = 16) {
    std::vector<TalbotNode<Float>> nodes;
    nodes.reserve(modes);
    constexpr Float sigma = static_cast<Float>(0.6407);
    constexpr Float mu = static_cast<Float>(0.5017);
    constexpr Float nu = static_cast<Float>(0.6122);
    constexpr Float eta = static_cast<Float>(0.2645);
    const Float pi = std::acos(static_cast<Float>(-1.0));

    for (std::size_t k = 0; k < modes; ++k) {
        const Float theta = -pi + ((static_cast<Float>(k) + static_cast<Float>(0.5)) *
                                   (static_cast<Float>(2.0) * pi / modes));
        const Float a = sigma * theta;
        const Float cot = std::cos(a) / std::sin(a);
        const Float csc2 = static_cast<Float>(1.0) / (std::sin(a) * std::sin(a));
        const Float re = (mu * theta * cot) - nu;
        const Float dre = mu * (cot - (a * csc2));
        const Float im = eta * theta;
        const Float dim = eta;
        const Float scale = static_cast<Float>(modes);
        const std::complex<Float> z(scale * re, scale * im);
        const std::complex<Float> dz(scale * dre, scale * dim);
        nodes.push_back(
            {z / t,
             std::exp(z) * dz / (std::complex<Float>(0.0, 1.0) * static_cast<Float>(modes) * t)});
    }
    return nodes;
}

} // namespace else_sim
