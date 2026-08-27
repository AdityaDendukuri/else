#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace else_sim {

/// @brief Lightweight, pure template row-major dense matrix.
template <typename Float = double>
class Matrix {
  public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(std::size_t rows, std::size_t cols, Float fill = static_cast<Float>(0))
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] Float *data() noexcept { return data_.data(); }
    [[nodiscard]] const Float *data() const noexcept { return data_.data(); }

    [[nodiscard]] Float &operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }
    [[nodiscard]] const Float &operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

  private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<Float> data_;
};

/// @brief Lightweight CSR sparse matrix.
template <typename Float = double, typename Index = std::size_t>
struct SparseMatrix {
    Index rows = 0;
    Index cols = 0;
    std::vector<Index> row_ptr;
    std::vector<Index> col_idx;
    std::vector<Float> values;

    [[nodiscard]] Float operator()(Index r, Index c) const noexcept {
        if (r >= rows || row_ptr.empty()) return static_cast<Float>(0);
        for (Index k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            if (col_idx[k] == c) return values[k];
        }
        return static_cast<Float>(0);
    }

    static SparseMatrix from_triplets(Index n_rows, Index n_cols,
                                      const std::vector<Index> &r,
                                      const std::vector<Index> &c,
                                      const std::vector<Float> &v) {
        SparseMatrix mat;
        mat.rows = n_rows;
        mat.cols = n_cols;
        mat.row_ptr.assign(n_rows + 1, 0);

        for (Index row : r) {
            if (row < n_rows) mat.row_ptr[row + 1]++;
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
template <typename Float = double, typename Index = std::size_t>
struct LUFactor {
    Matrix<Float> LU;
    bool singular = false;
};

/// @brief Compute a no-pivot dense LU factorization.
template <typename Float = double, typename Index = std::size_t>
inline LUFactor<Float, Index> factorize_lu(Matrix<Float> A) {
    using Real = decltype(std::abs(std::declval<Float>()));
    const Index n = A.rows();
    LUFactor<Float, Index> result;
    result.LU = std::move(A);
    auto &M = result.LU;
    constexpr Real tol = static_cast<Real>(1e-15);
    for (Index k = 0; k < n; ++k) {
        if (std::abs(M(k, k)) < tol) {
            result.singular = true;
            continue;
        }
        const Float inv_piv = static_cast<Float>(1) / M(k, k);
        const Float *row_k = M.data() + k * n;
        for (Index i = k + 1; i < n; ++i) {
            Float *row_i = M.data() + i * n;
            row_i[k] *= inv_piv;
            const Float m_ik = row_i[k];
            for (Index j = k + 1; j < n; ++j) row_i[j] -= m_ik * row_k[j];
        }
    }
    return result;
}

/// @brief Solve A x = b from a precomputed LU factorization.
template <typename Float = double, typename Index = std::size_t>
inline void lu_solve(const LUFactor<Float, Index> &f, const std::vector<Float> &b,
                     std::vector<Float> &x) {
    const Index n = f.LU.rows();
    x = b;
    const Float *data = f.LU.data();
    for (Index i = 0; i < n; ++i) {
        const Float *row_i = data + i * n;
        for (Index j = 0; j < i; ++j) x[i] -= row_i[j] * x[j];
    }
    for (Index i = n; i-- > 0;) {
        const Float *row_i = data + i * n;
        for (Index j = i + 1; j < n; ++j) x[i] -= row_i[j] * x[j];
        x[i] /= row_i[i];
    }
}

template <typename Float = double, typename Index = std::size_t>
inline void lu_solve(const LUFactor<Float, Index> &f, const Matrix<Float> &B,
                     Matrix<Float> &X) {
    const Index n = static_cast<Index>(f.LU.rows());
    const Index nrhs = static_cast<Index>(B.cols());
    if (B.rows() != n) throw std::invalid_argument("LU block right-hand side size mismatch");
    X = B;
    for (Index i = 0; i < n; ++i)
        for (Index j = 0; j < i; ++j) {
            const Float value = f.LU(i, j);
            for (Index c = 0; c < nrhs; ++c) X(i, c) -= value * X(j, c);
        }
    for (Index i = n; i-- > 0;) {
        for (Index j = i + 1; j < n; ++j) {
            const Float value = f.LU(i, j);
            for (Index c = 0; c < nrhs; ++c) X(i, c) -= value * X(j, c);
        }
        for (Index c = 0; c < nrhs; ++c) X(i, c) /= f.LU(i, i);
    }
}

/// @brief Solve A^T x = b from a precomputed LU factorization.
template <typename Float = double, typename Index = std::size_t>
inline void lu_solve_transpose(const LUFactor<Float, Index> &f,
                               const std::vector<Float> &b,
                               std::vector<Float> &x) {
    const Index n = f.LU.rows();
    x = b;
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < i; ++j) x[i] -= f.LU(j, i) * x[j];
        x[i] /= f.LU(i, i);
    }
    for (Index i = n; i-- > 0;)
        for (Index j = i + 1; j < n; ++j) x[i] -= f.LU(j, i) * x[j];
}

/// @brief Cholesky factorization result for symmetric positive definite matrices.
template <typename Float = double>
struct CholeskyFactor {
    Matrix<Float> L;
    bool success = true;
};

/// @brief Compute LL^T Cholesky factorization.
template <typename Float = double, typename Index = std::size_t>
inline CholeskyFactor<Float> factorize_cholesky(const Matrix<Float> &A) {
    const Index n = A.rows();
    CholeskyFactor<Float> result;
    result.L = Matrix<Float>(n, n, static_cast<Float>(0));
    auto &L = result.L;

    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j <= i; ++j) {
            Float sum = A(i, j);
            for (Index k = 0; k < j; ++k) {
                sum -= L(i, k) * L(j, k);
            }
            if (i == j) {
                if (sum <= static_cast<Float>(0)) {
                    result.success = false;
                    return result;
                }
                L(i, j) = std::sqrt(sum);
            } else {
                L(i, j) = sum / L(j, j);
            }
        }
    }
    result.success = true;
    return result;
}

/// @brief Solve LL^T x = b.
template <typename Float = double, typename Index = std::size_t>
inline void cholesky_solve(const CholeskyFactor<Float> &f, const std::vector<Float> &b, std::vector<Float> &x) {
    const Index n = f.L.rows();
    x = b;
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < i; ++j) x[i] -= f.L(i, j) * x[j];
        x[i] /= f.L(i, i);
    }
    for (Index i = n; i-- > 0;) {
        for (Index j = i + 1; j < n; ++j) x[i] -= f.L(j, i) * x[j];
        x[i] /= f.L(i, i);
    }
}

/// @brief Complex Talbot contour node.
template <typename Float = double>
struct TalbotNode {
    std::complex<Float> shift;
    std::complex<Float> weight;
};

/// @brief Generate modified Talbot contour nodes and weights for numerical Laplace inversion.
template <typename Float = double, typename Index = std::size_t>
inline std::vector<TalbotNode<Float>> talbot_contour(Float t, Index modes = 16) {
    std::vector<TalbotNode<Float>> nodes;
    nodes.reserve(modes);
    constexpr Float sigma = static_cast<Float>(0.6407);
    constexpr Float mu = static_cast<Float>(0.5017);
    constexpr Float nu = static_cast<Float>(0.6122);
    constexpr Float eta = static_cast<Float>(0.2645);
    const Float pi = std::acos(static_cast<Float>(-1.0));

    for (Index k = 0; k < modes; ++k) {
        const Float theta = -pi + ((static_cast<Float>(k) + static_cast<Float>(0.5)) * (static_cast<Float>(2.0) * pi / modes));
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
        nodes.push_back({z / t, std::exp(z) * dz / (std::complex<Float>(0.0, 1.0) * static_cast<Float>(modes) * t)});
    }
    return nodes;
}

} // namespace else_sim
