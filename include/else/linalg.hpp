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

/// @brief Dense matrix-vector product y = A * x.
template <typename Float = double>
inline void matvec(const Matrix<Float> &A, const std::vector<Float> &x, std::vector<Float> &y) {
    const auto m = A.rows();
    const auto n = A.cols();
    y.assign(m, static_cast<Float>(0));
    for (std::size_t i = 0; i < m; ++i) {
        Float sum = static_cast<Float>(0);
        for (std::size_t j = 0; j < n; ++j) {
            sum += A(i, j) * x[j];
        }
        y[i] = sum;
    }
}

/// @brief Packed LU factorization result supporting optional unpivoted mode.
template <typename Float = double, typename Index = std::size_t>
struct LUFactor {
    Matrix<Float> LU;
    std::vector<Index> piv;
    bool pivoted = false;
    bool singular = false;
};

/// @brief Compute LU factorization with optional pivoting (default: unpivoted for M-matrices).
template <typename Float = double, typename Index = std::size_t>
inline LUFactor<Float, Index> factorize_lu(Matrix<Float> A, bool pivot = false) {
    using Real = decltype(std::abs(std::declval<Float>()));
    const Index n = A.rows();
    LUFactor<Float, Index> result;
    result.LU = std::move(A);
    result.pivoted = pivot;
    result.singular = false;
    if (pivot) result.piv.resize(n);

    auto &M = result.LU;
    constexpr Real tol = static_cast<Real>(1e-15);

    for (Index k = 0; k < n; ++k) {
        if (pivot) {
            Index max_row = k;
            Real max_val = std::abs(M(k, k));
            for (Index i = k + 1; i < n; ++i) {
                Real v = std::abs(M(i, k));
                if (v > max_val) {
                    max_val = v;
                    max_row = i;
                }
            }
            result.piv[k] = max_row;
            if (max_row != k) {
                for (Index j = 0; j < n; ++j) {
                    std::swap(M(k, j), M(max_row, j));
                }
            }
        }

        if (std::abs(M(k, k)) < tol) {
            result.singular = true;
            continue;
        }

        const Float inv_piv = static_cast<Float>(1) / M(k, k);
        for (Index i = k + 1; i < n; ++i) {
            M(i, k) *= inv_piv;
            const Float m_ik = M(i, k);
            for (Index j = k + 1; j < n; ++j) {
                M(i, j) -= m_ik * M(k, j);
            }
        }
    }
    return result;
}

/// @brief Solve A x = b from a precomputed LU factorization.
template <typename Float = double, typename Index = std::size_t>
inline void lu_solve(const LUFactor<Float, Index> &f, const std::vector<Float> &b, std::vector<Float> &x) {
    const Index n = f.LU.rows();
    x = b;
    if (f.pivoted) {
        for (Index k = 0; k < n; ++k) {
            if (f.piv[k] != k) std::swap(x[k], x[f.piv[k]]);
        }
    }
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < i; ++j) {
            x[i] -= f.LU(i, j) * x[j];
        }
    }
    for (Index i = n; i-- > 0;) {
        for (Index j = i + 1; j < n; ++j) {
            x[i] -= f.LU(i, j) * x[j];
        }
        x[i] /= f.LU(i, i);
    }
}

/// @brief Solve A^T x = b from a precomputed LU factorization.
template <typename Float = double, typename Index = std::size_t>
inline void lu_solve_transpose(const LUFactor<Float, Index> &f, const std::vector<Float> &b, std::vector<Float> &x) {
    const Index n = f.LU.rows();
    x = b;
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < i; ++j) {
            x[i] -= f.LU(j, i) * x[j];
        }
        x[i] /= f.LU(i, i);
    }
    for (Index i = n; i-- > 0;) {
        for (Index j = i + 1; j < n; ++j) {
            x[i] -= f.LU(j, i) * x[j];
        }
    }
    if (f.pivoted) {
        for (Index k = n; k-- > 0;) {
            if (f.piv[k] != k) std::swap(x[k], x[f.piv[k]]);
        }
    }
}

/// @brief Cholesky factorization result for symmetric positive definite matrices.
template <typename Float = double>
struct CholeskyFactor {
    Matrix<Float> L;
    bool success = true;
};

/// @brief Compute LL^T Cholesky factorization.
template <typename Float = double, typename Index = std::size_t>
inline CholeskyFactor<Float> factorize_cholesky(Matrix<Float> A) {
    const Index n = A.rows();
    CholeskyFactor<Float> result;
    result.L = std::move(A);
    auto &L = result.L;

    for (Index j = 0; j < n; ++j) {
        Float sum = static_cast<Float>(0);
        for (Index k = 0; k < j; ++k) sum += L(j, k) * L(j, k);
        Float diag = L(j, j) - sum;
        if (diag <= static_cast<Float>(0)) {
            result.success = false;
            return result;
        }
        L(j, j) = std::sqrt(diag);
        const Float inv_diag = static_cast<Float>(1) / L(j, j);

        for (Index i = j + 1; i < n; ++i) {
            Float s = static_cast<Float>(0);
            for (Index k = 0; k < j; ++k) s += L(i, k) * L(j, k);
            L(i, j) = (L(i, j) - s) * inv_diag;
        }
    }
    for (Index i = 0; i < n; ++i) {
        for (Index j = i + 1; j < n; ++j) {
            L(i, j) = static_cast<Float>(0);
        }
    }
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
