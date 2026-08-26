#pragma once

#include "else/types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace else_sim {

/// @brief Perform x += y, update accumulated absolute error, and verify precision acceptability.
template <typename T>
inline bool safe_add(T &x, T &err, T y, T tolerance = static_cast<T>(1e-6)) {
    x += y;
    if (x <= static_cast<T>(0)) {
        return false;
    }
    err += std::numeric_limits<T>::epsilon() * std::abs(x);
    return (err / x) < tolerance;
}

/// @brief Computes normalized backward linear residual || A * x - b ||_inf / || b ||_inf.
template <typename Matrix, typename Vector, typename Float = double>
inline Float backward_linear_residual(const Matrix &A, const Vector &x, const Vector &b) {
    const auto n = b.size();
    Float max_res = static_cast<Float>(0);
    Float max_rhs = static_cast<Float>(0);
    for (std::size_t i = 0; i < n; ++i) {
        max_rhs = std::max(max_rhs, static_cast<Float>(std::abs(b[i])));
        Float reconstructed = static_cast<Float>(0);
        for (std::size_t j = 0; j < n; ++j) {
            reconstructed += static_cast<Float>(A(i, j) * x[j]);
        }
        max_res = std::max(max_res, static_cast<Float>(std::abs(reconstructed - b[i])));
    }
    return max_res / std::max(static_cast<Float>(1e-12), max_rhs);
}

} // namespace else_sim
