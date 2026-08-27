#pragma once

#include <cmath>
#include <limits>

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

} // namespace else_sim
