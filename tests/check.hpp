/// @file tests/check.hpp
/// @brief Minimal assertions that survive NDEBUG.
///
/// This project builds Release by default, so `assert` is compiled out and a
/// suite written on it reports success without verifying anything. These helpers
/// count failures and report through the process exit status instead, so a
/// broken check actually fails the build.
///
/// Every new suite should be run once against a deliberately impossible
/// tolerance to confirm that it can fail.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace check {

inline int failures = 0;
inline double worst_relative = 0.0;

/// Assert a condition.
inline void that(bool condition, const std::string &what) {
    if (!condition) {
        std::printf("  FAILED  %s\n", what.c_str());
        ++failures;
    }
}

/// Assert relative agreement, scaled by the larger of |actual|, |expected|, and 1.
inline void close(double actual, double expected, const std::string &what,
                  double tolerance = 1e-13) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    const double relative = std::abs(actual - expected) / scale;
    worst_relative = std::max(worst_relative, relative);
    if (!(relative <= tolerance)) {
        std::printf("  FAILED  %s: %.17g vs %.17g (relative %.3e)\n", what.c_str(), actual,
                    expected, relative);
        ++failures;
    }
}

/// Record that a named case finished. Printed after its checks, so any failure
/// above it is attributable to it.
inline void done(const char *name) { std::printf("  ok      %s\n", name); }

/// Print the summary and return the process exit status.
[[nodiscard]] inline int report(const char *suite) {
    if (failures > 0) {
        std::printf("\n%s: %d check(s) FAILED (worst relative %.3e)\n", suite, failures,
                    worst_relative);
        return 1;
    }
    std::printf("\n%s: all checks passed (worst relative %.3e)\n", suite, worst_relative);
    return 0;
}

} // namespace check
