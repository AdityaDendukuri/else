#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

namespace else_sim {

/// @brief Hash a sequence-valued discrete state.
template <typename State>
struct StateHash {
    std::size_t operator()(const State &s) const {
        std::size_t seed = s.size();
        for (const auto &value : s) {
            seed ^= std::hash<std::decay_t<decltype(value)>>{}(value) + 0x9e3779b9 + (seed << 6) +
                    (seed >> 2);
        }
        return seed;
    }
};

/// Default ordering: no block structure is supplied by the caller.
struct SingleBlockLevel {
    template <typename State>
    [[nodiscard]] constexpr std::size_t operator()(const State &) const noexcept {
        return 0;
    }
};

/// @brief Exit transition from a subnetwork interior state to an external state.
template <typename Index = std::size_t, typename State = std::vector<int>, typename Float = double>
struct BoundaryTransition {
    Index source = 0;
    State destination{};
    Float rate = static_cast<Float>(0);
};

/// @brief First and second occupation moment integrals from a shared factorization.
template <typename Matrix>
struct OccupationIntegrals {
    Matrix occupation;               ///< First moment: \int_0^\infty p(t) dt
    Matrix time_weighted_occupation; ///< Second moment: \int_0^\infty t p(t) dt
};

/// @brief Result of evaluating exit-time loss with floating-point error tracking.
template <typename Float = double>
struct CutTimeLossResult {
    Float loss = static_cast<Float>(0);            ///< Mean loss in exit time.
    Float estimated_error = static_cast<Float>(0); ///< Linear solve backward residual
    bool naive_fallback_used = false; ///< True if ground-truth scratch refactor was used
};

/// @brief Sampled macrostep trajectory containing discrete states and jump times.
template <typename State = std::vector<int>, typename Float = double>
struct Trajectory {
    std::vector<Float> times;
    std::vector<State> states;
};

/// @brief Options for ELSE dynamic simulation.
template <typename Index = std::size_t, typename Float = double>
struct ELSEOptions {
    Index capacity = 30;
    int expansion_depth = 1;
    Float tolerance = static_cast<Float>(1e-6);
    std::size_t maximum_steps = 100000;
    bool reuse_factorization = true;
};

/// @brief Transient probability density solution at a given time point.
template <typename State = std::vector<int>, typename Float = double>
struct DensitySolution {
    std::vector<State> states;
    std::vector<Float> probability;
};

} // namespace else_sim
