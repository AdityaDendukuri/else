#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace else_sim {

/// @brief Generic state hash functor for multi-index or sequence-like discrete states.
template <typename State>
struct StateHash {
    std::size_t operator()(const State &s) const {
        if constexpr (requires { s.begin(); s.end(); }) {
            std::size_t seed = s.size();
            for (const auto &elem : s) {
                seed ^= std::hash<std::decay_t<decltype(elem)>>{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        } else {
            return std::hash<State>{}(s);
        }
    }
};

/// @brief Directed transition between discrete states within a generator.
template <typename Index = std::size_t, typename Float = double>
struct Transition {
    Index source = 0;
    Index destination = 0;
    Float rate = static_cast<Float>(0);
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

    [[nodiscard]] auto size() const { return occupation.rows(); }
    [[nodiscard]] auto starts() const { return occupation.cols(); }
};

/// @brief Result of evaluating exit-time loss with floating-point error tracking.
template <typename Float = double>
struct CutTimeLossResult {
    Float loss = static_cast<Float>(0);             ///< Mean exit-time loss \f$\Delta \tau\f$
    Float estimated_error = static_cast<Float>(0);  ///< Linear solve backward residual
    bool naive_fallback_used = false;              ///< True if ground-truth scratch refactor was used
};

/// @brief Algorithmic strategy for state shedding.
enum class SheddingMethod : std::uint8_t {
    CholeskyWoodbury,      ///< Method 1: Exact Cholesky Woodbury principal block downdates
    NormalizedSymmetrized, ///< Method 2: Steady-state normalized symmetrized surrogate
    ExpectedVisits,        ///< Method 3: Flow-balanced expected visits O(n) approximation
    Auto,                  ///< Automatic selection based on reversibility
};

/// @brief Configuration parameters for state shedding and truncation.
template <typename Index = std::size_t, typename Float = double>
struct SheddingOptions {
    SheddingMethod method = SheddingMethod::Auto;
    Index target_capacity = 0;                     ///< Desired target capacity (0 = tolerance-only)
    Float tolerance = static_cast<Float>(1e-4);    ///< Maximum allowable relative exit-time loss
    bool fallback_to_stable = true;                ///< Enable fallback on ill-conditioning
};

/// @brief Diagnostics recorded during the state shedding process.
template <typename Index = std::size_t, typename Float = double>
struct SheddingDiagnostics {
    SheddingMethod requested_method = SheddingMethod::Auto;
    SheddingMethod effective_method = SheddingMethod::Auto;
    Index initial_states = 0;
    Index shed_states = 0;
    Index remaining_states = 0;
    Float initial_mean_exit_time = static_cast<Float>(0);
    Float total_estimated_loss = static_cast<Float>(0);
    Float relative_loss = static_cast<Float>(0);
    bool fallback_triggered = false;
    std::string fallback_reason;
};

/// @brief Result of a state space shedding operation.
template <typename Index = std::size_t, typename Float = double>
struct SheddingResult {
    std::vector<Index> kept_indices;
    std::vector<Index> shed_indices;
    std::vector<Float> state_losses;
    SheddingDiagnostics<Index, Float> diagnostics;
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
    int expansion_depth = 2;
    Float tolerance = static_cast<Float>(1e-6);
    std::size_t maximum_steps = 100000;
};

/// @brief Options for numerical contour inversion of the Laplace-domain density.
template <typename Index = std::size_t>
struct ContourOptions {
    Index nodes = 14;
};

/// @brief Transient probability density solution at a given time point.
template <typename State = std::vector<int>, typename Float = double>
struct DensitySolution {
    std::vector<State> states;
    std::vector<Float> probability;
};

} // namespace else_sim
