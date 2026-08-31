/// @file elsex/types.hpp
/// @brief Shared value types for ELSE.
#pragma once

#include "core/types.hpp"
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

namespace elsex {

using num::idx;
using num::real;

/// Hash a sequence-valued discrete state.
template <typename State>
struct StateHash {
    std::size_t operator()(const State &state) const {
        std::size_t seed = state.size();
        for (const auto &value : state) {
            seed ^= std::hash<std::decay_t<decltype(value)>>{}(value) + 0x9e3779b9 + (seed << 6) +
                    (seed >> 2);
        }
        return seed;
    }
};

/// A transition from an interior state of the subnetwork to a state outside it.
template <typename State = std::vector<int>>
struct BoundaryTransition {
    idx source = 0;      ///< Interior state the transition leaves from.
    State destination{}; ///< State outside the subnetwork.
    real rate = 0.0;     ///< Rate of this individual transition.
};

/// A sampled macrostep path: jump times and the state entered at each.
template <typename State = std::vector<int>>
struct Trajectory {
    std::vector<real> times;
    std::vector<State> states;
};

/// Transient probability over the represented states at one time.
template <typename State = std::vector<int>>
struct DensitySolution {
    std::vector<State> states;
    std::vector<real> probability;
};

/// Which rule the ensemble applies when the expanded set exceeds capacity.
enum class SheddingRule {
    ExpectedEntries, ///< score = (-Rjj) uj - rhoj.
    ExactCutTime,    ///< loss = uj qj / Zjj.
};

/// Options for the finite ensemble.
struct EnsembleOptions {
    idx capacity = 30;                 ///< Largest subnetwork to factor.
    int expansion_depth = 1;           ///< Reactions to grow outward from the entrances.
    real tolerance = 1e-6;             ///< Propensity below which a state is not admitted.
    std::size_t maximum_steps = 100000;
    SheddingRule rule = SheddingRule::ExpectedEntries;
    bool reuse_factorization = true;
};

} // namespace elsex
