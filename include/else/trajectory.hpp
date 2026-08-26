#pragma once

#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <random>
#include <span>
#include <vector>

namespace else_sim {

/// @brief Exit choice info for sampling outgoing boundary jumps.
template <typename State = std::vector<int>, typename Float = double>
struct ExitChoice {
    std::vector<State> destinations;
    std::vector<Float> rates;
};

/// @brief Builds categorical exit choice mappings for each boundary state in a subnetwork.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
inline std::vector<ExitChoice<State, Float>>
boundary_choices(const Subnetwork<Float, Index, State> &subnetwork) {
    std::vector<ExitChoice<State, Float>> choices(subnetwork.size());
    for (const auto &transition : subnetwork.boundary()) {
        auto &choice = choices[transition.source];
        choice.destinations.push_back(transition.destination);
        choice.rates.push_back(transition.rate);
    }
    return choices;
}

/// @brief Finalizes a trajectory by recording the terminal state at final_time.
template <typename State = std::vector<int>, typename Float = double>
inline void finish_trajectory(Trajectory<State, Float> &trajectory, Float &current_time, Float final_time) {
    current_time = final_time;
    if (trajectory.times.empty() || trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.empty() ? State{} : trajectory.states.back());
    }
}

} // namespace else_sim
