#pragma once

#include "else/linalg.hpp"
#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace else_sim {

template <typename Mat>
inline std::size_t matrix_rows_helper(const Mat &m) {
    if constexpr (requires { m.rows; }) return m.rows;
    else return m.n_rows();
}

/// @brief Connected neighborhood grown in descending stationary weight order.
template <typename SparseMat, typename Float = double, typename Index1 = std::size_t, typename Index2 = std::size_t>
inline std::vector<std::size_t>
laplacian_neighborhood(const SparseMat &laplacian,
                       const std::vector<Float> &stationary_sqrt,
                       Index1 initial, Index2 capacity) {
    const std::size_t n_rows = matrix_rows_helper(laplacian);
    const std::size_t init_idx = static_cast<std::size_t>(initial);
    const std::size_t cap = static_cast<std::size_t>(capacity);
    if (init_idx >= n_rows || cap == 0 || cap > n_rows) {
        throw std::invalid_argument("invalid Laplacian neighborhood request");
    }

    std::vector<bool> selected(n_rows, false);
    std::vector<bool> in_frontier(n_rows, false);
    std::priority_queue<std::pair<Float, std::size_t>> frontier;
    std::vector<std::size_t> states;
    states.reserve(cap);

    const auto expose = [&](std::size_t state) {
        const auto row_begin = laplacian.row_ptr()[state];
        const auto row_end = laplacian.row_ptr()[state + 1];
        for (auto k = row_begin; k < row_end; ++k) {
            const std::size_t neighbor = static_cast<std::size_t>(laplacian.col_idx()[k]);
            if (neighbor != state && !selected[neighbor] && !in_frontier[neighbor]) {
                in_frontier[neighbor] = true;
                frontier.push({stationary_sqrt[neighbor], neighbor});
            }
        }
    };

    selected[init_idx] = true;
    states.push_back(init_idx);
    expose(init_idx);

    while (states.size() < cap) {
        if (frontier.empty()) {
            throw std::invalid_argument("initial component is smaller than the requested capacity");
        }
        const std::size_t state = frontier.top().second;
        frontier.pop();
        if (!selected[state]) {
            selected[state] = true;
            states.push_back(state);
            expose(state);
        }
    }
    return states;
}

/// @brief Build successive restrictions of a symmetric Laplacian.
template <typename SparseMat, typename Float = double, typename Index1 = std::size_t, typename Index2 = std::size_t, typename State = std::vector<int>>
inline std::vector<Subnetwork<Float, std::size_t, State>>
laplacian_restrictions(const SparseMat &laplacian,
                       const std::vector<Float> &stationary_sqrt,
                       Index1 initial, Index2 capacity, int count) {
    if (count < 1) {
        throw std::invalid_argument("Laplacian restrictions require count >= 1");
    }

    const std::size_t n_rows = matrix_rows_helper(laplacian);
    auto states_idx = laplacian_neighborhood(laplacian, stationary_sqrt, initial, capacity);
    std::vector<bool> included(n_rows, false);
    for (std::size_t s : states_idx) included[s] = true;

    std::vector<Subnetwork<Float, std::size_t, State>> subnetworks;
    subnetworks.reserve(static_cast<std::size_t>(count));

    for (int step = 0; step < count; ++step) {
        const std::size_t n = states_idx.size();
        std::unordered_map<std::size_t, std::size_t> local_map;
        std::vector<State> sub_states;
        sub_states.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            local_map[states_idx[i]] = i;
            sub_states.push_back(State{static_cast<int>(states_idx[i])});
        }

        std::vector<std::size_t> r_rows, r_cols;
        std::vector<Float> r_vals;
        std::vector<BoundaryTransition<std::size_t, State, Float>> boundary;
        std::vector<Float> local_h(n);
        for (std::size_t i = 0; i < n; ++i) local_h[i] = stationary_sqrt[states_idx[i]];

        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t global_j = states_idx[j];
            Float col_sum = static_cast<Float>(0);
            const auto row_begin = laplacian.row_ptr()[global_j];
            const auto row_end = laplacian.row_ptr()[global_j + 1];
            for (auto k = row_begin; k < row_end; ++k) {
                const std::size_t global_i = static_cast<std::size_t>(laplacian.col_idx()[k]);
                if (global_i == global_j) continue;
                const Float rate = static_cast<Float>(laplacian.values()[k]);
                auto it = local_map.find(global_i);
                if (it != local_map.end()) {
                    r_rows.push_back(it->second);
                    r_cols.push_back(j);
                    r_vals.push_back(rate);
                } else {
                    boundary.push_back({j, State{static_cast<int>(global_i)}, rate});
                }
                col_sum += rate;
            }
            r_rows.push_back(j);
            r_cols.push_back(j);
            r_vals.push_back(-col_sum);
        }

        auto R = SparseMatrix<Float, std::size_t>::from_triplets(n, n, r_rows, r_cols, r_vals);
        subnetworks.emplace_back(std::move(sub_states), std::move(R), std::move(boundary), std::move(local_h));

        if (subnetworks.back().boundary().empty()) break;

        for (const auto &trans : subnetworks.back().boundary()) {
            const std::size_t g_dest = static_cast<std::size_t>(trans.destination[0]);
            if (!included[g_dest]) {
                included[g_dest] = true;
                states_idx.push_back(g_dest);
            }
        }
    }
    return subnetworks;
}

template <typename ReversibleLaplacianType, typename State = std::vector<int>, typename Float = double, typename Index = std::size_t>
inline Trajectory<State, Float>
laplacian_else_trajectory(const ReversibleLaplacianType &generator,
                          Index initial, Float initial_time,
                          Float final_time, ELSEOptions<Index, Float> options = {}, int seed = 42) {
    const auto &laplacian = generator.matrix();
    const auto &stationary_sqrt = generator.stationary_sqrt();
    const std::size_t n_rows = matrix_rows_helper(laplacian);
    if (initial >= n_rows || initial_time > final_time || options.capacity == 0 ||
        options.capacity > n_rows) {
        throw std::invalid_argument("invalid Laplacian trajectory request");
    }

    std::mt19937 random(seed);
    Trajectory<State, Float> trajectory;
    trajectory.times.push_back(initial_time);
    trajectory.states.push_back(State{static_cast<int>(initial)});

    Index current = initial;
    Float time = initial_time;
    std::size_t completed_steps = 0;

    for (; completed_steps < options.maximum_steps && time < final_time; ++completed_steps) {
        const auto states = laplacian_neighborhood(laplacian, stationary_sqrt, current, options.capacity);
        const Index n = states.size();
        std::unordered_map<Index, Index> local_map;
        std::vector<State> sub_states;
        sub_states.reserve(n);
        for (Index i = 0; i < n; ++i) {
            local_map[states[i]] = i;
            sub_states.push_back(State{static_cast<int>(states[i])});
        }

        std::vector<Index> r_rows, r_cols;
        std::vector<Float> r_vals;
        std::vector<BoundaryTransition<Index, State, Float>> boundary;
        std::vector<Float> local_h(n);
        for (Index i = 0; i < n; ++i) local_h[i] = stationary_sqrt[states[i]];

        for (Index j = 0; j < n; ++j) {
            const Index global_j = states[j];
            Float col_sum = static_cast<Float>(0);
            const auto row_begin = laplacian.row_ptr()[global_j];
            const auto row_end = laplacian.row_ptr()[global_j + 1];
            for (auto k = row_begin; k < row_end; ++k) {
                const Index global_i = laplacian.col_idx()[k];
                if (global_i == global_j) continue;
                const Float rate = laplacian.values()[k];
                auto it = local_map.find(global_i);
                if (it != local_map.end()) {
                    r_rows.push_back(it->second);
                    r_cols.push_back(j);
                    r_vals.push_back(rate);
                } else {
                    boundary.push_back({j, State{static_cast<int>(global_i)}, rate});
                }
                col_sum += rate;
            }
            r_rows.push_back(j);
            r_cols.push_back(j);
            r_vals.push_back(-col_sum);
        }

        auto R = SparseMatrix<Float, Index>::from_triplets(n, n, r_rows, r_cols, r_vals);
        Subnetwork<Float, Index, State> subnetwork(std::move(sub_states), std::move(R), std::move(boundary), std::move(local_h));
        if (subnetwork.boundary_states().empty()) break;

        Matrix<Float> source(subnetwork.size(), 1, static_cast<Float>(0));
        int pos = subnetwork.find(State{static_cast<int>(current)});
        if (pos >= 0) source(static_cast<std::size_t>(pos), 0) = static_cast<Float>(1);
        const auto integrals = subnetwork.occupation_integrals(source);
        const auto exit_probs = subnetwork.exit_probabilities(integrals);

        std::discrete_distribution<std::size_t> dist(exit_probs.begin(), exit_probs.end());
        const Index local_exit = subnetwork.boundary_states()[dist(random)];
        const Float waiting_time = subnetwork.conditional_exit_time(integrals, local_exit);

        std::vector<Index> destinations;
        std::vector<Float> rates;
        for (const auto &transition : subnetwork.boundary()) {
            if (transition.source == local_exit) {
                destinations.push_back(static_cast<Index>(transition.destination[0]));
                rates.push_back(transition.rate);
            }
        }
        if (!(waiting_time > static_cast<Float>(0)) || destinations.empty() || time + waiting_time >= final_time) {
            break;
        }

        time += waiting_time;
        std::discrete_distribution<std::size_t> rate_dist(rates.begin(), rates.end());
        current = destinations[rate_dist(random)];
        trajectory.times.push_back(time);
        trajectory.states.push_back(State{static_cast<int>(current)});
    }

    if (completed_steps < options.maximum_steps && time < final_time &&
        final_time < std::numeric_limits<Float>::infinity()) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(State{static_cast<int>(current)});
    }
    return trajectory;
}

} // namespace else_sim
