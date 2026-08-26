#pragma once

#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace else_sim {

template <typename State = std::vector<int>, typename Float = double>
struct ExitChoice {
    std::vector<State> destinations;
    std::vector<Float> rates;
};

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

template <typename State = std::vector<int>, typename Float = double>
inline void finish_trajectory(Trajectory<State, Float> &trajectory, Float &current_time, Float final_time) {
    current_time = final_time;
    if (trajectory.times.empty() || trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.empty() ? State{} : trajectory.states.back());
    }
}

template <typename ReactionSystem, typename State = std::vector<int>, typename Float = double, typename Index = std::size_t>
inline std::vector<Trajectory<State, Float>>
else_ensemble(const ReactionSystem &model,
              const std::vector<Float> &rates,
              const State &initial, std::size_t count,
              Float initial_time, Float final_time,
              ELSEOptions<Index, Float> options = {}, int seed = 42) {
    if (count == 0 || initial_time > final_time) {
        throw std::invalid_argument("invalid ELSE trajectory ensemble request");
    }

    std::vector<Trajectory<State, Float>> trajectories(count);
    std::vector<Float> times(count, initial_time);
    std::vector<State> states(count, initial);
    std::vector<std::size_t> steps(count, 0);
    std::vector<std::mt19937> generators;
    generators.reserve(count);
    for (std::size_t t = 0; t < count; ++t) {
        trajectories[t].times.push_back(initial_time);
        trajectories[t].states.push_back(initial);
        generators.emplace_back(seed + static_cast<int>(t));
    }

    while (true) {
        std::vector<std::size_t> unfinished;
        std::unordered_map<State, Float, StateHash<State>> entrances;
        for (std::size_t t = 0; t < count; ++t) {
            if (times[t] < final_time && steps[t] < options.maximum_steps) {
                unfinished.push_back(t);
                entrances[states[t]] += static_cast<Float>(1);
            }
        }
        if (unfinished.empty()) break;

        for (auto &[s, w] : entrances) {
            w /= static_cast<Float>(unfinished.size());
        }

        // Build reaction state space around entrances
        std::vector<State> workspace_states;
        std::unordered_set<State, StateHash<State>> visited;
        std::vector<State> frontier;
        for (const auto &[s, w] : entrances) {
            workspace_states.push_back(s);
            visited.insert(s);
            frontier.push_back(s);
        }

        for (int depth = 0; depth < options.expansion_depth; ++depth) {
            std::vector<State> next;
            for (const auto &state : frontier) {
                for (const auto &change : model.changes) {
                    State dest = state + change;
                    bool nonneg = true;
                    for (std::size_t i = 0; i < dest.size(); ++i) {
                        if (dest[i] < 0) { nonneg = false; break; }
                    }
                    if (!nonneg || !visited.insert(dest).second) continue;
                    Float total_rate = model.total_propensity(dest, rates, 0.0);
                    if (total_rate <= options.tolerance) continue;
                    workspace_states.push_back(dest);
                    next.push_back(dest);
                }
            }
            frontier = std::move(next);
        }

        // Restrict CME generator on workspace_states
        const Index n_ws = workspace_states.size();
        std::unordered_map<State, Index, StateHash<State>> ws_index;
        for (Index i = 0; i < n_ws; ++i) ws_index[workspace_states[i]] = i;

        std::vector<Index> r_rows, r_cols;
        std::vector<Float> r_vals;
        std::vector<BoundaryTransition<Index, State, Float>> boundary;

        for (Index j = 0; j < n_ws; ++j) {
            const auto &x = workspace_states[j];
            Float col_sum = static_cast<Float>(0);
            for (std::size_t k = 0; k < model.size(); ++k) {
                Float alpha = model.propensities[k](x, rates, 0.0);
                if (alpha <= static_cast<Float>(0)) continue;
                State y = x + model.changes[k];
                auto it = ws_index.find(y);
                if (it != ws_index.end()) {
                    r_rows.push_back(it->second);
                    r_cols.push_back(j);
                    r_vals.push_back(alpha);
                    col_sum += alpha;
                } else {
                    boundary.push_back({j, y, alpha});
                    col_sum += alpha;
                }
            }
            r_rows.push_back(j);
            r_cols.push_back(j);
            r_vals.push_back(-col_sum);
        }

        auto R = SparseMatrix<Float, Index>::from_triplets(n_ws, n_ws, r_rows, r_cols, r_vals);
        Subnetwork<Float, Index, State> subnetwork(workspace_states, std::move(R), std::move(boundary));

        if (subnetwork.boundary_states().empty()) {
            for (std::size_t t : unfinished) {
                finish_trajectory(trajectories[t], times[t], final_time);
            }
            break;
        }

        std::vector<State> entrance_states;
        std::unordered_map<State, Index, StateHash<State>> entrance_col;
        for (const auto &[s, w] : entrances) {
            entrance_col[s] = entrance_states.size();
            entrance_states.push_back(s);
        }

        Matrix<Float> rhs(subnetwork.size(), entrance_states.size(), static_cast<Float>(0));
        for (Index c = 0; c < entrance_states.size(); ++c) {
            int pos = subnetwork.find(entrance_states[c]);
            if (pos >= 0) rhs(static_cast<std::size_t>(pos), c) = static_cast<Float>(1);
        }

        const auto integrals = subnetwork.occupation_integrals(rhs);
        auto destinations = boundary_choices(subnetwork);

        for (std::size_t t : unfinished) {
            const Index col = entrance_col[states[t]];
            auto &rng = generators[t];
            const auto exit_weights = subnetwork.exit_probabilities(integrals, col);

            // Sample boundary position
            std::discrete_distribution<std::size_t> dist(exit_weights.begin(), exit_weights.end());
            const std::size_t boundary_pos = dist(rng);
            const Index b_state = subnetwork.boundary_states()[boundary_pos];
            const Float waiting_time = subnetwork.conditional_exit_time(integrals, b_state, col);
            auto &choice = destinations[b_state];

            if (!(waiting_time > static_cast<Float>(0)) || choice.destinations.empty() ||
                times[t] + waiting_time >= final_time) {
                finish_trajectory(trajectories[t], times[t], final_time);
                continue;
            }

            times[t] += waiting_time;
            std::discrete_distribution<std::size_t> rate_dist(choice.rates.begin(), choice.rates.end());
            states[t] = choice.destinations[rate_dist(rng)];
            trajectories[t].times.push_back(times[t]);
            trajectories[t].states.push_back(states[t]);
            ++steps[t];
        }
    }
    return trajectories;
}

template <typename ReactionSystem, typename State = std::vector<int>, typename Float = double, typename Index = std::size_t>
inline Trajectory<State, Float>
else_trajectory(const ReactionSystem &model,
                const std::vector<Float> &rates,
                const State &initial, Float initial_time,
                Float final_time, ELSEOptions<Index, Float> options = {}, int seed = 42) {
    auto ensemble = else_ensemble(model, rates, initial, 1, initial_time, final_time, options, seed);
    return std::move(ensemble.front());
}

} // namespace else_sim
