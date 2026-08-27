#pragma once

#include "else/restriction.hpp"
#include "else/shedding.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace else_sim {

template <typename State = std::vector<int>>
struct StateSpace {
    std::vector<State> states;
    std::unordered_map<State, int, StateHash<State>> index;

    [[nodiscard]] int size() const { return static_cast<int>(states.size()); }
    [[nodiscard]] bool empty() const { return states.empty(); }

    [[nodiscard]] int find(const State &x) const {
        auto it = index.find(x);
        return it != index.end() ? it->second : -1;
    }

    void add_state(const State &x) {
        if (index.count(x))
            return;
        int i = static_cast<int>(states.size());
        states.push_back(x);
        index[x] = i;
    }

    void remove_states(const std::vector<bool> &remove_mask) {
        if (std::none_of(remove_mask.begin(), remove_mask.end(), [](bool b) { return b; }))
            return;
        int n = static_cast<int>(states.size());
        std::vector<State> new_states;
        new_states.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (!remove_mask[i]) {
                new_states.push_back(states[i]);
            }
        }
        states = std::move(new_states);
        index.clear();
        for (int i = 0; i < static_cast<int>(states.size()); ++i) {
            index[states[i]] = i;
        }
    }
};

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
inline void finish_trajectory(Trajectory<State, Float> &trajectory, Float &current_time,
                              Float final_time) {
    current_time = final_time;
    if (trajectory.times.empty() || trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.empty() ? State{} : trajectory.states.back());
    }
}

template <typename ReactionSystem, typename State = std::vector<int>, typename Float = double,
          typename Index = std::size_t, typename LevelFunction = SingleBlockLevel>
inline std::vector<Trajectory<State, Float>>
else_ensemble(const ReactionSystem &model, const std::vector<Float> &rates, const State &initial,
              std::size_t count, Float initial_time, Float final_time,
              ELSEOptions<Index, Float> options = {}, int seed = 42, LevelFunction level = {}) {
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

    StateSpace<State> workspace;
    // The current solve supplies the shedding scores used before the next factorization.
    std::unordered_map<State, Float, StateHash<State>> previous_scores;
    while (true) {
        std::vector<std::size_t> unfinished;
        std::unordered_map<State, Float, StateHash<State>> entrances;
        for (std::size_t t = 0; t < count; ++t) {
            if (times[t] < final_time && steps[t] < options.maximum_steps) {
                unfinished.push_back(t);
                entrances[states[t]] += static_cast<Float>(1);
            }
        }
        if (unfinished.empty())
            break;

        for (auto &[s, w] : entrances) {
            w /= static_cast<Float>(unfinished.size());
        }

        std::vector<State> frontier;
        std::unordered_set<State, StateHash<State>> visited;
        for (const auto &[s, w] : entrances) {
            workspace.add_state(s);
            frontier.push_back(s);
            visited.insert(s);
        }

        for (int depth = 0; depth < options.expansion_depth; ++depth) {
            std::vector<State> next;
            for (const auto &state : frontier) {
                for (const auto &change : model.changes) {
                    State dest = state + change;
                    bool nonneg = true;
                    for (std::size_t i = 0; i < dest.size(); ++i) {
                        if (dest[i] < 0) {
                            nonneg = false;
                            break;
                        }
                    }
                    if (!nonneg || !visited.insert(dest).second)
                        continue;
                    Float total_rate = model.total_propensity(dest, rates, static_cast<Float>(0));
                    if (total_rate <= options.tolerance)
                        continue;
                    workspace.add_state(dest);
                    next.push_back(dest);
                }
            }
            frontier = std::move(next);
        }
        auto build_subnetwork = [&]() {
            return cme_subnetwork<ReactionSystem, std::vector<Float>, State, Float, Index>(
                model, rates, workspace.states, level);
        };

        if (workspace.size() > static_cast<int>(options.capacity)) {
            std::vector<Index> protected_indices;
            for (const auto &[s, w] : entrances) {
                int pos = workspace.find(s);
                if (pos >= 0)
                    protected_indices.push_back(static_cast<Index>(pos));
            }

            std::vector<Float> scores(workspace.size(), static_cast<Float>(0));
            if (options.expansion_depth == 0 && !previous_scores.empty()) {
                for (Index i = 0; i < static_cast<Index>(workspace.size()); ++i) {
                    auto found = previous_scores.find(workspace.states[i]);
                    if (found != previous_scores.end())
                        scores[i] = found->second;
                }
            } else {
                auto expanded = build_subnetwork();
                std::vector<Float> entrance_vec(workspace.size(), static_cast<Float>(0));
                for (const auto &[s, w] : entrances) {
                    int pos = workspace.find(s);
                    if (pos >= 0)
                        entrance_vec[static_cast<std::size_t>(pos)] = w;
                }
                const auto occ = expanded.occupation(entrance_vec);
                std::vector<Index> indices(workspace.size());
                std::iota(indices.begin(), indices.end(), static_cast<Index>(0));
                scores = expected_visit_scores(expanded, occ, std::span<const Index>(indices));
            }
            const Index remove_count = static_cast<Index>(workspace.size()) - options.capacity;
            const auto shed = lowest_scores<Float, Index>(scores, protected_indices, remove_count);
            std::vector<bool> remove(workspace.size(), false);
            for (Index i : shed)
                remove[i] = true;
            workspace.remove_states(remove);
        }

        auto subnetwork = build_subnetwork();

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
            if (pos >= 0)
                rhs(static_cast<std::size_t>(pos), c) = static_cast<Float>(1);
        }

        const auto integrals = subnetwork.occupation_integrals(rhs);

        std::vector<Float> mixture_occupation(subnetwork.size(), static_cast<Float>(0));
        for (Index c = 0; c < entrance_states.size(); ++c) {
            const Float weight = entrances.at(entrance_states[c]);
            for (Index i = 0; i < subnetwork.size(); ++i)
                mixture_occupation[i] += integrals.occupation(i, c) * weight;
        }
        std::vector<Index> score_indices(subnetwork.size());
        std::iota(score_indices.begin(), score_indices.end(), static_cast<Index>(0));
        auto scores = expected_visit_scores(subnetwork, mixture_occupation,
                                            std::span<const Index>(score_indices));
        previous_scores.clear();
        for (Index i = 0; i < subnetwork.size(); ++i) {
            auto entrance = entrances.find(subnetwork.states()[i]);
            if (entrance != entrances.end())
                scores[i] -= entrance->second;
            previous_scores[subnetwork.states()[i]] = scores[i];
        }

        auto destinations = boundary_choices(subnetwork);

        std::vector<std::discrete_distribution<std::size_t>> exit_dists;
        exit_dists.reserve(entrance_states.size());
        for (Index c = 0; c < entrance_states.size(); ++c) {
            const auto exit_weights = subnetwork.exit_probabilities(integrals, c);
            exit_dists.emplace_back(exit_weights.begin(), exit_weights.end());
        }

        for (std::size_t t : unfinished) {
            const Index col = entrance_col[states[t]];
            auto &rng = generators[t];
            const std::size_t boundary_pos = exit_dists[col](rng);
            const Index b_state = subnetwork.boundary_states()[boundary_pos];
            const Float waiting_time = subnetwork.conditional_exit_time(integrals, b_state, col);
            auto &choice = destinations[b_state];

            if (!(waiting_time > static_cast<Float>(0)) || choice.destinations.empty() ||
                times[t] + waiting_time >= final_time) {
                finish_trajectory(trajectories[t], times[t], final_time);
                continue;
            }

            times[t] += waiting_time;
            std::discrete_distribution<std::size_t> rate_dist(choice.rates.begin(),
                                                              choice.rates.end());
            states[t] = choice.destinations[rate_dist(rng)];
            trajectories[t].times.push_back(times[t]);
            trajectories[t].states.push_back(states[t]);
            ++steps[t];
        }
    }
    return trajectories;
}

template <typename ReactionSystem, typename State = std::vector<int>, typename Float = double,
          typename Index = std::size_t, typename LevelFunction = SingleBlockLevel>
inline Trajectory<State, Float>
else_trajectory(const ReactionSystem &model, const std::vector<Float> &rates, const State &initial,
                Float initial_time, Float final_time, ELSEOptions<Index, Float> options = {},
                int seed = 42, LevelFunction level = {}) {
    auto ensemble = else_ensemble(model, rates, initial, 1, initial_time, final_time, options, seed,
                                  std::move(level));
    return std::move(ensemble.front());
}

} // namespace else_sim
