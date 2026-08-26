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
struct StateSpace {
    std::vector<State> states;
    std::vector<Float> probs;
    std::unordered_map<State, int, StateHash<State>> index;

    [[nodiscard]] int size() const { return static_cast<int>(states.size()); }
    [[nodiscard]] bool empty() const { return states.empty(); }

    [[nodiscard]] int find(const State &x) const {
        auto it = index.find(x);
        return it != index.end() ? it->second : -1;
    }

    void add_state(const State &x, Float prob = static_cast<Float>(0)) {
        if (index.count(x)) return;
        int i = static_cast<int>(states.size());
        states.push_back(x);
        probs.push_back(prob);
        index[x] = i;
    }

    void remove_states(const std::vector<bool> &remove_mask) {
        if (std::none_of(remove_mask.begin(), remove_mask.end(), [](bool b) { return b; })) return;
        int n = static_cast<int>(states.size());
        std::vector<State> new_states;
        std::vector<Float> new_probs;
        new_states.reserve(n);
        new_probs.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (!remove_mask[i]) {
                new_states.push_back(states[i]);
                new_probs.push_back(probs[i]);
            }
        }
        states = std::move(new_states);
        probs = std::move(new_probs);
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

    StateSpace<State, Float> workspace;
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

        std::vector<State> frontier;
        std::unordered_set<State, StateHash<State>> visited;
        for (const auto &[s, w] : entrances) {
            workspace.add_state(s, w);
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
                        if (dest[i] < 0) { nonneg = false; break; }
                    }
                    if (!nonneg || !visited.insert(dest).second) continue;
                    Float total_rate = model.total_propensity(dest, rates, static_cast<Float>(0));
                    if (total_rate <= options.tolerance) continue;
                    workspace.add_state(dest);
                    next.push_back(dest);
                }
            }
            frontier = std::move(next);
        }

        auto build_subnetwork = [&]() {
            const Index n_ws = workspace.size();
            std::vector<Index> r_rows, r_cols;
            std::vector<Float> r_vals;
            std::vector<BoundaryTransition<Index, State, Float>> boundary;

            for (Index j = 0; j < n_ws; ++j) {
                const auto &x = workspace.states[j];
                Float col_sum = static_cast<Float>(0);
                for (std::size_t k = 0; k < model.size(); ++k) {
                    Float alpha = static_cast<Float>(model.propensities[k](x, rates, static_cast<Float>(0)));
                    if (alpha <= static_cast<Float>(0)) continue;
                    State y = x + model.changes[k];
                    int pos = workspace.find(y);
                    if (pos >= 0) {
                        r_rows.push_back(static_cast<Index>(pos));
                        r_cols.push_back(j);
                        r_vals.push_back(alpha);
                    } else {
                        boundary.push_back({j, y, alpha});
                    }
                    col_sum += alpha;
                }
                r_rows.push_back(j);
                r_cols.push_back(j);
                r_vals.push_back(-col_sum);
            }
            auto R = SparseMatrix<Float, Index>::from_triplets(n_ws, n_ws, r_rows, r_cols, r_vals);
            return Subnetwork<Float, Index, State>(workspace.states, std::move(R), std::move(boundary));
        };

        if (workspace.size() > static_cast<int>(options.capacity)) {
            auto expanded = build_subnetwork();
            std::vector<Float> entrance_vec(workspace.size(), static_cast<Float>(0));
            for (const auto &[s, w] : entrances) {
                int pos = workspace.find(s);
                if (pos >= 0) entrance_vec[static_cast<std::size_t>(pos)] = w;
            }
            const auto occ = expanded.occupation(entrance_vec);
            std::vector<Float> visits(workspace.size(), static_cast<Float>(0));
            for (Index i = 0; i < static_cast<Index>(workspace.size()); ++i) {
                Float incoming = entrance_vec[i];
                for (Index k = expanded.generator().row_ptr[i]; k < expanded.generator().row_ptr[i + 1]; ++k) {
                    Index j = expanded.generator().col_idx[k];
                    if (j != i) incoming += expanded.generator().values[k] * occ[j];
                }
                visits[i] = incoming;
            }
            std::vector<std::pair<Float, int>> candidates;
            for (int i = 0; i < workspace.size(); ++i) {
                if (!entrances.count(workspace.states[i])) {
                    candidates.push_back({visits[i], i});
                }
            }
            std::sort(candidates.begin(), candidates.end());
            const std::size_t remove_count = static_cast<std::size_t>(workspace.size() - static_cast<int>(options.capacity));
            std::vector<bool> remove(workspace.size(), false);
            for (std::size_t i = 0; i < remove_count && i < candidates.size(); ++i) {
                remove[candidates[i].second] = true;
            }
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
            if (pos >= 0) rhs(static_cast<std::size_t>(pos), c) = static_cast<Float>(1);
        }

        const auto integrals = subnetwork.occupation_integrals(rhs);
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
