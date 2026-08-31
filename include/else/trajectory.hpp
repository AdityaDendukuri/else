#pragma once

#include "else/factor_update.hpp"
#include "else/restriction.hpp"
#include "else/shedding.hpp"
#include "else/state_graph.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <numeric>
#include <optional>
#include <random>
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
inline void finish_trajectory(Trajectory<State, Float> &trajectory, Float &current_time,
                              Float final_time) {
    current_time = final_time;
    if (trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.back());
    }
}

template <typename ReactionSystem, typename State, typename Float, typename Index>
void expand_workspace(const ReactionSystem &model, const std::vector<Float> &rates,
                      const std::unordered_map<State, Float, StateHash<State>> &entrances,
                      int depth, Float tolerance, StateGraph<State, Index> &graph,
                      ActiveSlots<Index> &workspace) {
    // Grow the active state set outward from the occupied entrance states.
    std::vector<State> frontier;
    std::unordered_set<State, StateHash<State>> visited;
    for (const auto &entrance : entrances) {
        const State &state = entrance.first;
        workspace.insert(graph.insert(state));
        frontier.push_back(state);
        visited.insert(state);
    }

    while (depth-- > 0) {
        std::vector<State> next;
        for (const State &state : frontier) {
            for (const auto &change : model.changes) {
                State destination = apply_change(state, change);
                const bool valid = std::all_of(destination.begin(), destination.end(),
                                               [](auto value) { return value >= 0; });
                if (!valid || !visited.insert(destination).second)
                    continue;
                if (model.total_propensity(destination, rates, static_cast<Float>(0)) <= tolerance)
                    continue;
                workspace.insert(graph.insert(destination));
                next.push_back(std::move(destination));
            }
        }
        frontier = std::move(next);
    }
}

template <typename Float, typename Index, typename State>
[[nodiscard]] OccupationIntegrals<Matrix<Float>> occupation_integrals_from_base(
    const Subnetwork<Float, Index, State> &current, const Subnetwork<Float, Index, State> &base,
    const SparseMatrix<Float, Index> &base_generator, std::span<const Index> changed,
    const Matrix<Float> &right_hand_side) {
    if (changed.empty())
        return base.occupation_integrals(right_hand_side);

    // Correct the stored factorization for rows and columns whose state changed.
    const auto solve_base = [&](const std::vector<Float> &b, std::vector<Float> &x) {
        x = base.occupation(b);
    };
    const auto correction = factor_correction(base_generator, current.generator(), changed,
                                              solve_base, static_cast<Float>(-1));

    OccupationIntegrals<Matrix<Float>> result;
    solve_correction(correction, right_hand_side, result.occupation, solve_base);
    solve_correction(correction, result.occupation, result.time_weighted_occupation, solve_base);

    // A failed correction is refactorized by the caller.
    std::vector<Float> b(right_hand_side.rows()), x(right_hand_side.rows());
    for (Index i = 0; i < right_hand_side.rows(); ++i) {
        b[i] = right_hand_side(i, 0);
        x[i] = result.occupation(i, 0);
    }
    if (relative_residual<Float, Index>(current.generator(), x, b, static_cast<Float>(-1)) >
        static_cast<Float>(1e-8))
        throw std::runtime_error("factor correction residual is too large");
    return result;
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

    StateGraph<State, Index> graph;
    ActiveSlots<Index> workspace;
    // The current solve supplies the scores used for shedding on the next iteration.
    std::unordered_map<State, Float, StateHash<State>> previous_scores;
    using Network = Subnetwork<Float, Index, State>;
    std::optional<Network> base_network;
    SparseMatrix<Float, Index> base_generator;
    std::vector<Index> base_state_ids;
    while (true) {
        // Trajectories at the same entrance share the same matrix solves.
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

        for (auto &entrance : entrances)
            entrance.second /= static_cast<Float>(unfinished.size());

        const Index old_size = workspace.size();
        const int expansion_depth = old_size < options.capacity ? options.expansion_depth : 0;
        expand_workspace(model, rates, entrances, expansion_depth, options.tolerance, graph,
                         workspace);
        auto build_subnetwork = [&](bool factorize = true) {
            return cme_subnetwork<ReactionSystem, std::vector<Float>, State, Float, Index>(
                model, rates, graph, workspace, level, factorize);
        };

        // Shed before factorizing the generator used for advancement.
        if (workspace.size() > options.capacity) {
            std::vector<Index> protected_indices;
            for (const auto &entrance : entrances) {
                const Index pos = workspace.find(graph.find(entrance.first));
                if (pos != ActiveSlots<Index>::invalid_slot())
                    protected_indices.push_back(pos);
            }

            std::vector<Float> scores(workspace.size(), static_cast<Float>(0));
            if (expansion_depth == 0 && !previous_scores.empty()) {
                for (Index i = 0; i < workspace.size(); ++i) {
                    auto found = previous_scores.find(graph[workspace.state_ids[i]]);
                    if (found != previous_scores.end())
                        scores[i] = found->second;
                }
            } else {
                auto expanded = build_subnetwork();
                std::vector<Float> entrance_vec(workspace.size(), static_cast<Float>(0));
                for (const auto &entrance : entrances) {
                    const Index pos = workspace.find(graph.find(entrance.first));
                    if (pos != ActiveSlots<Index>::invalid_slot())
                        entrance_vec[pos] = entrance.second;
                }
                const auto occupation = expanded.occupation(entrance_vec);
                std::vector<Index> indices(workspace.size());
                std::iota(indices.begin(), indices.end(), static_cast<Index>(0));
                scores =
                    expected_visit_scores(expanded, occupation, std::span<const Index>(indices));
            }
            const Index remove_count = static_cast<Index>(workspace.size()) - options.capacity;
            const auto shed = lowest_scores<Float, Index>(scores, protected_indices, remove_count);
            std::vector<bool> remove(workspace.size(), false);
            for (Index i : shed)
                remove[i] = true;
            workspace.retain(remove, old_size);
        }

        // Stable slots make the new operator A = A0 + UV^T with rank at most twice
        // the number of replaced states.  Beyond three replacements, refactoring wins.
        std::vector<Index> changed;
        if (base_network && base_state_ids.size() == workspace.state_ids.size())
            for (Index i = 0; i < workspace.size(); ++i)
                if (workspace.state_ids[i] != base_state_ids[i])
                    changed.push_back(i);
        constexpr std::size_t maximum_changed_slots = 3;
        const bool reuse_factors = options.reuse_factorization && base_network &&
                                   base_state_ids.size() == workspace.state_ids.size() &&
                                   changed.size() <= maximum_changed_slots;
        auto subnetwork = build_subnetwork(!reuse_factors);

        if (subnetwork.boundary_states().empty()) {
            for (std::size_t t : unfinished) {
                finish_trajectory(trajectories[t], times[t], final_time);
            }
            break;
        }

        std::vector<State> entrance_states;
        std::unordered_map<State, Index, StateHash<State>> entrance_col;
        for (const auto &entrance : entrances) {
            entrance_col[entrance.first] = entrance_states.size();
            entrance_states.push_back(entrance.first);
        }

        Matrix<Float> entrance_matrix(subnetwork.size(), entrance_states.size(),
                                      static_cast<Float>(0));
        for (Index c = 0; c < entrance_states.size(); ++c) {
            const Index position = subnetwork.find(entrance_states[c]);
            if (position < subnetwork.size())
                entrance_matrix(position, c) = static_cast<Float>(1);
        }

        OccupationIntegrals<Matrix<Float>> integrals;
        bool store_as_base = !reuse_factors;
        if (reuse_factors) {
            try {
                integrals = occupation_integrals_from_base(
                    subnetwork, *base_network, base_generator, std::span<const Index>(changed),
                    entrance_matrix);
            } catch (const std::runtime_error &) {
                subnetwork = build_subnetwork();
                integrals = subnetwork.occupation_integrals(entrance_matrix);
                store_as_base = true;
            }
        } else {
            integrals = subnetwork.occupation_integrals(entrance_matrix);
        }

        // Average the entrance-specific columns U = Z E using the group weights.
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

        // Each trajectory samples independently from its entrance-specific exit law.
        for (std::size_t t : unfinished) {
            const Index col = entrance_col[states[t]];
            auto &rng = generators[t];
            const std::size_t boundary_pos = exit_dists[col](rng);
            const Index exit_state = subnetwork.boundary_states()[boundary_pos];
            const Float waiting_time = subnetwork.conditional_exit_time(integrals, exit_state, col);
            auto &choice = destinations[exit_state];

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

        if (options.reuse_factorization && store_as_base && workspace.size() == options.capacity) {
            base_state_ids = workspace.state_ids;
            base_generator = subnetwork.generator();
            base_network.emplace(std::move(subnetwork));
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
