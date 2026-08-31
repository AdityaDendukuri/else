// Choose the state set for one ELSE step.
#pragma once

#include "elsex/restriction.hpp"
#include "elsex/shedding.hpp"
#include "elsex/state_graph.hpp"
#include "elsex/subnetwork.hpp"
#include "elsex/types.hpp"
#include <algorithm>
#include <random>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace elsex {

// Draw a weighted support without replacement and retain the original masses.
// The density calculation subsequently conditions on the represented mass.
template <typename State, typename RNG>
[[nodiscard]] std::unordered_map<State, real, StateHash<State>>
subsample_support(const std::unordered_map<State, real, StateHash<State>> &distribution,
                  idx capacity, RNG &random) {
    if (distribution.size() <= capacity)
        return distribution;

    std::vector<State> states;
    std::vector<real> weights;
    states.reserve(distribution.size());
    weights.reserve(distribution.size());
    for (const auto &[state, weight] : distribution) {
        states.push_back(state);
        weights.push_back(std::max(weight, 0.0));
    }

    std::unordered_map<State, real, StateHash<State>> selected;
    selected.reserve(capacity);
    for (idx draw = 0; draw < capacity; ++draw) {
        std::discrete_distribution<idx> choose(weights.begin(), weights.end());
        const idx position = choose(random);
        selected.emplace(states[position], distribution.at(states[position]));
        weights[position] = 0.0;
    }
    return selected;
}

template <typename ReactionSystem, typename Rates, typename State>
void expand_workspace(const ReactionSystem &model, const Rates &rates,
                      const std::vector<State> &entrances, int depth, real tolerance,
                      StateGraph<State> &graph, ActiveSlots &active) {
    std::vector<State> frontier;
    std::unordered_set<State, StateHash<State>> visited;
    for (const State &state : entrances) {
        active.insert(graph.insert(state));
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
                if (model.total_propensity(destination, rates, real(0)) <= tolerance)
                    continue;
                active.insert(graph.insert(destination));
                next.push_back(std::move(destination));
            }
        }
        frontier = std::move(next);
    }
}

/// Grow a state set outward from `seeds` by `depth` reactions.
///
/// States whose total propensity does not exceed `tolerance` are not admitted,
/// and neither are states with a negative coordinate.
template <typename ReactionSystem, typename Rates, typename State>
[[nodiscard]] std::vector<State> expanded_states(const ReactionSystem &model, const Rates &rates,
                                                 const std::vector<State> &seeds, int depth,
                                                 real tolerance) {
    std::vector<State> collected;
    std::unordered_set<State, StateHash<State>> seen;
    std::vector<State> frontier;
    for (const State &state : seeds) {
        if (seen.insert(state).second) {
            collected.push_back(state);
            frontier.push_back(state);
        }
    }

    while (depth-- > 0 && !frontier.empty()) {
        std::vector<State> next;
        for (const State &state : frontier) {
            for (const auto &change : model.changes) {
                State destination = apply_change(state, change);
                const bool representable = std::all_of(destination.begin(), destination.end(),
                                                       [](auto value) { return value >= 0; });
                if (!representable || !seen.insert(destination).second) {
                    continue;
                }
                if (model.total_propensity(destination, rates, real(0)) <= tolerance) {
                    continue;
                }
                collected.push_back(destination);
                next.push_back(std::move(destination));
            }
        }
        frontier = std::move(next);
    }
    return collected;
}

/// Drop the lowest-scoring states until only `capacity` remain, never a protected one.
///
/// `mixture` is \f$\rho\f$ over `states`, in the same order. This factors the
/// expanded operator to score it, and the caller then factors the kept operator
/// again, as in the paper's finite-ensemble algorithm. Avoiding that
/// second factorization is what the Woodbury update is for.
template <typename ReactionSystem, typename Rates, typename State>
[[nodiscard]] std::vector<State>
shed_to_capacity(const ReactionSystem &model, const Rates &rates, std::vector<State> states,
                 const num::Vector &mixture, std::span<const idx> protected_states,
                 const EnsembleOptions &options) {
    if (states.size() <= options.capacity) {
        return states;
    }

    const Subnetwork<State> expanded = restriction(model, rates, states);
    const SheddingState state = shedding_state(expanded, mixture);
    const num::Vector scores = options.rule == SheddingRule::ExactCutTime
                                   ? exact_cut_time_losses(expanded, state)
                                   : expected_entries(expanded, state);

    const std::vector<idx> shed = lowest_scores(std::span<const real>(scores.data(), scores.size()),
                                                protected_states, states.size() - options.capacity);

    std::vector<bool> remove(states.size(), false);
    for (idx j : shed) {
        remove[j] = true;
    }
    std::vector<State> kept;
    kept.reserve(states.size() - shed.size());
    for (idx j = 0; j < states.size(); ++j) {
        if (!remove[j]) {
            kept.push_back(std::move(states[j]));
        }
    }
    return kept;
}

} // namespace elsex
