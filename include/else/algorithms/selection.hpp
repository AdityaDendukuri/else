// Choose the state set for one ELSE step.
#pragma once

#include "else/core/state_graph.hpp"
#include "else/core/subnetwork.hpp"
#include "else/core/types.hpp"
#include "else/quantities/shedding.hpp"
#include "else/restriction/restriction.hpp"
#include "stochastic/categorical.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace else_sim {

// Draw a weighted support without replacement and retain the original masses.
// The density calculation subsequently conditions on the represented mass.
template <typename State, typename RNG>
[[nodiscard]] table<State, real> subsample_support(const table<State, real> &distribution,
                                                   idx capacity, RNG &random) {
    if (distribution.size() <= capacity)
        return distribution;

    array<State> states;
    array<real> weights;
    states.reserve(distribution.size());
    weights.reserve(distribution.size());
    for (const auto &[state, weight] : distribution) {
        states.push_back(state);
        weights.push_back(std::max(weight, 0.0));
    }

    real remaining = std::accumulate(weights.begin(), weights.end(), 0.0);
    table<State, real> selected;
    selected.reserve(capacity);
    for (idx draw = 0; draw < capacity && remaining > 0.0; ++draw) {
        const idx position = num::sample_categorical(view<const real>(weights), random);
        selected.emplace(states[position], distribution.at(states[position]));
        remaining -= weights[position];
        weights[position] = 0.0;
    }
    return selected;
}

// Draw density particles with replacement and aggregate repeated states.
// Each of the `particles` draws carries the same fraction of the input mass,
// so the returned weights sum to the represented mass of `distribution`.
template <typename State, typename RNG>
[[nodiscard]] table<State, real>
multinomial_particle_resample(const table<State, real> &distribution, idx particles, RNG &random) {
    if (particles == 0 || distribution.empty())
        return {};

    array<State> states;
    array<real> weights;
    states.reserve(distribution.size());
    weights.reserve(distribution.size());
    real total = 0.0;
    for (const auto &[state, weight] : distribution) {
        const real nonnegative = std::max(weight, 0.0);
        states.push_back(state);
        weights.push_back(nonnegative);
        total += nonnegative;
    }
    if (!(total > 0.0))
        return {};

    num::categorical_sampler choose{view<const real>(weights)};
    table<State, real> selected;
    selected.reserve(std::min<idx>(states.size(), particles));
    const real particle_mass = total / static_cast<real>(particles);
    for (idx draw = 0; draw < particles; ++draw)
        selected[states[choose(random)]] += particle_mass;
    return selected;
}

template <typename State, typename RNG>
[[nodiscard]] table<State, real> resample_density_support(const table<State, real> &distribution,
                                                          const EnsembleOptions &options,
                                                          RNG &random) {
    switch (options.density_resampling) {
    case DensityResampling::WeightedSupport:
        return subsample_support(distribution, options.capacity, random);
    case DensityResampling::MultinomialParticles: {
        const idx particles =
            options.density_particles == 0 ? options.capacity : options.density_particles;
        if (particles > options.capacity)
            throw std::invalid_argument(
                "density particle count must not exceed subnetwork capacity");
        return multinomial_particle_resample(distribution, particles, random);
    }
    }
    return {};
}

// Depth-bounded reaction-neighborhood BFS shared by `expand_workspace` and
// `expanded_states`; they differ only in what `admit` does with each state.
template <typename ReactionSystem, typename Rates, typename State, typename Admit>
void expand_frontier(const ReactionSystem &model, const Rates &rates, const array<State> &seeds,
                     int depth, real tolerance, Admit &&admit) {
    key_set<State> visited;
    array<State> frontier;
    for (const State &state : seeds) {
        if (visited.insert(state).second) {
            admit(state);
            frontier.push_back(state);
        }
    }

    while (depth-- > 0 && !frontier.empty()) {
        array<State> next;
        for (const State &state : frontier) {
            for (const auto &change : model.changes) {
                State destination = apply_change(state, change);
                const bool representable = std::all_of(destination.begin(), destination.end(),
                                                       [](auto value) { return value >= 0; });
                if (!representable || !visited.insert(destination).second)
                    continue;
                if (model.total_propensity(destination, rates, real(0)) <= tolerance)
                    continue;
                admit(destination);
                next.push_back(std::move(destination));
            }
        }
        frontier = std::move(next);
    }
}

// Grow the workspace outward from `entrances` by `depth` reactions, recording
// every reachable state into `graph`/`active`.
template <typename ReactionSystem, typename Rates, typename State>
void expand_workspace(const ReactionSystem &model, const Rates &rates,
                      const array<State> &entrances, int depth, real tolerance,
                      StateGraph<State> &graph, ActiveSlots &active) {
    expand_frontier(model, rates, entrances, depth, tolerance,
                    [&](const State &state) { insert(active, insert(graph, state)); });
}

// Grow a state set outward from `seeds` by `depth` reactions, admitting only
// non-negative states with total propensity above `tolerance`.
template <typename ReactionSystem, typename Rates, typename State>
[[nodiscard]] array<State> expanded_states(const ReactionSystem &model, const Rates &rates,
                                           const array<State> &seeds, int depth, real tolerance) {
    array<State> collected;
    expand_frontier(model, rates, seeds, depth, tolerance,
                    [&](const State &state) { collected.push_back(state); });
    return collected;
}

// Drop the lowest-scoring states until only `capacity` remain, never a
// protected one. `mixture` is rho over `states`, in the same order.
template <typename ReactionSystem, typename Rates, typename State>
[[nodiscard]] array<State> shed_to_capacity(const ReactionSystem &model, const Rates &rates,
                                            array<State> states, const num::vec &mixture,
                                            view<const idx> protected_states,
                                            const EnsembleOptions &options) {
    if (states.size() <= options.capacity) {
        return states;
    }

    const Subnetwork<State> expanded = restriction(model, rates, states);
    const SheddingState state = shedding_state(expanded, mixture);
    const num::vec scores = score_states(expanded, state, options);

    const array<idx> shed =
        lowest_scores(scores.span(), protected_states, states.size() - options.capacity);

    array<bool> remove(states.size(), false);
    for (idx j : shed) {
        remove[j] = true;
    }
    array<State> kept;
    kept.reserve(states.size() - shed.size());
    for (idx j = 0; j < states.size(); ++j) {
        if (!remove[j]) {
            kept.push_back(std::move(states[j]));
        }
    }
    return kept;
}

} // namespace else_sim
