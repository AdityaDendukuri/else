// Restrict a reaction system to a finite state set.
#pragma once

#include "else/core/state_graph.hpp"
#include "else/core/subnetwork.hpp"
#include "else/core/types.hpp"
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace else_sim {

// Apply a stoichiometric change to a state.
template <typename State, typename Change>
[[nodiscard]] State apply_change(State state, const Change &change) {
    for (idx i = 0; i < change.size(); ++i) {
        state[i] += change[i];
    }
    return state;
}

// Block level of every state, or an empty vector when `level` is `nullptr`.
template <typename State, typename LevelFunction>
[[nodiscard]] array<idx> compute_levels(const array<State> &states, LevelFunction level) {
    if constexpr (std::is_same_v<LevelFunction, std::nullptr_t>) {
        return {};
    } else {
        array<idx> levels(states.size());
        for (idx i = 0; i < states.size(); ++i)
            levels[i] = static_cast<idx>(level(states[i]));
        return levels;
    }
}

namespace detail {

template <typename State>
struct CachedTransition {
    State destination{};
    real rate = 0.0;
};

template <typename State>
struct RestrictionCache {
    array<State> states;
    array<array<CachedTransition<State>>> outgoing;
    array<real> values;
    array<idx> columns;
    array<idx> row_ptr;
};

// The generator triplets and boundary transitions for a finite state set.
template <typename State>
struct GeneratorBuild {
    array<idx> rows, columns;
    array<real> values;
    array<BoundaryTransition<State>> boundary;
};

// Row convention: entry (i,j) is the rate from state i to state j; the
// diagonal is the total outgoing rate, so row sums are -(R*1).
template <typename ReactionSystem, typename Rates, typename State>
[[nodiscard]] GeneratorBuild<State> build_generator(const ReactionSystem &model, const Rates &rates,
                                                    const array<State> &states) {
    table<State, idx> position;
    position.reserve(states.size());
    for (idx i = 0; i < states.size(); ++i) {
        position.emplace(states[i], i);
    }

    GeneratorBuild<State> built;
    for (idx source = 0; source < states.size(); ++source) {
        real outgoing = 0.0;
        for (idx reaction = 0; reaction < model.changes.size(); ++reaction) {
            const real rate =
                static_cast<real>(model.propensities[reaction](states[source], rates, real(0)));
            if (rate <= 0.0) {
                continue;
            }

            State destination = apply_change(states[source], model.changes[reaction]);
            const auto found = position.find(destination);
            if (found == position.end()) {
                built.boundary.push_back({source, std::move(destination), rate});
            } else {
                built.rows.push_back(source);
                built.columns.push_back(found->second);
                built.values.push_back(rate);
            }
            outgoing += rate;
        }

        built.rows.push_back(source);
        built.columns.push_back(source);
        built.values.push_back(-outgoing);
    }
    return built;
}

} // namespace detail

template <typename ReactionSystem, typename Rates, typename State>
[[nodiscard]] detail::RestrictionCache<State> make_restriction_cache(const ReactionSystem &model,
                                                                     const Rates &rates,
                                                                     const array<State> &states) {
    detail::RestrictionCache<State> cache;
    cache.states = states;
    cache.outgoing.resize(states.size());
    for (idx source = 0; source < states.size(); ++source) {
        for (idx reaction = 0; reaction < model.changes.size(); ++reaction) {
            const real rate =
                static_cast<real>(model.propensities[reaction](states[source], rates, real(0)));
            if (rate > 0.0)
                cache.outgoing[source].push_back(
                    {apply_change(states[source], model.changes[reaction]), rate});
        }
    }
    return cache;
}

template <typename ReactionSystem, typename Rates, typename State>
void refresh_restriction_cache(const ReactionSystem &model, const Rates &rates,
                               const array<State> &states, detail::RestrictionCache<State> &cache) {
    if (cache.states.size() != states.size() || cache.outgoing.size() != states.size()) {
        cache = make_restriction_cache(model, rates, states);
        return;
    }
    for (idx source = 0; source < states.size(); ++source) {
        if (cache.states[source] == states[source])
            continue;
        cache.outgoing[source].clear();
        for (idx reaction = 0; reaction < model.changes.size(); ++reaction) {
            const real rate =
                static_cast<real>(model.propensities[reaction](states[source], rates, real(0)));
            if (rate > 0.0)
                cache.outgoing[source].push_back(
                    {apply_change(states[source], model.changes[reaction]), rate});
        }
    }
    cache.states = states;
}

template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] Subnetwork<State>
restriction_cached(const ReactionSystem &model, const Rates &rates, array<State> states,
                   detail::RestrictionCache<State> &cache, LevelFunction level = nullptr,
                   bool factorize = true) {
    refresh_restriction_cache(model, rates, states, cache);
    table<State, idx> position;
    position.reserve(states.size());
    for (idx i = 0; i < states.size(); ++i)
        position.emplace(states[i], i);
    array<idx> columns, row_ptr(states.size() + 1, 0);
    array<real> values;
    array<BoundaryTransition<State>> boundary;
    for (idx source = 0; source < states.size(); ++source) {
        real outgoing = 0.0;
        for (const auto &transition : cache.outgoing[source]) {
            const auto found = position.find(transition.destination);
            if (found == position.end())
                boundary.push_back({source, transition.destination, transition.rate});
            else {
                columns.push_back(found->second);
                values.push_back(transition.rate);
            }
            outgoing += transition.rate;
        }
        columns.push_back(source);
        values.push_back(-outgoing);
        row_ptr[source + 1] = columns.size();
    }
    cache.columns = columns;
    cache.values = values;
    cache.row_ptr = row_ptr;
    num::spmat generator(states.size(), states.size(), std::move(values), std::move(columns),
                         std::move(row_ptr));
    const array<idx> levels = compute_levels(states, level);
    return make_subnetwork(std::move(states), std::move(generator), std::move(boundary), {},
                           view<const idx>(levels), factorize);
}

// The principal restriction R = R_bar(S,S) of a reaction system. `model` must
// expose `changes` (stoichiometry per reaction) and `propensities`.
template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] Subnetwork<State> restriction(const ReactionSystem &model, const Rates &rates,
                                            array<State> states, LevelFunction level = nullptr,
                                            bool factorize = true) {
    detail::GeneratorBuild<State> built = detail::build_generator(model, rates, states);
    auto generator = num::spmat::from_triplets(states.size(), states.size(), built.rows,
                                               built.columns, built.values);
    const array<idx> levels = compute_levels(states, level);
    return make_subnetwork(std::move(states), std::move(generator), std::move(built.boundary), {},
                           view<const idx>(levels), factorize);
}

// Restrict a reversible reaction system and factor its symmetric normalization.
template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] Subnetwork<State>
reversible_restriction(const ReactionSystem &model, const Rates &rates, array<State> states,
                       view<const real> stationary_sqrt, LevelFunction level = nullptr) {
    if (stationary_sqrt.size() != states.size())
        throw std::invalid_argument("one stationary weight is required per state");

    Subnetwork<State> unfactored = restriction(model, rates, states, nullptr, false);
    num::vec stationary(states.size());
    for (idx i = 0; i < states.size(); ++i)
        stationary[i] = stationary_sqrt[i] * stationary_sqrt[i];

    const array<idx> levels = compute_levels(states, level);
    return make_subnetwork(std::move(states), std::move(unfactored.generator),
                           std::move(unfactored.boundary), std::move(stationary),
                           view<const idx>(levels));
}

template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] Subnetwork<State>
restriction(const ReactionSystem &model, const Rates &rates, const StateGraph<State> &graph,
            const ActiveSlots &active, LevelFunction level = nullptr, bool factorize = true) {
    return restriction(model, rates, active_states(graph, active), level, factorize);
}

} // namespace else_sim
