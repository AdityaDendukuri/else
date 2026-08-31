// Restrict a reaction system to a finite state set.
#pragma once

#include "elsex/subnetwork.hpp"
#include "elsex/state_graph.hpp"
#include "elsex/types.hpp"
#include <unordered_map>
#include <utility>
#include <type_traits>
#include <span>
#include <vector>

namespace elsex {

/// Apply a stoichiometric change to a state.
template <typename State, typename Change>
[[nodiscard]] State apply_change(State state, const Change &change) {
    for (std::size_t i = 0; i < change.size(); ++i) {
        state[i] += change[i];
    }
    return state;
}

/// The principal restriction \f$R = \bar R_{S,S}\f$ of a reaction system.
///
/// Row convention: entry `(i, j)` is the rate from state `i` to state `j`. The
/// diagonal carries the total outgoing rate *including* transitions that leave
/// the set, so the row sums are negative by exactly the escaping rate and
/// \f$w = -(R\mathbf 1)\f$ holds. Every leaving transition is also recorded
/// individually as a boundary transition.
///
/// `model` must expose `changes` (stoichiometry per reaction) and `propensities`
/// (a callable per reaction taking state, rates, and time).
template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] Subnetwork<State> restriction(const ReactionSystem &model, const Rates &rates,
                                            std::vector<State> states,
                                            LevelFunction level = nullptr,
                                            bool factorize = true) {
    std::unordered_map<State, idx, StateHash<State>> position;
    position.reserve(states.size());
    for (idx i = 0; i < states.size(); ++i) {
        position.emplace(states[i], i);
    }

    std::vector<idx> rows, columns;
    std::vector<real> values;
    std::vector<BoundaryTransition<State>> boundary;

    for (idx source = 0; source < states.size(); ++source) {
        real outgoing = 0.0;
        for (std::size_t reaction = 0; reaction < model.changes.size(); ++reaction) {
            const real rate =
                static_cast<real>(model.propensities[reaction](states[source], rates, real(0)));
            if (rate <= 0.0) {
                continue;
            }

            State destination = apply_change(states[source], model.changes[reaction]);
            const auto found = position.find(destination);
            if (found == position.end()) {
                boundary.push_back({source, std::move(destination), rate});
            } else {
                rows.push_back(source);
                columns.push_back(found->second);
                values.push_back(rate);
            }
            outgoing += rate;
        }

        rows.push_back(source);
        columns.push_back(source);
        values.push_back(-outgoing);
    }

    auto generator =
        num::SparseMatrix::from_triplets(states.size(), states.size(), rows, columns, values);
    if constexpr (std::is_same_v<LevelFunction, std::nullptr_t>) {
        return Subnetwork<State>(std::move(states), std::move(generator), std::move(boundary), {},
                                 {}, factorize);
    } else {
        std::vector<idx> levels(states.size());
        for (idx i = 0; i < states.size(); ++i)
            levels[i] = static_cast<idx>(level(states[i]));
        return Subnetwork<State>(std::move(states), std::move(generator), std::move(boundary),
                                 {}, std::span<const idx>(levels), factorize);
    }
}

template <typename ReactionSystem, typename Rates, typename State,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] Subnetwork<State> restriction(const ReactionSystem &model, const Rates &rates,
                                            const StateGraph<State> &graph,
                                            const ActiveSlots &active,
                                            LevelFunction level = nullptr,
                                            bool factorize = true) {
    return restriction(model, rates, active_states(graph, active), level, factorize);
}

} // namespace elsex
