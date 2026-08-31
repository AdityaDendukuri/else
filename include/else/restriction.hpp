#pragma once

#include "else/state_graph.hpp"
#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <unordered_map>
#include <vector>

namespace else_sim {

template <typename State, typename Change>
inline State apply_change(State state, const Change &change) {
    for (std::size_t i = 0; i < change.size(); ++i)
        state[i] += change[i];
    return state;
}

namespace detail {

template <typename ReactionSystem, typename Rates, typename State, typename Float, typename Index,
          typename LevelFunction, typename FindState>
inline Subnetwork<Float, Index, State>
build_cme_subnetwork(const ReactionSystem &model, const Rates &rates, std::vector<State> states,
                     FindState find_state, LevelFunction level, bool factorize,
                     const std::vector<Float> *stationary_sqrt = nullptr) {
    std::vector<Index> rows, columns;
    std::vector<Float> values;
    std::vector<BoundaryTransition<Index, State, Float>> boundary;

    // Internal reactions enter R; reactions leaving the state set enter the boundary.
    for (Index column = 0; column < states.size(); ++column) {
        Float total_rate = static_cast<Float>(0);
        for (std::size_t reaction = 0; reaction < model.changes.size(); ++reaction) {
            const Float rate = static_cast<Float>(
                model.propensities[reaction](states[column], rates, static_cast<Float>(0)));
            if (rate <= static_cast<Float>(0))
                continue;

            State destination = apply_change(states[column], model.changes[reaction]);
            const Index row = find_state(destination);
            if (row == states.size())
                boundary.push_back({column, std::move(destination), rate});
            else {
                rows.push_back(row);
                columns.push_back(column);
                values.push_back(rate);
            }
            total_rate += rate;
        }
        // The diagonal contains every outgoing rate, including boundary reactions.
        rows.push_back(column);
        columns.push_back(column);
        values.push_back(-total_rate);
    }

    auto generator = SparseMatrix<Float, Index>::from_triplets(states.size(), states.size(), rows,
                                                               columns, values);
    auto levels = make_block_levels<Index>(states, level);
    if (stationary_sqrt)
        return {std::move(states), std::move(generator), std::move(boundary), *stationary_sqrt,
                std::move(levels)};
    return {std::move(states), std::move(generator), std::move(boundary), std::move(levels),
            factorize};
}

} // namespace detail

/// Restrict a reaction system to `states`; transitions leaving them form the boundary.
template <typename ReactionSystem, typename Rates, typename State = std::vector<int>,
          typename Float = double, typename Index = std::size_t,
          typename LevelFunction = SingleBlockLevel>
inline Subnetwork<Float, Index, State>
cme_subnetwork(const ReactionSystem &model, const Rates &rates, const std::vector<State> &states,
               LevelFunction level = {}, bool factorize = true) {
    std::unordered_map<State, Index, StateHash<State>> index;
    for (Index i = 0; i < states.size(); ++i)
        index[states[i]] = i;
    const auto find_state = [&](const State &state) {
        auto found = index.find(state);
        return found == index.end() ? static_cast<Index>(states.size()) : found->second;
    };
    return detail::build_cme_subnetwork<ReactionSystem, Rates, State, Float, Index>(
        model, rates, states, find_state, level, factorize);
}

/// Build a reversible restriction and factor its symmetric normalization.
template <typename ReactionSystem, typename Rates, typename State = std::vector<int>,
          typename Float = double, typename Index = std::size_t,
          typename LevelFunction = SingleBlockLevel>
inline Subnetwork<Float, Index, State>
reversible_cme_subnetwork(const ReactionSystem &model, const Rates &rates,
                          const std::vector<State> &states,
                          const std::vector<Float> &stationary_sqrt, LevelFunction level = {}) {
    std::unordered_map<State, Index, StateHash<State>> index;
    for (Index i = 0; i < states.size(); ++i)
        index[states[i]] = i;
    const auto find_state = [&](const State &state) {
        auto found = index.find(state);
        return found == index.end() ? static_cast<Index>(states.size()) : found->second;
    };
    return detail::build_cme_subnetwork<ReactionSystem, Rates, State, Float, Index>(
        model, rates, states, find_state, level, true, &stationary_sqrt);
}

/// Restrict a reaction system using persistent graph IDs and stable active slots.
template <typename ReactionSystem, typename Rates, typename State, typename Float = double,
          typename Index = std::size_t, typename LevelFunction = SingleBlockLevel>
inline Subnetwork<Float, Index, State>
cme_subnetwork(const ReactionSystem &model, const Rates &rates,
               const StateGraph<State, Index> &graph, const ActiveSlots<Index> &active,
               LevelFunction level = {}, bool factorize = true) {
    auto states = active_states(graph, active);
    const auto find_state = [&](const State &state) {
        const Index id = graph.find(state);
        if (id == StateGraph<State, Index>::invalid_id())
            return active.size();
        const Index slot = active.find(id);
        return slot == ActiveSlots<Index>::invalid_slot() ? active.size() : slot;
    };
    return detail::build_cme_subnetwork<ReactionSystem, Rates, State, Float, Index>(
        model, rates, std::move(states), find_state, level, factorize);
}

} // namespace else_sim
