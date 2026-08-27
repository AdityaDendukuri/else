#pragma once

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

/// Restrict a reaction system to `states`; transitions leaving them form the boundary.
template <typename ReactionSystem, typename Rates, typename State = std::vector<int>,
          typename Float = double, typename Index = std::size_t,
          typename LevelFunction = SingleBlockLevel>
inline Subnetwork<Float, Index, State>
cme_subnetwork(const ReactionSystem &model, const Rates &rates, const std::vector<State> &states,
               LevelFunction level = {}) {
    std::unordered_map<State, Index, StateHash<State>> index;
    for (Index i = 0; i < states.size(); ++i)
        index[states[i]] = i;

    std::vector<Index> rows, columns;
    std::vector<Float> values;
    std::vector<BoundaryTransition<Index, State, Float>> boundary;
    for (Index column = 0; column < states.size(); ++column) {
        Float total_rate = static_cast<Float>(0);
        for (std::size_t reaction = 0; reaction < model.changes.size(); ++reaction) {
            const Float rate = static_cast<Float>(
                model.propensities[reaction](states[column], rates, static_cast<Float>(0)));
            if (rate <= static_cast<Float>(0))
                continue;

            State destination = apply_change(states[column], model.changes[reaction]);
            auto found = index.find(destination);
            if (found == index.end()) {
                boundary.push_back({column, std::move(destination), rate});
            } else {
                rows.push_back(found->second);
                columns.push_back(column);
                values.push_back(rate);
            }
            total_rate += rate;
        }
        rows.push_back(column);
        columns.push_back(column);
        values.push_back(-total_rate);
    }

    auto generator = SparseMatrix<Float, Index>::from_triplets(states.size(), states.size(), rows,
                                                               columns, values);
    auto levels = make_block_levels<Index>(states, level);
    return {states, std::move(generator), std::move(boundary), std::move(levels)};
}

} // namespace else_sim
