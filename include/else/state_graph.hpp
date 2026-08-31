#pragma once

#include "else/types.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace else_sim {

template <typename State = std::vector<int>, typename StateId = std::size_t>
struct StateGraph {
    // A state keeps the same ID whenever it reappears in the subnetwork.
    std::vector<State> states;
    std::unordered_map<State, StateId, StateHash<State>> ids;

    [[nodiscard]] StateId find(const State &state) const {
        auto found = ids.find(state);
        return found == ids.end() ? invalid_id() : found->second;
    }

    StateId insert(const State &state) {
        auto [found, added] = ids.emplace(state, static_cast<StateId>(states.size()));
        if (added)
            states.push_back(state);
        return found->second;
    }

    [[nodiscard]] const State &operator[](StateId id) const { return states[id]; }
    [[nodiscard]] static constexpr StateId invalid_id() { return static_cast<StateId>(-1); }
};

template <typename Index = std::size_t>
struct ActiveSlots {
    // Slots are the row and column indices of the current generator.
    std::vector<Index> state_ids;
    std::unordered_map<Index, Index> slots;

    [[nodiscard]] Index size() const { return static_cast<Index>(state_ids.size()); }

    [[nodiscard]] Index find(Index id) const {
        auto found = slots.find(id);
        return found == slots.end() ? invalid_slot() : found->second;
    }

    Index insert(Index id) {
        auto found = slots.find(id);
        if (found != slots.end())
            return found->second;
        const Index slot = size();
        state_ids.push_back(id);
        slots.emplace(id, slot);
        return slot;
    }

    // New states fill holes left by shedding, so unchanged rows keep their slots.
    void retain(const std::vector<bool> &remove, Index old_size) {
        if (remove.size() != state_ids.size() || old_size > size())
            throw std::invalid_argument("invalid active-slot removal mask");

        const Index target_size =
            static_cast<Index>(std::count(remove.begin(), remove.end(), false));
        if (old_size > target_size)
            throw std::invalid_argument("cannot preserve more slots than the target capacity");

        std::vector<Index> retained(target_size, invalid_slot());
        std::vector<Index> holes;
        std::vector<Index> appended;
        for (Index slot = 0; slot < old_size; ++slot) {
            if (remove[slot])
                holes.push_back(slot);
            else
                retained[slot] = state_ids[slot];
        }
        for (Index slot = old_size; slot < target_size; ++slot)
            holes.push_back(slot);
        for (Index slot = old_size; slot < size(); ++slot)
            if (!remove[slot])
                appended.push_back(state_ids[slot]);
        if (holes.size() != appended.size())
            throw std::invalid_argument("active-slot replacement must preserve capacity");

        for (Index k = 0; k < holes.size(); ++k)
            retained[holes[k]] = appended[k];
        state_ids = std::move(retained);
        slots.clear();
        for (Index slot = 0; slot < size(); ++slot)
            slots[state_ids[slot]] = slot;
    }

    [[nodiscard]] static constexpr Index invalid_slot() { return static_cast<Index>(-1); }
};

template <typename State, typename Index>
[[nodiscard]] std::vector<State> active_states(const StateGraph<State, Index> &graph,
                                               const ActiveSlots<Index> &active) {
    std::vector<State> states;
    states.reserve(active.state_ids.size());
    for (Index id : active.state_ids)
        states.push_back(graph[id]);
    return states;
}

} // namespace else_sim
