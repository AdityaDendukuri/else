#pragma once

#include "elsex/types.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace elsex {

template <typename State>
struct StateGraph {
    std::vector<State> states;
    std::unordered_map<State, idx, StateHash<State>> ids;

    [[nodiscard]] idx find(const State &state) const {
        const auto found = ids.find(state);
        return found == ids.end() ? invalid() : found->second;
    }

    idx insert(const State &state) {
        const auto [found, added] = ids.emplace(state, states.size());
        if (added)
            states.push_back(state);
        return found->second;
    }

    [[nodiscard]] idx size() const { return states.size(); }
    [[nodiscard]] const State &operator[](idx id) const { return states[id]; }
    [[nodiscard]] static constexpr idx invalid() { return static_cast<idx>(-1); }
};

struct ActiveSlots {
    std::vector<idx> state_ids;
    std::unordered_map<idx, idx> slots;

    [[nodiscard]] idx size() const { return state_ids.size(); }
    [[nodiscard]] const std::vector<idx> &identities() const { return state_ids; }
    [[nodiscard]] std::vector<idx> changed_since(std::span<const idx> previous) const {
        if (previous.size() != size())
            return {};
        std::vector<idx> changed;
        for (idx slot = 0; slot < size(); ++slot)
            if (state_ids[slot] != previous[slot])
                changed.push_back(slot);
        return changed;
    }
    [[nodiscard]] idx find(idx id) const {
        const auto found = slots.find(id);
        return found == slots.end() ? invalid() : found->second;
    }
    idx insert(idx id) {
        const auto found = slots.find(id);
        if (found != slots.end())
            return found->second;
        const idx slot = size();
        state_ids.push_back(id);
        slots.emplace(id, slot);
        return slot;
    }

    // Fill removed old slots with newly appended states. Unchanged states keep
    // their row and column indices.
    void retain(const std::vector<bool> &remove, idx old_size) {
        if (remove.size() != size() || old_size > size())
            throw std::invalid_argument("invalid active-slot removal mask");
        const idx target = std::count(remove.begin(), remove.end(), false);
        if (old_size > target)
            throw std::invalid_argument("cannot preserve more slots than the target size");

        std::vector<idx> kept(target, invalid());
        std::vector<idx> holes;
        std::vector<idx> appended;
        for (idx slot = 0; slot < old_size; ++slot) {
            if (remove[slot])
                holes.push_back(slot);
            else
                kept[slot] = state_ids[slot];
        }
        for (idx slot = old_size; slot < target; ++slot)
            holes.push_back(slot);
        for (idx slot = old_size; slot < size(); ++slot)
            if (!remove[slot])
                appended.push_back(state_ids[slot]);
        if (holes.size() != appended.size())
            throw std::invalid_argument("active-slot replacement changed the capacity");
        for (idx k = 0; k < holes.size(); ++k)
            kept[holes[k]] = appended[k];

        state_ids = std::move(kept);
        slots.clear();
        for (idx slot = 0; slot < size(); ++slot)
            slots.emplace(state_ids[slot], slot);
    }

    [[nodiscard]] static constexpr idx invalid() { return static_cast<idx>(-1); }
};

template <typename State>
[[nodiscard]] std::vector<State> active_states(const StateGraph<State> &graph,
                                               const ActiveSlots &active) {
    std::vector<State> states;
    states.reserve(active.size());
    for (idx id : active.state_ids)
        states.push_back(graph[id]);
    return states;
}

} // namespace elsex
