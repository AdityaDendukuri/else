// Workspace bookkeeping: which states are known, and which occupy a slot in
// the active subnetwork.
#pragma once

#include "else/core/types.hpp"
#include <algorithm>
#include <stdexcept>

namespace else_sim {

// Sentinel for "no such state/slot", shared by StateGraph and ActiveSlots.
inline constexpr idx invalid_index = static_cast<idx>(-1);

// Every state seen so far, with a stable identity assigned on first sight.
template <typename State>
struct StateGraph {
    array<State> states;
    table<State, idx> ids;
};

template <typename State>
[[nodiscard]] idx size(const StateGraph<State> &graph) {
    return graph.states.size();
}

template <typename State>
[[nodiscard]] idx find(const StateGraph<State> &graph, const State &state) {
    const auto found = graph.ids.find(state);
    return found == graph.ids.end() ? invalid_index : found->second;
}

template <typename State>
idx insert(StateGraph<State> &graph, const State &state) {
    const auto [found, added] = graph.ids.emplace(state, graph.states.size());
    if (added)
        graph.states.push_back(state);
    return found->second;
}

// Which state identities currently occupy a slot in the active subnetwork.
// `state_ids[slot]` is the identity at that slot; `slots` inverts that map.
struct ActiveSlots {
    array<idx> state_ids;
    table<idx, idx> slots;
};

[[nodiscard]] inline idx size(const ActiveSlots &active) {
    return active.state_ids.size();
}

[[nodiscard]] inline array<idx> changed_since(const ActiveSlots &active, view<const idx> previous) {
    if (previous.size() != size(active))
        return {};
    array<idx> changed;
    for (idx slot = 0; slot < size(active); ++slot)
        if (active.state_ids[slot] != previous[slot])
            changed.push_back(slot);
    return changed;
}

[[nodiscard]] inline idx find(const ActiveSlots &active, idx id) {
    const auto found = active.slots.find(id);
    return found == active.slots.end() ? invalid_index : found->second;
}

inline idx insert(ActiveSlots &active, idx id) {
    const auto found = active.slots.find(id);
    if (found != active.slots.end()) {

        return found->second;
    }
    const idx slot = size(active);
    active.state_ids.push_back(id);
    active.slots.emplace(id, slot);
    return slot;
}

// Fill removed old slots with newly appended states, so unchanged slots
// never move. `remove` marks every slot (old or appended) that won't survive.
inline void retain(ActiveSlots &active, const array<bool> &remove, idx old_size) {
    if (remove.size() != size(active) || old_size > size(active))
        throw std::invalid_argument("invalid active-slot removal mask");
    const idx target = std::count(remove.begin(), remove.end(), false);
    if (old_size > target)
        throw std::invalid_argument("cannot preserve more slots than the target size");

    array<idx> kept(target, invalid_index);
    array<idx> holes;
    array<idx> appended;
    for (idx slot = 0; slot < old_size; ++slot) {
        if (remove[slot])
            holes.push_back(slot);
        else
            kept[slot] = active.state_ids[slot];
    }
    for (idx slot = old_size; slot < target; ++slot)
        holes.push_back(slot);
    for (idx slot = old_size; slot < size(active); ++slot)
        if (!remove[slot])
            appended.push_back(active.state_ids[slot]);
    if (holes.size() != appended.size())
        throw std::invalid_argument("active-slot replacement changed the capacity");
    for (idx k = 0; k < holes.size(); ++k)
        kept[holes[k]] = appended[k];

    active.state_ids = std::move(kept);
    active.slots.clear();
    for (idx slot = 0; slot < size(active); ++slot)
        active.slots.emplace(active.state_ids[slot], slot);
}

template <typename State>
[[nodiscard]] array<State> active_states(const StateGraph<State> &graph,
                                         const ActiveSlots &active) {
    array<State> states;
    states.reserve(size(active));
    for (idx id : active.state_ids)
        states.push_back(graph.states[id]);
    return states;
}

} // namespace else_sim
