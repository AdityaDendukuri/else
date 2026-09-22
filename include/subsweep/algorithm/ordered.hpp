// Ordered subsweep (Algorithm 1): labeled walkers advance through one
// shared subnetwork per sweep, each sampling its own exit.
#pragma once

#include "subsweep/algorithm/row_system.hpp"
#include "subsweep/algorithm/system.hpp"
#include "subsweep/sweep/subnetwork.hpp"
#include "stochastic/categorical.hpp"
#include "stochastic/rng.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <optional>
#include <stdexcept>

namespace subsweep {

namespace detail {

// The distinct current slots i_1..i_d of the unfinished walkers with the
// number of walkers c_k at each.
struct current_set {
    array<idx> slots, counts;
    table<idx, idx> row;
};

[[nodiscard]] inline current_set group_by_slot(const array<idx> &walker_slot,
                                               const array<bool> &unfinished) {
    current_set current;
    for (idx t = 0; t < walker_slot.size(); ++t) {
        if (!unfinished[t])
            continue;
        const auto [entry, added] = current.row.emplace(walker_slot[t], current.slots.size());
        if (added) {
            num::append(current.slots, walker_slot[t]);
            num::append(current.counts, 0);
        }
        ++current.counts[entry->second];
    }
    return current;
}

template <typename S>
void remove(S &s, subnetwork &sn, idx slot, array<idx> &walker_slot, array<idx> &others) {
    const idx last = sn.size() - 1;
    s.discard(slot);
    sn.discard(slot);
    for (idx &w : walker_slot)
        if (w == last)
            w = slot;
    for (idx &o : others)
        if (o == last)
            o = slot;
}

// Grow by `expansion_depth` layers around the current slots, then shed to
// capacity: with fresh scores after an expansion, with the last sweep's
// scores otherwise. Slots are removed from the highest down.
template <typename S>
void select_active_states(S &s, subnetwork &sn, array<idx> &walker_slot,
                          const current_set &current, const std::optional<num::vec> &last_scores) {
    const sweep_options &options = sn.options();
    bool expanded = false;
    if constexpr (expandable_system<S>) {
        array<idx> layer = current.slots;
        for (int depth = 0; depth < options.expansion_depth && sn.size() < options.capacity;
             ++depth) {
            array<idx> next;
            for (idx slot : layer)
                s.expand(slot, [&](const grown_state &g) {
                    if (!g.fresh)
                        return;
                    sn.add(g.total_rate, g.row, g.column, g.level, g.stationary);
                    expanded = true;
                    num::append(next, g.slot);
                });
            std::sort(next.begin(), next.end(), std::greater<idx>());
            array<idx> admitted;
            for (idx slot : next) {
                if (sn.total_rate(slot) > options.tolerance)
                    num::append(admitted, slot);
                else
                    remove(s, sn, slot, walker_slot, admitted);
            }
            layer = std::move(admitted);
        }
    }
    if (sn.size() <= options.capacity)
        return;
    num::vec scores(sn.size(), 0.0);
    if (!expanded && last_scores) {
        for (idx j = 0; j < std::min(sn.size(), last_scores->size()); ++j)
            scores[j] = (*last_scores)[j];
    } else {
        num::vec rho(sn.size(), 0.0);
        idx walkers = 0;
        for (idx count : current.counts)
            walkers += count;
        for (idx k = 0; k < current.slots.size(); ++k)
            rho[current.slots[k]] = static_cast<real>(current.counts[k]) / walkers;
        scores = sn.scores(std::move(rho));
    }
    array<idx> shed =
        lowest_scores(scores.span(), current.slots, sn.size() - options.capacity, sn.levels());
    std::sort(shed.begin(), shed.end(), std::greater<idx>());
    array<idx> none;
    for (idx slot : shed)
        remove(s, sn, slot, walker_slot, none);
}

} // namespace detail

// Advance the walkers from their slots to `final_time`. Walkers at the same
// slot share the rows U = EZ and V = EZ^2; time advances by the conditional
// mean sweep duration. Returns each walker's jump times, one per `grow`.
template <system S>
array<array<real>> ordered_subsweep(S &s, subnetwork &sn, array<idx> walker_slot,
                                    real initial_time, real final_time, unsigned seed = 42,
                                    run_diagnostics *diagnostics = nullptr) {
    using clock = std::chrono::steady_clock;
    const auto seconds = [](clock::time_point start) {
        return std::chrono::duration<double>(clock::now() - start).count();
    };
    const sweep_options &options = sn.options();
    const idx count = walker_slot.size();
    if (count == 0 || initial_time > final_time || options.capacity == 0)
        throw std::invalid_argument("ordered subsweep needs walkers, a horizon, and a capacity");

    array<array<real>> jumps(count);
    array<real> times(count, initial_time);
    array<idx> sweeps(count, 0);
    array<num::rng> generators;
    for (idx t = 0; t < count; ++t)
        generators.emplace_back(seed + static_cast<unsigned>(t));
    std::optional<num::vec> last_scores;
    sweep_solution solution; // kept across sweeps so its buffers are reused

    while (true) {
        // A walker at a slot with no outgoing rate is absorbed.
        array<bool> unfinished(count, false);
        idx active = 0;
        for (idx t = 0; t < count; ++t) {
            unfinished[t] = times[t] < final_time && sweeps[t] < options.maximum_sweeps &&
                            sn.total_rate(walker_slot[t]) > 0.0;
            active += unfinished[t] ? 1 : 0;
        }
        if (active == 0)
            break;
        auto current = detail::group_by_slot(walker_slot, unfinished);

        const auto selection_start = clock::now();
        detail::select_active_states(s, sn, walker_slot, current, last_scores);
        current = detail::group_by_slot(walker_slot, unfinished);
        const double selection_seconds = seconds(selection_start);

        sn.solve(current.slots, current.counts, solution);

        const auto score_start = clock::now();
        last_scores = sn.scores(solution);
        const double score_seconds = seconds(score_start);

        const auto exit_setup_start = clock::now();
        array<num::categorical_sampler> exit_sampler;
        for (idx k = 0; k < current.slots.size(); ++k)
            exit_sampler.emplace_back(solution.exit_law(k).span());
        const double exit_setup_seconds = seconds(exit_setup_start);

        // Sample each walker's pre-exit slot and duration; the system samples
        // the exterior destination and the subnetwork grows by it.
        const auto sampling_start = clock::now();
        for (idx t = 0; t < count; ++t) {
            if (!unfinished[t])
                continue;
            const idx k = current.row.at(walker_slot[t]);
            const idx j = exit_sampler[k](generators[t]);
            const real duration = solution.duration(k, j);
            if (!(duration > 0.0) || times[t] + duration >= final_time) {
                times[t] = final_time;
                continue;
            }
            const grown_state g = s.grow(t, j, solution.size());
            if (g.fresh)
                sn.add(g.total_rate, g.row, g.column, g.level, g.stationary);
            times[t] += duration;
            walker_slot[t] = g.slot;
            num::append(jumps[t], times[t]);
            ++sweeps[t];
        }
        const double sampling_seconds = seconds(sampling_start);

        if (diagnostics) {
            sweep_diagnostics d = solution.diagnostics;
            d.active_walkers = active;
            d.selection_seconds = selection_seconds;
            d.score_seconds = score_seconds;
            d.exit_setup_seconds = exit_setup_seconds;
            d.sampling_seconds = sampling_seconds;
            num::append(diagnostics->sweeps, d);
        }
    }
    return jumps;
}

// A row system's walkers from its initial state, as labeled trajectories.
template <typename State, typename R, typename Level, typename Stationary>
[[nodiscard]] array<trajectory<State>>
ordered_paths(row_system<State, R, Level, Stationary> &s, const sweep_options &options,
              real initial_time, real final_time, unsigned seed = 42,
              run_diagnostics *diagnostics = nullptr) {
    subnetwork sn(options);
    const grown_state &g = s.initial();
    sn.add(g.total_rate, g.row, g.column, g.level, g.stationary);
    const auto jumps = ordered_subsweep(s, sn, array<idx>(s.paths().size(), 0), initial_time,
                                        final_time, seed, diagnostics);
    return s.trajectories(initial_time, jumps, final_time);
}

} // namespace subsweep
