// Finite ensemble ELSE.
#pragma once

#include "else/algorithms/selection.hpp"
#include "else/core/state_graph.hpp"
#include "else/core/subnetwork.hpp"
#include "else/core/types.hpp"
#include "else/quantities/factor_update.hpp"
#include "else/quantities/law.hpp"
#include "else/quantities/shedding.hpp"
#include "else/restriction/restriction.hpp"
#include "stochastic/categorical.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace else_sim {

namespace detail {

// The distinct entrances of the active trajectories with their multiplicities.
template <typename State>
struct Entrances {
    array<State> states;   // i_k
    array<idx> counts;     // c_k
    table<State, idx> row; // i_k -> k
    idx active = 0;        // m_active
};

template <typename State>
[[nodiscard]] idx size(const Entrances<State> &entrances) {
    return entrances.states.size();
}

template <typename State>
[[nodiscard]] Entrances<State> group_by_entrance(const array<State> &current,
                                                 const array<bool> &unfinished) {
    Entrances<State> entrances;
    for (idx t = 0; t < current.size(); ++t) {
        if (!unfinished[t]) {
            continue;
        }
        ++entrances.active;
        const auto [entry, added] = entrances.row.emplace(current[t], entrances.states.size());
        if (added) {
            entrances.states.push_back(current[t]);
            entrances.counts.push_back(0);
        }
        ++entrances.counts[entry->second];
    }
    return entrances;
}

// rho = sum_k (c_k/m_active) at entrance i_k.
template <typename State>
[[nodiscard]] num::vec entrance_mixture(const Subnetwork<State> &subnetwork,
                                        const Entrances<State> &entrances) {
    num::vec mixture(size(subnetwork), 0.0);
    for (idx k = 0; k < size(entrances); ++k) {
        const idx position = find(subnetwork, entrances.states[k]);
        if (position < size(subnetwork)) {
            mixture[position] =
                static_cast<real>(entrances.counts[k]) / static_cast<real>(entrances.active);
        }
    }
    return mixture;
}

// Transitions leaving each escape state, indexed by position in `escape_states`.
template <typename State>
struct Channels {
    array<array<State>> destinations;
    array<array<real>> rates;
};

template <typename State>
[[nodiscard]] Channels<State> escape_channels(const Subnetwork<State> &subnetwork) {
    const auto &escape_states = subnetwork.escape_states;
    table<idx, idx> position;
    for (idx p = 0; p < escape_states.size(); ++p) {
        position.emplace(escape_states[p], p);
    }

    Channels<State> channels;
    channels.destinations.resize(escape_states.size());
    channels.rates.resize(escape_states.size());
    for (const auto &transition : subnetwork.boundary) {
        const auto found = position.find(transition.source);
        if (found != position.end()) {
            channels.destinations[found->second].push_back(transition.destination);
            channels.rates[found->second].push_back(transition.rate);
        }
    }
    return channels;
}

// Close a trajectory out at `final_time`.
template <typename State>
void finish(Trajectory<State> &trajectory, real &time, real final_time) {
    time = final_time;
    if (final_time < std::numeric_limits<real>::infinity() &&
        trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.back());
    }
}

// Grow the workspace around `entrances`, then shed back to `options.capacity`
// if needed. When nothing expanded (`depth == 0`), scores are reused from
// `previous_scores` (set by `record_heuristic_scores` last macrostep) instead
// of a fresh restriction and factorization.
template <typename ReactionSystem, typename Rates, typename State>
void select_active_states(const ReactionSystem &model, const Rates &rates,
                          const Entrances<State> &entrances, const EnsembleOptions &options,
                          const table<State, real> &previous_scores, StateGraph<State> &graph,
                          ActiveSlots &active) {
    const idx old_size = size(active);
    const int depth = old_size < options.capacity ? options.expansion_depth : 0;
    expand_workspace(model, rates, entrances.states, depth, options.tolerance, graph, active);

    if (size(active) <= options.capacity) {
        return;
    }

    array<idx> protected_states;
    protected_states.reserve(size(entrances));
    for (const State &entrance : entrances.states) {
        const idx state_id = find(graph, entrance);
        const idx position = find(active, state_id);
        if (position < size(active)) {
            protected_states.push_back(position);
        }
    }

    num::vec scores(size(active), 0.0);
    if (depth == 0 && !previous_scores.empty()) {
        for (idx i = 0; i < size(active); ++i) {
            const auto found = previous_scores.find(graph.states[active.state_ids[i]]);
            if (found != previous_scores.end())
                scores[i] = found->second;
        }
    } else {
        const Subnetwork<State> expanded = restriction(model, rates, graph, active);
        const SheddingState state = shedding_state(expanded, entrance_mixture(expanded, entrances));
        scores = score_states(expanded, state, options);
    }

    const auto shed =
        lowest_scores(scores.span(), protected_states, size(active) - options.capacity);
    array<bool> remove(size(active), false);
    for (idx slot : shed)
        remove[slot] = true;
    retain(active, remove, old_size);
}

// The subnetwork and entrance law for one macrostep, and whether it should
// become the next reuse base.
template <typename State>
struct SubnetworkResolution {
    Subnetwork<State> subnetwork;
    EntranceLaw law;
    bool store_as_base = false;
    bool stopped = false; // No escape from any entrance; the ensemble is done.
    idx changed_state_slots = 0;
    bool reuse_attempted = false;
    bool reuse_accepted = false;
    idx block_count = 0;
    idx reused_prefix_blocks = 0;
    idx reused_prefix_states = 0;
    double restriction_and_factor_seconds = 0.0;
    double factor_update_seconds = 0.0;
    double entrance_law_seconds = 0.0;
};

// Slots whose state id differs between `base_ids` and `active` (empty if
// their sizes differ).
[[nodiscard]] inline array<idx> changed_slots(const array<idx> &base_ids,
                                              const ActiveSlots &active) {
    array<idx> changed;
    if (base_ids.size() != size(active))
        return changed;
    for (idx i = 0; i < size(active); ++i)
        if (base_ids[i] != active.state_ids[i])
            changed.push_back(i);
    return changed;
}

// Position of each entrance within `subnetwork`.
template <typename State>
[[nodiscard]] array<idx> compute_entrance_rows(const Subnetwork<State> &subnetwork,
                                               const Entrances<State> &entrances) {
    array<idx> rows(size(entrances));
    for (idx k = 0; k < size(entrances); ++k) {
        rows[k] = find(subnetwork, entrances.states[k]);
        if (rows[k] >= size(subnetwork)) {
            throw std::runtime_error("an entrance was shed from its own subnetwork");
        }
    }
    return rows;
}

// Woodbury-updated entrance law against `base`, or nullopt if its residual
// check fails and the caller should fall back to a fresh factorization.
template <typename State>
[[nodiscard]] std::optional<EntranceLaw>
try_reuse_law(const Subnetwork<State> &base, const Subnetwork<State> &subnetwork,
              view<const idx> entrance_rows, view<const idx> changed) {
    try {
        const LowRankDelta delta =
            row_column_delta(base.operator_matrix, subnetwork.operator_matrix, changed);
        const auto update = make_factor_update(base, delta.left, delta.right);
        EntranceLaw law = entrance_law(subnetwork, entrance_rows, update);

        num::vec rhs(size(subnetwork), 0.0);
        rhs[entrance_rows.front()] = 1.0;
        num::vec solution(size(subnetwork), 0.0);
        for (idx j = 0; j < size(subnetwork); ++j)
            solution[j] = law.occupation(0, j);
        if (relative_residual(num::transpose(subnetwork.operator_matrix), solution, rhs) > 1e-8)
            return std::nullopt;
        return law;
    } catch (const std::runtime_error &) {
        return std::nullopt;
    }
}

// Restrict the active workspace and compute its entrance law. Block factors
// reuse their unchanged prefix and refactor the affected suffix; dense factors
// use Woodbury for a small row-and-column update. If the subnetwork has no
// escape state, every unfinished trajectory is closed out here.
template <typename ReactionSystem, typename Rates, typename State, typename LevelFunction>
[[nodiscard]] SubnetworkResolution<State> resolve_subnetwork_and_law(
    const ReactionSystem &model, const Rates &rates, const Entrances<State> &entrances,
    const StateGraph<State> &graph, const ActiveSlots &active, LevelFunction level,
    const EnsembleOptions &options, const std::optional<Subnetwork<State>> &base,
    const array<idx> &base_state_ids, array<Trajectory<State>> &trajectories, num::vec &times,
    const array<bool> &unfinished, real final_time,
    detail::RestrictionCache<State> *restriction_cache = nullptr) {
    const array<idx> changed = changed_slots(base_state_ids, active);
    const bool reusable_base = options.reuse_factorization && base &&
                               base_state_ids.size() == size(active);
    const bool block_suffix = reusable_base && uses_block_factor(*base);
    const idx automatic_reuse_slots = 3;
    const idx maximum_reuse_slots =
        options.maximum_reuse_slots == 0 ? automatic_reuse_slots : options.maximum_reuse_slots;
    const bool reuse = reusable_base &&
                       (block_suffix || changed.size() <= maximum_reuse_slots);

    const auto restriction_start = std::chrono::steady_clock::now();
    const auto active_state_values = active_states(graph, active);
    Subnetwork<State> subnetwork =
        (restriction_cache && reusable_base)
            ? restriction_cached(model, rates, active_state_values, *restriction_cache, level, !reuse)
            : restriction(model, rates, graph, active, level, !reuse);
    if (restriction_cache && options.reuse_factorization && !reusable_base &&
        size(active) >= options.capacity)
        *restriction_cache = make_restriction_cache(model, rates, active_state_values);
    double restriction_and_factor_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - restriction_start).count();

    if (subnetwork.escape_states.empty()) {
        for (idx t = 0; t < trajectories.size(); ++t) {
            if (unfinished[t]) {
                finish(trajectories[t], times[t], final_time);
            }
        }
        return {.subnetwork = std::move(subnetwork),
                .law = {},
                .store_as_base = false,
                .stopped = true,
                .changed_state_slots = changed.size(),
                .reuse_attempted = reuse,
                .reuse_accepted = false};
    }

    // Solve UM = E once, then VM = U, for every distinct entrance at once.
    const array<idx> entrance_rows = compute_entrance_rows(subnetwork, entrances);

    EntranceLaw law;
    bool store_as_base = !reuse;
    bool reuse_accepted = false;
    BlockSuffixInfo suffix_info;
    const auto law_start = std::chrono::steady_clock::now();
    double factor_update_seconds = 0.0;
    if (reuse && !changed.empty()) {
        bool suffix_updated = false;
        if (block_suffix) {
            const auto update_start = std::chrono::steady_clock::now();
            suffix_updated = refactor_block_suffix(*base, subnetwork, changed, &suffix_info);
            factor_update_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start)
                    .count();
        }
        if (suffix_updated) {
            law = entrance_law(subnetwork, view<const idx>(entrance_rows));
            store_as_base = true;
            reuse_accepted = true;
        } else if (!block_suffix) {
            if (auto reused = try_reuse_law(*base, subnetwork, entrance_rows, changed)) {
                law = std::move(*reused);
                reuse_accepted = true;
            } else {
                subnetwork = restriction_cached(model, rates, active_state_values, *restriction_cache,
                                                level, true);
                law = entrance_law(subnetwork, view<const idx>(entrance_rows));
                store_as_base = true;
            }
        } else {
            subnetwork = restriction_cached(model, rates, active_state_values, *restriction_cache,
                                            level, true);
            law = entrance_law(subnetwork, view<const idx>(entrance_rows));
            store_as_base = true;
        }
    } else if (reuse) {
        law = entrance_law(*base, view<const idx>(entrance_rows));
        reuse_accepted = true;
    } else {
        law = entrance_law(subnetwork, view<const idx>(entrance_rows));
    }
    const double entrance_law_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - law_start).count() -
        factor_update_seconds;

    return {.subnetwork = std::move(subnetwork),
            .law = std::move(law),
            .store_as_base = store_as_base,
            .stopped = false,
            .changed_state_slots = changed.size(),
            .reuse_attempted = reuse,
            .reuse_accepted = reuse_accepted,
            .block_count = suffix_info.block_count,
            .reused_prefix_blocks = suffix_info.reused_prefix_blocks,
            .reused_prefix_states = suffix_info.reused_prefix_states,
            .restriction_and_factor_seconds = restriction_and_factor_seconds,
            .factor_update_seconds = factor_update_seconds,
            .entrance_law_seconds = entrance_law_seconds};
}

// Cache a per-state score for the next macrostep's `select_active_states`
// when nothing expands: the diagonal-weighted mixture occupation, minus this
// state's own share of the entrance mixture when it is itself an entrance.
template <typename State>
void record_heuristic_scores(const Subnetwork<State> &subnetwork, const EntranceLaw &law,
                             const Entrances<State> &entrances,
                             table<State, real> &previous_scores) {
    num::vec mixture_occupation(size(subnetwork), 0.0);
    previous_scores.clear();
    const num::vec diagonal = num::diagonal(subnetwork.operator_matrix);
    for (idx k = 0; k < size(entrances); ++k) {
        const real weight = static_cast<real>(entrances.counts[k]) / entrances.active;
        for (idx j = 0; j < size(subnetwork); ++j)
            mixture_occupation[j] += weight * law.occupation(k, j);
    }
    for (idx j = 0; j < size(subnetwork); ++j) {
        real score = diagonal[j] * mixture_occupation[j];
        const auto found = entrances.row.find(subnetwork.states[j]);
        if (found != entrances.row.end())
            score -= static_cast<real>(entrances.counts[found->second]) / entrances.active;
        previous_scores[subnetwork.states[j]] = score;
    }
}

} // namespace detail

// Advance `count` trajectories together, sharing one subnetwork per
// macrostep: trajectories are grouped by current state, so the d distinct
// entrances need only one factorization's U = EZ and V = EZ^2.
template <typename ReactionSystem, typename Rates, typename State = num::multi_index,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] array<Trajectory<State>>
else_ensemble(const ReactionSystem &model, const Rates &rates, const State &initial, idx count,
              real initial_time, real final_time, EnsembleOptions options = {}, unsigned seed = 42,
              LevelFunction level = nullptr, EnsembleDiagnostics *diagnostics = nullptr) {
    if (count == 0) {
        throw std::invalid_argument("ensemble requires at least one trajectory");
    }
    if (initial_time > final_time) {
        throw std::invalid_argument("initial time must not exceed final time");
    }
    if (options.capacity == 0) {
        throw std::invalid_argument("capacity must be positive");
    }

    array<Trajectory<State>> trajectories(count);
    array<State> current(count, initial);
    num::vec times(count, initial_time);
    array<idx> steps(count, 0);
    array<num::rng> generators;
    generators.reserve(count);
    for (idx t = 0; t < count; ++t) {
        trajectories[t].times.push_back(initial_time);
        trajectories[t].states.push_back(initial);
        generators.emplace_back(seed + static_cast<unsigned>(t));
    }

    StateGraph<State> graph;
    ActiveSlots active;
    table<State, real> previous_scores;
    std::optional<Subnetwork<State>> base;
    array<idx> base_state_ids;
    detail::RestrictionCache<State> restriction_cache;

    while (true) {
        array<bool> unfinished(count, false);
        for (idx t = 0; t < count; ++t) {
            unfinished[t] = times[t] < final_time && steps[t] < options.maximum_steps;
        }
        const auto entrances = detail::group_by_entrance(current, unfinished);
        if (entrances.active == 0) {
            break;
        }

        const auto selection_start = std::chrono::steady_clock::now();
        detail::select_active_states(model, rates, entrances, options, previous_scores, graph,
                                     active);
        const double selection_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - selection_start)
                .count();

        auto resolution = detail::resolve_subnetwork_and_law(
            model, rates, entrances, graph, active, level, options, base, base_state_ids,
            trajectories, times, unfinished, final_time, &restriction_cache);
        if (resolution.stopped) {
            break;
        }
        Subnetwork<State> &subnetwork = resolution.subnetwork;
        const EntranceLaw &law = resolution.law;

        const auto score_start = std::chrono::steady_clock::now();
        detail::record_heuristic_scores(subnetwork, law, entrances, previous_scores);
        const double score_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - score_start).count();

        const auto escape_setup_start = std::chrono::steady_clock::now();
        array<num::categorical_sampler> escape_sampler;
        escape_sampler.reserve(size(entrances));
        for (idx k = 0; k < size(entrances); ++k) {
            const num::vec beta = escape_distribution(subnetwork, law, k);
            escape_sampler.emplace_back(beta.span());
        }
        const auto channels = detail::escape_channels(subnetwork);
        const double escape_setup_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - escape_setup_start)
                .count();

        const auto sampling_start = std::chrono::steady_clock::now();
        for (idx t = 0; t < count; ++t) {
            if (!unfinished[t]) {
                continue;
            }
            const idx k = entrances.row.at(current[t]);
            auto &generator = generators[t];

            // Sample the escape state J, then the channel C leaving it.
            const idx escape_position = escape_sampler[k](generator);
            const idx escape_state = subnetwork.escape_states[escape_position];
            const real waiting = conditional_escape_time(law, k, escape_state);
            const auto &destinations = channels.destinations[escape_position];

            if (!(waiting > 0.0) || destinations.empty() || times[t] + waiting >= final_time) {
                detail::finish(trajectories[t], times[t], final_time);
                continue;
            }

            num::categorical_sampler channel{view<const real>(channels.rates[escape_position])};
            times[t] += waiting;
            current[t] = destinations[channel(generator)];
            trajectories[t].times.push_back(times[t]);
            trajectories[t].states.push_back(current[t]);
            ++steps[t];
        }
        const double sampling_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - sampling_start).count();

        if (diagnostics) {
            diagnostics->steps.push_back({.active_trajectories = entrances.active,
                                          .distinct_entrances = detail::size(entrances),
                                          .subnetwork_size = size(subnetwork),
                                          .changed_state_slots = resolution.changed_state_slots,
                                          .reuse_attempted = resolution.reuse_attempted,
                                          .reuse_accepted = resolution.reuse_accepted,
                                          .fresh_factorization = !resolution.reuse_accepted,
                                          .block_count = resolution.block_count,
                                          .reused_prefix_blocks = resolution.reused_prefix_blocks,
                                          .reused_prefix_states = resolution.reused_prefix_states,
                                          .selection_seconds = selection_seconds,
                                          .restriction_and_factor_seconds =
                                              resolution.restriction_and_factor_seconds,
                                          .factor_update_seconds =
                                              resolution.factor_update_seconds,
                                          .entrance_law_seconds = resolution.entrance_law_seconds,
                                          .score_seconds = score_seconds,
                                          .escape_setup_seconds = escape_setup_seconds,
                                          .sampling_seconds = sampling_seconds});
        }

        if (options.reuse_factorization && resolution.store_as_base &&
            size(active) == options.capacity) {
            base_state_ids = active.state_ids;
            base.emplace(std::move(subnetwork));
        }
    }
    return trajectories;
}

// One trajectory, the `count == 1` case of `else_ensemble`.
template <typename ReactionSystem, typename Rates, typename State = num::multi_index,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] Trajectory<State> else_trajectory(const ReactionSystem &model, const Rates &rates,
                                                const State &initial, real initial_time,
                                                real final_time, EnsembleOptions options = {},
                                                unsigned seed = 42, LevelFunction level = nullptr) {
    auto ensemble =
        else_ensemble(model, rates, initial, 1, initial_time, final_time, options, seed, level);
    return std::move(ensemble.front());
}

} // namespace else_sim
