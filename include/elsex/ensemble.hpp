// Finite ensemble ELSE.
#pragma once

#include "elsex/factor_update.hpp"
#include "elsex/law.hpp"
#include "elsex/restriction.hpp"
#include "elsex/selection.hpp"
#include "elsex/shedding.hpp"
#include "elsex/subnetwork.hpp"
#include "elsex/types.hpp"
#include "stochastic/categorical.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace elsex {

namespace detail {

/// The distinct entrances of the active trajectories with their multiplicities.
template <typename State>
struct Entrances {
    std::vector<State> states;                            ///< \f$i_k\f$
    std::vector<idx> counts;                              ///< \f$c_k\f$
    std::unordered_map<State, idx, StateHash<State>> row; ///< \f$i_k \mapsto k\f$
    idx active = 0;                                       ///< \f$m_{\rm active}\f$

    [[nodiscard]] idx size() const { return states.size(); }
};

template <typename State>
[[nodiscard]] Entrances<State> group_by_entrance(const std::vector<State> &current,
                                                 const std::vector<bool> &unfinished) {
    Entrances<State> entrances;
    for (std::size_t t = 0; t < current.size(); ++t) {
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

/// \f$\rho = \sum_k (c_k/m_{\rm active})\delta_{i_k}\f$.
template <typename State>
[[nodiscard]] num::Vector entrance_mixture(const Subnetwork<State> &subnetwork,
                                           const Entrances<State> &entrances) {
    num::Vector mixture(subnetwork.size(), 0.0);
    for (idx k = 0; k < entrances.size(); ++k) {
        const idx position = subnetwork.find(entrances.states[k]);
        if (position < subnetwork.size()) {
            mixture[position] =
                static_cast<real>(entrances.counts[k]) / static_cast<real>(entrances.active);
        }
    }
    return mixture;
}

/// Transitions leaving each escape state, indexed by position in `escape_states()`.
template <typename State>
struct Channels {
    std::vector<std::vector<State>> destinations;
    std::vector<std::vector<real>> rates;
};

template <typename State>
[[nodiscard]] Channels<State> escape_channels(const Subnetwork<State> &subnetwork) {
    const auto &escape_states = subnetwork.escape_states();
    std::unordered_map<idx, idx> position;
    for (idx p = 0; p < escape_states.size(); ++p) {
        position.emplace(escape_states[p], p);
    }

    Channels<State> channels;
    channels.destinations.resize(escape_states.size());
    channels.rates.resize(escape_states.size());
    for (const auto &transition : subnetwork.boundary()) {
        const auto found = position.find(transition.source);
        if (found != position.end()) {
            channels.destinations[found->second].push_back(transition.destination);
            channels.rates[found->second].push_back(transition.rate);
        }
    }
    return channels;
}

/// Close a trajectory out at `final_time`.
template <typename State>
void finish(Trajectory<State> &trajectory, real &time, real final_time) {
    time = final_time;
    if (final_time < std::numeric_limits<real>::infinity() &&
        trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.back());
    }
}

} // namespace detail

/// Advance `count` trajectories together, sharing one subnetwork per macrostep.
///
/// Trajectories are grouped by their current state, so the \f$d\f$ distinct
/// entrances need only \f$U = EZ\f$ and \f$V = EZ^2\f$ from a single
/// factorization. Sharing changes neither the escape distribution nor the
/// conditional mean: every trajectory still samples from its own entrance row.
template <typename ReactionSystem, typename Rates, typename State = std::vector<int>,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] std::vector<Trajectory<State>>
else_ensemble(const ReactionSystem &model, const Rates &rates, const State &initial,
              std::size_t count, real initial_time, real final_time, EnsembleOptions options = {},
              unsigned seed = 42, LevelFunction level = nullptr) {
    if (count == 0) {
        throw std::invalid_argument("ensemble requires at least one trajectory");
    }
    if (initial_time > final_time) {
        throw std::invalid_argument("initial time must not exceed final time");
    }
    if (options.capacity == 0) {
        throw std::invalid_argument("capacity must be positive");
    }

    std::vector<Trajectory<State>> trajectories(count);
    std::vector<State> current(count, initial);
    std::vector<real> times(count, initial_time);
    std::vector<std::size_t> steps(count, 0);
    std::vector<std::mt19937> generators;
    generators.reserve(count);
    for (std::size_t t = 0; t < count; ++t) {
        trajectories[t].times.push_back(initial_time);
        trajectories[t].states.push_back(initial);
        generators.emplace_back(seed + static_cast<unsigned>(t));
    }

    StateGraph<State> graph;
    ActiveSlots active;
    std::unordered_map<State, real, StateHash<State>> previous_scores;
    std::optional<Subnetwork<State>> base;
    std::vector<idx> base_state_ids;

    while (true) {
        std::vector<bool> unfinished(count, false);
        for (std::size_t t = 0; t < count; ++t) {
            unfinished[t] = times[t] < final_time && steps[t] < options.maximum_steps;
        }
        const auto entrances = detail::group_by_entrance(current, unfinished);
        if (entrances.active == 0) {
            break;
        }

        const idx old_size = active.size();
        const int depth = old_size < options.capacity ? options.expansion_depth : 0;
        expand_workspace(model, rates, entrances.states, depth, options.tolerance, graph, active);

        if (active.size() > options.capacity) {
            std::vector<idx> protected_states;
            protected_states.reserve(entrances.size());
            for (const State &entrance : entrances.states) {
                const idx state_id = graph.find(entrance);
                const idx position = active.find(state_id);
                if (position < active.size()) {
                    protected_states.push_back(position);
                }
            }
            num::Vector scores(active.size(), 0.0);
            if (depth == 0 && !previous_scores.empty()) {
                for (idx i = 0; i < active.size(); ++i) {
                    const auto found = previous_scores.find(graph[active.state_ids[i]]);
                    if (found != previous_scores.end())
                        scores[i] = found->second;
                }
            } else {
                const Subnetwork<State> expanded = restriction(model, rates, graph, active);
                const SheddingState state =
                    shedding_state(expanded, detail::entrance_mixture(expanded, entrances));
                scores = options.rule == SheddingRule::ExactCutTime
                             ? exact_cut_time_losses(expanded, state)
                             : expected_entries(expanded, state);
            }
            const auto shed = lowest_scores(std::span<const real>(scores.data(), scores.size()),
                                            protected_states, active.size() - options.capacity);
            std::vector<bool> remove(active.size(), false);
            for (idx slot : shed)
                remove[slot] = true;
            active.retain(remove, old_size);
        }

        std::vector<idx> changed;
        if (base && base_state_ids.size() == active.size())
            for (idx i = 0; i < active.size(); ++i)
                if (base_state_ids[i] != active.state_ids[i])
                    changed.push_back(i);
        constexpr idx maximum_changed_slots = 3;
        const bool reuse = options.reuse_factorization && base &&
                           base_state_ids.size() == active.size() &&
                           changed.size() <= maximum_changed_slots;
        Subnetwork<State> subnetwork = restriction(model, rates, graph, active, level, !reuse);

        if (subnetwork.escape_states().empty()) {
            for (std::size_t t = 0; t < count; ++t) {
                if (unfinished[t]) {
                    detail::finish(trajectories[t], times[t], final_time);
                }
            }
            break;
        }

        // Solve UM = E once, then VM = U, for every distinct entrance at once.
        std::vector<idx> entrance_rows(entrances.size());
        for (idx k = 0; k < entrances.size(); ++k) {
            entrance_rows[k] = subnetwork.find(entrances.states[k]);
            if (entrance_rows[k] >= subnetwork.size()) {
                throw std::runtime_error("an entrance was shed from its own subnetwork");
            }
        }
        EntranceLaw law;
        bool store_as_base = !reuse;
        if (reuse && !changed.empty()) {
            try {
                const LowRankDelta delta = row_column_delta(base->operator_matrix(),
                                                            subnetwork.operator_matrix(), changed);
                const FactorUpdate update(*base, delta.left, delta.right);
                law = entrance_law(subnetwork, std::span<const idx>(entrance_rows), update);

                num::Vector rhs(subnetwork.size(), 0.0);
                rhs[entrance_rows.front()] = 1.0;
                num::Vector solution(subnetwork.size(), 0.0);
                for (idx j = 0; j < subnetwork.size(); ++j)
                    solution[j] = law.occupation(0, j);
                if (relative_residual(num::transpose(subnetwork.operator_matrix()), solution, rhs) >
                    1e-8)
                    throw std::runtime_error("factor update residual is too large");
            } catch (const std::runtime_error &) {
                subnetwork = restriction(model, rates, graph, active, level, true);
                law = entrance_law(subnetwork, std::span<const idx>(entrance_rows));
                store_as_base = true;
            }
        } else if (reuse) {
            law = entrance_law(*base, std::span<const idx>(entrance_rows));
        } else {
            law = entrance_law(subnetwork, std::span<const idx>(entrance_rows));
        }

        num::Vector mixture_occupation(subnetwork.size(), 0.0);
        previous_scores.clear();
        const num::Vector diagonal = num::diagonal(subnetwork.operator_matrix());
        for (idx k = 0; k < entrances.size(); ++k) {
            const real weight = static_cast<real>(entrances.counts[k]) / entrances.active;
            for (idx j = 0; j < subnetwork.size(); ++j)
                mixture_occupation[j] += weight * law.occupation(k, j);
        }
        for (idx j = 0; j < subnetwork.size(); ++j) {
            real score = diagonal[j] * mixture_occupation[j];
            const auto found = entrances.row.find(subnetwork.states()[j]);
            if (found != entrances.row.end())
                score -= static_cast<real>(entrances.counts[found->second]) / entrances.active;
            previous_scores[subnetwork.states()[j]] = score;
        }

        std::vector<num::CategoricalSampler> escape_sampler;
        escape_sampler.reserve(entrances.size());
        for (idx k = 0; k < entrances.size(); ++k) {
            const num::Vector beta = escape_distribution(subnetwork, law, k);
            escape_sampler.emplace_back(std::span<const real>(beta.data(), beta.size()));
        }
        const auto channels = detail::escape_channels(subnetwork);

        for (std::size_t t = 0; t < count; ++t) {
            if (!unfinished[t]) {
                continue;
            }
            const idx k = entrances.row.at(current[t]);
            auto &generator = generators[t];

            // Sample the escape state J, then the channel C leaving it.
            const idx escape_position = escape_sampler[k](generator);
            const idx escape_state = subnetwork.escape_states()[escape_position];
            const real waiting = conditional_escape_time(law, k, escape_state);
            const auto &destinations = channels.destinations[escape_position];

            if (!(waiting > 0.0) || destinations.empty() || times[t] + waiting >= final_time) {
                detail::finish(trajectories[t], times[t], final_time);
                continue;
            }

            num::CategoricalSampler channel{std::span<const real>(channels.rates[escape_position])};
            times[t] += waiting;
            current[t] = destinations[channel(generator)];
            trajectories[t].times.push_back(times[t]);
            trajectories[t].states.push_back(current[t]);
            ++steps[t];
        }

        if (options.reuse_factorization && store_as_base && active.size() == options.capacity) {
            base_state_ids = active.state_ids;
            base.emplace(std::move(subnetwork));
        }
    }
    return trajectories;
}

/// One trajectory, the `count == 1` case of `else_ensemble`.
template <typename ReactionSystem, typename Rates, typename State = std::vector<int>,
          typename LevelFunction = std::nullptr_t>
[[nodiscard]] Trajectory<State> else_trajectory(const ReactionSystem &model, const Rates &rates,
                                                const State &initial, real initial_time,
                                                real final_time, EnsembleOptions options = {},
                                                unsigned seed = 42, LevelFunction level = nullptr) {
    auto ensemble =
        else_ensemble(model, rates, initial, 1, initial_time, final_time, options, seed, level);
    return std::move(ensemble.front());
}

} // namespace elsex
