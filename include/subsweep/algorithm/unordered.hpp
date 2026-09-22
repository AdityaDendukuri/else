// Unordered subsweep (Algorithm 2): sweep a distribution through successive
// subnetworks, resampling its support between them, and keep the chain for
// reconstruction.
#pragma once

#include "subsweep/algorithm/reconstruct.hpp"
#include "subsweep/algorithm/row_system.hpp"
#include "stochastic/categorical.hpp"
#include "stochastic/rng.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace subsweep {

template <typename State = num::multi_index>
struct unordered_chain {
    array<chain_link> links;
    array<State> states; // label -> state
    idx start = 0;       // slot of the initial state in the first link
};

template <typename State = num::multi_index>
struct transient_distribution {
    array<State> states;
    num::vec probability;
};

// The states within `depth` layers of `seeds` whose total rate exceeds
// `tolerance`; a seed with none is absorbing and its mass is lost.
template <typename State, rate_matrix<State> R>
[[nodiscard]] array<State> expanded_states(const R &r, const array<State> &seeds, int depth,
                                           real tolerance) {
    const auto total_rate = [&](const State &state) {
        real total = 0.0;
        r(state, [&](const State &, real rate) { total += rate; });
        return total;
    };
    key_set<State> visited;
    array<State> collected, frontier;
    for (const State &state : seeds)
        if (visited.insert(state).second && total_rate(state) > tolerance) {
            num::append(collected, state);
            num::append(frontier, state);
        }
    while (depth-- > 0 && !frontier.empty()) {
        array<State> next;
        for (const State &state : frontier)
            r(state, [&](State destination, real) {
                if (!visited.insert(destination).second || total_rate(destination) <= tolerance)
                    return;
                num::append(collected, destination);
                num::append(next, std::move(destination));
            });
        frontier = std::move(next);
    }
    return collected;
}

// Resample a support: distinct states without replacement keeping their
// masses, or m particle draws with replacement of mass 1/m each.
template <typename State>
[[nodiscard]] table<State, real> resample_support(const table<State, real> &distribution,
                                                  const sweep_options &options, num::rng &random) {
    array<State> states;
    array<real> weights;
    for (const auto &[state, weight] : distribution) {
        num::append(states, state);
        num::append(weights, std::max(weight, 0.0));
    }
    table<State, real> selected;
    if (options.resampling == support_resampling::weighted_support) {
        if (distribution.size() <= options.capacity)
            return distribution;
        real remaining = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (idx draw = 0; draw < options.capacity && remaining > 0.0; ++draw) {
            const idx k = num::sample_categorical(view<const real>(weights), random);
            selected.emplace(states[k], distribution.at(states[k]));
            remaining -= weights[k];
            weights[k] = 0.0;
        }
        return selected;
    }
    const idx particles = options.particles == 0 ? options.capacity : options.particles;
    const real total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(total > 0.0))
        return selected;
    num::categorical_sampler choose{view<const real>(weights)};
    for (idx draw = 0; draw < particles; ++draw)
        selected[states[choose(random)]] += total / static_cast<real>(particles);
    return selected;
}

// Up to `count` subnetworks from the arrival mass at `initial`: each window is
// the support expanded by `expansion_depth` layers and shed to capacity; rho
// is swept to b = B^T Z^T rho and its support resampled for the next window.
// `window` overrides the first window.
template <typename State, rate_matrix<State> R, typename Level = std::nullptr_t,
          typename Stationary = std::nullptr_t>
[[nodiscard]] unordered_chain<State>
unordered_subsweep(const R &r, const State &initial, int count, sweep_options options = {},
                   unsigned seed = 42, Level level = nullptr, Stationary stationary = nullptr,
                   array<State> window = {}) {
    unordered_chain<State> chain;
    table<State, idx> label;
    const auto label_of = [&](const State &state) {
        const auto [entry, added] = label.emplace(state, chain.states.size());
        if (added)
            num::append(chain.states, state);
        return entry->second;
    };
    table<State, real> arrival{{initial, 1.0}}, scale{{initial, 1.0}};
    num::rng random(seed);
    table<State, idx> previous_position;
    array<State> previous_states;

    for (int step = 0; step < count; ++step) {
        real mass = 0.0;
        array<State> support;
        for (const auto &[state, weight] : arrival) {
            mass += weight;
            num::append(support, state);
        }
        if (!(mass > options.tolerance))
            break;
        array<State> states = step == 0 && !window.empty()
                                  ? window
                                  : expanded_states(r, support, options.expansion_depth,
                                                    options.tolerance);
        table<State, idx> position;
        for (idx i = 0; i < states.size(); ++i)
            position.emplace(states[i], i);
        const auto rho_on = [&](const table<State, idx> &where, idx n) {
            num::vec rho(n, 0.0);
            for (const auto &[state, weight] : arrival)
                if (const auto found = where.find(state); found != where.end())
                    rho[found->second] = weight / mass;
            return rho;
        };
        if (states.size() > options.capacity) {
            subnetwork expanded = restriction(r, states, options, level, stationary);
            array<idx> protected_slots;
            for (const State &state : support)
                if (const auto found = position.find(state); found != position.end())
                    num::append(protected_slots, found->second);
            const num::vec scores = expanded.scores(rho_on(position, states.size()));
            const array<idx> shed = lowest_scores(scores.span(), protected_slots,
                                                  states.size() - options.capacity);
            array<bool> drop(states.size(), false);
            for (idx j : shed)
                drop[j] = true;
            array<State> kept;
            for (idx j = 0; j < states.size(); ++j)
                if (!drop[j])
                    num::append(kept, std::move(states[j]));
            states = std::move(kept);
            position.clear();
            for (idx i = 0; i < states.size(); ++i)
                position.emplace(states[i], i);
        }

        if (step == 0)
            chain.start = position.at(initial);
        subnetwork sn = restriction(r, states, options, level, stationary);
        chain_link link{sn.matrix(), {}, {}, num::vec(states.size(), 0.0)};
        for (idx i = 0; i < states.size(); ++i) {
            num::append(link.label, label_of(states[i]));
            const auto found = scale.find(states[i]);
            link.arrival_scale[i] = found == scale.end() ? 0.0 : found->second;
        }
        if (!chain.links.empty())
            for (idx i = 0; i < previous_states.size(); ++i)
                r(previous_states[i], [&](const State &destination, real rate) {
                    if (previous_position.contains(destination))
                        return;
                    if (const auto found = position.find(destination); found != position.end())
                        num::append(chain.links.back().exits, i, found->second, rate);
                });
        num::append(chain.links, std::move(link));
        previous_states = states;
        previous_position = position;

        const num::vec u = sn.factor().solve_transpose(rho_on(position, states.size()));
        table<State, real> next;
        for (idx i = 0; i < states.size(); ++i)
            r(states[i], [&](const State &destination, real rate) {
                if (!position.contains(destination))
                    next[destination] += rate * u[i];
            });
        if (next.empty())
            break;
        arrival = resample_support(next, options, random);
        real exact_total = 0.0, sampled_total = 0.0;
        for (const auto &[state, weight] : next)
            exact_total += weight;
        for (const auto &[state, weight] : arrival)
            sampled_total += weight;
        scale.clear();
        for (const auto &[state, sampled] : arrival)
            scale[state] = (sampled / sampled_total) / (next.at(state) / exact_total);
    }
    return chain;
}

template <typename State>
[[nodiscard]] transient_distribution<State>
reconstruct_distribution(const unordered_chain<State> &chain, real time, idx modes = 14,
                         bool normalize = true) {
    transient_distribution<State> out{chain.states, num::vec(chain.states.size(), 0.0)};
    for (const auto &[label, value] :
         reconstruct(chain.links, chain.start, time, modes, normalize))
        out.probability[label] = value;
    return out;
}

} // namespace subsweep
