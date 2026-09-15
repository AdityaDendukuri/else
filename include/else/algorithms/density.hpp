// Unlabeled ELSE: density evolution by composing subnetwork resolvents.
#pragma once

#include "else/algorithms/selection.hpp"
#include "else/core/subnetwork.hpp"
#include "else/core/types.hpp"
#include "else/restriction/restriction.hpp"
#include "linear/solvers/auto_resolvent.hpp"
#include "linear/sparse/sparse.hpp"
#include "quadrature/talbot.hpp"
#include "stochastic/rng.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace else_sim {

struct BoundaryEntry {
    idx source;
    idx target;
    real rate;
};

template <typename State = num::multi_index>
struct DensityChain {
    array<Subnetwork<State>> subnetworks;
    array<State> states;
    table<State, idx> position;
    array<array<BoundaryEntry>> boundary;
    array<num::vec> entrance_scale; // one weight per state of each subnetwork
    array<num::auto_resolvent_solver> resolvents;
};

template <typename State>
[[nodiscard]] DensityChain<State>
make_density_chain(array<Subnetwork<State>> subnetworks,
                   const array<table<State, real>> &entrance_scale = {}) {
    if (subnetworks.empty())
        throw std::invalid_argument("density composition requires at least one subnetwork");

    DensityChain<State> chain;
    chain.subnetworks = std::move(subnetworks);
    chain.boundary.resize(chain.subnetworks.size());
    chain.entrance_scale.resize(chain.subnetworks.size());
    for (idx r = 0; r < chain.subnetworks.size(); ++r) {
        const auto &subnetwork = chain.subnetworks[r];
        chain.entrance_scale[r] = num::vec(size(subnetwork), 1.0);
        if (!entrance_scale.empty()) {
            if (entrance_scale.size() != chain.subnetworks.size())
                throw std::invalid_argument("one entrance scale is required per subnetwork");
            for (idx j = 0; j < size(subnetwork); ++j) {
                const auto found = entrance_scale[r].find(subnetwork.states[j]);
                chain.entrance_scale[r][j] = found == entrance_scale[r].end() ? 0.0 : found->second;
            }
        }
        for (const State &state : subnetwork.states)
            if (chain.position.emplace(state, chain.states.size()).second)
                chain.states.push_back(state);
        chain.resolvents.emplace_back(num::transpose(subnetwork.generator));
    }
    for (idx r = 0; r + 1 < chain.subnetworks.size(); ++r) {
        const auto &next = chain.subnetworks[r + 1];
        for (const auto &transition : chain.subnetworks[r].boundary) {
            const idx target = find(next, transition.destination);
            if (target < size(next))
                chain.boundary[r].push_back({transition.source, target, transition.rate});
        }
    }
    return chain;
}

template <typename State>
[[nodiscard]] array<num::cplx> apply_resolvent(DensityChain<State> &chain, idx r, num::cplx shift,
                                               const array<num::cplx> &arrival) {
    chain.resolvents[r].factorize(shift);
    array<num::cplx> local;
    chain.resolvents[r].solve(arrival, local);
    return local;
}

// Add iteration r's contribution p_hat^(r)(s)^T = a_hat_r(s)^T F_r(s) into
// the running density, weighted by this contour node's quadrature weight.
template <typename State>
void accumulate_density_contribution(const DensityChain<State> &chain, idx r, num::cplx weight,
                                     const array<num::cplx> &local, num::vec &density) {
    const auto &subnetwork = chain.subnetworks[r];
    for (idx j = 0; j < size(subnetwork); ++j)
        density[chain.position.at(subnetwork.states[j])] += (weight * local[j]).real();
}

// Advance the arrival density via a_hat_{r+1}(s)^T = a_hat_r(s)^T G_r(s),
// applying the boundary rates directly rather than forming G_r(s).
template <typename State>
[[nodiscard]] array<num::cplx> advance_arrival(const DensityChain<State> &chain, idx r,
                                               const array<num::cplx> &local) {
    array<num::cplx> next(size(chain.subnetworks[r + 1]), num::cplx(0.0, 0.0));
    for (const BoundaryEntry &entry : chain.boundary[r])
        next[entry.target] += entry.rate * local[entry.source];
    for (idx j = 0; j < next.size(); ++j)
        next[j] *= chain.entrance_scale[r + 1][j];
    return next;
}

template <typename State>
[[nodiscard]] DensitySolution<State> inverse_laplace_density(DensityChain<State> &chain,
                                                             const State &initial, real time,
                                                             idx modes = 14) {
    if (!(time > 0.0))
        throw std::invalid_argument("density time must be positive");
    const idx start = find(chain.subnetworks.front(), initial);
    if (start >= size(chain.subnetworks.front()))
        throw std::invalid_argument("initial state is outside the first subnetwork");

    num::vec probability(chain.states.size(), 0.0);
    num::inverse_laplace_accumulate(time, modes, [&](num::cplx shift, num::cplx weight) {
        array<num::cplx> arrival(size(chain.subnetworks.front()), num::cplx(0.0, 0.0));
        arrival[start] = num::cplx(1.0, 0.0);

        // p_hat(s) = p0 sum_r G_1(s)...G_{r-1}(s) F_r(s).
        for (idx r = 0; r < chain.subnetworks.size(); ++r) {
            const auto local = apply_resolvent(chain, r, shift, arrival);
            accumulate_density_contribution(chain, r, weight, local, probability);
            if (r + 1 == chain.subnetworks.size())
                break;
            arrival = advance_arrival(chain, r, local);
        }
    });

    real total = 0.0;
    for (real &value : probability) {
        value = std::max(0.0, value);
        total += value;
    }
    if (total > 0.0)
        for (real &value : probability)
            value /= total;
    return {chain.states, std::move(probability)};
}

// Choose the subnetwork chain from the zero-frequency density recursion:
// a_{r+1}^T = a_r^T Z_r B_r, with Z_r = M_r^-1.
template <typename ReactionSystem, typename Rates, typename State = num::multi_index>
[[nodiscard]] array<Subnetwork<State>>
density_subnetworks(const ReactionSystem &model, const Rates &rates, const State &initial,
                    int count, EnsembleOptions options = {}, unsigned seed = 42) {
    if (count < 1) {
        throw std::invalid_argument("density evolution requires at least one subnetwork");
    }

    table<State, real> arrival{{initial, 1.0}};
    array<Subnetwork<State>> chain;
    chain.reserve(static_cast<idx>(count));
    num::rng random(seed);

    for (int step = 0; step < count; ++step) {
        real mass = 0.0;
        for (const auto &[state, weight] : arrival) {
            mass += weight;
        }
        if (!(mass > options.tolerance)) {
            break;
        }

        array<State> support;
        support.reserve(arrival.size());
        for (const auto &[state, weight] : arrival) {
            support.push_back(state);
        }

        auto states =
            expanded_states(model, rates, support, options.expansion_depth, options.tolerance);

        // rho is the normalized arrival, and its support is protected.
        const auto mixture_over = [&](const Subnetwork<State> &subnetwork) {
            num::vec mixture(size(subnetwork), 0.0);
            for (const auto &[state, weight] : arrival) {
                const idx position = find(subnetwork, state);
                if (position < size(subnetwork)) {
                    mixture[position] = weight / mass;
                }
            }
            return mixture;
        };

        if (states.size() > options.capacity) {
            const Subnetwork<State> expanded = restriction(model, rates, states);
            array<idx> protected_states;
            protected_states.reserve(support.size());
            for (const State &state : support) {
                const idx position = find(expanded, state);
                if (position < size(expanded)) {
                    protected_states.push_back(position);
                }
            }
            states = shed_to_capacity(model, rates, std::move(states), mixture_over(expanded),
                                      view<const idx>(protected_states), options);
        }

        chain.push_back(restriction(model, rates, std::move(states)));
        const Subnetwork<State> &subnetwork = chain.back();
        if (subnetwork.boundary.empty()) {
            break;
        }

        // s=0 version of the same recursion used by inverse_laplace_density.
        const num::vec occupancy = solve_transpose(subnetwork, mixture_over(subnetwork));
        table<State, real> next;
        for (const auto &transition : subnetwork.boundary) {
            next[transition.destination] += transition.rate * occupancy[transition.source];
        }
        arrival = resample_density_support(next, options, random);
    }
    return chain;
}

} // namespace else_sim
