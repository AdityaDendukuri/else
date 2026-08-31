#pragma once

#include "elsex/restriction.hpp"
#include "elsex/selection.hpp"
#include "elsex/subnetwork.hpp"
#include "elsex/types.hpp"
#include "linear/solvers/auto_resolvent.hpp"
#include "linear/sparse/sparse.hpp"
#include "quadrature/talbot.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace elsex {

struct BoundaryEntry {
    idx source;
    idx target;
    real rate;
};

template <typename State = std::vector<int>>
struct DensityChain {
    std::vector<Subnetwork<State>> subnetworks;
    std::vector<State> states;
    std::unordered_map<State, idx, StateHash<State>> position;
    std::vector<std::vector<BoundaryEntry>> boundary;
    std::vector<num::AutoResolventSolver> resolvents;
};

template <typename State>
[[nodiscard]] DensityChain<State> make_density_chain(std::vector<Subnetwork<State>> subnetworks) {
    if (subnetworks.empty())
        throw std::invalid_argument("density composition requires at least one subnetwork");

    DensityChain<State> chain;
    chain.subnetworks = std::move(subnetworks);
    chain.boundary.resize(chain.subnetworks.size());
    for (const auto &subnetwork : chain.subnetworks) {
        for (const State &state : subnetwork.states())
            if (chain.position.emplace(state, chain.states.size()).second)
                chain.states.push_back(state);
        chain.resolvents.emplace_back(num::transpose(subnetwork.generator()));
    }
    for (idx r = 0; r + 1 < chain.subnetworks.size(); ++r) {
        const auto &next = chain.subnetworks[r + 1];
        for (const auto &transition : chain.subnetworks[r].boundary()) {
            const idx target = next.find(transition.destination);
            if (target < next.size())
                chain.boundary[r].push_back({transition.source, target, transition.rate});
        }
    }
    return chain;
}

template <typename State>
[[nodiscard]] std::vector<num::cplx> apply_resolvent(DensityChain<State> &chain, idx r,
                                                     num::cplx shift,
                                                     const std::vector<num::cplx> &arrival) {
    chain.resolvents[r].factorize(shift);
    std::vector<num::cplx> local;
    chain.resolvents[r].solve(arrival, local);
    return local;
}

template <typename State>
void add_F(const DensityChain<State> &chain, idx r, num::cplx weight,
           const std::vector<num::cplx> &local, std::vector<real> &density) {
    const auto &subnetwork = chain.subnetworks[r];
    for (idx j = 0; j < subnetwork.size(); ++j)
        density[chain.position.at(subnetwork.states()[j])] += (weight * local[j]).real();
}

template <typename State>
[[nodiscard]] std::vector<num::cplx> apply_G(const DensityChain<State> &chain, idx r,
                                             const std::vector<num::cplx> &local) {
    std::vector<num::cplx> next(chain.subnetworks[r + 1].size(), num::cplx(0.0, 0.0));
    for (const BoundaryEntry &entry : chain.boundary[r])
        next[entry.target] += entry.rate * local[entry.source];
    return next;
}

template <typename State>
[[nodiscard]] DensitySolution<State> inverse_laplace_density(DensityChain<State> &chain,
                                                             const State &initial, real time,
                                                             idx modes = 14) {
    if (!(time > 0.0))
        throw std::invalid_argument("density time must be positive");
    const idx start = chain.subnetworks.front().find(initial);
    if (start >= chain.subnetworks.front().size())
        throw std::invalid_argument("initial state is outside the first subnetwork");

    std::vector<real> probability(chain.states.size(), 0.0);
    num::inverse_laplace_accumulate(time, modes, [&](num::cplx shift, num::cplx weight) {
        std::vector<num::cplx> arrival(chain.subnetworks.front().size(), num::cplx(0.0, 0.0));
        arrival[start] = num::cplx(1.0, 0.0);

        // p_hat(s) = p0 sum_r G_1(s)...G_{r-1}(s) F_r(s).
        for (idx r = 0; r < chain.subnetworks.size(); ++r) {
            const auto local = apply_resolvent(chain, r, shift, arrival);
            add_F(chain, r, weight, local, probability);
            if (r + 1 == chain.subnetworks.size())
                break;
            arrival = apply_G(chain, r, local);
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
// a_{r+1}^T = a_r^T Z_r B_r, with Z_r=(-R_r)^-1.
template <typename ReactionSystem, typename Rates, typename State = std::vector<int>>
[[nodiscard]] std::vector<Subnetwork<State>>
density_subnetworks(const ReactionSystem &model, const Rates &rates, const State &initial,
                    int count, EnsembleOptions options = {}, unsigned seed = 42) {
    if (count < 1) {
        throw std::invalid_argument("density evolution requires at least one subnetwork");
    }

    std::unordered_map<State, real, StateHash<State>> arrival{{initial, 1.0}};
    std::vector<Subnetwork<State>> chain;
    chain.reserve(static_cast<std::size_t>(count));
    std::mt19937 random(seed);

    for (int step = 0; step < count; ++step) {
        real mass = 0.0;
        for (const auto &[state, weight] : arrival) {
            mass += weight;
        }
        if (!(mass > options.tolerance)) {
            break;
        }

        std::vector<State> support;
        support.reserve(arrival.size());
        for (const auto &[state, weight] : arrival) {
            support.push_back(state);
        }

        auto states =
            expanded_states(model, rates, support, options.expansion_depth, options.tolerance);

        // rho is the normalized arrival, and its support is protected.
        const auto mixture_over = [&](const Subnetwork<State> &subnetwork) {
            num::Vector mixture(subnetwork.size(), 0.0);
            for (const auto &[state, weight] : arrival) {
                const idx position = subnetwork.find(state);
                if (position < subnetwork.size()) {
                    mixture[position] = weight / mass;
                }
            }
            return mixture;
        };

        if (states.size() > options.capacity) {
            const Subnetwork<State> expanded = restriction(model, rates, states);
            std::vector<idx> protected_states;
            protected_states.reserve(support.size());
            for (const State &state : support) {
                const idx position = expanded.find(state);
                if (position < expanded.size()) {
                    protected_states.push_back(position);
                }
            }
            states = shed_to_capacity(model, rates, std::move(states), mixture_over(expanded),
                                      std::span<const idx>(protected_states), options);
        }

        chain.push_back(restriction(model, rates, std::move(states)));
        const Subnetwork<State> &subnetwork = chain.back();
        if (subnetwork.boundary().empty()) {
            break;
        }

        // s=0 version of the same recursion used by inverse_laplace_density.
        const num::Vector occupancy =
            num::solve_transpose(subnetwork.factor(), mixture_over(subnetwork));
        std::unordered_map<State, real, StateHash<State>> next;
        for (const auto &transition : subnetwork.boundary()) {
            next[transition.destination] += transition.rate * occupancy[transition.source];
        }
        arrival = subsample_support(next, options.capacity, random);
    }
    return chain;
}

} // namespace elsex
