// Reversible subnetworks drawn from a symmetric graph Laplacian.
#pragma once

#include "elsex/law.hpp"
#include "elsex/selection.hpp"
#include "elsex/subnetwork.hpp"
#include "elsex/types.hpp"
#include "stochastic/categorical.hpp"
#include "structures/containers/indexed_priority_queue.hpp"
#include <functional>
#include <random>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace elsex {

/// A connected neighborhood of `origin`, grown in descending stationary weight.
///
/// Growth stops at `capacity` states or when the frontier is exhausted, so the
/// result is always connected and contains `origin`.
template <typename SparseMat>
[[nodiscard]] std::vector<idx> laplacian_neighborhood(const SparseMat &laplacian,
                                                      std::span<const real> stationary_sqrt,
                                                      idx origin, idx capacity) {
    const idx n = laplacian.n_rows();
    if (origin >= n || capacity == 0 || capacity > n) {
        throw std::invalid_argument("invalid Laplacian neighborhood request");
    }
    if (stationary_sqrt.size() != n) {
        throw std::invalid_argument("stationary weights must have one entry per state");
    }

    std::vector<bool> selected(n, false);
    num::IndexedPriorityQueue<real, idx, std::greater<real>> frontier(n);
    std::vector<idx> states;
    states.reserve(capacity);

    const auto expose = [&](idx state) {
        for (auto k = laplacian.row_ptr()[state]; k < laplacian.row_ptr()[state + 1]; ++k) {
            const idx neighbor = static_cast<idx>(laplacian.col_idx()[k]);
            if (neighbor != state && !selected[neighbor] && !frontier.contains(neighbor)) {
                frontier.push(neighbor, stationary_sqrt[neighbor]);
            }
        }
    };

    selected[origin] = true;
    states.push_back(origin);
    expose(origin);

    while (states.size() < capacity && !frontier.empty()) {
        const idx state = frontier.top_index();
        frontier.pop();
        selected[state] = true;
        states.push_back(state);
        expose(state);
    }
    return states;
}

/// Restrict a symmetric Laplacian to `states` as \f$R = -H^{-1}LH\f$.
///
/// The paper writes \f$R_r = -H_r^{-1}L_rH_r\f$ with \f$L_r\f$ a grounded
/// symmetric Laplacian and \f$H_r = \mathrm{diag}(\sqrt{\pi})\f$, so
/// \f$R_{ab} = -L_{ab}h_b/h_a\f$ with diagonal \f$-L_{aa}\f$, and detailed
/// balance holds with \f$\pi = h^2\f$.
///
/// Note the conservative vector of \f$R\f$ is \f$h^{-1}\f$ rather than
/// \f$\mathbf 1\f$, since \f$Lh = 0\f$ for an ungrounded Laplacian. Escape rates
/// therefore come from the named boundary transitions, as everywhere else; a
/// grounded Laplacian's absorption is not modelled as an escape.
///
/// States are labelled by their global index, so a restriction composes with the
/// density chain the same way a reaction-system restriction does.
template <typename SparseMat, typename State = std::vector<int>>
[[nodiscard]] Subnetwork<State> laplacian_subnetwork(const SparseMat &laplacian,
                                                     std::span<const real> stationary_sqrt,
                                                     const std::vector<idx> &states) {
    std::unordered_map<idx, idx> local;
    std::vector<State> labels;
    std::vector<real> stationary;
    local.reserve(states.size());
    labels.reserve(states.size());
    stationary.reserve(states.size());
    for (idx i = 0; i < states.size(); ++i) {
        local.emplace(states[i], i);
        labels.push_back(State{static_cast<int>(states[i])});
        stationary.push_back(stationary_sqrt[states[i]] * stationary_sqrt[states[i]]);
    }

    std::vector<idx> rows, columns;
    std::vector<real> values;
    std::vector<BoundaryTransition<State>> boundary;

    for (idx a = 0; a < states.size(); ++a) {
        const idx global_a = states[a];
        for (auto k = laplacian.row_ptr()[global_a]; k < laplacian.row_ptr()[global_a + 1]; ++k) {
            const idx global_b = static_cast<idx>(laplacian.col_idx()[k]);
            const real entry = static_cast<real>(laplacian.values()[k]);

            if (global_b == global_a) {
                rows.push_back(a);
                columns.push_back(a);
                values.push_back(-entry);
                continue;
            }

            const real rate = -entry * stationary_sqrt[global_b] / stationary_sqrt[global_a];
            if (rate <= 0.0) {
                continue;
            }

            const auto found = local.find(global_b);
            if (found == local.end()) {
                boundary.push_back({a, State{static_cast<int>(global_b)}, rate});
            } else {
                rows.push_back(a);
                columns.push_back(found->second);
                values.push_back(rate);
            }
        }
    }

    auto generator =
        num::SparseMatrix::from_triplets(states.size(), states.size(), rows, columns, values);
    return Subnetwork<State>(std::move(labels), std::move(generator), std::move(boundary),
                             std::move(stationary));
}

/// Successive restrictions, each grown from the boundary of the last.
///
/// Suitable as the input to `make_density_chain`.
template <typename SparseMat, typename State = std::vector<int>>
[[nodiscard]] std::vector<Subnetwork<State>>
laplacian_restrictions(const SparseMat &laplacian, std::span<const real> stationary_sqrt,
                       idx origin, idx capacity, int count, unsigned seed = 42) {
    if (count < 1) {
        throw std::invalid_argument("Laplacian restrictions require count >= 1");
    }

    std::vector<idx> window = laplacian_neighborhood(laplacian, stationary_sqrt, origin, capacity);
    std::unordered_map<State, real, StateHash<State>> arrival{
        {State{static_cast<int>(origin)}, 1.0}};
    std::mt19937 random(seed);

    std::vector<Subnetwork<State>> chain;
    chain.reserve(static_cast<std::size_t>(count));

    for (int step = 0; step < count; ++step) {
        chain.push_back(laplacian_subnetwork<SparseMat, State>(laplacian, stationary_sqrt, window));
        if (step + 1 == count || chain.back().boundary().empty()) {
            break;
        }

        real mass = 0.0;
        for (const auto &[state, weight] : arrival)
            mass += weight;
        num::Vector entrance(chain.back().size(), 0.0);
        for (const auto &[state, weight] : arrival) {
            const idx position = chain.back().find(state);
            if (position < chain.back().size())
                entrance[position] = weight / mass;
        }
        const num::Vector occupation = chain.back().solve_transpose(entrance);

        std::unordered_map<State, real, StateHash<State>> next_arrival;
        for (const auto &transition : chain.back().boundary()) {
            next_arrival[transition.destination] += transition.rate * occupation[transition.source];
        }
        if (next_arrival.empty()) {
            break;
        }
        arrival = subsample_support(next_arrival, capacity, random);
        window.clear();
        window.reserve(arrival.size());
        for (const auto &[state, weight] : arrival)
            window.push_back(static_cast<idx>(state[0]));
    }
    return chain;
}

/// One ELSE trajectory over a reversible Laplacian.
///
/// Rebuilds the neighborhood around the current state at every macrostep, so the
/// window follows the walk.
template <typename ReversibleLaplacian, typename State = std::vector<int>>
[[nodiscard]] Trajectory<State>
laplacian_else_trajectory(const ReversibleLaplacian &generator, idx origin, real initial_time,
                          real final_time, EnsembleOptions options = {}, unsigned seed = 42) {
    const auto &laplacian = generator.matrix();
    const std::span<const real> stationary_sqrt(generator.stationary_sqrt());
    if (origin >= laplacian.n_rows() || initial_time > final_time) {
        throw std::invalid_argument("invalid Laplacian trajectory request");
    }

    std::mt19937 random(seed);
    Trajectory<State> trajectory;
    trajectory.times.push_back(initial_time);
    trajectory.states.push_back(State{static_cast<int>(origin)});

    idx current = origin;
    real time = initial_time;

    for (std::size_t step = 0; step < options.maximum_steps && time < final_time; ++step) {
        const auto window =
            laplacian_neighborhood(laplacian, stationary_sqrt, current, options.capacity);
        const auto subnetwork =
            laplacian_subnetwork<decltype(laplacian), State>(laplacian, stationary_sqrt, window);
        if (subnetwork.escape_states().empty()) {
            break;
        }

        const idx entrance = subnetwork.find(State{static_cast<int>(current)});
        const std::vector<idx> entrances{entrance};
        const EntranceLaw law = entrance_law(subnetwork, std::span<const idx>(entrances));

        const num::Vector beta = escape_distribution(subnetwork, law, 0);
        num::CategoricalSampler escape{std::span<const real>(beta.data(), beta.size())};
        const idx escape_state = subnetwork.escape_states()[escape(random)];
        const real waiting = conditional_escape_time(law, 0, escape_state);

        std::vector<idx> destinations;
        std::vector<real> rates;
        for (const auto &transition : subnetwork.boundary()) {
            if (transition.source == escape_state) {
                destinations.push_back(static_cast<idx>(transition.destination[0]));
                rates.push_back(transition.rate);
            }
        }
        if (!(waiting > 0.0) || destinations.empty() || time + waiting >= final_time) {
            break;
        }

        num::CategoricalSampler channel{std::span<const real>(rates)};
        time += waiting;
        current = destinations[channel(random)];
        trajectory.times.push_back(time);
        trajectory.states.push_back(State{static_cast<int>(current)});
    }

    if (final_time < std::numeric_limits<real>::infinity() &&
        trajectory.times.back() < final_time) {
        trajectory.times.push_back(final_time);
        trajectory.states.push_back(trajectory.states.back());
    }
    return trajectory;
}

} // namespace elsex
