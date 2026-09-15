#pragma once

#include "else/restriction/laplacian.hpp"
#include "linear/eigen/jacobi_eig.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace laplacian_validation {

constexpr num::idx committor_count = 5;
using Curves = std::array<num::array<double>, committor_count>;

template <typename SparseMat>
[[nodiscard]] Curves
ssa_committor_means(const SparseMat &laplacian, num::view<const double> stationary_sqrt,
                    const num::array<num::array<double>> &committors, num::idx initial,
                    num::view<const double> observation_times, num::idx samples, unsigned seed) {
    if (samples == 0 || observation_times.empty())
        throw std::invalid_argument("the validation ensemble must be nonempty");

#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
#else
    const int thread_count = 1;
#endif
    num::array<Curves> partial(static_cast<num::idx>(thread_count));
    for (auto &thread_curves : partial)
        for (auto &curve : thread_curves)
            curve.assign(observation_times.size(), 0.0);

#pragma omp parallel for schedule(static)
    for (num::idx sample = 0; sample < samples; ++sample) {
#ifdef _OPENMP
        const int thread = omp_get_thread_num();
#else
        const int thread = 0;
#endif
        auto &local = partial[static_cast<num::idx>(thread)];
        num::rng random(seed + static_cast<unsigned>(sample));
        num::idx state = initial;
        double time = 0.0;
        num::idx observation = 0;
        while (observation < observation_times.size()) {
            double total_rate = 0.0;
            for (auto k = laplacian.row_ptr()[state]; k < laplacian.row_ptr()[state + 1]; ++k) {
                const auto destination = static_cast<num::idx>(laplacian.col_idx()[k]);
                if (destination != state)
                    total_rate += -laplacian.values()[k] * stationary_sqrt[destination] /
                                  stationary_sqrt[state];
            }
            if (!(total_rate > 0.0))
                throw std::runtime_error("the direct Laplacian chain cannot leave its state");

            std::exponential_distribution<double> holding(total_rate);
            const double next_time = time + holding(random);
            while (observation < observation_times.size() &&
                   observation_times[observation] < next_time) {
                for (num::idx j = 0; j < committor_count; ++j)
                    local[j][observation] += committors[state][j];
                ++observation;
            }
            if (observation == observation_times.size())
                break;

            std::uniform_real_distribution<double> target_distribution(0.0, total_rate);
            const double target = target_distribution(random);
            double cumulative = 0.0;
            num::idx destination = state;
            for (auto k = laplacian.row_ptr()[state]; k < laplacian.row_ptr()[state + 1]; ++k) {
                const auto candidate = static_cast<num::idx>(laplacian.col_idx()[k]);
                if (candidate == state)
                    continue;
                cumulative +=
                    -laplacian.values()[k] * stationary_sqrt[candidate] / stationary_sqrt[state];
                if (cumulative >= target) {
                    destination = candidate;
                    break;
                }
            }
            state = destination;
            time = next_time;
        }
    }

    Curves result;
    for (auto &curve : result)
        curve.assign(observation_times.size(), 0.0);
    for (const auto &thread_curves : partial)
        for (num::idx j = 0; j < committor_count; ++j)
            for (num::idx k = 0; k < observation_times.size(); ++k)
                result[j][k] += thread_curves[j][k] / static_cast<double>(samples);
    return result;
}

inline num::vec symmetric_semigroup_row(const num::eigen_result &eigen, double time, num::idx row) {
    const num::idx n = eigen.values.size();
    num::vec result(n, 0.0);
    for (num::idx mode = 0; mode < n; ++mode) {
        const double coefficient = eigen.vectors(row, mode) * std::exp(-time * eigen.values[mode]);
        for (num::idx column = 0; column < n; ++column)
            result[column] += coefficient * eigen.vectors(column, mode);
    }
    return result;
}

template <typename SparseMat>
num::array<num::idx> graph_distance_levels(const SparseMat &laplacian,
                                           const num::array<num::idx> &states, num::idx origin) {
    num::table<num::idx, num::idx> local;
    local.reserve(states.size());
    for (num::idx index = 0; index < states.size(); ++index)
        local.emplace(states[index], index);

    num::array<num::idx> levels(states.size(), states.size());
    std::queue<num::idx> frontier;
    levels[local.at(origin)] = 0;
    frontier.push(origin);
    while (!frontier.empty()) {
        const num::idx state = frontier.front();
        frontier.pop();
        const num::idx level = levels[local.at(state)];
        for (auto entry = laplacian.row_ptr()[state]; entry < laplacian.row_ptr()[state + 1];
             ++entry) {
            const num::idx neighbor = static_cast<num::idx>(laplacian.col_idx()[entry]);
            const auto found = local.find(neighbor);
            if (neighbor != state && found != local.end() &&
                levels[found->second] == states.size()) {
                levels[found->second] = level + 1;
                frontier.push(neighbor);
            }
        }
    }
    return levels;
}

inline double conditional_survival_probability(const num::eigen_result &eigen, num::idx entrance,
                                               num::idx exit_source, double time) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (num::idx mode = 0; mode < eigen.values.size(); ++mode) {
        const double coefficient =
            eigen.vectors(entrance, mode) * eigen.vectors(exit_source, mode) / eigen.values[mode];
        denominator += coefficient;
        numerator += coefficient * std::exp(-time * eigen.values[mode]);
    }
    return std::clamp(numerator / denominator, 0.0, 1.0);
}

template <typename RNG>
double sample_conditional_exit_time(const num::eigen_result &eigen, num::idx entrance,
                                    num::idx exit_source, RNG &random) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double target_survival = 1.0 - uniform(random);

    double denominator = 0.0;
    double mean = 0.0;
    for (num::idx mode = 0; mode < eigen.values.size(); ++mode) {
        const double coefficient = eigen.vectors(entrance, mode) * eigen.vectors(exit_source, mode);
        denominator += coefficient / eigen.values[mode];
        mean += coefficient / (eigen.values[mode] * eigen.values[mode]);
    }
    mean /= denominator;
    double lower = 0.0;
    double upper = std::max(2.0 * mean, 1e-16);
    while (conditional_survival_probability(eigen, entrance, exit_source, upper) > target_survival)
        upper *= 2.0;

    for (int iteration = 0; iteration < 56; ++iteration) {
        const double midpoint = 0.5 * (lower + upper);
        if (conditional_survival_probability(eigen, entrance, exit_source, midpoint) >
            target_survival)
            lower = midpoint;
        else
            upper = midpoint;
    }
    return 0.5 * (lower + upper);
}

template <typename SparseMat>
[[nodiscard]] Curves
labeled_subsweep_committor_means(const SparseMat &laplacian,
                                 num::view<const double> stationary_sqrt,
                                 const num::array<num::array<double>> &committors, num::idx initial,
                                 num::view<const double> observation_times, num::idx samples,
                                 unsigned seed, num::idx subnetwork_capacity) {
    if (samples == 0 || observation_times.empty() || subnetwork_capacity == 0)
        throw std::invalid_argument(
            "the labeled subsweep ensemble and its subnetworks must be nonempty");

#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
#else
    const int thread_count = 1;
#endif
    num::array<Curves> partial(static_cast<num::idx>(thread_count));
    for (auto &thread_curves : partial)
        for (auto &curve : thread_curves)
            curve.assign(observation_times.size(), 0.0);

#pragma omp parallel for schedule(dynamic)
    for (num::idx sample = 0; sample < samples; ++sample) {
#ifdef _OPENMP
        const int thread = omp_get_thread_num();
#else
        const int thread = 0;
#endif
        auto &local = partial[static_cast<num::idx>(thread)];
        num::rng random(seed + static_cast<unsigned>(sample));
        num::idx current = initial;
        double time = 0.0;
        num::idx observation = 0;

        while (observation < observation_times.size()) {
            const auto window = else_sim::laplacian_neighborhood(laplacian, stationary_sqrt,
                                                                 current, subnetwork_capacity);
            const auto subnetwork =
                else_sim::laplacian_subnetwork(laplacian, stationary_sqrt, window);
            const auto entrance =
                else_sim::find(subnetwork, num::multi_index{static_cast<int>(current)});
            const num::vec local_stationary_sqrt =
                else_sim::detail::stationary_weights(subnetwork.stationary.span());
            const num::spmat symmetric = else_sim::detail::similarity_scaled(
                subnetwork.operator_matrix, local_stationary_sqrt.span());
            const num::array<num::idx> levels = graph_distance_levels(laplacian, window, current);
            const num::block_cholesky_factor factor = num::factor_block_cholesky(symmetric, levels);
            num::vec entrance_indicator(else_sim::size(subnetwork), 0.0);
            entrance_indicator[entrance] = 1.0;
            num::vec symmetric_occupation(else_sim::size(subnetwork), 0.0);
            num::solve(factor, entrance_indicator, symmetric_occupation);
            num::vec exit_source_weights(else_sim::size(subnetwork), 0.0);
            for (num::idx state = 0; state < else_sim::size(subnetwork); ++state)
                exit_source_weights[state] = std::max(
                    0.0, symmetric_occupation[state] * local_stationary_sqrt[state] /
                             local_stationary_sqrt[entrance] * subnetwork.escape_rates[state]);
            num::categorical_sampler exit_source_sampler{
                num::view<const double>(exit_source_weights)};
            const num::idx exit_source = exit_source_sampler(random);

            const num::eigen_result eigen =
                num::eig_sym(num::assume_symmetric(num::dense(symmetric)));
            const double waiting =
                sample_conditional_exit_time(eigen, entrance, exit_source, random);
            const double exit_time = time + waiting;

            while (observation < observation_times.size() &&
                   observation_times[observation] < exit_time) {
                const double elapsed = observation_times[observation] - time;
                const num::vec left = symmetric_semigroup_row(eigen, elapsed, entrance);
                const num::vec right =
                    symmetric_semigroup_row(eigen, waiting - elapsed, exit_source);
                num::vec bridge_weights(else_sim::size(subnetwork), 0.0);
                for (num::idx state = 0; state < else_sim::size(subnetwork); ++state)
                    bridge_weights[state] = std::max(0.0, left[state] * right[state]);
                num::categorical_sampler bridge_sampler{num::view<const double>(bridge_weights)};
                const num::idx local_state = bridge_sampler(random);
                const auto global_state = static_cast<num::idx>(subnetwork.states[local_state][0]);
                for (num::idx j = 0; j < committor_count; ++j)
                    local[j][observation] += committors[global_state][j];
                ++observation;
            }
            if (observation == observation_times.size())
                break;

            num::array<num::idx> destinations;
            num::array<double> rates;
            for (const auto &transition : subnetwork.boundary) {
                if (transition.source == exit_source) {
                    destinations.push_back(static_cast<num::idx>(transition.destination[0]));
                    rates.push_back(transition.rate);
                }
            }
            num::categorical_sampler channel{num::view<const double>(rates)};
            current = destinations[channel(random)];
            time = exit_time;
        }
    }

    Curves result;
    for (auto &curve : result)
        curve.assign(observation_times.size(), 0.0);
    for (const auto &thread_curves : partial)
        for (num::idx j = 0; j < committor_count; ++j)
            for (num::idx k = 0; k < observation_times.size(); ++k)
                result[j][k] += thread_curves[j][k] / static_cast<double>(samples);
    return result;
}

[[nodiscard]] inline double maximum_discrepancy(const Curves &left, const Curves &right) {
    double discrepancy = 0.0;
    for (num::idx j = 0; j < committor_count; ++j)
        for (num::idx k = 0; k < left[j].size(); ++k)
            discrepancy = std::max(discrepancy, std::abs(left[j][k] - right[j][k]));
    return discrepancy;
}

} // namespace laplacian_validation
