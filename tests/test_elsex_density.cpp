/// @file tests/test_elsex_density.cpp
/// @brief Unlabeled ELSE density evolution.
///
/// Both checks are against a dense matrix exponential rather than against the
/// legacy tree. On a closed state space a single subnetwork has no boundary, so
/// the Talbot inversion must reproduce exp(R t) outright; and for a chain of
/// subnetworks the paper's exactness theorem says the composition converges to
/// the same thing as the chain lengthens.

#include "check.hpp"

#include "else/algorithms/density.hpp"
#include "else/restriction/restriction.hpp"

#include "linear/expv/expv.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

using State = num::multi_index;
using Rates = num::array<num::real>;
using num::idx;
using num::real;

constexpr int capacity_states = 9; // copy numbers 0..8
constexpr real birth = 1.2;
constexpr real death = 0.5;
constexpr real horizon = 0.6;

/// Birth-death on 0..capacity_states-1 with reflecting ends, so the space is
/// closed and no probability escapes it.
struct System {
    num::array<num::array<int>> changes;
    num::array<std::function<real(const State &, const Rates &, real)>> propensities;

    [[nodiscard]] real total_propensity(const State &state, const Rates &rates, real time) const {
        real total = 0.0;
        for (const auto &propensity : propensities) {
            total += propensity(state, rates, time);
        }
        return total;
    }
};

System closed_chain() {
    System model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const State &x, const Rates &r, real) {
            return x[0] + 1 < capacity_states ? r[0] : 0.0;
        },
        [](const State &x, const Rates &r, real) { return r[1] * static_cast<real>(x[0]); },
    };
    return model;
}

num::array<State> all_states() {
    num::array<State> states;
    for (int copies = 0; copies < capacity_states; ++copies) {
        states.push_back(State{copies});
    }
    return states;
}

/// Row `start` of exp(R t) for the full closed generator.
num::array<real> reference_density(real time, idx start) {
    const auto model = closed_chain();
    const Rates rates{birth, death};
    const auto subnetwork = else_sim::restriction(model, rates, all_states());

    // p^T = p0^T exp(Rt), and (exp(R^T t) delta_start)_j = (exp(Rt))_{start,j},
    // so a transpose exponential-vector product gives the row directly.
    num::vec initial(capacity_states, 0.0);
    initial[start] = 1.0;
    const num::vec propagated =
        num::expv(time, num::transpose(subnetwork.generator), initial, 30, 1e-12);

    num::array<real> density(capacity_states, 0.0);
    for (idx j = 0; j < static_cast<idx>(capacity_states); ++j) {
        density[j] = propagated[j];
    }
    return density;
}

/// Total variation between an ELSE solution and the reference, over all states.
real total_variation(const else_sim::DensitySolution<State> &solution,
                     const num::array<real> &reference) {
    num::array<real> aligned(capacity_states, 0.0);
    for (num::idx k = 0; k < solution.states.size(); ++k) {
        aligned[static_cast<idx>(solution.states[k][0])] = solution.probability[k];
    }
    real distance = 0.0;
    for (idx j = 0; j < static_cast<idx>(capacity_states); ++j) {
        distance += std::abs(aligned[j] - reference[j]);
    }
    return 0.5 * distance;
}

} // namespace

void test_single_subnetwork_reproduces_the_matrix_exponential() {
    const auto model = closed_chain();
    const Rates rates{birth, death};

    // The whole space in one subnetwork: no boundary, so the composition is a
    // single resolvent and its inversion must be exp(R t).
    num::array<else_sim::Subnetwork<State>> chain;
    chain.push_back(else_sim::restriction(model, rates, all_states()));
    check::that(chain.front().boundary.empty(), "a closed space has no boundary");

    auto density = else_sim::make_density_chain(std::move(chain));
    const auto solution = else_sim::inverse_laplace_density(density, State{0}, horizon, 24);
    const auto reference = reference_density(horizon, 0);

    for (num::idx k = 0; k < solution.states.size(); ++k) {
        const idx j = static_cast<idx>(solution.states[k][0]);
        check::close(solution.probability[k], reference[j], "Talbot inversion equals exp(Rt)",
                     1e-8);
    }
    check::done("one subnetwork over a closed space reproduces exp(R t)");
}

void test_composition_converges_to_the_matrix_exponential() {
    const auto model = closed_chain();
    const Rates rates{birth, death};
    const auto reference = reference_density(horizon, 0);

    else_sim::EnsembleOptions options;
    options.capacity = 64; // no shedding; truncation is the chain length alone
    options.expansion_depth = 1;

    const auto distance_after = [&](int steps) {
        auto chain = else_sim::density_subnetworks(model, rates, State{0}, steps, options);
        auto density = else_sim::make_density_chain(std::move(chain));
        return total_variation(else_sim::inverse_laplace_density(density, State{0}, horizon, 24),
                               reference);
    };

    const real few = distance_after(1);
    const real several = distance_after(4);
    const real many = distance_after(12);

    check::that(several < few, "composing more subnetworks improves the density");
    check::that(many <= several, "the improvement does not reverse");
    check::that(many < 1e-3, "a long chain matches exp(R t) closely");
    std::printf("          (total variation: %.2e at N=1, %.2e at N=4, %.2e at N=12)\n", few,
                several, many);
    check::done("composition converges to exp(R t) as the chain lengthens");
}

void test_entrance_resampling_scales_boundary_arrival() {
    const auto model = closed_chain();
    const Rates rates{birth, death};
    num::array<else_sim::Subnetwork<State>> subnetworks;
    subnetworks.push_back(
        else_sim::restriction(model, rates, num::array<State>{{0}, {1}, {2}, {3}}));
    subnetworks.push_back(
        else_sim::restriction(model, rates, num::array<State>{{4}, {5}, {6}, {7}, {8}}));
    num::array<num::table<State, real>> scales(2);
    scales[0][State{0}] = 1.0;
    scales[1][State{4}] = 3.0;
    auto density = else_sim::make_density_chain(std::move(subnetworks), scales);

    num::array<num::cplx> local(4, num::cplx(0.0, 0.0));
    local[3] = num::cplx(2.0, 0.0);
    const auto arrival = else_sim::advance_arrival(density, 0, local);
    check::close(arrival[0].real(), 6.0 * birth,
                 "the empirical-to-exact ratio scales the next entrance", 1e-12);
    check::done("density composition applies entrance-resampling weights");
}

void test_density_is_a_distribution() {
    const auto model = closed_chain();
    const Rates rates{birth, death};
    else_sim::EnsembleOptions options;
    options.capacity = 64;

    auto chain = else_sim::density_subnetworks(model, rates, State{3}, 6, options);
    check::that(!chain.empty(), "the chain is non-empty");
    auto density = else_sim::make_density_chain(std::move(chain));
    const auto solution = else_sim::inverse_laplace_density(density, State{3}, horizon, 24);

    check::that(solution.states.size() == solution.probability.size(),
                "one probability per represented state");
    real total = 0.0;
    for (const real value : solution.probability) {
        check::that(value >= 0.0, "probabilities are non-negative");
        total += value;
    }
    check::close(total, 1.0, "probabilities sum to one", 1e-12);
    check::done("the reported density is a normalized distribution");
}

void test_multinomial_particle_resampling() {
    const num::table<State, real> exit_distribution{
        {State{0}, 0.6}, {State{1}, 0.3}, {State{2}, 0.1}};
    constexpr idx particles = 20;
    num::rng random(7);

    const auto sample =
        else_sim::multinomial_particle_resample(exit_distribution, particles, random);
    real total = 0.0;
    for (const auto &[state, weight] : sample) {
        total += weight;
        const real count = weight * static_cast<real>(particles);
        check::close(count, std::round(count), "particle weights are empirical counts", 1e-12);
    }
    check::close(total, 1.0, "particle resampling preserves total mass", 1e-12);
    check::that(sample.size() <= particles, "repeated particle draws are aggregated");

    constexpr idx repetitions = 5000;
    num::array<real> mean(3, 0.0);
    for (idx repetition = 0; repetition < repetitions; ++repetition) {
        const auto resampled =
            else_sim::multinomial_particle_resample(exit_distribution, particles, random);
        for (const auto &[state, weight] : resampled)
            mean[static_cast<idx>(state[0])] += weight / static_cast<real>(repetitions);
    }
    check::close(mean[0], 0.6, "particle mean recovers the first exit weight", 8e-3);
    check::close(mean[1], 0.3, "particle mean recovers the second exit weight", 8e-3);
    check::close(mean[2], 0.1, "particle mean recovers the third exit weight", 8e-3);
    std::printf("          (particle mean after %zu resamples: %.4f, %.4f, %.4f)\n",
                static_cast<num::idx>(repetitions), mean[0], mean[1], mean[2]);
    check::done("multinomial density particles recover the exit distribution in mean");
}

void test_density_resampling_policy() {
    const num::table<State, real> exit_distribution{
        {State{0}, 0.4}, {State{1}, 0.3}, {State{2}, 0.2}, {State{3}, 0.1}};
    else_sim::EnsembleOptions options;
    options.capacity = 3;
    options.density_resampling = else_sim::DensityResampling::MultinomialParticles;
    options.density_particles = 3;
    num::rng random(11);

    const auto sample = else_sim::resample_density_support(exit_distribution, options, random);
    real total = 0.0;
    for (const auto &[state, weight] : sample)
        total += weight;
    check::close(total, 1.0, "selected particle policy preserves mass", 1e-12);
    check::that(sample.size() <= options.density_particles,
                "particle policy returns no more unique states than walkers");
    check::done("density options select multinomial-particle resampling");
}

void test_particle_resampling_runs_the_density_pipeline() {
    const auto model = closed_chain();
    const Rates rates{birth, death};
    const auto reference = reference_density(horizon, 0);

    const auto distance_for = [&](else_sim::DensityResampling resampling) {
        else_sim::EnsembleOptions options;
        options.capacity = 3;
        options.expansion_depth = 1;
        options.density_resampling = resampling;
        options.density_particles = 3;
        auto chain = else_sim::density_subnetworks(model, rates, State{0}, 12, options, 17);
        auto density = else_sim::make_density_chain(std::move(chain));
        const auto solution = else_sim::inverse_laplace_density(density, State{0}, horizon, 18);
        real total = 0.0;
        for (const real probability : solution.probability)
            total += probability;
        check::close(total, 1.0, "resampled density pipeline returns unit mass", 1e-12);
        return total_variation(solution, reference);
    };

    const real support_error = distance_for(else_sim::DensityResampling::WeightedSupport);
    const real particle_error = distance_for(else_sim::DensityResampling::MultinomialParticles);
    check::that(std::isfinite(support_error), "weighted-support density error is finite");
    check::that(std::isfinite(particle_error), "particle-resampled density error is finite");
    std::printf("          (TV at capacity 3: support %.3e, particles %.3e)\n", support_error,
                particle_error);
    check::done("both density policies run through density reconstruction");
}

int main() {
    std::printf("else density\n");
    test_single_subnetwork_reproduces_the_matrix_exponential();
    test_composition_converges_to_the_matrix_exponential();
    test_entrance_resampling_scales_boundary_arrival();
    test_density_is_a_distribution();
    test_multinomial_particle_resampling();
    test_density_resampling_policy();
    test_particle_resampling_runs_the_density_pipeline();
    return check::report("else density");
}
