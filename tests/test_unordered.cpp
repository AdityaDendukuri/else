// The unordered loop against a dense matrix exponential: one subnetwork on
// a closed space reproduces exp(R t) outright, and a chain of subnetworks
// converges to it as it lengthens.
#include "check.hpp"
#include "common/rate_matrix.hpp"
#include "linear/expv/expv.hpp"
#include <cmath>
#include <cstdio>
#include <functional>

namespace {

using namespace subsweep;
using namespace subsweep::examples;
using State = num::multi_index;
using Rates = array<real>;

constexpr int capacity_states = 9; // copy numbers 0..8
constexpr real birth = 1.2, death = 0.5, horizon = 0.6;

struct System {
    array<array<int>> changes;
    array<std::function<real(const State &, const Rates &, real)>> propensities;
};

// Birth-death on 0..8 with reflecting ends, so no probability exits.
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

array<State> all_states() {
    array<State> states;
    for (int copies = 0; copies < capacity_states; ++copies)
        num::append(states, State{copies});
    return states;
}

// Row `start` of exp(R t): (exp(R^T t) e_start)_j.
num::vec reference_density(real time, idx start) {
    const Rates rates{birth, death};
    const auto model = closed_chain();
    const subnetwork sn = restriction(reaction_rate_matrix(model, rates), all_states(), {});
    num::vec initial(capacity_states, 0.0);
    initial[start] = 1.0;
    return num::expv(time, num::scaled(num::transpose(sn.matrix()), -1.0), initial, 30, 1e-12);
}

real total_variation(const transient_distribution<State> &solution, const num::vec &reference) {
    num::vec aligned(capacity_states, 0.0);
    for (idx k = 0; k < solution.states.size(); ++k)
        aligned[static_cast<idx>(solution.states[k][0])] = solution.probability[k];
    real distance = 0.0;
    for (idx j = 0; j < capacity_states; ++j)
        distance += std::abs(aligned[j] - reference[j]);
    return 0.5 * distance;
}

} // namespace

void test_single_subnetwork_reproduces_the_matrix_exponential() {
    const Rates rates{birth, death};
    const auto model = closed_chain();
    const auto c = reaction_rate_matrix(model, rates);
    sweep_options options;
    options.capacity = 64;
    const auto chain = unordered_subsweep(c, State{0}, 1, options, 42, nullptr, nullptr, all_states());
    check::that(chain.links.size() == 1 && chain.links.front().exits.empty(),
                "a closed space has no boundary");
    const auto solution = reconstruct_distribution(chain, horizon, 24);
    const num::vec reference = reference_density(horizon, 0);
    for (idx k = 0; k < solution.states.size(); ++k)
        check::close(solution.probability[k], reference[static_cast<idx>(solution.states[k][0])],
                     "Talbot inversion equals exp(Rt)", 1e-8);
    check::done("one subnetwork over a closed space reproduces exp(R t)");
}

void test_composition_converges_to_the_matrix_exponential() {
    const Rates rates{birth, death};
    const auto model = closed_chain();
    const auto c = reaction_rate_matrix(model, rates);
    const num::vec reference = reference_density(horizon, 0);
    sweep_options options;
    options.capacity = 64; // no shedding; truncation is the chain length alone
    const auto distance_after = [&](int steps) {
        const auto chain = unordered_subsweep(c, State{0}, steps, options);
        return total_variation(reconstruct_distribution(chain, horizon, 24), reference);
    };
    const real few = distance_after(1), several = distance_after(4), many = distance_after(12);
    check::that(several < few, "composing more subnetworks improves the density");
    check::that(many <= several, "the improvement does not reverse");
    check::that(many < 1e-3, "a long chain matches exp(R t) closely");
    std::printf("          (total variation: %.2e at N=1, %.2e at N=4, %.2e at N=12)\n", few,
                several, many);
    check::done("composition converges to exp(R t) as the chain lengthens");
}

void test_shedding_and_resampling_keep_a_distribution() {
    const Rates rates{birth, death};
    const auto model = closed_chain();
    const auto c = reaction_rate_matrix(model, rates);
    const num::vec reference = reference_density(horizon, 0);
    for (const support_resampling resampling :
         {support_resampling::weighted_support, support_resampling::multinomial_particles}) {
        sweep_options options;
        options.capacity = 3;
        options.resampling = resampling;
        options.particles = 3;
        const auto chain = unordered_subsweep(c, State{0}, 12, options, 17);
        for (const chain_link &link : chain.links)
            check::that(link.matrix.n_rows() <= 3, "every link is within capacity");
        const auto solution = reconstruct_distribution(chain, horizon, 18);
        real total = 0.0;
        for (const real value : solution.probability) {
            check::that(value >= 0.0, "probabilities are non-negative");
            total += value;
        }
        check::close(total, 1.0, "probabilities sum to one", 1e-12);
        std::printf("          (TV at capacity 3: %.3e)\n", total_variation(solution, reference));
    }
    check::done("shedding and both resampling policies return a distribution");
}

void test_multinomial_particles_recover_the_exit_weights() {
    const table<State, real> exits{{State{0}, 0.6}, {State{1}, 0.3}, {State{2}, 0.1}};
    sweep_options options;
    options.capacity = 20;
    options.resampling = support_resampling::multinomial_particles;
    num::rng random(7);
    num::vec mean(3, 0.0);
    constexpr idx repetitions = 5000;
    for (idx repetition = 0; repetition < repetitions; ++repetition) {
        const auto sample = resample_support(exits, options, random);
        real total = 0.0;
        for (const auto &[state, weight] : sample) {
            total += weight;
            mean[static_cast<idx>(state[0])] += weight / repetitions;
        }
        check::close(total, 1.0, "particle resampling preserves mass", 1e-12);
    }
    check::close(mean[0], 0.6, "first exit weight", 8e-3);
    check::close(mean[1], 0.3, "second exit weight", 8e-3);
    check::close(mean[2], 0.1, "third exit weight", 8e-3);
    check::done("multinomial particles recover the exit distribution in mean");
}

int main() {
    std::printf("subsweep unordered\n");
    test_single_subnetwork_reproduces_the_matrix_exponential();
    test_composition_converges_to_the_matrix_exponential();
    test_shedding_and_resampling_keep_a_distribution();
    test_multinomial_particles_recover_the_exit_weights();
    return check::report("subsweep unordered");
}
