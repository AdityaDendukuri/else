/// @file tests/test_elsex_density.cpp
/// @brief Unlabeled ELSE density evolution.
///
/// Both checks are against a dense matrix exponential rather than against the
/// legacy tree. On a closed state space a single subnetwork has no boundary, so
/// the Talbot inversion must reproduce exp(R t) outright; and for a chain of
/// subnetworks the paper's exactness theorem says the composition converges to
/// the same thing as the chain lengthens.

#include "check.hpp"

#include "elsex/density.hpp"
#include "elsex/restriction.hpp"

#include "linear/expv/expv.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"

#include <cmath>
#include <functional>
#include <vector>

namespace {

using State = std::vector<int>;
using Rates = std::vector<num::real>;
using num::idx;
using num::real;

constexpr int capacity_states = 9; // copy numbers 0..8
constexpr real birth = 1.2;
constexpr real death = 0.5;
constexpr real horizon = 0.6;

/// Birth-death on 0..capacity_states-1 with reflecting ends, so the space is
/// closed and no probability escapes it.
struct System {
    std::vector<std::vector<int>> changes;
    std::vector<std::function<real(const State &, const Rates &, real)>> propensities;

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

std::vector<State> all_states() {
    std::vector<State> states;
    for (int copies = 0; copies < capacity_states; ++copies) {
        states.push_back(State{copies});
    }
    return states;
}

/// Row `start` of exp(R t) for the full closed generator.
std::vector<real> reference_density(real time, idx start) {
    const auto model = closed_chain();
    const Rates rates{birth, death};
    const auto subnetwork = elsex::restriction(model, rates, all_states());

    // p^T = p0^T exp(Rt), and (exp(R^T t) delta_start)_j = (exp(Rt))_{start,j},
    // so a transpose exponential-vector product gives the row directly.
    num::Vector initial(capacity_states, 0.0);
    initial[start] = 1.0;
    const num::Vector propagated =
        num::expv(time, num::transpose(subnetwork.generator()), initial, 30, 1e-12);

    std::vector<real> density(capacity_states, 0.0);
    for (idx j = 0; j < static_cast<idx>(capacity_states); ++j) {
        density[j] = propagated[j];
    }
    return density;
}

/// Total variation between an ELSE solution and the reference, over all states.
real total_variation(const elsex::DensitySolution<State> &solution,
                     const std::vector<real> &reference) {
    std::vector<real> aligned(capacity_states, 0.0);
    for (std::size_t k = 0; k < solution.states.size(); ++k) {
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
    std::vector<elsex::Subnetwork<State>> chain;
    chain.push_back(elsex::restriction(model, rates, all_states()));
    check::that(chain.front().boundary().empty(), "a closed space has no boundary");

    auto density = elsex::make_density_chain(std::move(chain));
    const auto solution = elsex::inverse_laplace_density(density, State{0}, horizon, 24);
    const auto reference = reference_density(horizon, 0);

    for (std::size_t k = 0; k < solution.states.size(); ++k) {
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

    elsex::EnsembleOptions options;
    options.capacity = 64; // no shedding; truncation is the chain length alone
    options.expansion_depth = 1;

    const auto distance_after = [&](int steps) {
        auto chain = elsex::density_subnetworks(model, rates, State{0}, steps, options);
        auto density = elsex::make_density_chain(std::move(chain));
        return total_variation(elsex::inverse_laplace_density(density, State{0}, horizon, 24),
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

void test_density_is_a_distribution() {
    const auto model = closed_chain();
    const Rates rates{birth, death};
    elsex::EnsembleOptions options;
    options.capacity = 64;

    auto chain = elsex::density_subnetworks(model, rates, State{3}, 6, options);
    check::that(!chain.empty(), "the chain is non-empty");
    auto density = elsex::make_density_chain(std::move(chain));
    const auto solution = elsex::inverse_laplace_density(density, State{3}, horizon, 24);

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

int main() {
    std::printf("elsex density\n");
    test_single_subnetwork_reproduces_the_matrix_exponential();
    test_composition_converges_to_the_matrix_exponential();
    test_density_is_a_distribution();
    return check::report("elsex density");
}
