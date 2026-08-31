/// @file tests/test_elsex_ensemble.cpp
/// @brief Finite-m ensemble ELSE.
///
/// Two of these checks are analytic rather than comparative. A pure birth
/// process has only one escape route, so its conditional mean escape time is the
/// unconditional one and the macrostep length is known in closed form. And the
/// paper's claim that sharing a subnetwork changes nothing is checked directly:
/// the grouped block solve must reproduce what a solve per entrance would give.

#include "check.hpp"

#include "elsex/ensemble.hpp"
#include "elsex/law.hpp"
#include "elsex/restriction.hpp"
#include "elsex/state_graph.hpp"

#include <functional>
#include <span>
#include <vector>

namespace {

using State = std::vector<int>;
using Rates = std::vector<num::real>;
using num::idx;
using num::real;

/// The minimal reaction-system interface `restriction` and the ensemble require.
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

/// Pure birth at a constant rate: one reaction, one escape route.
System pure_birth() {
    System model;
    model.changes = {{1}};
    model.propensities = {[](const State &, const Rates &r, real) { return r[0]; }};
    return model;
}

/// Birth-death with constant birth and linear death.
System birth_death() {
    System model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const State &, const Rates &r, real) { return r[0]; },
        [](const State &x, const Rates &r, real) { return r[1] * static_cast<real>(x[0]); },
    };
    return model;
}

} // namespace

void test_pure_birth_macrostep_is_analytic() {
    // Depth-1 expansion around a single entrance gives the window {x, x+1}.
    // A pure birth process must traverse both before escaping, so the escape
    // time is the sum of two exponentials and its mean is exactly 2/lambda.
    const real lambda = 2.5;
    const Rates rates{lambda};
    const auto model = pure_birth();

    elsex::EnsembleOptions options;
    options.capacity = 32;
    options.expansion_depth = 1;
    options.maximum_steps = 4;

    const auto trajectory =
        elsex::else_trajectory(model, rates, State{0}, 0.0, 1e9, options, 7u);

    check::that(trajectory.times.size() >= 5, "four macrosteps were recorded");
    for (std::size_t step = 1; step + 1 < trajectory.times.size(); ++step) {
        check::close(trajectory.times[step] - trajectory.times[step - 1], 2.0 / lambda,
                     "macrostep length is 2/lambda", 1e-12);
        check::that(trajectory.states[step][0] == trajectory.states[step - 1][0] + 2,
                    "each macrostep advances two states");
    }
    check::done("pure-birth macrostep matches its closed form");
}

void test_shared_subnetwork_matches_per_entrance_solves() {
    // The paper's remark: grouping trajectories changes neither the escape
    // distribution nor the conditional mean. Solve the block once for all
    // entrances, then once per entrance, and require agreement.
    const Rates rates{1.4, 0.3};
    const auto model = birth_death();

    std::vector<State> states;
    for (int copies = 1; copies <= 9; ++copies) {
        states.push_back(State{copies});
    }
    const auto subnetwork = elsex::restriction(model, rates, states);

    const std::vector<idx> entrances{0, 3, 4, 8};
    const auto grouped = elsex::entrance_law(subnetwork, std::span<const idx>(entrances));

    for (idx k = 0; k < entrances.size(); ++k) {
        const std::vector<idx> single{entrances[k]};
        const auto alone = elsex::entrance_law(subnetwork, std::span<const idx>(single));

        for (idx j = 0; j < subnetwork.size(); ++j) {
            check::close(grouped.occupation(k, j), alone.occupation(0, j),
                         "grouped U row equals the solitary solve");
            check::close(grouped.second_occupation(k, j), alone.second_occupation(0, j),
                         "grouped V row equals the solitary solve");
        }

        const auto grouped_beta = elsex::escape_distribution(subnetwork, grouped, k);
        const auto alone_beta = elsex::escape_distribution(subnetwork, alone, 0);
        for (idx b = 0; b < grouped_beta.size(); ++b) {
            check::close(grouped_beta[b], alone_beta[b], "grouped beta equals the solitary beta");
        }
        for (idx b : subnetwork.escape_states()) {
            check::close(elsex::conditional_escape_time(grouped, k, b),
                         elsex::conditional_escape_time(alone, 0, b),
                         "grouped mu equals the solitary mu");
        }
    }
    check::done("sharing a subnetwork does not change the entrance laws");
}

void test_trajectories_are_well_formed() {
    const Rates rates{1.4, 0.3};
    const auto model = birth_death();

    elsex::EnsembleOptions options;
    options.capacity = 12; // small enough that shedding runs
    options.expansion_depth = 1;

    const auto ensemble =
        elsex::else_ensemble(model, rates, State{4}, 24, 0.0, 6.0, options, 2026u);

    check::that(ensemble.size() == 24, "one trajectory per requested path");
    for (const auto &trajectory : ensemble) {
        check::that(trajectory.times.size() == trajectory.states.size(),
                    "each recorded time has a state");
        check::that(!trajectory.times.empty(), "trajectory is non-empty");
        check::close(trajectory.times.front(), 0.0, "starts at the initial time");
        check::that(trajectory.states.front() == State{4}, "starts at the initial state");
        check::close(trajectory.times.back(), 6.0, "ends at the final time");

        for (std::size_t step = 1; step < trajectory.times.size(); ++step) {
            check::that(trajectory.times[step] >= trajectory.times[step - 1],
                        "times are non-decreasing");
            check::that(trajectory.states[step][0] >= 0, "copy numbers stay non-negative");
        }
    }
    check::done("trajectories are well formed under shedding");
}

void test_ensemble_is_reproducible() {
    const Rates rates{1.4, 0.3};
    const auto model = birth_death();
    elsex::EnsembleOptions options;
    options.capacity = 10;

    const auto first = elsex::else_ensemble(model, rates, State{3}, 8, 0.0, 4.0, options, 99u);
    const auto again = elsex::else_ensemble(model, rates, State{3}, 8, 0.0, 4.0, options, 99u);

    for (std::size_t t = 0; t < first.size(); ++t) {
        check::that(first[t].times.size() == again[t].times.size(),
                    "same seed gives the same step count");
        for (std::size_t step = 0; step < first[t].times.size(); ++step) {
            check::close(first[t].times[step], again[t].times[step], "same seed gives same times");
            check::that(first[t].states[step] == again[t].states[step],
                        "same seed gives same states");
        }
    }
    check::done("the ensemble is reproducible at a fixed seed");
}

void test_state_graph_identities_persist() {
    elsex::StateGraph<State> graph;
    const idx first = graph.insert(State{2});
    const idx second = graph.insert(State{5});

    check::that(first != second, "distinct states get distinct identities");
    check::that(graph.insert(State{2}) == first, "re-inserting returns the original identity");
    check::that(graph.find(State{5}) == second, "lookup finds an inserted state");
    check::that(graph.find(State{9}) == elsex::StateGraph<State>::invalid(),
                "lookup reports an unknown state");
    check::that(graph[first] == State{2}, "identities index back to their state");
    check::that(graph.size() == 2, "size counts distinct states");
    check::done("state identities persist across re-entry");
}

int main() {
    std::printf("elsex ensemble\n");
    test_pure_birth_macrostep_is_analytic();
    test_shared_subnetwork_matches_per_entrance_solves();
    test_trajectories_are_well_formed();
    test_ensemble_is_reproducible();
    test_state_graph_identities_persist();
    return check::report("elsex ensemble");
}
