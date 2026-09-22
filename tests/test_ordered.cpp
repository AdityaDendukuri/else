// The ordered loop on a row system: the closed-form pure-birth sweep,
// well-formed reproducible walkers, agreement across regimes, and reuse
// against fresh factors.
#include "check.hpp"
#include "common/rate_matrix.hpp"
#include <cstdio>
#include <functional>

namespace {

using namespace subsweep;
using namespace subsweep::examples;
using State = num::multi_index;
using Rates = array<real>;

struct System {
    array<array<int>> changes;
    array<std::function<real(const State &, const Rates &, real)>> propensities;
};

System birth_death() {
    System model;
    model.changes = {{1}, {-1}};
    model.propensities = {
        [](const State &, const Rates &r, real) { return r[0]; },
        [](const State &x, const Rates &r, real) { return r[1] * static_cast<real>(x[0]); },
    };
    return model;
}

const auto copy_number_level = [](const State &state) { return static_cast<idx>(state[0]); };

void expect_same_walkers(const array<trajectory<State>> &a, const array<trajectory<State>> &b,
                         const char *what) {
    check::that(a.size() == b.size(), std::string(what) + ": walker count");
    for (idx t = 0; t < a.size(); ++t) {
        check::that(a[t].states == b[t].states, std::string(what) + ": states");
        check::that(a[t].times.size() == b[t].times.size(), std::string(what) + ": sweep count");
        for (idx step = 0; step < std::min(a[t].times.size(), b[t].times.size()); ++step)
            check::close(a[t].times[step], b[t].times[step], std::string(what) + ": times", 1e-10);
    }
}

} // namespace

void test_pure_birth_sweep_is_analytic() {
    // Depth-1 expansion around one state gives the window {x, x+1}; pure
    // birth traverses both before exiting, so the mean sweep is 2/lambda.
    const real lambda = 2.5;
    System model;
    model.changes = {{1}};
    model.propensities = {[](const State &, const Rates &r, real) { return r[0]; }};
    const Rates rates{lambda};
    const auto c = reaction_rate_matrix(model, rates);
    sweep_options options;
    options.capacity = 32;
    options.maximum_sweeps = 4;
    auto s = rows_system(c, State{0}, 1, 7u);
    const auto path = ordered_paths(s, options, 0.0, 1e9, 7u).front();
    check::that(path.times.size() >= 5, "four sweeps were recorded");
    for (idx step = 1; step + 1 < path.times.size(); ++step) {
        check::close(path.times[step] - path.times[step - 1], 2.0 / lambda, "sweep is 2/lambda",
                     1e-12);
        check::that(path.states[step][0] == path.states[step - 1][0] + 2,
                    "each sweep advances two states");
    }
    check::done("the pure-birth sweep matches its closed form");
}

void test_walkers_are_well_formed_and_reproducible() {
    const Rates rates{1.4, 0.3};
    const auto model = birth_death();
    const auto c = reaction_rate_matrix(model, rates);
    sweep_options options;
    options.capacity = 12; // small enough that shedding runs
    auto s = rows_system(c, State{4}, 24, 2026u);
    const auto walkers = ordered_paths(s, options, 0.0, 6.0, 2026u);
    check::that(walkers.size() == 24, "one trajectory per walker");
    for (const auto &path : walkers) {
        check::that(path.times.size() == path.states.size() && !path.times.empty(),
                    "each recorded time has a state");
        check::close(path.times.front(), 0.0, "starts at the initial time");
        check::that(path.states.front() == State{4}, "starts at the initial state");
        check::close(path.times.back(), 6.0, "ends at the final time");
        for (idx step = 1; step < path.times.size(); ++step)
            check::that(path.times[step] >= path.times[step - 1] && path.states[step][0] >= 0,
                        "times are nondecreasing and copy numbers nonnegative");
    }
    auto s2 = rows_system(c, State{4}, 24, 2026u);
    expect_same_walkers(walkers, ordered_paths(s2, options, 0.0, 6.0, 2026u), "same seed");
    check::done("walkers are well formed under shedding and reproducible at a fixed seed");
}

void test_regimes_sample_the_same_walkers() {
    const Rates rates{1.4, 0.3};
    const auto model = birth_death();
    const auto c = reaction_rate_matrix(model, rates);
    const auto run = [&](solve_regime regime) {
        sweep_options options;
        options.capacity = 16;
        options.regime = regime;
        options.reuse_factors = false;
        auto s = rows_system(c, State{6}, 12, 5u, copy_number_level);
        return ordered_paths(s, options, 0.0, 8.0, 5u);
    };
    const auto dense = run(solve_regime::dense);
    expect_same_walkers(dense, run(solve_regime::block), "dense vs block");
    expect_same_walkers(dense, run(solve_regime::sparse), "dense vs sparse");
    check::done("the three regimes sample the same walkers from the same seed");
}

void test_factor_reuse_matches_fresh_factors() {
    // The block suffix update is exact, so its walkers agree with fresh
    // factors under expected visits. Woodbury rows differ at round-off,
    // which reorders ties among states far from rho, so the dense run and
    // the cut-time rule are checked for acceptance only.
    const Rates rates{1.4, 0.3};
    const auto model = birth_death();
    const auto c = reaction_rate_matrix(model, rates);
    const auto run = [&](const sweep_options &options, run_diagnostics *diagnostics = nullptr) {
        auto s = rows_system(c, State{20}, 96, 2027u, copy_number_level);
        return ordered_paths(s, options, 0.0, 50.0, 2027u, diagnostics);
    };
    for (const shedding_rule rule : {shedding_rule::expected_visits, shedding_rule::cut_time}) {
        for (const solve_regime regime : {solve_regime::block, solve_regime::dense}) {
            sweep_options reused;
            reused.capacity = 9;
            reused.rule = rule;
            reused.regime = regime;
            reused.maximum_sweeps = 40;
            auto fresh = reused;
            fresh.reuse_factors = false;
            run_diagnostics diagnostics;
            const auto walkers = run(reused, &diagnostics);
            if (regime == solve_regime::block && rule == shedding_rule::expected_visits)
                expect_same_walkers(walkers, run(fresh), "block suffix reuse");
            idx accepted = 0;
            for (const auto &sweep : diagnostics.sweeps)
                accepted += sweep.reuse_accepted ? 1 : 0;
            check::that(accepted > 0, "the run reused at least one factor");
            for (const auto &path : walkers)
                check::that(path.times.back() <= 50.0, "walkers end by the final time");
        }
    }
    check::done("factor reuse reproduces fresh factorizations in both regimes");
}

int main() {
    std::printf("subsweep ordered\n");
    test_pure_birth_sweep_is_analytic();
    test_walkers_are_well_formed_and_reproducible();
    test_regimes_sample_the_same_walkers();
    test_factor_reuse_matches_fresh_factors();
    return check::report("subsweep ordered");
}
