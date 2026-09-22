// The subnetwork: grow and discard keep R_bar consistent, the first-exit
// quantities match their definitions, and reuse across sweeps reproduces
// fresh solves.
#include "check.hpp"
#include "fixtures.hpp"
#include "container/matrix_expr.hpp"
#include <cstdio>

namespace {

using namespace subsweep;
using fixtures::with_regime;
constexpr idx order = 24;

// The dense inverse Z = R_bar^-1 by basis solves.
num::mat inverse(const subnetwork &sn) {
    const factorization f = sn.factor();
    return f.solve(num::identity(sn.size()));
}

} // namespace

void test_discard_moves_the_last_slot() {
    subnetwork sn = fixtures::birth_death(6, {});
    const num::mat before = num::dense(sn.matrix());
    sn.discard(2);
    check::that(sn.size() == 5, "one slot fewer");
    const num::mat after = num::dense(sn.matrix());
    const array<idx> old_of{0, 1, 5, 3, 4};
    for (idx i = 0; i < 5; ++i)
        for (idx j = 0; j < 5; ++j)
            check::close(after(i, j), before(old_of[i], old_of[j]), "R_bar after discard");
    check::that(sn.levels()[2] == 5, "the level moved with the slot");
    check::done("discard moves the last slot into the hole");
}

void test_exit_law_and_duration_match_their_definitions() {
    subnetwork sn = fixtures::birth_death(order, {});
    const num::mat z = inverse(sn);
    const num::vec w = sn.exit_rates();
    const array<idx> current{0, 3, 4, 8}, counts{1, 2, 1, 3};
    const sweep_solution solution = sn.solve(current, counts);
    for (idx k = 0; k < current.size(); ++k) {
        const num::vec law = solution.exit_law(k);
        real total = 0.0;
        for (idx j = 0; j < order; ++j) {
            check::close(solution.occupation(k, j), z(current[k], j), "U = EZ", 1e-11);
            real z2 = 0.0;
            for (idx l = 0; l < order; ++l)
                z2 += z(current[k], l) * z(l, j);
            check::close(solution.second_occupation(k, j), z2, "V = EZ^2", 1e-10);
            check::close(law[j], w[j] * z(current[k], j), "beta_k(j) = w_j U(k,j)", 1e-11);
            total += law[j];
        }
        check::close(total, 1.0, "the exit law is normalized", 1e-11);
        for (idx j : {idx(0), idx(order - 1)})
            check::close(solution.duration(k, j),
                         solution.second_occupation(k, j) / solution.occupation(k, j),
                         "duration is V/U", 1e-12);
        const sweep_solution alone = sn.solve(array<idx>{current[k]}, array<idx>{1});
        for (idx j = 0; j < order; ++j)
            check::close(alone.occupation(0, j), solution.occupation(k, j), "grouped rows", 1e-12);
    }
    check::close(solution.rho[3], 2.0 / 7.0, "rho is the walker fraction", 1e-15);
    check::done("exit law, duration, and grouped rows match their definitions");
}

void test_scores_match_their_definitions() {
    using namespace num::ops;
    for (const solve_regime regime : {solve_regime::dense, solve_regime::block}) {
        subnetwork sn = fixtures::birth_death(order, with_regime(regime, shedding_rule::cut_time));
        const num::mat z = inverse(sn);
        num::vec rho(order, 0.0);
        rho[5] = 0.25;
        rho[9] = 0.75;
        const num::vec u = num::transpose(z) * rho;
        const num::vec q = z * num::vec(order, 1.0);
        const num::vec scores = sn.scores(rho);
        for (idx j = 0; j < order; ++j)
            check::close(scores[j], u[j] * q[j] / z(j, j), "cut-time loss u_j q_j / Z_jj", 1e-10);
        // Exactness of the loss: removing j changes the mean exit time by the score.
        const idx j = 14;
        subnetwork without = fixtures::birth_death(order, with_regime(regime));
        without.discard(j);
        num::vec rho_without(order - 1, 0.0);
        for (idx i = 0; i < order - 1; ++i)
            rho_without[i] = rho[i];
        rho_without[j] = rho[order - 1];
        const num::mat z_without = inverse(without);
        const num::vec q_without = z_without * num::vec(order - 1, 1.0);
        const num::vec u_without = num::transpose(z_without) * rho_without;
        real before = 0.0, after = 0.0;
        for (idx i = 0; i < order; ++i)
            before += rho[i] * q[i];
        for (idx i = 0; i < order - 1; ++i)
            after += rho_without[i] * q_without[i];
        check::close(before - after, scores[j], "the cut-time loss is exact", 1e-9);

        subnetwork visits = fixtures::birth_death(order, with_regime(regime));
        const num::vec expected = visits.scores(rho);
        for (idx i = 0; i < order; ++i)
            check::close(expected[i], visits.total_rate(i) * u[i] - rho[i],
                         "expected visits R_jj u_j - rho_j", 1e-10);
    }
    check::done("both shedding scores match their definitions");
}

void test_lowest_scores_break_ties_toward_the_highest_level() {
    const array<real> scores{0.0, 0.0, 0.5, 0.0, 0.2};
    const array<idx> levels{3, 1, 0, 3, 2};
    const array<idx> chosen = lowest_scores(scores, array<idx>{0}, 2, levels);
    check::that(chosen.size() == 2 && chosen[0] == 3 && chosen[1] == 1,
                "highest level first among ties, protected slots excluded");
    const array<idx> plain = lowest_scores(scores, {}, 3);
    check::that(plain[0] == 0 && plain[1] == 1 && plain[2] == 3, "index order without levels");
    check::done("lowest scores select by score, then level, then index");
}

// Replace the state at level `level` by one with different rates to the
// same neighbors, as a sweep's discard and grow would.
void replace(subnetwork &sn, idx level) {
    const auto slot_of_level = [&](idx wanted) {
        for (idx j = 0; j < sn.size(); ++j)
            if (sn.levels()[j] == wanted)
                return j;
        return sn.size();
    };
    sn.discard(slot_of_level(level));
    array<slot_rate> row, column;
    real total = 0.0;
    for (const idx neighbor : {level - 1, level + 1}) {
        const idx slot = slot_of_level(neighbor);
        if (slot == sn.size())
            continue;
        num::append(row, slot, 0.9 + 0.1 * static_cast<real>(neighbor));
        num::append(column, slot, 1.3);
        total += 0.9 + 0.1 * static_cast<real>(neighbor);
    }
    sn.add(total + 0.2, row, column, level);
}

void test_reuse_reproduces_fresh_solves() {
    for (const solve_regime regime : {solve_regime::dense, solve_regime::block}) {
        for (const shedding_rule rule : {shedding_rule::expected_visits, shedding_rule::cut_time}) {
            sweep_options options = with_regime(regime, rule);
            options.capacity = order;
            options.woodbury_cutoff = 4;
            subnetwork reused = fixtures::birth_death(order, options);
            options.reuse_factors = false;
            subnetwork fresh = fixtures::birth_death(order, options);
            const array<idx> current{2, 9, 12, 20, 22}, counts{1, 1, 1, 2, 1};
            (void)reused.solve(current, counts);
            for (subnetwork *sn : {&reused, &fresh}) {
                replace(*sn, 15);
                replace(*sn, 6);
            }
            const sweep_solution a = reused.solve(current, counts), b = fresh.solve(current, counts);
            check::that(a.diagnostics.reuse_attempted, "reuse was attempted");
            check::that(a.diagnostics.reuse_accepted, "reuse was accepted");
            check::that(a.diagnostics.changed_state_slots == 3, "three slots changed");
            if (regime == solve_regime::block)
                check::that(a.diagnostics.reused_prefix_states == 6, "levels before 6 are reused");
            for (idx k = 0; k < current.size(); ++k)
                for (idx j = 0; j < order; ++j) {
                    check::close(a.occupation(k, j), b.occupation(k, j), "reused U", 1e-10);
                    check::close(a.second_occupation(k, j), b.second_occupation(k, j),
                                 "reused V", 1e-9);
                }
            const num::vec sa = reused.scores(a), sb = fresh.scores(b);
            for (idx j = 0; j < order; ++j)
                check::close(sa[j], sb[j], "reused scores", 1e-9);
            // One more replacement: the block regime updates the cached rows
            // by the Woodbury identity rather than solving them again.
            for (subnetwork *sn : {&reused, &fresh})
                replace(*sn, 17);
            const sweep_solution c = reused.solve(current, counts), d = fresh.solve(current, counts);
            check::that(c.diagnostics.reuse_accepted, "reuse was accepted again");
            if (regime == solve_regime::block)
                check::that(c.diagnostics.reused_current_rows == 5, "cached rows were updated");
            for (idx k = 0; k < current.size(); ++k)
                for (idx j = 0; j < order; ++j) {
                    check::close(c.occupation(k, j), d.occupation(k, j), "cached U", 1e-10);
                    check::close(c.second_occupation(k, j), d.second_occupation(k, j), "cached V",
                                 1e-9);
                }
            const sweep_solution again = reused.solve(array<idx>{2, 20}, array<idx>{1, 1});
            check::that(again.diagnostics.reuse_accepted, "an unchanged subnetwork reuses");
            if (regime == solve_regime::block)
                check::that(again.diagnostics.changed_state_slots == 0, "the base is current");
        }
    }
    check::done("dense Woodbury and block suffix reuse reproduce fresh solves");
}

int main() {
    std::printf("subsweep subnetwork\n");
    test_discard_moves_the_last_slot();
    test_exit_law_and_duration_match_their_definitions();
    test_scores_match_their_definitions();
    test_lowest_scores_break_ties_toward_the_highest_level();
    test_reuse_reproduces_fresh_solves();
    return check::report("subsweep subnetwork");
}
