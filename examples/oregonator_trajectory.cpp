// One Oregonator walker with a loop of one's own in the block regime. The
// system below implements the concept of system.hpp by hand: it enumerates
// the reactions, keeps the states behind the slots, and answers grow,
// expand, and discard. The loop is the ordered algorithm for one walker.
#include "plot/plot.hpp"
#include "subsweep/subsweep.hpp"

using namespace subsweep;
using state = num::multi_index; // (X, Y, Z)

struct oregonator_system {
    // The model: five reactions, destination x + change[r] at rate propensity(x, r).
    real k1, k2, k3, k4, k5;
    static constexpr int change[5][3] = {{1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-2, 0, 0}, {0, 1, -1}};

    [[nodiscard]] real propensity(const state &x, int r) const {
        if (x[0] < 0 || x[1] < 0 || x[2] < 0)
            return 0.0;
        const real a[] = {k1 * x[1], k2 * x[0] * x[1], k3 * x[0],
                          x[0] < 2 ? 0.0 : 0.5 * k4 * x[0] * (x[0] - 1), k5 * x[2]};
        return a[r];
    }

    [[nodiscard]] static state successor(const state &x, int r) {
        return {x[0] + change[r][0], x[1] + change[r][1], x[2] + change[r][2]};
    }

    [[nodiscard]] static state predecessor(const state &x, int r) {
        return {x[0] - change[r][0], x[1] - change[r][1], x[2] - change[r][2]};
    }

    // The states behind the slots [0, n) of the subnetwork, both ways.
    array<state> states;
    table<state, idx> slot_of;
    num::rng random{42};

    // A state as the next slot: its total rate, its rates to the slots
    // present (its row), and their rates to it (its column, read from the
    // predecessors x - change[r]). A state already present is not fresh.
    grown_state place(const state &x) {
        grown_state g;
        if (const auto found = slot_of.find(x); found != slot_of.end()) {
            g.slot = found->second;
            return g;
        }
        g.slot = states.size();
        g.fresh = true;
        g.level = (x[0] + x[1] + x[2]) / 2; // block regime: half the total copy number
        for (int r = 0; r < 5; ++r) {
            const real rate = propensity(x, r);
            g.total_rate += rate;
            if (const auto found = slot_of.find(successor(x, r)); rate > 0.0 && found != slot_of.end())
                num::append(g.row, found->second, rate);
            if (const auto found = slot_of.find(predecessor(x, r)); found != slot_of.end())
                num::append(g.column, found->second, propensity(predecessor(x, r), r));
        }
        slot_of.emplace(x, g.slot);
        num::append(states, x);
        return g;
    }

    // grow: the walker leaves through slot `from`. Sample one of its
    // reactions whose product lies outside the slots [0, n), the subnetwork
    // the exit law was computed on, and place the product.
    grown_state grow(idx, idx from, idx n) {
        array<int> reactions;
        array<real> weights;
        for (int r = 0; r < 5; ++r) {
            const auto found = slot_of.find(successor(states[from], r));
            if (propensity(states[from], r) > 0.0 && (found == slot_of.end() || found->second >= n)) {
                num::append(reactions, r);
                num::append(weights, propensity(states[from], r));
            }
        }
        const int r = reactions[num::sample_categorical(view<const real>(weights), random)];
        return place(successor(states[from], r));
    }

    // expand: every product of `from` not yet in the subnetwork, placed.
    template <typename Visit> void expand(idx from, Visit &&visit) {
        const state x = states[from];
        for (int r = 0; r < 5; ++r)
            if (propensity(x, r) > 0.0 && !slot_of.contains(successor(x, r)))
                visit(place(successor(x, r)));
    }

    // discard: forget slot j; the last slot moves into it.
    void discard(idx j) {
        slot_of.erase(states[j]);
        if (j + 1 != states.size()) {
            states[j] = states.back();
            slot_of[states[j]] = j;
        }
        states.pop_back();
    }
};
static_assert(expandable_system<oregonator_system>);

int main() {
    constexpr real final_time = 1.5;
    const sweep_options options{
        .capacity = 120, .rule = shedding_rule::cut_time, .regime = solve_regime::block};
    oregonator_system sys{2.0, 0.1, 104.0, 0.016, 26.0};

    // The subnetwork starts as the initial state alone, in slot 0.
    subnetwork sn(options);
    grown_state g = sys.place({500, 1000, 2000});
    idx at = sn.add(g.total_rate, g.row, g.column, g.level);

    sweep_solution solution;
    array<real> times{0.0};
    array<state> path{sys.states[at]};
    while (times.back() < final_time) {
        // Solve for the current slot: the rows U = EZ and V = EZ^2.
        sn.solve(array<idx>{at}, array<idx>{1}, solution);

        // Sample the pre-exit slot j from beta(j) = w_j U(j) and the sweep's
        // duration V(j) / U(j).
        const idx j = num::sample_categorical(solution.exit_law(0).span(), sys.random);
        num::append(times, times.back() + solution.duration(0, j));

        // The system samples the product outside the subnetwork; the
        // subnetwork grows by it, and the walker is there now.
        g = sys.grow(0, j, sn.size());
        at = g.fresh ? sn.add(g.total_rate, g.row, g.column, g.level) : g.slot;
        num::append(path, sys.states[at]);

        // Shed the slot of lowest cut-time score when over capacity. The
        // scores are those of the sweep just solved; the new slot has none
        // and is kept, and the walker's slot number follows the swap.
        if (sn.size() > options.capacity) {
            const num::vec scores = sn.scores(solution);
            const idx lowest = lowest_scores(scores.span(), array<idx>{at}, 1, sn.levels()).front();
            sys.discard(lowest);
            sn.discard(lowest);
            if (at == sn.size())
                at = lowest;
        }
    }

    num::plt::subplot(3, 1);
    for (const int c : {0, 1, 2}) {
        array<real> counts;
        for (const state &x : path)
            num::append(counts, static_cast<real>(x[c]));
        num::plt::plot(times, counts, "", "steps lw 1.2");
        num::plt::ylabel(c == 0 ? "X" : c == 1 ? "Y" : "Z");
        num::plt::xlim(0.0, final_time);
        if (c < 2)
            num::plt::next();
    }
    num::plt::xlabel("time");
    num::plt::savefig("oregonator_trajectory.png");
}
