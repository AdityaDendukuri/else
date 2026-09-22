// The genetic toggle switch: phase-plane paths of SSA and ordered subsweep
// at a high and a low barrier, and how many distinct current states an
// ensemble of shared walkers occupies.
#include "common/models.hpp"
#include "common/rate_matrix.hpp"
#include "common/summary.hpp"
#include "ssa/ssa.hpp"
#include <iostream>

using namespace subsweep;
using namespace subsweep::examples;

namespace {

constexpr real final_time = 500.0;
constexpr idx max_steps = 100000, shown = 15;

array<trajectory<state>> sweep_paths(const reaction_model &model, idx walkers, idx capacity,
                                     unsigned seed, real horizon = final_time,
                                     idx sweeps = max_steps) {
    sweep_options options;
    options.capacity = capacity;
    options.tolerance = 1e-12;
    options.maximum_sweeps = sweeps;
    options.regime = solve_regime::block;
    auto system = rows_system(reaction_rate_matrix(model.system, model.rates), model.initial,
                              walkers, seed, toggle_level);
    return ordered_paths(system, options, 0.0, horizon, seed);
}

array<markovkit::trajectory> ssa_paths(const reaction_model &model, unsigned first_seed,
                                       int trapped_below = -1) {
    array<markovkit::trajectory> paths;
    for (unsigned seed = first_seed; paths.size() < shown && seed < first_seed + 100; ++seed) {
        auto path = ssa::gillespie(model.system, model.rates, model.initial, 0.0, final_time,
                                   static_cast<int>(seed), max_steps);
        int max_v = 0;
        for (const auto &x : path.states)
            max_v = std::max(max_v, x[1]);
        if (trapped_below < 0 || max_v < trapped_below)
            num::append(paths, std::move(path));
    }
    return paths;
}

} // namespace

int main() {
    const reaction_model high = toggle_switch(0.20, {17, 1}), low = toggle_switch(0.08, {7, 1});
    num::plt::subplot(2, 2);
    phase_panel(ssa_paths(high, 1, 30), shown, "SSA trajectories", "#e67e22", high.initial, 76.0,
                83.0, 105.0, "High barrier (s=0.20): SSA");
    num::plt::next();
    phase_panel(sweep_paths(high, 50, 600, 8), shown, "Ordered subsweep trajectories", "#2980b9",
                high.initial, 76.0, 83.0, 105.0, "High barrier (s=0.20): ordered subsweep");
    num::plt::next();
    phase_panel(ssa_paths(low, 10), shown, "SSA trajectories", "#e67e22", low.initial, 30.5, 33.5,
                45.0, "Low barrier (s=0.08): SSA");
    num::plt::next();
    phase_panel(sweep_paths(low, shown, 250, 42), shown, "Ordered subsweep trajectories",
                "#2980b9", low.initial, 30.5, 33.5, 45.0, "Low barrier (s=0.08): ordered subsweep");
    num::plt::savefig("toggle_trajectories.png");

    // Distinct current states after 15 sweeps of m shared walkers.
    array<real> counts, fractions;
    for (idx m : {10, 20, 50, 100, 250, 500, 1000}) {
        const auto paths = sweep_paths(low, m, std::max<idx>(300, m + 100), 42, 15.0, 15);
        key_set<state> current;
        for (const auto &path : paths)
            current.insert(path.states.back());
        num::append(counts, static_cast<real>(m));
        num::append(fractions, std::min(1.0, real(current.size()) / real(m)));
        std::cout << "walkers " << m << ": distinct current / walkers = " << fractions.back()
                  << '\n';
    }
    num::plt::subplot(1, 1);
    num::plt::plot(counts, fractions, "Distinct current / walkers",
                   "linespoints pt 7 lw 2 lc rgb '#2980b9'");
    num::plt::xlabel("walkers (m)");
    num::plt::ylabel("distinct current states / walkers");
    num::plt::semilogx();
    num::plt::ylim(0.0, 1.05);
    num::plt::savefig("toggle_sharing.png");
}
