// A simulation loop of one's own on the lattice system: the mean absorption
// time from the origin, checked against q = Z 1 on the whole interior.
#include "common/lattice.hpp"
#include <iostream>

using namespace subsweep;
using namespace subsweep::examples;

// One walker from the origin to absorption. The subnetwork is the states
// visited so far, shed back to capacity by the score: no expansion step.
real absorption_time(lattice_system &sys, const sweep_options &options) {
    subnetwork sn(options);
    grown_state g = sys.place({0, 0});
    idx at = sn.add(g.total_rate, g.row, g.column, g.level);
    sweep_solution solution;
    real time = 0.0;
    while (sn.total_rate(at) > 0.0) {
        if (sn.size() > options.capacity) {
            num::vec rho(sn.size(), 0.0);
            rho[at] = 1.0;
            array<idx> shed = lowest_scores(sn.scores(rho), array<idx>{at}, sn.size() - options.capacity, sn.levels());
            std::sort(shed.rbegin(), shed.rend());
            for (idx j : shed) {
                sys.discard(j), sn.discard(j);
                at = at == sn.size() ? j : at;
            }
        }
        sn.solve(array<idx>{at}, array<idx>{1}, solution);
        const idx j = num::sample_categorical(solution.exit_law(0).span(), sys.random);
        time += solution.duration(0, j);
        g = sys.grow(0, j, sn.size());
        at = g.fresh ? sn.add(g.total_rate, g.row, g.column, g.level) : g.slot;
    }
    return time;
}
int main() {
    constexpr int radius = 10, walkers = 1000;
    const sweep_options options{.capacity = 40, .regime = solve_regime::block};
    real mean = 0.0, square = 0.0;
    for (int w = 0; w < walkers; ++w) {
        lattice_system sys{radius, 0.4, num::rng(7 + w)};
        const real time = absorption_time(sys, options);
        mean += time / walkers, square += time * time / walkers;
    }
    // Exact mean: the whole interior as one subnetwork, q = Z 1 at the origin.
    lattice_system all{radius, 0.4};
    subnetwork interior(options);
    for (int x = -radius; x < radius; ++x)
        for (int y = -radius; y < radius; ++y)
            if (norm({x, y}) < radius) {
                const grown_state g = all.place({x, y});
                interior.add(g.total_rate, g.row, g.column, g.level);
            }
    const num::vec q = interior.factor().solve(num::vec(interior.size(), 1.0));
    std::cout << "mean absorption time: " << mean << " +- " << std::sqrt((square - mean * mean) / walkers)
              << " (subsweep), " << q[all.slot_of.at({0, 0})] << " (exact)\n";
}
