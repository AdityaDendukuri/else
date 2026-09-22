// Ordered subsweep walkers on the Laplacian chain against SSA: committor
// expectations from each basin on a logarithmic time grid.
#include "common/laplacian.hpp"
#include <iostream>

using namespace subsweep;
using namespace subsweep::examples;

int main() {
    const laplacian_problem p = load_laplacian("laplacians/cs-medium.json");
    constexpr idx samples = 1000, reference_samples = 5000, capacity = 60;
    const num::vec grid = num::logspace(-12.0, -5.0, 29);
    const array<real> times(grid.begin(), grid.end());

    std::array<curve_summary, committor_count> ordered, ssa;
#pragma omp parallel for schedule(dynamic)
    for (idx basin = 0; basin < committor_count; ++basin) {
        ordered[basin] = ordered_committor_means(p, p.starts[basin], times, samples,
                                                 20000 + static_cast<unsigned>(basin), capacity);
        ssa[basin] = ssa_committor_means(p, p.starts[basin], times, reference_samples,
                                         30000 + static_cast<unsigned>(basin));
    }

    num::plt::subplot(1, committor_count);
    for (idx basin = 0; basin < committor_count; ++basin) {
        plot_curves(times, ordered[basin].mean, "lines lw 2", true);
        plot_curves(times, ssa[basin].mean, "lines dt 2 lw 1.2", false);
        finish_panel(basin);
        std::cout << "basin " << basin + 1 << " maximum discrepancy = "
                  << maximum_discrepancy(ordered[basin].mean, ssa[basin].mean) << '\n';
    }
    num::plt::savefig("laplacian_trajectories.png");
}
