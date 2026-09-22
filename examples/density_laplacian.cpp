// Unordered subsweep density on the Laplacian chain: the reconstructed
// committor expectations from each basin against SSA and ordered walkers.
#include "common/laplacian.hpp"
#include <fstream>
#include <iostream>
#include <string>

using namespace subsweep;
using namespace subsweep::examples;

int main(int argc, char **argv) {
    const laplacian_problem p = load_laplacian("laplacians/cs-medium.json");
    const int subnetworks = argc > 1 ? std::stoi(argv[1]) : 64;
    const idx capacity = argc > 2 ? std::stoul(argv[2]) : 4000;
    const idx ordered_samples = argc > 3 ? std::stoul(argv[3]) : 1000;
    constexpr idx reference_samples = 5000, ordered_capacity = 60;
    const num::vec grid = num::logspace(-12.0, -5.0, 29);
    const array<real> times(grid.begin(), grid.end());

    std::array<curves, committor_count> unordered;
    std::array<curve_summary, committor_count> ordered, ssa;
#pragma omp parallel for schedule(dynamic)
    for (idx basin = 0; basin < committor_count; ++basin) {
        unordered[basin] = unordered_committor_means(p, p.starts[basin], times, subnetworks,
                                                     capacity, 100 + static_cast<unsigned>(basin));
        ssa[basin] = ssa_committor_means(p, p.starts[basin], times, reference_samples,
                                         10000 + static_cast<unsigned>(basin));
        ordered[basin] =
            ordered_committor_means(p, p.starts[basin], times, ordered_samples,
                                    20000 + static_cast<unsigned>(basin), ordered_capacity);
    }

    num::plt::subplot(1, committor_count);
    real ordered_worst = 0.0, unordered_worst = 0.0;
    for (idx basin = 0; basin < committor_count; ++basin) {
        plot_curves(times, ssa[basin].mean, "lines dt 2 lw 0.8", false);
        plot_curves(times, unordered[basin], "lines lw 2", true);
        plot_curves(times, ordered[basin].mean, "points pt 13 ps 1.35", false);
        finish_panel(basin);
        ordered_worst = std::max(ordered_worst, maximum_discrepancy(ordered[basin].mean, ssa[basin].mean));
        unordered_worst = std::max(unordered_worst, maximum_discrepancy(unordered[basin], ssa[basin].mean));
    }
    num::plt::savefig("committor_comparison.png");

    std::ofstream summary("laplacian_accuracy.csv");
    summary << "comparison,samples,max_abs_difference\n"
            << "ordered," << ordered_samples << ',' << ordered_worst << '\n'
            << "unordered,1," << unordered_worst << '\n';
    std::cout << "ordered: max = " << ordered_worst << ", unordered: max = " << unordered_worst
              << '\n';
}
