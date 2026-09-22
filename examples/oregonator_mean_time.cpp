// Ordered subsweep versus SSA on the Oregonator: ensemble means of the
// three species to the horizon, with 95% bands.
#include "common/models.hpp"
#include "common/rate_matrix.hpp"
#include "common/summary.hpp"
#include "ssa/ssa.hpp"
#include <fstream>
#include <iostream>

using namespace subsweep;
using namespace subsweep::examples;

int main() {
    constexpr idx paths = 100, ssa_paths = 200, samples = 601;
    constexpr real final_time = 1.5;
    const reaction_model model = oregonator();
    sweep_options options;
    options.capacity = 120;
    options.maximum_sweeps = 600000;
    options.regime = solve_regime::block;

    run_diagnostics diagnostics;
    auto system = rows_system(reaction_rate_matrix(model.system, model.rates), model.initial,
                              paths, 42, oregonator_level);
    const auto walkers = ordered_paths(system, options, 0.0, final_time, 42, &diagnostics);
    array<markovkit::trajectory> ssa(ssa_paths);
#pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < static_cast<int>(ssa_paths); ++p)
        ssa[p] = ssa::gillespie(model.system, model.rates, model.initial, 0.0, final_time, 42 + p,
                                options.maximum_sweeps);

    idx sweep_exits = 0, ssa_events = 0, reused = 0;
    for (const auto &path : walkers)
        sweep_exits += path.states.size() - 1;
    for (const auto &path : ssa)
        ssa_events += path.reactions.size();
    for (const auto &sweep : diagnostics.sweeps)
        reused += sweep.reuse_accepted ? 1 : 0;

    array<real> times(samples);
    for (idx k = 0; k < samples; ++k)
        times[k] = final_time * static_cast<real>(k) / static_cast<real>(samples - 1);
    const char *labels[] = {"X", "Y", "Z"};
    std::ofstream accuracy("oregonator_accuracy.csv");
    accuracy << "species,subsweep_vs_ssa_relative_rmse,ssa_split_relative_rmse,standardized_rms\n";
    num::plt::subplot(1, 3);
    for (idx component = 0; component < 3; ++component) {
        const summary sweep = summarize(walkers, times, component);
        const summary reference = summarize(ssa, times, component);
        accuracy << labels[component] << ',' << relative_rmse(sweep.mean, reference.mean) << ','
                 << relative_rmse(summarize(ssa, times, component, 0, paths).mean,
                                  summarize(ssa, times, component, paths, 2 * paths).mean)
                 << ',' << standardized_rms(sweep, reference) << '\n';
        band(times, sweep, "#1f77b4");
        band(times, reference, "#ff7f0e");
        num::plt::plot(times, sweep.mean, "Ordered subsweep mean", "lines lw 2 lc rgb '#1f77b4'");
        num::plt::plot(times, reference.mean, "SSA mean", "lines lw 2 lc rgb '#ff7f0e'");
        num::plt::title(labels[component]);
        num::plt::xlabel("physical time");
        num::plt::ylabel("molecule count");
        num::plt::xlim(0.0, final_time);
        num::plt::legend();
        if (component + 1 < 3)
            num::plt::next();
    }
    num::plt::savefig("oregonator_mean_time.png");
    std::cout << "Oregonator: sweeps=" << diagnostics.sweeps.size() << ", reused=" << reused
              << ", subsweep exits=" << sweep_exits << ", SSA reaction events=" << ssa_events
              << ", event reduction=" << real(ssa_events) / real(sweep_exits) << "x\n";
}
