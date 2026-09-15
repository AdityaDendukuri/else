#include "else/algorithms/ensemble.hpp"
#include "markovkit/reaction_system.hpp"
#include <chrono>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
    const num::idx steps = argc > 1 ? std::stoul(argv[1]) : 5000;
    const num::idx paths = argc > 2 ? std::stoul(argv[2]) : 1;
    const num::idx capacity = argc > 3 ? std::stoul(argv[3]) : 600;
    const bool dense = argc > 4 && std::string_view(argv[4]) == "dense";
    const num::idx maximum_reuse_slots = argc > 5 ? std::stoul(argv[5]) : 3;

    const double y1 = 500.0, y2 = 1000.0, y3 = 2000.0;
    const double mu1 = 2000.0, mu2 = 50000.0;
    const num::array<double> rates = {mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1,
                                      2.0 * mu1 / (y1 * y1), (mu1 + mu2) / y3};
    markovkit::ReactionSystem model{
        .changes = {{1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-2, 0, 0}, {0, 1, -1}},
        .propensities = {
            [](const auto &x, const auto &r, double) { return r[0] * x[1]; },
            [](const auto &x, const auto &r, double) { return r[1] * x[0] * x[1]; },
            [](const auto &x, const auto &r, double) { return r[2] * x[0]; },
            [](const auto &x, const auto &r, double) {
                return x[0] < 2 ? 0.0 : 0.5 * r[3] * x[0] * (x[0] - 1);
            },
            [](const auto &x, const auto &r, double) { return r[4] * x[2]; },
        }};
    const auto level = [dense](const markovkit::State &x) {
        return dense ? 0 : (x[0] + x[1] + x[2]) / 2;
    };

    for (bool reuse : {false, true}) {
        else_sim::EnsembleOptions options{.capacity = capacity,
                                          .expansion_depth = 1,
                                          .tolerance = 1e-12,
                                          .maximum_steps = steps,
                                          .reuse_factorization = reuse,
                                          .maximum_reuse_slots = maximum_reuse_slots};
        const auto start = std::chrono::steady_clock::now();
        else_sim::EnsembleDiagnostics diagnostics;
        const auto trajectories =
            else_sim::else_ensemble(model, rates, markovkit::State{500, 1000, 2000}, paths, 0.0,
                                    std::numeric_limits<double>::infinity(), options, 42, level,
                                    &diagnostics);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        double mean_time = 0.0;
        for (const auto &trajectory : trajectories)
            mean_time += trajectory.times.back() / trajectories.size();
        num::idx attempted = 0, accepted = 0, factored = 0, changed = 0;
        double block_fraction = 0.0, state_fraction = 0.0;
        double selection = 0.0, restriction_factor = 0.0, factor_update = 0.0, law = 0.0,
               score = 0.0, escape_setup = 0.0, sampling = 0.0;
        for (const auto &step : diagnostics.steps) {
            attempted += step.reuse_attempted ? 1 : 0;
            accepted += step.reuse_accepted ? 1 : 0;
            factored += step.fresh_factorization ? 1 : 0;
            if (step.reuse_accepted) {
                changed += step.changed_state_slots;
                block_fraction += static_cast<double>(step.reused_prefix_blocks) /
                                  std::max<num::idx>(step.block_count, 1);
                state_fraction += static_cast<double>(step.reused_prefix_states) /
                                  std::max<num::idx>(step.subnetwork_size, 1);
            }
            selection += step.selection_seconds;
            restriction_factor += step.restriction_and_factor_seconds;
            factor_update += step.factor_update_seconds;
            law += step.entrance_law_seconds;
            score += step.score_seconds;
            escape_setup += step.escape_setup_seconds;
            sampling += step.sampling_seconds;
        }
        std::cout << (reuse ? "reuse" : "scratch") << " wall=" << seconds
                  << " mean_time=" << mean_time
                  << " first_steps=" << trajectories.front().times.size() - 1
                  << " attempted=" << attempted << " accepted=" << accepted
                  << " factored=" << factored;
        if (accepted > 0)
            std::cout << " mean_changed=" << static_cast<double>(changed) / accepted
                      << " mean_block_prefix=" << block_fraction / accepted
                      << " mean_state_prefix=" << state_fraction / accepted;
        std::cout << " phases(selection=" << selection
                  << ",restriction_factor=" << restriction_factor
                  << ",factor_update=" << factor_update << ",law=" << law
                  << ",score=" << score << ",escape_setup=" << escape_setup
                  << ",sampling=" << sampling << ")\n";
    }
}
