// Wall time of 1,000 ordered sweeps of one Oregonator walker in the block
// regime, recomputing every block factor versus refactoring from the first
// changed level, for each capacity in the paper. Medians of three runs.
#include "common/models.hpp"
#include "common/rate_matrix.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

using namespace subsweep;
using namespace subsweep::examples;
constexpr idx sweeps = 1000;
constexpr unsigned seed = 42;

struct run_result {
    double wall = 0.0;
    idx reuse_updates = 0, fresh_factorizations = 0;
    double mean_block_prefix = 0.0, mean_state_prefix = 0.0;
    double selection = 0.0, restriction_factor = 0.0, factor_update = 0.0, current_rows = 0.0,
           score = 0.0, exit_setup = 0.0, sampling = 0.0;
};

run_result run(const reaction_model &model, idx capacity, bool reuse) {
    sweep_options options;
    options.capacity = capacity;
    options.maximum_sweeps = sweeps;
    options.regime = solve_regime::block;
    options.reuse_factors = reuse;
    run_diagnostics diagnostics;
    const auto start = std::chrono::steady_clock::now();
    auto system = rows_system(reaction_rate_matrix(model.system, model.rates), model.initial, 1,
                              seed, oregonator_level);
    const auto walker = ordered_paths(system, options, 0.0, std::numeric_limits<real>::infinity(),
                                      seed, &diagnostics);
    run_result result;
    result.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (walker.front().times.size() != sweeps + 1)
        throw std::runtime_error("the walker did not complete every sweep");
    idx accepted_blocks = 0;
    for (const auto &sweep : diagnostics.sweeps) {
        result.reuse_updates += sweep.reuse_accepted ? 1 : 0;
        result.fresh_factorizations += sweep.fresh_factorization ? 1 : 0;
        if (sweep.reuse_accepted && sweep.block_count > 0) {
            ++accepted_blocks;
            result.mean_block_prefix += double(sweep.reused_prefix_blocks) / sweep.block_count;
            result.mean_state_prefix += double(sweep.reused_prefix_states) / sweep.subnetwork_size;
        }
        result.selection += sweep.selection_seconds;
        result.restriction_factor += sweep.restriction_and_factor_seconds;
        result.factor_update += sweep.factor_update_seconds;
        result.current_rows += sweep.current_rows_seconds;
        result.score += sweep.score_seconds;
        result.exit_setup += sweep.exit_setup_seconds;
        result.sampling += sweep.sampling_seconds;
    }
    if (accepted_blocks > 0) {
        result.mean_block_prefix /= accepted_blocks;
        result.mean_state_prefix /= accepted_blocks;
    }
    return result;
}

run_result median_run(const reaction_model &model, idx capacity, bool reuse) {
    array<run_result> results;
    for (int repetition = 0; repetition < 3; ++repetition)
        num::append(results, run(model, capacity, reuse));
    std::sort(results.begin(), results.end(),
              [](const run_result &a, const run_result &b) { return a.wall < b.wall; });
    return results[1];
}

} // namespace

int main(int argc, char **argv) {
    const reaction_model model = oregonator();
    std::ofstream csv(argc > 1 ? argv[1] : "computational_scaling.csv");
    csv << "capacity,scratch_wall,reuse_wall,reuse_updates,fresh_factorizations,"
           "mean_block_prefix,mean_state_prefix,"
           "scratch_selection,scratch_restriction_factor,scratch_factor_update,"
           "scratch_current_rows,scratch_score,scratch_exit_setup,scratch_sampling,"
           "reuse_selection,reuse_restriction_factor,reuse_factor_update,"
           "reuse_current_rows,reuse_score,reuse_exit_setup,reuse_sampling\n";
    std::cout << "capacity  scratch (s)  reuse (s)  updates  fresh  block prefix  state prefix\n";
    for (idx capacity : {60, 120, 240, 360, 480, 600}) {
        const run_result scratch = median_run(model, capacity, false);
        const run_result reuse = median_run(model, capacity, true);
        csv << capacity << ',' << std::setprecision(10) << scratch.wall << ',' << reuse.wall << ','
            << reuse.reuse_updates << ',' << reuse.fresh_factorizations << ','
            << reuse.mean_block_prefix << ',' << reuse.mean_state_prefix;
        for (const run_result *result : {&scratch, &reuse})
            csv << ',' << result->selection << ',' << result->restriction_factor << ','
                << result->factor_update << ',' << result->current_rows << ',' << result->score
                << ',' << result->exit_setup << ',' << result->sampling;
        csv << '\n';
        std::cout << std::setw(8) << capacity << std::fixed << std::setprecision(4) << std::setw(13)
                  << scratch.wall << std::setw(11) << reuse.wall << std::setw(9)
                  << reuse.reuse_updates << std::setw(7) << reuse.fresh_factorizations
                  << std::setw(14) << reuse.mean_block_prefix << std::setw(14)
                  << reuse.mean_state_prefix << '\n';
    }
}
