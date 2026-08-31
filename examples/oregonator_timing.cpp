/// Time the ELSE portion of the Oregonator on both trees, no plotting.
///
/// The legacy call passes a block level and reuses factorizations across
/// macrosteps; the new tree has neither yet, so this is the honest gap that the
/// block-tridiagonal port and the Woodbury wiring have to close.
#include "else/trajectory.hpp"
#include "elsex/ensemble.hpp"
#include "markovkit.hpp"
#include <chrono>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t simulation_steps = 100000;
constexpr num::idx subnetwork_capacity = 60;
constexpr double final_time = std::numeric_limits<double>::infinity();

markovkit::ReactionSystem oregonator() {
    return markovkit::ReactionSystem{
        .changes = {{1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-2, 0, 0}, {0, 1, -1}},
        .propensities = {
            [](const markovkit::State &x, const auto &r, double) { return r[0] * x[1]; },
            [](const markovkit::State &x, const auto &r, double) { return r[1] * x[0] * x[1]; },
            [](const markovkit::State &x, const auto &r, double) { return r[2] * x[0]; },
            [](const markovkit::State &x, const auto &r, double) {
                return x[0] < 2 ? 0.0 : 0.5 * r[3] * x[0] * (x[0] - 1);
            },
            [](const markovkit::State &x, const auto &r, double) { return r[4] * x[2]; },
        }};
}

template <typename Run> double milliseconds(Run &&run) {
    const auto start = std::chrono::high_resolution_clock::now();
    run();
    const auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

} // namespace

int main(int argc, char **argv) {
    const std::size_t paths = argc > 1 ? std::stoul(argv[1]) : 20;

    const double y1 = 500.0, y2 = 1000.0, y3 = 2000.0, mu1 = 2000.0, mu2 = 50000.0;
    const std::vector<double> rates = {
        mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1, 2.0 * mu1 / (y1 * y1), (mu1 + mu2) / y3,
    };
    const auto model = oregonator();
    const markovkit::State initial{500, 1000, 2000};
    const auto block_level = [](const markovkit::State &x) { return (x[0] + x[1] + x[2]) / 2; };

    std::printf("Oregonator ELSE, capacity %zu, %zu steps, %zu trajectories\n\n",
                subnetwork_capacity, simulation_steps, paths);

    std::size_t legacy_steps = 0;
    const double legacy_block_reuse = milliseconds([&] {
        const auto paths_out = else_sim::else_ensemble(
            model, rates, initial, paths, 0.0, final_time,
            {.capacity = subnetwork_capacity, .maximum_steps = simulation_steps}, 42, block_level);
        legacy_steps = paths_out.front().times.size();
    });

    const double legacy_block_scratch = milliseconds([&] {
        (void)else_sim::else_ensemble(model, rates, initial, paths, 0.0, final_time,
                                      {.capacity = subnetwork_capacity,
                                       .maximum_steps = simulation_steps,
                                       .reuse_factorization = false},
                                      42, block_level);
    });

    const double legacy_dense_reuse = milliseconds([&] {
        (void)else_sim::else_ensemble(
            model, rates, initial, paths, 0.0, final_time,
            {.capacity = subnetwork_capacity, .maximum_steps = simulation_steps}, 42);
    });

    std::size_t new_steps = 0;
    const double sparse_scratch = milliseconds([&] {
        const auto paths_out = elsex::else_ensemble(
            model, rates, initial, paths, 0.0, final_time,
            {.capacity = subnetwork_capacity, .maximum_steps = simulation_steps}, 42);
        new_steps = paths_out.front().times.size();
    });

    std::printf("  legacy  block-tridiagonal + Woodbury reuse   %9.1f ms\n", legacy_block_reuse);
    std::printf("  legacy  block-tridiagonal, refactor each step %9.1f ms\n", legacy_block_scratch);
    std::printf("  legacy  dense LU + Woodbury reuse             %9.1f ms\n", legacy_dense_reuse);
    std::printf("  elsex   sparse LU, refactor each step         %9.1f ms\n", sparse_scratch);
    std::printf("\n  macrosteps recorded: legacy %zu, elsex %zu\n", legacy_steps, new_steps);
    return 0;
}
