/// Compare refactorization with Woodbury reuse on the Oregonator.
#include "else/algorithms/ensemble.hpp"
#include "markovkit/reaction_system.hpp"
#include <chrono>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr num::idx simulation_steps = 100000;
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

template <typename Run>
double milliseconds(Run &&run) {
    const auto start = std::chrono::high_resolution_clock::now();
    run();
    const auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

} // namespace

int main(int argc, char **argv) {
    const num::idx paths = argc > 1 ? std::stoul(argv[1]) : 20;

    const double y1 = 500.0, y2 = 1000.0, y3 = 2000.0, mu1 = 2000.0, mu2 = 50000.0;
    const num::array<double> rates = {
        mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1, 2.0 * mu1 / (y1 * y1), (mu1 + mu2) / y3,
    };
    const auto model = oregonator();
    const markovkit::State initial{500, 1000, 2000};
    const auto block_level = [](const markovkit::State &x) { return (x[0] + x[1] + x[2]) / 2; };

    std::printf("Oregonator ELSE, capacity %zu, %zu steps, %zu trajectories\n\n",
                subnetwork_capacity, simulation_steps, paths);

    num::idx recorded_steps = 0;
    const double block_reuse = milliseconds([&] {
        const auto paths_out = else_sim::else_ensemble(
            model, rates, initial, paths, 0.0, final_time,
            {.capacity = subnetwork_capacity, .maximum_steps = simulation_steps}, 42, block_level);
        recorded_steps = paths_out.front().times.size();
    });

    const double block_scratch = milliseconds([&] {
        (void)else_sim::else_ensemble(model, rates, initial, paths, 0.0, final_time,
                                      {.capacity = subnetwork_capacity,
                                       .maximum_steps = simulation_steps,
                                       .reuse_factorization = false},
                                      42, block_level);
    });

    const double dense_reuse = milliseconds([&] {
        (void)else_sim::else_ensemble(
            model, rates, initial, paths, 0.0, final_time,
            {.capacity = subnetwork_capacity, .maximum_steps = simulation_steps}, 42);
    });

    const double dense_scratch = milliseconds([&] {
        (void)else_sim::else_ensemble(model, rates, initial, paths, 0.0, final_time,
                                      {.capacity = subnetwork_capacity,
                                       .maximum_steps = simulation_steps,
                                       .reuse_factorization = false},
                                      42);
    });

    std::printf("  block-tridiagonal + Woodbury reuse    %9.1f ms\n", block_reuse);
    std::printf("  block-tridiagonal, refactor each step %9.1f ms\n", block_scratch);
    std::printf("  dense LU + Woodbury reuse             %9.1f ms\n", dense_reuse);
    std::printf("  dense LU, refactor each step          %9.1f ms\n", dense_scratch);
    std::printf("\n  macrosteps recorded: %zu\n", recorded_steps);
    return 0;
}
