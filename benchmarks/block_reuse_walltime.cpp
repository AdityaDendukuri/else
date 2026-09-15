/// End-to-end wall time for the block-tridiagonal subsweep path. The median
/// suppresses one-off scheduler noise.
#include "else/algorithms/ensemble.hpp"
#include "markovkit/reaction_system.hpp"
#include "plot/plot.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

markovkit::ReactionSystem oregonator() {
    return markovkit::ReactionSystem{
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
}

double run(const markovkit::ReactionSystem &model, const num::array<double> &rates,
           num::idx capacity, num::idx steps, unsigned seed) {
    const auto level = [](const markovkit::State &x) { return (x[0] + x[1] + x[2]) / 2; };
    const auto start = std::chrono::steady_clock::now();
    (void)else_sim::else_ensemble(
        model, rates, markovkit::State{500, 1000, 2000}, 1, 0.0,
        std::numeric_limits<double>::infinity(),
        {.capacity = capacity,
         .expansion_depth = 1,
         .tolerance = 1e-12,
         .maximum_steps = steps,
         .reuse_factorization = true},
        seed, level);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

double median(num::array<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
    const num::idx steps = argc > 1 ? std::stoul(argv[1]) : 1000;
    const int repetitions = argc > 2 ? std::stoi(argv[2]) : 5;
    const std::string prefix = argc > 3 ? argv[3] : "subsweep_walltime";
    if (repetitions < 1)
        throw std::invalid_argument("the repetition count must be positive");

    const double y1 = 500.0, y2 = 1000.0, y3 = 2000.0;
    const double mu1 = 2000.0, mu2 = 50000.0;
    const num::array<double> rates = {mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1,
                                      2.0 * mu1 / (y1 * y1), (mu1 + mu2) / y3};
    const auto model = oregonator();
    const num::array<num::idx> capacities{60, 120, 240, 360, 480, 600};
    num::array<double> x, wall_time;

    (void)run(model, rates, capacities.front(), 20, 9000);

    std::ofstream csv(prefix + ".csv");
    csv << "capacity,wall_seconds\n";
    std::cout << "capacity  wall time (s)\n";
    for (num::idx capacity : capacities) {
        num::array<double> samples;
        for (int repetition = 0; repetition < repetitions; ++repetition)
            samples.push_back(run(model, rates, capacity, steps, 1000));
        const double elapsed = median(std::move(samples));
        x.push_back(static_cast<double>(capacity));
        wall_time.push_back(elapsed);
        csv << capacity << ',' << std::setprecision(10) << elapsed << '\n';
        std::cout << std::setw(8) << capacity << std::fixed << std::setprecision(3)
                  << std::setw(15) << elapsed << '\n';
    }

    num::plt::plot(x, wall_time, "labeled subsweep", "linespoints lw 2 pt 7 ps 1.3");
    num::plt::xlabel("subnetwork capacity");
    num::plt::ylabel("wall time for 1,000 updates (s)");
    num::plt::savefig(prefix + ".png");
}
