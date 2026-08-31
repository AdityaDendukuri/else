#include "markovkit.hpp"
#include <chrono>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
    const std::size_t steps = argc > 1 ? std::stoul(argv[1]) : 5000;
    const std::size_t paths = argc > 2 ? std::stoul(argv[2]) : 1;
    const std::size_t capacity = argc > 3 ? std::stoul(argv[3]) : 600;
    const bool dense = argc > 4 && std::string_view(argv[4]) == "dense";

    const double y1 = 500.0, y2 = 1000.0, y3 = 2000.0;
    const double mu1 = 2000.0, mu2 = 50000.0;
    const std::vector<double> rates = {mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1,
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
        else_sim::ELSEOptions options{.capacity = capacity,
                                      .expansion_depth = 1,
                                      .tolerance = 1e-12,
                                      .maximum_steps = steps,
                                      .reuse_factorization = reuse};
        const auto start = std::chrono::steady_clock::now();
        const auto trajectories =
            else_sim::else_ensemble(model, rates, markovkit::State{500, 1000, 2000}, paths, 0.0,
                                    std::numeric_limits<double>::infinity(), options, 42, level);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        double mean_time = 0.0;
        for (const auto &trajectory : trajectories)
            mean_time += trajectory.times.back() / trajectories.size();
        std::cout << (reuse ? "reuse" : "scratch") << " wall=" << seconds
                  << " mean_time=" << mean_time
                  << " first_steps=" << trajectories.front().times.size() - 1 << '\n';
    }
}
