#include "container/util/math.hpp"
#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "laplacian_validation.hpp"
#include "plot/plot.hpp"
#include "stats/selection.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <span>
#include <string>
#include <vector>

int main() {
    // Load the graph Laplacian, stationary weights, and committors.
    const auto data = num::io::read_json("laplacians/cs-medium.json");
    const auto laplacian = num::io::sparse_matrix(data.at("L"));
    const auto h = num::io::json_vector<double>(data.at("h"));
    const auto committors = num::io::json_matrix<double>(data.at("C"));
    constexpr num::idx count = 5;
    constexpr num::idx samples = 1000;
    constexpr num::idx reference_samples = 5000;
    constexpr num::idx capacity = 60;
    std::array<num::idx, count> starts{};
    // Start each ensemble at its dominant committor state.
    for (num::idx observable = 0; observable < count; ++observable) {
        starts[observable] = num::argmax(
            laplacian.n_rows(), [&](num::idx state) { return committors[state][observable]; });
    }

    // Observe each path on a logarithmic time grid.
    const auto times = num::logspace(-12.0, -5.0, 29);

    std::array<laplacian_validation::Curves, count> curves;
    std::array<laplacian_validation::Curves, count> reference;
    std::array<double, count> discrepancies{};
#pragma omp parallel for schedule(dynamic)
    for (num::idx panel = 0; panel < count; ++panel) {
        curves[panel] = laplacian_validation::labeled_subsweep_committor_means(
            laplacian, num::view<const double>(h), committors, starts[panel],
            num::view<const double>(times), samples, 20000 + static_cast<unsigned>(panel),
            capacity);
        reference[panel] = laplacian_validation::ssa_committor_means(
            laplacian, num::view<const double>(h), committors, starts[panel],
            num::view<const double>(times), reference_samples,
            30000 + static_cast<unsigned>(panel));
        discrepancies[panel] =
            laplacian_validation::maximum_discrepancy(curves[panel], reference[panel]);
    }

    num::plt::subplot(1, 5);
    for (num::idx panel = 0; panel < count; ++panel) {
        for (num::idx observable = 0; observable < count; ++observable) {
            const std::string color = std::to_string(observable + 1);
            num::plt::plot(times, curves[panel][observable], "C" + std::to_string(observable + 1),
                           "lines lw 2 lc " + color);
            num::plt::plot(times, reference[panel][observable], std::string{},
                           "lines dt 2 lw 1.2 lc " + color);
        }
        num::plt::title("start in basin " + std::to_string(panel + 1));
        num::plt::xlabel("time");
        num::plt::ylabel("empirical committor mean");
        num::plt::semilogx();
        num::plt::ylim(-0.02, 1.02);
        if (panel == 0)
            num::plt::legend("bottom left");
        num::plt::next();
    }
    num::plt::savefig("laplacian_trajectories.png");
    for (num::idx panel = 0; panel < count; ++panel)
        std::cout << "basin " << panel + 1 << " maximum discrepancy = " << discrepancies[panel]
                  << '\n';
}
