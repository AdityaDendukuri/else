/// Committor evolution from each basin, on the numerics-based ELSE.
///
/// A direct port of density_laplacian.cpp: same data, same subnetwork sizes,
/// same contour, same plot. Only the ELSE calls differ.
#include "container/util/math.hpp"
#include "elsex/density.hpp"
#include "elsex/laplacian.hpp"
#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "plot/plot.hpp"
#include "stats/probability.hpp"
#include "stats/selection.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <span>
#include <vector>

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // Load the Laplacian, square-root stationary weights, and committors.
    const auto data = num::io::read_json("laplacians/cs-medium.json");
    const auto laplacian = num::io::sparse_matrix(data.at("L"));
    const auto h = num::io::json_vector<double>(data.at("h"));
    const auto committors = num::io::json_matrix<double>(data.at("C"));

    constexpr std::size_t count = 5;
    std::array<num::idx, count> starts{};
    for (std::size_t obs = 0; obs < count; ++obs) {
        starts[obs] =
            num::argmax(laplacian.n_rows(), [&](num::idx state) { return committors[state][obs]; });
    }

    const auto times = num::logspace(-12.0, -4.0, 41);
    std::array<std::array<std::vector<double>, count>, count> curves;

// Propagate the local density by Talbot inversion.
#pragma omp parallel for schedule(dynamic)
    for (std::size_t panel = 0; panel < count; ++panel) {
        // Build two consecutive subnetworks around the starting state.
        auto subnetworks = elsex::laplacian_restrictions(laplacian, std::span<const double>(h),
                                                         starts[panel], 500, 2);

        auto density = elsex::make_density_chain(std::move(subnetworks));

        for (double time : times) {
            const auto solution = elsex::inverse_laplace_density(
                density, std::vector<int>{static_cast<int>(starts[panel])}, time, /*modes=*/14);

            // Project the density onto each committor.
            for (std::size_t obs = 0; obs < count; ++obs) {
                const double mean = num::weighted_sum(
                    std::span<const double>(solution.probability), [&](num::idx index) {
                        const auto state = static_cast<num::idx>(solution.states[index][0]);
                        return committors[state][obs];
                    });
                curves[panel][obs].push_back(mean);
            }
        }
    }

    // Plot the committor evolution from each starting basin.
    num::plt::subplot(2, 3);
    for (std::size_t panel = 0; panel < count; ++panel) {
        for (std::size_t obs = 0; obs < count; ++obs) {
            num::plt::plot(times, curves[panel][obs], "C" + std::to_string(obs + 1), "lines lw 2");
        }
        num::plt::title("start = argmax(C" + std::to_string(panel + 1) + ")");
        num::plt::xlabel("t");
        num::plt::ylabel("C^T p(t)");
        num::plt::semilogx();
        num::plt::ylim(-0.02, 1.02);
        if (panel == 0) {
            num::plt::legend();
        }
        num::plt::next();
    }
    num::plt::savefig("elsex_density_laplacian.png");

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "elsex_density_laplacian completed in " << total_ms
              << " ms -> saved elsex_density_laplacian.png\n";

    return 0;
}
