/// @file examples/density_laplacian.cpp
/// @brief Demonstrates localized Markov density propagation using Expanding Local Subnetwork
///        Enumeration (ELSE) with the Laplace density solver over Weideman–Talbot contour nodes.
///
/// Mathematical Formulation:
///   We solve the Continuous-Time Markov Master Equation:
///       dp/dt = Q * p,    p(0) = p0
///   via Numerical Inverse Laplace Transformation (NILT):
///       p(t) = L^{-1}[(zI - Q)^{-1} p0](t) = (1 / 2*pi*i) \oint_\Gamma e^{z*t} (zI - Q)^{-1} p0 dz
///
///   1. Computational Object : else_sim::LaplaceDensitySolver
///   2. Integration Contour   : Weideman–Talbot hyperbolic contour nodes {z_k, w_k}_{k=1}^14
///   3. Resolvent Solvers     : num::AutoResolventSolver solving shifted systems (z_k*I - Q) x = b

#include "else/density.hpp"
#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "markovkit.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Load Graph Laplacian (L), Stationary Distribution (h), and Committors (C)
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

    // 2. Propagate local density using the Laplace solver with Talbot contour quadrature nodes
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t panel = 0; panel < count; ++panel) {
        // Build localized subnetwork around starting state
        auto subnetworks = else_sim::laplacian_restrictions(laplacian, h, starts[panel], 500, 2);

        // LaplaceDensitySolver sets up shifted linear resolvents (z_k * I - Q)^{-1}
        else_sim::LaplaceDensitySolver density(std::move(subnetworks));

        for (double time : times) {
            // Solve transient density p(t) by integrating resolvents over 14 Talbot contour nodes
            const auto solution = density.solve(markovkit::State{static_cast<int>(starts[panel])},
                                                time, /*nodes=*/14);

            // Project density onto each committor: <C_j>(t) = \sum_i C_ij * p_i(t)
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

    // 3. Plot 6-panel committor evolution
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
    num::plt::savefig("density_laplacian.png");

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "density_laplacian completed in " << total_ms << " ms -> saved density_laplacian.png\n";

    return 0;
}
