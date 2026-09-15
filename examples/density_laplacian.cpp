#include "container/util/math.hpp"
#include "else/algorithms/density.hpp"
#include "else/restriction/laplacian.hpp"
#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "laplacian_validation.hpp"
#include "plot/plot.hpp"
#include "stats/probability.hpp"
#include "stats/selection.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char **argv) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // Load the Laplacian, square-root stationary weights, and committors.
    const auto data = num::io::read_json("laplacians/cs-medium.json");
    const auto laplacian = num::io::sparse_matrix(data.at("L"));
    const auto h = num::io::json_vector<double>(data.at("h"));
    const auto committors = num::io::json_matrix<double>(data.at("C"));

    constexpr num::idx count = 5;
    std::array<num::idx, count> starts{};
    for (num::idx obs = 0; obs < count; ++obs) {
        starts[obs] =
            num::argmax(laplacian.n_rows(), [&](num::idx state) { return committors[state][obs]; });
    }

    const int subnetwork_count = argc > 1 ? std::stoi(argv[1]) : 64;
    const num::idx capacity = argc > 2 ? static_cast<num::idx>(std::stoul(argv[2])) : 4000;
    const num::idx particles = capacity;
    const num::idx labeled_samples = argc > 3 ? static_cast<num::idx>(std::stoull(argv[3])) : 1000;
    constexpr num::idx labeled_capacity = 60;
    constexpr num::idx reference_samples = 5000;
    const auto times = num::logspace(-12.0, -5.0, 29);
    std::array<std::array<num::array<double>, count>, count> curves;
    std::array<laplacian_validation::Curves, count> labeled_curves;
    std::array<laplacian_validation::Curves, count> reference;
    std::array<double, count> unlabeled_discrepancies{};
    std::array<double, count> labeled_discrepancies{};
    std::array<num::idx, count> realized_subnetworks{};
    std::array<num::idx, count> maximum_subnetwork_size{};

// Propagate the local density by Talbot inversion.
#pragma omp parallel for schedule(dynamic)
    for (num::idx panel = 0; panel < count; ++panel) {
        // Sample successive entrance supports and keep every restriction at
        // capacity.
        num::array<num::table<num::multi_index, double>> entrance_scales;
        auto subnetworks = else_sim::laplacian_restrictions(
            laplacian, num::view<const double>(h), starts[panel], capacity, subnetwork_count,
            100 + static_cast<unsigned>(panel), else_sim::DensityResampling::MultinomialParticles,
            particles, &entrance_scales);
        realized_subnetworks[panel] = subnetworks.size();
        for (const auto &subnetwork : subnetworks)
            maximum_subnetwork_size[panel] =
                std::max(maximum_subnetwork_size[panel], else_sim::size(subnetwork));

        auto density = else_sim::make_density_chain(std::move(subnetworks), entrance_scales);
        for (double time : times) {
            const auto solution = else_sim::inverse_laplace_density(
                density, num::multi_index{static_cast<int>(starts[panel])}, time,
                /*modes=*/14);

            // Project the density onto each committor.
            for (num::idx obs = 0; obs < count; ++obs) {
                const double mean = num::weighted_sum(
                    num::view<const double>(solution.probability), [&](num::idx index) {
                        const auto state = static_cast<num::idx>(solution.states[index][0]);
                        return committors[state][obs];
                    });
                curves[panel][obs].push_back(mean);
            }
        }

        reference[panel] = laplacian_validation::ssa_committor_means(
            laplacian, num::view<const double>(h), committors, starts[panel],
            num::view<const double>(times), reference_samples,
            10000 + static_cast<unsigned>(panel));
        labeled_curves[panel] = laplacian_validation::labeled_subsweep_committor_means(
            laplacian, num::view<const double>(h), committors, starts[panel],
            num::view<const double>(times), labeled_samples, 20000 + static_cast<unsigned>(panel),
            labeled_capacity);
        unlabeled_discrepancies[panel] =
            laplacian_validation::maximum_discrepancy(curves[panel], reference[panel]);
        labeled_discrepancies[panel] =
            laplacian_validation::maximum_discrepancy(labeled_curves[panel], reference[panel]);
    }

    num::plt::subplot(1, 5);
    for (num::idx panel = 0; panel < count; ++panel) {
        for (num::idx obs = 0; obs < count; ++obs) {
            const std::string color = std::to_string(obs + 1);
            num::plt::plot(times, reference[panel][obs], std::string{},
                           "lines dt 2 lw 0.8 lc " + color);
            num::plt::plot(times, curves[panel][obs], "C" + std::to_string(obs + 1),
                           "lines lw 2 lc " + color);
            num::plt::plot(times, labeled_curves[panel][obs], std::string{},
                           "points pt 13 ps 1.35 lc " + color);
        }
        num::plt::title("start in basin " + std::to_string(panel + 1));
        num::plt::xlabel("time");
        num::plt::ylabel("committor expectation");
        num::plt::semilogx();
        num::plt::ylim(-0.02, 1.02);
        if (panel == 0)
            num::plt::legend("bottom left");
        num::plt::next();
    }
    num::plt::savefig("committor_comparison.png");

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "density_laplacian completed in " << total_ms
              << " ms -> saved committor_comparison.png\n";
    for (num::idx panel = 0; panel < count; ++panel)
        std::cout << "basin " << panel + 1
                  << " labeled maximum discrepancy = " << labeled_discrepancies[panel]
                  << ", unlabeled maximum discrepancy = " << unlabeled_discrepancies[panel]
                  << ", subnetworks = " << realized_subnetworks[panel]
                  << ", maximum size = " << maximum_subnetwork_size[panel] << '\n';

    return 0;
}
