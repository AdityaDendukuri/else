#include "else/algorithms/density.hpp"
#include "else/restriction/laplacian.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numerics.hpp>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr num::idx states = 120;
constexpr num::idx basins = 5;
constexpr num::idx capacity = 90;
constexpr int subnetworks = 20;
constexpr num::idx talbot_nodes = 20;

double l1_error(const num::vec &reference,
                const else_sim::DensitySolution<num::multi_index> &approximation) {
    num::vec embedded(reference.size(), 0.0);
    for (num::idx k = 0; k < approximation.probability.size(); ++k)
        embedded[static_cast<num::idx>(approximation.states[k][0])] = approximation.probability[k];
    double error = 0.0;
    for (num::idx i = 0; i < reference.size(); ++i)
        error += std::abs(reference[i] - embedded[i]);
    return error;
}

} // namespace

int main() {
    num::rng64 random(12345);
    const auto graph = num::structures::erdos_renyi(states, 0.08, random, true, 0.5, 2.0);
    const auto laplacian = num::linear::laplacian(graph);
    const auto generator = num::scaled(laplacian, -1.0);
    const num::array<double> stationary_sqrt(states, 1.0);
    const auto times = num::logspace(-2.0, 0.0, 31);
    const std::array<num::idx, basins> starts = {0, 24, 48, 72, 96};
    std::array<std::array<num::array<double>, basins>, basins> ib_curves;
    std::array<std::array<num::array<double>, basins>, basins> reference_curves;
    std::array<num::array<double>, basins> errors;

    for (num::idx panel = 0; panel < basins; ++panel) {
        auto restrictions = else_sim::laplacian_restrictions(
            laplacian, num::view<const double>(stationary_sqrt), starts[panel], capacity,
            subnetworks, 100 + static_cast<unsigned>(panel));
        auto density = else_sim::make_density_chain(std::move(restrictions));
        num::vec initial(states, 0.0);
        initial[starts[panel]] = 1.0;
        for (double time : times) {
            const auto approximation = else_sim::inverse_laplace_density(
                density, num::multi_index{static_cast<int>(starts[panel])}, time, talbot_nodes);
            const num::vec reference = num::expv(time, generator, initial, 80, 1e-13);
            errors[panel].push_back(std::max(1e-16, l1_error(reference, approximation)));
            for (num::idx basin = 0; basin < basins; ++basin) {
                const num::idx first = static_cast<num::idx>(basin) * (states / basins);
                const num::idx last = first + (states / basins);
                double ib_mass = 0.0;
                for (num::idx k = 0; k < approximation.probability.size(); ++k) {
                    const num::idx state = static_cast<num::idx>(approximation.states[k][0]);
                    if (state >= first && state < last)
                        ib_mass += approximation.probability[k];
                }
                double reference_mass = 0.0;
                for (num::idx state = first; state < last; ++state)
                    reference_mass += reference[state];
                ib_curves[panel][basin].push_back(ib_mass);
                reference_curves[panel][basin].push_back(reference_mass);
            }
        }
    }

    num::plt::subplot(2, 3);
    for (num::idx panel = 0; panel < basins; ++panel) {
        for (num::idx basin = 0; basin < basins; ++basin) {
            num::plt::plot(times, ib_curves[panel][basin], "B" + std::to_string(basin + 1),
                           "lines lw 2");
            num::plt::plot(times, reference_curves[panel][basin], std::string{},
                           "points pt 6 ps 0.35");
        }
        num::plt::title("start in B" + std::to_string(panel + 1));
        num::plt::xlabel("physical time");
        num::plt::ylabel("basin probability");
        num::plt::semilogx();
        num::plt::ylim(0.0, 1.0);
        if (panel == 0)
            num::plt::legend();
        num::plt::next();
    }
    for (num::idx panel = 0; panel < basins; ++panel)
        num::plt::plot(times, errors[panel], "start B" + std::to_string(panel + 1), "lines lw 2");
    num::plt::title("Full-density error");
    num::plt::xlabel("physical time");
    num::plt::ylabel("L1 error");
    num::plt::loglog();
    num::plt::legend();
    num::plt::savefig("random_laplacian_validation.png");

    std::cout << "random Laplacian: n=" << states << ", nnz=" << laplacian.nnz()
              << ", capacity=" << capacity << ", subnetworks=" << subnetworks
              << ", Talbot nodes=" << talbot_nodes << "\n";
    for (num::idx panel = 0; panel < basins; ++panel)
        std::cout << "start B" << panel + 1
                  << ": max L1=" << *std::max_element(errors[panel].begin(), errors[panel].end())
                  << "\n";
}
