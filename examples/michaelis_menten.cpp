#include "markovkit.hpp"
#include <array>
#include <map>
#include <vector>

int main() {
    // Substrate and enzyme bind, unbind, and form product.
    const markovkit::ReactionSystem model{
        .changes = {{-1, -1, 1, 0}, {1, 1, -1, 0}, {0, 1, -1, 1}},
        .propensities = {
            [](const markovkit::State &x, const auto &rates, double) {
                return rates[0] * x[0] * x[1];
            },
            [](const markovkit::State &x, const auto &rates, double) { return rates[1] * x[2]; },
            [](const markovkit::State &x, const auto &rates, double) { return rates[2] * x[2]; },
        }};
    const std::vector<double> rates{0.01, 0.1, 0.1};
    const markovkit::State initial{50, 10, 0, 0};
    constexpr double time = 10.0;

    // Both solvers start from the same point mass and horizon.
    const fsp::FSPProblem problem{
        .model = model,
        .u0 = initial,
        .t0 = 0.0,
        .tf = time,
        .rates = rates,
        .bc = fsp::rect_boundary({50, 50, 50, 50}),
    };
    auto [fsp_solution, diagnostics] =
        fsp::solve_adaptive_fsp(problem, {.eps_dt = 0.1, .flux_tolerance = 1e-10});

    // Solve the same terminal density through ELSE subnetworks.
    auto subnetworks = else_sim::density_subnetworks(model, rates, initial, 30,
                                                     {.capacity = 60, .expansion_depth = 0});
    const auto else_solution =
        else_sim::TalbotDensitySolver(std::move(subnetworks)).solve(initial, time);

    std::array<std::map<int, double>, 4> fsp_marginals;
    std::array<std::map<int, double>, 4> else_marginals;
    // Form comparable single-species marginals.
    for (std::size_t species = 0; species < 4; ++species) {
        fsp_marginals[species] = fsp_solution.marginal(fsp_solution.n_snapshots() - 1, species);
        else_marginals[species] = markovkit::marginal_distribution(
            else_solution.states, else_solution.probability, species);
    }

    // Plot matching FSP and ELSE marginals for every species.
    const std::array names{"S", "E", "C", "P"};
    num::plt::subplot(2, 2);
    for (std::size_t species = 0; species < 4; ++species) {
        std::vector<double> fsp_x;
        std::vector<double> fsp_y;
        std::vector<double> else_x;
        std::vector<double> else_y;
        for (const auto &[state, probability] : fsp_marginals[species]) {
            fsp_x.push_back(state);
            fsp_y.push_back(probability);
        }
        for (const auto &[state, probability] : else_marginals[species]) {
            else_x.push_back(state);
            else_y.push_back(probability);
        }
        num::plt::plot(fsp_x, fsp_y, "FSP", "lines lw 2");
        num::plt::plot(else_x, else_y, "ELSE", "boxes");
        num::plt::xlabel(names[species]);
        num::plt::ylabel("probability");
        num::plt::legend();
        if (species < 3) {
            num::plt::next();
        }
    }
    num::plt::savefig("michaelis_menten.png");
}
