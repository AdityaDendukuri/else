#include "markovkit.hpp"
#include <array>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t simulation_steps = 100000;
constexpr num::idx subnetwork_capacity = 60;
constexpr double final_time = std::numeric_limits<double>::infinity();
constexpr std::array<const char *, 3> labels = {"X", "Y", "Z"};
constexpr std::array<const char *, 3> colors = {"#2980b9", "#c0392b", "#27ae60"};

} // namespace

int main() {
    // These scales reproduce the stochastic Oregonator regime.
    const double y1 = 500.0;
    const double y2 = 1000.0;
    const double y3 = 2000.0;
    const double mu1 = 2000.0;
    const double mu2 = 50000.0;
    const std::vector<double> rates = {
        mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1, 2.0 * mu1 / (y1 * y1), (mu1 + mu2) / y3,
    };

    // Five reactions couple the three oscillator species.
    markovkit::ReactionSystem model{
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

    const markovkit::State initial{500, 1000, 2000};
    // Compare matched ELSE and SSA trajectory ensembles.
    const auto one_else = else_sim::else_ensemble(
        model, rates, initial, 1, 0.0, final_time,
        {.capacity = subnetwork_capacity, .maximum_steps = simulation_steps}, 42);
    const std::vector<markovkit::Trajectory> one_ssa{
        ssa::gillespie(model, rates, initial, 0.0, final_time, 42, simulation_steps)};
    const auto many_else = else_sim::else_ensemble(
        model, rates, initial, 20, 0.0, final_time,
        {.capacity = subnetwork_capacity, .maximum_steps = simulation_steps}, 42);
    std::vector<markovkit::Trajectory> many_ssa(20);
    // Use matching seed ranges for the SSA ensemble with OpenMP parallelism.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 20; ++i) {
        const int seed = 42 + i;
        many_ssa[i] = ssa::gillespie(model, rates, initial, 0.0, final_time, seed, simulation_steps);
    }

    // Plot single paths above their ensemble counterparts.
    num::plt::subplot(2, 2);
    num::plt::plot_paths(one_else, labels, colors, "ELSE: 1 trajectory");
    num::plt::next();
    num::plt::plot_paths(one_ssa, labels, colors, "SSA: 1 trajectory");
    num::plt::next();
    num::plt::plot_paths(many_else, labels, colors, "ELSE: 20 trajectories");
    num::plt::next();
    num::plt::plot_paths(many_ssa, labels, colors, "SSA: 20 trajectories");
    num::plt::savefig("oregonator_else_ssa.png");
}
