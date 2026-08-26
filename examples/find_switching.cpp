#include "markovkit.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

void test_params(double scale, const std::string &name) {
    const double alpha = 20.0 * scale;
    const double beta = 400.0 * scale;
    const double K = 100.0 * scale;
    const double K3 = K * K * K;
    const double d_u = 1.0 + (0.1 / 1.1);
    const double d_v = 1.0;

    const std::vector<double> rates = {alpha, beta, K3, d_u, alpha, beta, K3, d_v};

    markovkit::ReactionSystem model;
    model.changes = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    model.propensities = {
        [](const markovkit::State &x, const std::vector<double> &r, double) {
            double v3 = static_cast<double>(x[1]) * x[1] * x[1];
            return r[0] + (r[1] * r[2] / (r[2] + v3));
        },
        [](const markovkit::State &x, const std::vector<double> &r, double) { return r[3] * x[0]; },
        [](const markovkit::State &x, const std::vector<double> &r, double) {
            double u3 = static_cast<double>(x[0]) * x[0] * x[0];
            return r[4] + (r[5] * r[6] / (r[6] + u3));
        },
        [](const markovkit::State &x, const std::vector<double> &r, double) { return r[7] * x[1]; },
    };

    const markovkit::State initial_state{static_cast<int>(std::round(17.0 * scale / 0.2)), 1};
    constexpr double final_time = 500.0;

    std::cout << "\n=== Testing " << name << " (scale=" << scale << ", start=[" << initial_state[0] << ", " << initial_state[1] << "]) ===\n";

    // Test SSA
    int ssa_switches = 0;
    for (int seed = 1; seed <= 10; ++seed) {
        auto traj = ssa::gillespie(model, rates, initial_state, 0.0, final_time, seed, 200000);
        int max_u = 0, max_v = 0;
        for (const auto &st : traj.states) {
            max_u = std::max(max_u, st[0]);
            max_v = std::max(max_v, st[1]);
        }
        double threshold = 0.5 * beta;
        if (max_u > threshold && max_v > threshold) {
            ++ssa_switches;
        }
    }
    std::cout << "  SSA switches: " << ssa_switches << "/10 paths\n";

    // Test ELSE
    else_sim::ELSEOptions opt{.capacity = 300, .expansion_depth = 1, .tolerance = 1e-12, .maximum_steps = 100000};
    auto ens = else_sim::else_ensemble(model, rates, initial_state, 20, 0.0, final_time, opt, 42);
    int else_switches = 0;
    for (const auto &traj : ens) {
        int max_u = 0, max_v = 0;
        for (const auto &st : traj.states) {
            max_u = std::max(max_u, st[0]);
            max_v = std::max(max_v, st[1]);
        }
        double threshold = 0.5 * beta;
        if (max_u > threshold && max_v > threshold) {
            ++else_switches;
        }
    }
    std::cout << "  ELSE switches: " << else_switches << "/20 paths\n";
}

int main() {
    test_params(0.20, "Original (scale = 0.20)");
    test_params(0.12, "Moderate scale (scale = 0.12)");
    test_params(0.10, "Frequent switching scale (scale = 0.10)");
    test_params(0.08, "High switching scale (scale = 0.08)");
    return 0;
}
