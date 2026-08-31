#include "markovkit.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

struct ToggleSystem {
    markovkit::ReactionSystem model;
    std::vector<double> rates;
    markovkit::State initial_state;
    std::vector<double> basin_u_x;
    std::vector<double> basin_u_y;
    std::vector<double> basin_v_x;
    std::vector<double> basin_v_y;
    std::vector<double> start_x;
    std::vector<double> start_y;
    double plot_max;
};

ToggleSystem make_toggle(double scale, markovkit::State start, double u_basin, double v_basin,
                         double plot_max) {
    const double alpha = 20.0 * scale;
    const double beta = 400.0 * scale;
    const double K = 100.0 * scale;
    const double K3 = K * K * K;
    const double d_u = 1.0 + (0.1 / 1.1);
    const double d_v = 1.0;

    ToggleSystem sys;
    sys.rates = {alpha, beta, K3, d_u, alpha, beta, K3, d_v};
    sys.initial_state = start;
    sys.basin_u_x = {u_basin};
    sys.basin_u_y = {v_basin * 0.06};
    sys.basin_v_x = {u_basin * 0.06};
    sys.basin_v_y = {v_basin};
    sys.start_x = {static_cast<double>(start[0])};
    sys.start_y = {static_cast<double>(start[1])};
    sys.plot_max = plot_max;

    sys.model.changes = {
        {1, 0},  // R1: 0 -> U
        {-1, 0}, // R2: U -> 0
        {0, 1},  // R3: 0 -> V
        {0, -1}, // R4: V -> 0
    };
    sys.model.propensities = {
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
    return sys;
}

int main() {
    std::cout << "=== ELSE Toggle-Switch Multi-Trajectory Multi-Regime Phase Plane ===\n\n";

    constexpr double final_time = 500.0;
    constexpr std::size_t max_steps = 100000;
    constexpr std::size_t n_paths = 15;

    // Regime 1: High Barrier / Metastable (scale = 0.20)
    std::cout << "1. Simulating Regime 1: High Barrier (scale = 0.20, " << n_paths
              << " trajectories)...\n";
    auto sys1 = make_toggle(0.20, {17, 1}, 76.0, 83.0, 105.0);

    // Independent SSA paths for Regime 1 (trapped in U-high basin)
    std::vector<std::vector<double>> ssa1_u, ssa1_v;
    for (int seed = 1; ssa1_u.size() < n_paths && seed <= 100; ++seed) {
        auto traj = ssa::gillespie(sys1.model, sys1.rates, sys1.initial_state, 0.0, final_time,
                                   seed, max_steps);
        int max_v = 0;
        for (const auto &st : traj.states)
            max_v = std::max(max_v, st[1]);
        if (max_v < 30) { // Trapped trajectory
            std::vector<double> u, v;
            for (const auto &st : traj.states) {
                u.push_back(static_cast<double>(st[0]));
                v.push_back(static_cast<double>(st[1]));
            }
            ssa1_u.push_back(std::move(u));
            ssa1_v.push_back(std::move(v));
        }
    }

    // Shared ELSE ensemble
    const auto block_level = [](const markovkit::State &x) { return x[0]; };
    else_sim::ELSEOptions opt1{
        .capacity = 600, .expansion_depth = 1, .tolerance = 1e-12, .maximum_steps = max_steps};
    auto ens1 = else_sim::else_ensemble(sys1.model, sys1.rates, sys1.initial_state, 50, 0.0,
                                        final_time, opt1, 8, block_level);
    std::vector<std::vector<double>> else1_u(n_paths), else1_v(n_paths);
    for (std::size_t i = 0; i < n_paths && i < ens1.size(); ++i) {
        for (const auto &st : ens1[i].states) {
            else1_u[i].push_back(static_cast<double>(st[0]));
            else1_v[i].push_back(static_cast<double>(st[1]));
        }
    }

    // Regime 2: Lower Barrier / Fast Switching (scale = 0.08)
    std::cout << "2. Simulating Regime 2: Lower Barrier (scale = 0.08, " << n_paths
              << " trajectories)...\n";
    auto sys2 = make_toggle(0.08, {7, 1}, 30.5, 33.5, 45.0);

    // Independent SSA paths
    std::vector<std::vector<double>> ssa2_u(n_paths), ssa2_v(n_paths);
    for (std::size_t i = 0; i < n_paths; ++i) {
        auto traj = ssa::gillespie(sys2.model, sys2.rates, sys2.initial_state, 0.0, final_time,
                                   static_cast<int>(10 + i), max_steps);
        for (const auto &st : traj.states) {
            ssa2_u[i].push_back(static_cast<double>(st[0]));
            ssa2_v[i].push_back(static_cast<double>(st[1]));
        }
    }

    // Shared ELSE ensemble
    else_sim::ELSEOptions opt2{
        .capacity = 250, .expansion_depth = 1, .tolerance = 1e-12, .maximum_steps = max_steps};
    auto ens2 = else_sim::else_ensemble(sys2.model, sys2.rates, sys2.initial_state, n_paths, 0.0,
                                        final_time, opt2, 42, block_level);
    std::vector<std::vector<double>> else2_u(n_paths), else2_v(n_paths);
    for (std::size_t i = 0; i < n_paths && i < ens2.size(); ++i) {
        for (const auto &st : ens2[i].states) {
            else2_u[i].push_back(static_cast<double>(st[0]));
            else2_v[i].push_back(static_cast<double>(st[1]));
        }
    }

    // Plot the phase-plane comparison.
    std::cout << "3. Rendering 2x2 multi-trajectory phase plane plot...\n";
    num::plt::subplot(2, 2);

    // Row 1, Col 1: High Barrier SSA (Independent Trajectories Trapped)
    for (std::size_t i = 0; i < n_paths; ++i) {
        std::string label = (i == 0) ? "SSA paths (15)" : "";
        num::plt::plot(ssa1_u[i], ssa1_v[i], label, "lines lw 1.1 lc rgb '#e67e22'");
    }
    num::plt::plot(sys1.start_x, sys1.start_y, "Start (17, 1)",
                   "points pt 9 ps 1.8 lc rgb '#000000'");
    num::plt::plot(sys1.basin_u_x, sys1.basin_u_y, "U-high basin",
                   "points pt 7 ps 2.0 lc rgb '#2980b9'");
    num::plt::plot(sys1.basin_v_x, sys1.basin_v_y, "V-high basin (unreached)",
                   "points pt 7 ps 2.0 lc rgb '#d35400'");
    num::plt::title("High Barrier (s=0.20): Direct SSA (Trapped in U-basin)");
    num::plt::xlabel("U");
    num::plt::ylabel("V");
    num::plt::xlim(0.0, sys1.plot_max);
    num::plt::ylim(0.0, sys1.plot_max);
    num::plt::legend();
    num::plt::next();

    // Row 1, Col 2: High Barrier ELSE (Shared Trajectories Escaping & Finding V-basin)
    for (std::size_t i = 0; i < n_paths; ++i) {
        std::string label = (i == 0) ? "Shared ELSE paths (15)" : "";
        num::plt::plot(else1_u[i], else1_v[i], label, "lines lw 1.3 lc rgb '#2980b9'");
    }
    num::plt::plot(sys1.start_x, sys1.start_y, "Start (17, 1)",
                   "points pt 9 ps 1.8 lc rgb '#000000'");
    num::plt::plot(sys1.basin_u_x, sys1.basin_u_y, "U-high basin",
                   "points pt 7 ps 2.0 lc rgb '#2980b9'");
    num::plt::plot(sys1.basin_v_x, sys1.basin_v_y, "V-high basin (found)",
                   "points pt 7 ps 2.0 lc rgb '#d35400'");
    num::plt::title("High Barrier (s=0.20): Shared ELSE (Escapes to V-basin)");
    num::plt::xlabel("U");
    num::plt::ylabel("V");
    num::plt::xlim(0.0, sys1.plot_max);
    num::plt::ylim(0.0, sys1.plot_max);
    num::plt::legend();
    num::plt::next();

    // Row 2, Col 1: Low Barrier SSA (Independent Trajectories Transitioning)
    for (std::size_t i = 0; i < n_paths; ++i) {
        std::string label = (i == 0) ? "SSA paths (15)" : "";
        num::plt::plot(ssa2_u[i], ssa2_v[i], label, "lines lw 1.1 lc rgb '#e67e22'");
    }
    num::plt::plot(sys2.start_x, sys2.start_y, "Start (7, 1)",
                   "points pt 9 ps 1.8 lc rgb '#000000'");
    num::plt::plot(sys2.basin_u_x, sys2.basin_u_y, "U-high basin",
                   "points pt 7 ps 2.0 lc rgb '#2980b9'");
    num::plt::plot(sys2.basin_v_x, sys2.basin_v_y, "V-high basin",
                   "points pt 7 ps 2.0 lc rgb '#d35400'");
    num::plt::title("Low Barrier (s=0.08): Direct SSA (Transitions)");
    num::plt::xlabel("U");
    num::plt::ylabel("V");
    num::plt::xlim(0.0, sys2.plot_max);
    num::plt::ylim(0.0, sys2.plot_max);
    num::plt::legend();
    num::plt::next();

    // Row 2, Col 2: Low Barrier ELSE (Shared Trajectories Transitioning)
    for (std::size_t i = 0; i < n_paths; ++i) {
        std::string label = (i == 0) ? "Shared ELSE paths (15)" : "";
        num::plt::plot(else2_u[i], else2_v[i], label, "lines lw 1.3 lc rgb '#2980b9'");
    }
    num::plt::plot(sys2.start_x, sys2.start_y, "Start (7, 1)",
                   "points pt 9 ps 1.8 lc rgb '#000000'");
    num::plt::plot(sys2.basin_u_x, sys2.basin_u_y, "U-high basin",
                   "points pt 7 ps 2.0 lc rgb '#2980b9'");
    num::plt::plot(sys2.basin_v_x, sys2.basin_v_y, "V-high basin",
                   "points pt 7 ps 2.0 lc rgb '#d35400'");
    num::plt::title("Low Barrier (s=0.08): Shared ELSE (Transitions)");
    num::plt::xlabel("U");
    num::plt::ylabel("V");
    num::plt::xlim(0.0, sys2.plot_max);
    num::plt::ylim(0.0, sys2.plot_max);
    num::plt::legend();

    num::plt::savefig("toggle_trajectories.png");
    std::cout << "[SUCCESS] Saved toggle_trajectories.png\n\n";

    // Measure how many entrance solves are shared across trajectories.
    std::cout << "4. Measuring trajectory entrance sharing across ensemble sizes...\n";
    const std::vector<std::size_t> test_counts = {10, 20, 50, 100, 250, 500, 1000};
    std::vector<double> traj_counts_dbl;
    std::vector<double> sharing_fractions;

    for (std::size_t m : test_counts) {
        const num::idx cap = std::max<num::idx>(300, static_cast<num::idx>(m + 100));
        auto ens = else_sim::else_ensemble(sys2.model, sys2.rates, sys2.initial_state, m, 0.0, 15.0,
                                           {.capacity = cap, .maximum_steps = 15}, 42, block_level);

        std::unordered_set<markovkit::State> entrances;
        for (const auto &path : ens) {
            if (!path.states.empty()) {
                entrances.insert(path.states.back());
            }
        }
        double frac = std::min(1.0, static_cast<double>(entrances.size()) /
                                        std::max(1.0, static_cast<double>(m)));

        traj_counts_dbl.push_back(static_cast<double>(m));
        sharing_fractions.push_back(frac);
        std::cout << "  Trajectories: " << std::setw(6) << m
                  << " | Distinct Entrances / Trajectories: " << std::fixed << std::setprecision(4)
                  << frac << "\n";
    }

    num::plt::subplot(1, 1);
    num::plt::plot(traj_counts_dbl, sharing_fractions, "Distinct entrances / Trajectories",
                   "linespoints pt 7 lw 2 lc rgb '#2980b9'");
    num::plt::title("Distinct Entrance Columns Relative to Shared Trajectories");
    num::plt::xlabel("Trajectories (m)");
    num::plt::ylabel("Distinct Entrances / Trajectories");
    num::plt::semilogx();
    num::plt::ylim(0.0, 1.05);
    num::plt::savefig("toggle_sharing.png");
    std::cout << "[SUCCESS] Saved toggle_sharing.png\n";

    return 0;
}
