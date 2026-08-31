#include "else/restriction.hpp"
#include "markovkit.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    std::cout << "========================================================================\n";
    std::cout << "  ELSE WOODBURY CUT-TIME UPDATE: FLOATING-POINT ERROR & TIMING BENCHMARK \n";
    std::cout << "========================================================================\n\n";

    constexpr num::idx n = 40;
    constexpr double birth_rate = 12.0;
    constexpr double death_rate = 1.0;

    // Define the immigration-death reaction network.
    markovkit::ReactionSystem model{
        .changes = {{1}, {-1}},
        .propensities = {
            [](const markovkit::State &, const auto &rates, double) { return rates[0]; },
            [](const markovkit::State &state, const auto &rates, double) {
                return rates[1] * state[0];
            },
        }};
    const std::vector<double> rates{birth_rate, death_rate};

    std::vector<markovkit::State> states;
    for (num::idx state = 0; state < n; ++state) {
        states.push_back(markovkit::State{static_cast<int>(state)});
    }

    // Stationary Poisson weights for reversible scaling
    std::vector<double> h(n, 1.0);
    for (num::idx state = 1; state < n; ++state) {
        h[state] = h[state - 1] * std::sqrt(birth_rate / (death_rate * state));
    }

    auto subnetwork = else_sim::reversible_cme_subnetwork(model, rates, states, h);

    // Initial entrance in state 0
    num::Vector entrance(n, 0.0);
    entrance[0] = 1.0;
    const auto u = subnetwork.occupation(entrance);

    // Compare each Woodbury score with a fresh reduced solve.
    std::cout
        << "--- 1. Component-Wise Floating-Point Error (Woodbury vs Naive Ground Truth) ---\n\n";
    std::cout << std::left << std::setw(8) << "State" << std::setw(16) << "Woodbury Loss"
              << std::setw(16) << "Naive Loss" << std::setw(18) << "Relative Error" << std::setw(20)
              << "safe_add Bound" << std::setw(20) << "Backward Residual" << "\n";
    std::cout << std::string(98, '-') << "\n";

    std::vector<double> state_indices(n);
    std::vector<double> rel_errors(n);
    std::vector<double> safe_add_bounds(n);
    std::vector<double> backward_residuals(n);

    double max_rel_error = 0.0;
    double max_safe_add_bound = 0.0;
    double max_backward_res = 0.0;

    for (num::idx i = 0; i < n; ++i) {
        state_indices[i] = static_cast<double>(i);
        const std::vector<num::idx> block{i};

        auto diag = subnetwork.cut_time_loss_with_diagnostics(u, block, 1e-6);
        double loss_woodbury = diag.loss;
        double loss_naive = subnetwork.naive_cut_time_loss(u, block);

        double err = std::abs(loss_woodbury - loss_naive);
        double rel_err = err / std::max(1e-12, loss_naive);
        double bres = diag.estimated_error;

        // safe_add precision tracking metric: eps_mach * |loss| / loss
        double x = loss_woodbury;
        double acc_err = 0.0;
        num::safe_add(x, acc_err, 0.0, 1e-6);
        double safe_bound = acc_err / std::max(1e-12, loss_woodbury);

        rel_errors[i] = std::max(1e-18, rel_err);
        safe_add_bounds[i] = std::max(1e-18, safe_bound);
        backward_residuals[i] = std::max(1e-18, bres);

        max_rel_error = std::max(max_rel_error, rel_err);
        max_safe_add_bound = std::max(max_safe_add_bound, safe_bound);
        max_backward_res = std::max(max_backward_res, bres);

        if (i < 8 || i >= n - 5 || i % 5 == 0) {
            std::cout << std::left << std::setw(8) << i << std::setw(16) << std::scientific
                      << std::setprecision(3) << loss_woodbury << std::setw(16) << std::scientific
                      << std::setprecision(3) << loss_naive << std::setw(18) << std::scientific
                      << std::setprecision(3) << rel_err << std::setw(20) << std::scientific
                      << std::setprecision(3) << safe_bound << std::setw(20) << std::scientific
                      << std::setprecision(3) << bres << "\n";
        }
    }
    std::cout << "\nMax Component-Wise Relative Error:     " << std::scientific << max_rel_error
              << "\n";
    std::cout << "Max safe_add Precision Bound:         " << std::scientific << max_safe_add_bound
              << "\n";
    std::cout << "Max Component-Wise Backward Residual: " << std::scientific << max_backward_res
              << "\n\n";

    // Measure how the update behaves as the removed block grows.
    std::cout
        << "--- 2. Block Cut Set Benchmark: Woodbury Principal Inversion vs Naive Scratch ---\n\n";
    std::cout << std::left << std::setw(14) << "Block Size k" << std::setw(18) << "Woodbury (us)"
              << std::setw(18) << "Naive (us)" << std::setw(16) << "Rel Discrepancy" << "\n";
    std::cout << std::string(66, '-') << "\n";

    const std::vector<num::idx> block_sizes = {1, 2, 4, 8, 12, 16, 20};
    constexpr int repetitions = 300;
    constexpr double precision_tolerance = 1e-6;
    std::vector<double> plotted_block_sizes;
    std::vector<double> accumulated_error_checks;
    std::vector<double> block_relative_discrepancies;
    std::vector<double> precision_tolerances;

    for (num::idx k : block_sizes) {
        std::vector<num::idx> block(k);
        for (num::idx i = 0; i < k; ++i) {
            block[i] = i + 1; // Nested blocks; preserve entrance state 0.
        }

        // Warmup & discrepancy
        const auto diagnostics =
            subnetwork.cut_time_loss_with_diagnostics(u, block, precision_tolerance);
        double w_loss = diagnostics.loss;
        double n_loss = subnetwork.naive_cut_time_loss(u, block);
        double rel_diff = std::abs(w_loss - n_loss) / std::max(1e-12, n_loss);

        plotted_block_sizes.push_back(static_cast<double>(k));
        accumulated_error_checks.push_back(std::max(1e-18, diagnostics.estimated_error));
        block_relative_discrepancies.push_back(std::max(1e-18, rel_diff));
        precision_tolerances.push_back(precision_tolerance);

        // Benchmark Woodbury
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repetitions; ++r) {
            volatile double l = subnetwork.cut_time_loss(u, block);
            (void)l;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double w_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / repetitions;

        // Benchmark Naive
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repetitions; ++r) {
            volatile double l = subnetwork.naive_cut_time_loss(u, block);
            (void)l;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double n_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / repetitions;

        std::cout << std::left << std::setw(14) << k << std::setw(18) << std::fixed
                  << std::setprecision(2) << w_us << std::setw(18) << std::fixed
                  << std::setprecision(2) << n_us << std::setw(16) << std::scientific
                  << std::setprecision(2) << rel_diff << "\n";
    }
    std::cout << "\n";

    // Plot error and runtime against removed block size.
    num::plt::plot(state_indices, rel_errors,
                   "Relative Discrepancy |Loss_{Woodbury} - Loss_{Naive}| / Loss_{Naive}",
                   "lines lw 2 lc rgb '#c0392b'");
    num::plt::plot(state_indices, safe_add_bounds,
                   "safe\\_add Precision Bound (\\varepsilon_{mach} |x| / x)",
                   "lines dt 2 lw 2 lc rgb '#27ae60'");
    num::plt::plot(state_indices, backward_residuals,
                   "Linear Residual ||Z_{SS} c - u_S||_{\\infty}",
                   "lines dt 3 lw 2 lc rgb '#2980b9'");
    num::plt::title("ELSE Cut-Time Update Precision & Floating-Point Error Metrics");
    num::plt::xlabel("Removed State Index j");
    num::plt::ylabel("Relative Precision / Error");
    num::plt::semilogy();
    num::plt::legend();
    num::plt::savefig("else_shedding_comparison.png");

    std::cout << "[SUCCESS] Floating-point error plot saved to else_shedding_comparison.png\n";

    // Plot the actual error estimate accumulated by safe_add while evaluating
    // q_S^T (Z_SS)^-1 u_S.  Each point contains all additions for that block.
    num::plt::plot(plotted_block_sizes, accumulated_error_checks,
                   "Accumulated safe\\_add estimate e/x",
                   "linespoints pt 7 ps 0.8 lw 2 lc rgb '#27ae60'");
    num::plt::plot(plotted_block_sizes, block_relative_discrepancies,
                   "Measured Woodbury--direct discrepancy",
                   "linespoints pt 5 ps 0.8 lw 2 lc rgb '#c0392b'");
    num::plt::plot(plotted_block_sizes, precision_tolerances, "Acceptance tolerance 10^{-6}",
                   "lines dt 2 lw 2 lc rgb '#2c3e50'");
    num::plt::title("Accumulated Floating-Point Check for Block Shedding");
    num::plt::xlabel("Removed block size |S|");
    num::plt::ylabel("Relative error estimate");
    num::plt::semilogy();
    num::plt::legend();
    num::plt::savefig("else_safe_add_accumulation.png");

    std::cout << "[SUCCESS] Accumulated safe_add plot saved to "
                 "else_safe_add_accumulation.png\n";
    return 0;
}
