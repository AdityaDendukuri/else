#include "else/else.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    std::cout << "========================================================================\n";
    std::cout << "     ELSE PURE TEMPLATE LIBRARY (ZERO DEPENDENCIES): BENCHMARK          \n";
    std::cout << "========================================================================\n\n";

    constexpr std::size_t n = 40;
    constexpr double birth_rate = 12.0;
    constexpr double death_rate = 1.0;

    std::vector<std::vector<int>> states(n);
    for (std::size_t i = 0; i < n; ++i)
        states[i] = {static_cast<int>(i)};

    std::vector<std::size_t> rows, cols;
    std::vector<double> vals;
    std::vector<else_sim::BoundaryTransition<std::size_t, std::vector<int>, double>> boundary;

    for (std::size_t j = 0; j < n; ++j) {
        double col_sum = 0.0;
        // Birth: j -> j + 1
        if (j + 1 < n) {
            rows.push_back(j + 1);
            cols.push_back(j);
            vals.push_back(birth_rate);
            col_sum += birth_rate;
        } else {
            // Exit boundary transition
            boundary.push_back({j, {static_cast<int>(n)}, birth_rate});
            col_sum += birth_rate;
        }
        // Death: j -> j - 1
        if (j > 0) {
            double d = death_rate * j;
            rows.push_back(j - 1);
            cols.push_back(j);
            vals.push_back(d);
            col_sum += d;
        }
        // Diagonal: -col_sum
        rows.push_back(j);
        cols.push_back(j);
        vals.push_back(-col_sum);
    }
    auto R = else_sim::SparseMatrix<double>::from_triplets(n, n, rows, cols, vals);

    // Stationary Poisson weights for reversible scaling: h_j = sqrt(pi_j)
    std::vector<double> h(n, 1.0);
    for (std::size_t i = 1; i < n; ++i) {
        h[i] = h[i - 1] * std::sqrt(birth_rate / (death_rate * i));
    }

    else_sim::Subnetwork<double, std::size_t, std::vector<int>> subnetwork(states, std::move(R),
                                                                           std::move(boundary), h);

    std::vector<double> p0(n, 0.0);
    p0[0] = 1.0;
    const auto u = subnetwork.occupation(p0);

    std::cout << std::left << std::setw(14) << "Block Size k" << std::setw(18) << "Woodbury (us)"
              << std::setw(18) << "Naive (us)" << std::setw(16) << "Speedup" << std::setw(18)
              << "Rel Discrepancy" << "\n";
    std::cout << std::string(84, '-') << "\n";

    for (std::size_t k : {1, 2, 4, 8, 12, 16, 20}) {
        std::vector<std::size_t> block(k);
        for (std::size_t i = 0; i < k; ++i)
            block[i] = n - 1 - i;

        double w_loss = subnetwork.cut_time_loss(u, block);
        double n_loss = subnetwork.naive_cut_time_loss(u, block);
        double diff = std::abs(w_loss - n_loss) / std::max(1e-12, n_loss);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < 200; ++r) {
            volatile double l = subnetwork.cut_time_loss(u, block);
            (void)l;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double w_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200.0;

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < 200; ++r) {
            volatile double l = subnetwork.naive_cut_time_loss(u, block);
            (void)l;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double n_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / 200.0;

        double speedup = n_us / std::max(1e-6, w_us);

        std::cout << std::left << std::setw(14) << k << std::setw(18) << std::fixed
                  << std::setprecision(2) << w_us << std::setw(18) << std::fixed
                  << std::setprecision(2) << n_us << std::setw(16)
                  << (std::to_string(static_cast<int>(speedup)) + " x") << std::setw(18)
                  << std::scientific << std::setprecision(2) << diff << "\n";
    }
    std::cout << "\n[SUCCESS] Pure template zero-dependency ELSE benchmark completed.\n";
    return 0;
}
