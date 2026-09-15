#include "else/quantities/factor_update.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "plot/plot.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

num::spmat make_matrix(num::idx n, num::view<const num::idx> changed = {}) {
    num::array<bool> replaced(n, false);
    for (num::idx state : changed)
        replaced[state] = true;

    num::array<num::idx> rows, columns;
    num::array<double> values;
    for (num::idx i = 0; i < n; ++i) {
        for (num::idx j = 0; j < n; ++j) {
            double value = 0.0;
            if (i == j)
                value = replaced[i] ? 2.5 : 2.0;
            else if (replaced[i] || replaced[j])
                value = -0.004 / (1.0 + std::abs(static_cast<double>(i) - j));
            else
                value = -0.002 / (1.0 + std::abs(static_cast<double>(i) - j));
            rows.push_back(i);
            columns.push_back(j);
            values.push_back(value);
        }
    }
    return num::spmat::from_triplets(n, n, rows, columns, values);
}

double maximum_error(const num::vec &a, const num::vec &b) {
    double error = 0.0;
    for (num::idx i = 0; i < a.size(); ++i)
        error = std::max(error, std::abs(a[i] - b[i]));
    return error;
}

template <typename Work>
double milliseconds(Work work, int repetitions) {
    const auto start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition)
        work();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
               .count() /
           repetitions;
}

} // namespace

int main(int argc, char **argv) {
    const num::idx n = argc > 1 ? std::stoul(argv[1]) : 240;
    constexpr int repetitions = 8;
    const num::spmat base = make_matrix(n);
    // The matrix is dense, so force the dense LU: at the default `dense_limit`
    // of 32 the solver would pick KLU, and the "refactor" column would time a
    // sparse factorization of a dense pattern rather than the dense LU the
    // Woodbury update is being compared against.
    const num::auto_linear_options dense{.dense_limit = n};
    const num::auto_linear_solver base_factor(base, dense);
    num::vec right_hand_side(n, 0.0);
    for (num::idx i = 0; i < n; ++i)
        right_hand_side[i] = 1.0 + static_cast<double>(i) / n;

    num::array<num::idx> order(n);
    for (num::idx i = 0; i < n; ++i)
        order[i] = (17 + 53 * i) % n;
    const num::array<num::idx> counts{1, 2, 4, 8, 12, 16, 24, 32, 48, 64};

    num::array<double> ranks, rebuild_times, update_times;
    std::cout << "Dense LU reuse, n=" << n
              << "\n\nchanged  rank  refactor (ms)  update (ms)  speedup  error\n";
    for (num::idx count : counts) {
        if (count > n)
            break;
        const num::view<const num::idx> changed(order.data(), count);
        const num::spmat current = make_matrix(n, changed);
        const auto delta = else_sim::row_column_delta(base, current, changed);
        const auto update = else_sim::make_factor_update(base_factor, delta.left, delta.right);

        num::vec reference;
        num::auto_linear_solver(current, dense).solve_transpose(right_hand_side, reference);
        const num::vec reused = else_sim::solve_transpose(update, right_hand_side);
        const double error = maximum_error(reference, reused);
        if (error > 1e-10)
            throw std::runtime_error("factor update failed its accuracy check");

        const double rebuild = milliseconds(
            [&] {
                volatile num::auto_linear_solver factor(current, dense);
                (void)factor;
            },
            repetitions);
        const double correction = milliseconds(
            [&] {
                volatile auto factor =
                    else_sim::make_factor_update(base_factor, delta.left, delta.right);
                (void)factor;
            },
            repetitions);

        std::cout << std::fixed << std::setprecision(3) << std::setw(7) << count << std::setw(6)
                  << 2 * count << std::setw(15) << rebuild << std::setw(13) << correction
                  << std::setw(9) << rebuild / correction << "  " << std::scientific << error
                  << '\n';
        ranks.push_back(static_cast<double>(count));
        rebuild_times.push_back(rebuild);
        update_times.push_back(correction);
    }

    num::plt::subplot(1, 2);
    num::plt::semilogy();
    num::plt::plot(ranks, rebuild_times, "refactor", "linespoints lw 2");
    num::plt::plot(ranks, update_times, "Woodbury reuse", "linespoints lw 2");
    num::plt::title("Factorization versus low-rank reuse");
    num::plt::xlabel("replaced states (update rank / 2)");
    num::plt::ylabel("milliseconds (log scale)");
    num::plt::legend();

    num::array<double> speedup(ranks.size());
    for (num::idx i = 0; i < ranks.size(); ++i)
        speedup[i] = rebuild_times[i] / update_times[i];
    num::plt::next();
    num::plt::plot(ranks, speedup, "LU", "linespoints lw 2");
    num::plt::title("Reuse speedup");
    num::plt::xlabel("replaced states (update rank / 2)");
    num::plt::ylabel("refactor time / reuse time");
    num::plt::savefig("dense_factor_update.png");
}
