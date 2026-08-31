#include "else/factor_update.hpp"
#include "plot/plot.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Matrix = else_sim::Matrix<double>;
using Vector = std::vector<double>;

Matrix make_matrix(std::size_t n) {
    Matrix matrix(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        matrix(i, i) = 2.0;
        for (std::size_t j = 0; j < i; ++j) {
            const double value = -0.002 / (1.0 + std::abs(static_cast<double>(i) - j));
            matrix(i, j) = value;
            matrix(j, i) = value;
        }
    }
    return matrix;
}

Matrix replace_states(Matrix matrix, std::span<const std::size_t> changed) {
    for (std::size_t state : changed) {
        for (std::size_t j = 0; j < matrix.rows(); ++j) {
            if (j == state)
                continue;
            const double distance = std::abs(static_cast<double>(state) - j);
            const double value = -0.004 / (1.0 + distance);
            matrix(state, j) = value;
            matrix(j, state) = value;
        }
        matrix(state, state) = 2.5;
    }
    return matrix;
}

double maximum_error(const Vector &left, const Vector &right) {
    double error = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i)
        error = std::max(error, std::abs(left[i] - right[i]));
    return error;
}

template <typename Work>
double milliseconds(Work work, int repetitions) {
    const auto start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition)
        work();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() / repetitions;
}

struct Result {
    std::size_t changed;
    double lu_rebuild;
    double lu_update;
    double cholesky_rebuild;
    double cholesky_update;
    double error;
};

Result measure(const Matrix &base, const else_sim::LUFactor<double> &base_lu,
               const else_sim::CholeskyFactor<double> &base_cholesky, const Vector &right_hand_side,
               std::span<const std::size_t> changed, int repetitions) {
    const Matrix updated = replace_states(base, changed);
    const auto solve_lu = [&](const Vector &b, Vector &x) { else_sim::lu_solve(base_lu, b, x); };
    const auto solve_cholesky = [&](const Vector &b, Vector &x) {
        else_sim::cholesky_solve(base_cholesky, b, x);
    };

    const auto updated_lu = else_sim::factorize_lu(updated);
    Vector reference;
    else_sim::lu_solve(updated_lu, right_hand_side, reference);
    const auto lu_correction = else_sim::factor_correction(base, updated, changed, solve_lu);
    const auto cholesky_correction =
        else_sim::factor_correction(base, updated, changed, solve_cholesky);
    Vector lu_solution, cholesky_solution;
    else_sim::solve_correction(lu_correction, right_hand_side, lu_solution, solve_lu);
    else_sim::solve_correction(cholesky_correction, right_hand_side, cholesky_solution,
                               solve_cholesky);
    const double error = std::max(maximum_error(reference, lu_solution),
                                  maximum_error(reference, cholesky_solution));
    if (error > 1e-11)
        throw std::runtime_error("dense factor update failed its accuracy check");

    return {
        .changed = changed.size(),
        .lu_rebuild = milliseconds(
            [&] {
                volatile auto factor = else_sim::factorize_lu(updated);
                (void)factor;
            },
            repetitions),
        .lu_update = milliseconds(
            [&] {
                volatile auto correction =
                    else_sim::factor_correction(base, updated, changed, solve_lu);
                (void)correction;
            },
            repetitions),
        .cholesky_rebuild = milliseconds(
            [&] {
                volatile auto factor = else_sim::factorize_cholesky(updated);
                (void)factor;
            },
            repetitions),
        .cholesky_update = milliseconds(
            [&] {
                volatile auto correction =
                    else_sim::factor_correction(base, updated, changed, solve_cholesky);
                (void)correction;
            },
            repetitions),
        .error = error,
    };
}

} // namespace

int main(int argc, char **argv) {
    const std::size_t n = argc > 1 ? std::stoul(argv[1]) : 240;
    constexpr int repetitions = 8;
    const Matrix base = make_matrix(n);
    Vector right_hand_side(n);
    for (std::size_t i = 0; i < n; ++i)
        right_hand_side[i] = 1.0 + double(i) / n;

    const auto base_lu = else_sim::factorize_lu(base);
    const auto base_cholesky = else_sim::factorize_cholesky(base);

    std::vector<std::size_t> replacement_order(n);
    for (std::size_t i = 0; i < n; ++i)
        replacement_order[i] = (17 + 53 * i) % n;
    const std::vector<std::size_t> counts = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64};

    std::cout << "Dense factor reuse, n=" << n << "\n\n"
              << "changed rank  LU rebuild  LU update  speedup  Chol rebuild  Chol update  speedup "
                 " error\n";
    std::vector<double> ranks;
    std::vector<double> lu_rebuild, lu_update, chol_rebuild, chol_update;
    for (std::size_t count : counts) {
        if (count > n)
            break;
        const Result result =
            measure(base, base_lu, base_cholesky, right_hand_side,
                    std::span<const std::size_t>(replacement_order.data(), count), repetitions);
        std::cout << std::fixed << std::setprecision(3) << std::setw(7) << result.changed << ' '
                  << std::setw(4) << 2 * result.changed << "  " << std::setw(10)
                  << result.lu_rebuild << "  " << std::setw(9) << result.lu_update << "  "
                  << std::setw(7) << result.lu_rebuild / result.lu_update << "  " << std::setw(12)
                  << result.cholesky_rebuild << "  " << std::setw(11) << result.cholesky_update
                  << "  " << std::setw(7) << result.cholesky_rebuild / result.cholesky_update
                  << "  " << std::scientific << result.error << '\n';
        ranks.push_back(static_cast<double>(result.changed));
        lu_rebuild.push_back(result.lu_rebuild);
        lu_update.push_back(result.lu_update);
        chol_rebuild.push_back(result.cholesky_rebuild);
        chol_update.push_back(result.cholesky_update);
    }

    // The figure shows the crossover directly: reuse is worthwhile while the
    // low-rank correction remains cheaper than rebuilding the factor.
    num::plt::subplot(1, 2);
    num::plt::semilogy();
    num::plt::plot(ranks, lu_rebuild, "LU refactor", "linespoints lw 2");
    num::plt::plot(ranks, lu_update, "LU reuse", "linespoints lw 2");
    num::plt::plot(ranks, chol_rebuild, "Cholesky refactor", "linespoints lw 2");
    num::plt::plot(ranks, chol_update, "Cholesky reuse", "linespoints lw 2");
    num::plt::title("Factorization versus low-rank reuse");
    num::plt::xlabel("replaced states (update rank / 2)");
    num::plt::ylabel("milliseconds (log scale)");
    num::plt::legend();

    std::vector<double> lu_speedup, chol_speedup;
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        lu_speedup.push_back(lu_rebuild[i] / lu_update[i]);
        chol_speedup.push_back(chol_rebuild[i] / chol_update[i]);
    }
    num::plt::next();
    num::plt::plot(ranks, lu_speedup, "LU", "linespoints lw 2");
    num::plt::plot(ranks, chol_speedup, "Cholesky", "linespoints lw 2");
    num::plt::title("Reuse speedup");
    num::plt::xlabel("replaced states (update rank / 2)");
    num::plt::ylabel("refactor time / reuse time");
    num::plt::legend();
    num::plt::savefig("dense_factor_update.png");
    std::cout << "Saved dense_factor_update.png\n";
}
