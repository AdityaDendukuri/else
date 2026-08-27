#include "else/linalg.hpp"
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

struct Correction {
    Matrix transformed_columns;
    Matrix rows;
    else_sim::LUFactor<double> small_factor;
};

template <typename Solve>
Correction factor_correction(const Matrix &base, const Matrix &updated,
                             std::span<const std::size_t> changed, Solve solve_base) {
    const std::size_t n = base.rows();
    const std::size_t rank = 2 * changed.size();
    Matrix columns(n, rank, 0.0);
    Matrix rows(rank, n, 0.0);
    std::vector<bool> changed_row(n, false);
    for (std::size_t state : changed) changed_row[state] = true;

    // A - A0 = E Delta[S,:] + Delta[:,S] E^T.  The second term omits
    // rows S because their entries were already supplied by the first term.
    for (std::size_t k = 0; k < changed.size(); ++k) {
        const std::size_t state = changed[k];
        columns(state, k) = 1.0;
        rows(changed.size() + k, state) = 1.0;
        for (std::size_t i = 0; i < n; ++i) {
            rows(k, i) = updated(state, i) - base(state, i);
            if (!changed_row[i])
                columns(i, changed.size() + k) = updated(i, state) - base(i, state);
        }
    }

    Matrix transformed(n, rank);
    Vector rhs(n), solution;
    for (std::size_t column = 0; column < rank; ++column) {
        for (std::size_t i = 0; i < n; ++i) rhs[i] = columns(i, column);
        solve_base(rhs, solution);
        for (std::size_t i = 0; i < n; ++i) transformed(i, column) = solution[i];
    }

    Matrix small(rank, rank, 0.0);
    for (std::size_t i = 0; i < rank; ++i) {
        small(i, i) = 1.0;
        for (std::size_t j = 0; j < rank; ++j)
            for (std::size_t k = 0; k < n; ++k)
                small(i, j) += rows(i, k) * transformed(k, j);
    }
    auto factor = else_sim::factorize_lu(std::move(small));
    if (factor.singular) throw std::runtime_error("singular dense update correction");
    return {std::move(transformed), std::move(rows), std::move(factor)};
}

template <typename Solve>
Vector solve(const Correction &correction, const Vector &right_hand_side, Solve solve_base) {
    Vector base_solution, reduced(correction.rows.rows(), 0.0), coefficients, solution;
    solve_base(right_hand_side, base_solution);
    for (std::size_t i = 0; i < correction.rows.rows(); ++i)
        for (std::size_t j = 0; j < correction.rows.cols(); ++j)
            reduced[i] += correction.rows(i, j) * base_solution[j];
    else_sim::lu_solve(correction.small_factor, reduced, coefficients);

    solution = base_solution;
    for (std::size_t i = 0; i < solution.size(); ++i)
        for (std::size_t j = 0; j < coefficients.size(); ++j)
            solution[i] -= correction.transformed_columns(i, j) * coefficients[j];
    return solution;
}

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
            if (j == state) continue;
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
    for (int repetition = 0; repetition < repetitions; ++repetition) work();
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
               const else_sim::CholeskyFactor<double> &base_cholesky,
               const Vector &right_hand_side, std::span<const std::size_t> changed,
               int repetitions) {
    const Matrix updated = replace_states(base, changed);
    const auto solve_lu = [&](const Vector &b, Vector &x) { else_sim::lu_solve(base_lu, b, x); };
    const auto solve_cholesky = [&](const Vector &b, Vector &x) {
        else_sim::cholesky_solve(base_cholesky, b, x);
    };

    const auto updated_lu = else_sim::factorize_lu(updated);
    Vector reference;
    else_sim::lu_solve(updated_lu, right_hand_side, reference);
    const auto lu_correction = factor_correction(base, updated, changed, solve_lu);
    const auto cholesky_correction = factor_correction(base, updated, changed, solve_cholesky);
    const double error = std::max(
        maximum_error(reference, solve(lu_correction, right_hand_side, solve_lu)),
        maximum_error(reference, solve(cholesky_correction, right_hand_side, solve_cholesky)));
    if (error > 1e-11)
        throw std::runtime_error("dense factor update failed its accuracy check");

    return {
        .changed = changed.size(),
        .lu_rebuild = milliseconds([&] {
            volatile auto factor = else_sim::factorize_lu(updated);
            (void)factor;
        }, repetitions),
        .lu_update = milliseconds([&] {
            volatile auto correction = factor_correction(base, updated, changed, solve_lu);
            (void)correction;
        }, repetitions),
        .cholesky_rebuild = milliseconds([&] {
            volatile auto factor = else_sim::factorize_cholesky(updated);
            (void)factor;
        }, repetitions),
        .cholesky_update = milliseconds([&] {
            volatile auto correction = factor_correction(base, updated, changed, solve_cholesky);
            (void)correction;
        }, repetitions),
        .error = error,
    };
}

} // namespace

int main(int argc, char **argv) {
    const std::size_t n = argc > 1 ? std::stoul(argv[1]) : 240;
    constexpr int repetitions = 8;
    const Matrix base = make_matrix(n);
    Vector right_hand_side(n);
    for (std::size_t i = 0; i < n; ++i) right_hand_side[i] = 1.0 + double(i) / n;

    const auto base_lu = else_sim::factorize_lu(base);
    const auto base_cholesky = else_sim::factorize_cholesky(base);

    std::vector<std::size_t> replacement_order(n);
    for (std::size_t i = 0; i < n; ++i) replacement_order[i] = (17 + 53 * i) % n;
    const std::vector<std::size_t> counts = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64};

    std::cout << "Dense factor reuse, n=" << n << "\n\n"
              << "changed rank  LU rebuild  LU update  speedup  Chol rebuild  Chol update  speedup  error\n";
    for (std::size_t count : counts) {
        if (count > n) break;
        const Result result = measure(base, base_lu, base_cholesky, right_hand_side,
                                      std::span<const std::size_t>(replacement_order.data(), count),
                                      repetitions);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(7) << result.changed << ' '
                  << std::setw(4) << 2 * result.changed << "  "
                  << std::setw(10) << result.lu_rebuild << "  "
                  << std::setw(9) << result.lu_update << "  "
                  << std::setw(7) << result.lu_rebuild / result.lu_update << "  "
                  << std::setw(12) << result.cholesky_rebuild << "  "
                  << std::setw(11) << result.cholesky_update << "  "
                  << std::setw(7) << result.cholesky_rebuild / result.cholesky_update << "  "
                  << std::scientific << result.error << '\n';
    }
}
