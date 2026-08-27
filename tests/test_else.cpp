#include "else/else.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_unpivoted_lu() {
    // 2x2 M-matrix: [2, -1; -1, 3]
    else_sim::Matrix<double> A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = -1.0;
    A(1, 0) = -1.0;
    A(1, 1) = 3.0;

    // 1. Unpivoted LU
    auto lu_unpiv = else_sim::factorize_lu(A);
    assert(!lu_unpiv.singular);
    assert(std::abs(lu_unpiv.LU(0, 0) - 2.0) < 1e-15);
    assert(std::abs(lu_unpiv.LU(1, 0) - (-0.5)) < 1e-15); // L(1, 0) = -0.5
    assert(std::abs(lu_unpiv.LU(1, 1) - 2.5) < 1e-15);    // U(1, 1) = 3 - (-0.5)*(-1) = 2.5

    std::vector<double> b = {1.0, 0.0};
    std::vector<double> x(2);
    else_sim::lu_solve(lu_unpiv, b, x);
    // Exact inv: [0.6, 0.2; 0.2, 0.4] => x = [0.6, 0.2]
    assert(std::abs(x[0] - 0.6) < 1e-15);
    assert(std::abs(x[1] - 0.2) < 1e-15);

    std::cout << "[PASS] test_unpivoted_lu\n";
}

void test_block_tridiagonal_lu() {
    constexpr std::size_t n = 40;
    std::vector<std::size_t> rows, cols;
    std::vector<double> values;
    for (std::size_t i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        values.push_back(i + 1 == n ? 2.0 : 3.0);
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            values.push_back(-1.0);
        }
        if (i + 1 < n) {
            rows.push_back(i);
            cols.push_back(i + 1);
            values.push_back(-1.0);
        }
    }
    auto A = else_sim::SparseMatrix<double>::from_triplets(n, n, rows, cols, values);
    std::vector<std::size_t> levels(n);
    for (std::size_t i = 0; i < n; ++i)
        levels[i] = i;
    auto factor = else_sim::factorize_block_lu(A, levels);

    std::vector<double> expected(n);
    for (std::size_t i = 0; i < n; ++i)
        expected[i] = 1.0 + static_cast<double>(i) / n;
    std::vector<double> b(n, 0.0), x;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            b[i] += A.values[p] * expected[A.col_idx[p]];
        }
    }
    else_sim::block_solve(factor, b, x);
    for (std::size_t i = 0; i < n; ++i)
        assert(std::abs(x[i] - expected[i]) < 1e-12);

    else_sim::Matrix<double> B(n, 2), X;
    for (std::size_t i = 0; i < n; ++i) {
        B(i, 0) = b[i];
        B(i, 1) = 2.0 * b[i];
    }
    else_sim::block_solve(factor, B, X);
    for (std::size_t i = 0; i < n; ++i) {
        assert(std::abs(X(i, 0) - expected[i]) < 1e-12);
        assert(std::abs(X(i, 1) - 2.0 * expected[i]) < 1e-12);
    }

    std::vector<std::vector<int>> states(n);
    for (std::size_t i = 0; i < n; ++i)
        states[i] = {static_cast<int>(i)};
    auto generator_values = values;
    for (double &value : generator_values)
        value = -value;
    auto R = else_sim::SparseMatrix<double>::from_triplets(n, n, rows, cols, generator_values);
    std::vector<else_sim::BoundaryTransition<std::size_t, std::vector<int>, double>> boundary;
    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(
        std::move(states), std::move(R), std::move(boundary), std::move(levels));
    assert(sub.uses_sparse_solver());
    auto occupation = sub.occupation(b);
    for (std::size_t i = 0; i < n; ++i)
        assert(std::abs(occupation[i] - expected[i]) < 1e-12);

    std::cout << "[PASS] test_block_tridiagonal_lu\n";
}

void test_block_tridiagonal_cholesky() {
    constexpr std::size_t n = 40;
    std::vector<std::size_t> rows, cols;
    std::vector<double> values;
    for (std::size_t i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        values.push_back(i + 1 == n ? 2.0 : 3.0);
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            values.push_back(-1.0);
        }
        if (i + 1 < n) {
            rows.push_back(i);
            cols.push_back(i + 1);
            values.push_back(-1.0);
        }
    }
    auto A = else_sim::SparseMatrix<double>::from_triplets(n, n, rows, cols, values);
    std::vector<std::size_t> levels(n);
    for (std::size_t i = 0; i < n; ++i)
        levels[i] = i;

    std::vector<double> expected(n);
    for (std::size_t i = 0; i < n; ++i)
        expected[i] = 1.0 + static_cast<double>(i) / n;
    std::vector<double> b(n, 0.0), x;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p)
            b[i] += A.values[p] * expected[A.col_idx[p]];

    auto factor = else_sim::factorize_block_cholesky(A, levels);
    else_sim::block_solve(factor, b, x);
    for (std::size_t i = 0; i < n; ++i)
        assert(std::abs(x[i] - expected[i]) < 1e-12);

    else_sim::Matrix<double> B(n, 2), X;
    for (std::size_t i = 0; i < n; ++i) {
        B(i, 0) = b[i];
        B(i, 1) = 2.0 * b[i];
    }
    else_sim::block_solve(factor, B, X);
    for (std::size_t i = 0; i < n; ++i) {
        assert(std::abs(X(i, 0) - expected[i]) < 1e-12);
        assert(std::abs(X(i, 1) - 2.0 * expected[i]) < 1e-12);
    }

    std::vector<std::vector<int>> states(n);
    for (std::size_t i = 0; i < n; ++i)
        states[i] = {static_cast<int>(i)};
    auto generator_values = values;
    for (double &value : generator_values)
        value = -value;
    auto R = else_sim::SparseMatrix<double>::from_triplets(n, n, rows, cols, generator_values);
    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(
        std::move(states), std::move(R), {}, std::vector<double>(n, 1.0), std::move(levels));
    assert(sub.is_reversible());
    assert(sub.uses_sparse_solver());
    const auto occupation = sub.occupation(b);
    for (std::size_t i = 0; i < n; ++i)
        assert(std::abs(occupation[i] - expected[i]) < 1e-12);

    std::cout << "[PASS] test_block_tridiagonal_cholesky\n";
}

void test_woodbury_safe_add() {
    double x = 1.0;
    double err = 0.0;

    assert(else_sim::safe_add(x, err, -0.2, 1e-6));
    assert(std::abs(x - 0.8) < 1e-15);
    assert(err > 0.0);

    assert(!else_sim::safe_add(x, err, -0.9, 1e-6));

    std::cout << "[PASS] test_woodbury_safe_add\n";
}

void test_subnetwork_solves_and_cut_time() {
    std::vector<std::vector<int>> states = {{0}, {1}};
    const std::vector<std::size_t> rows = {0, 0, 1, 1};
    const std::vector<std::size_t> cols = {0, 1, 0, 1};
    const std::vector<double> vals = {-2.0, 1.0, 1.0, -3.0};
    auto R = else_sim::SparseMatrix<double>::from_triplets(2, 2, rows, cols, vals);

    std::vector<else_sim::BoundaryTransition<std::size_t, std::vector<int>, double>> boundary = {
        {0, {100}, 1.0},
        {1, {100}, 2.0},
    };

    // Construct unpivoted Subnetwork
    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(states, std::move(R),
                                                                    std::move(boundary));

    std::vector<double> p0 = {1.0, 0.0};
    auto u = sub.occupation(p0);
    assert(std::abs(u[0] - 0.6) < 1e-12);
    assert(std::abs(u[1] - 0.2) < 1e-12);

    auto delta_t = sub.cut_time_losses(u);
    assert(std::abs(delta_t[0] - 0.8) < 1e-12);
    assert(std::abs(delta_t[1] - 0.3) < 1e-12);

    auto diag = sub.cut_time_loss_with_diagnostics(u, std::vector<std::size_t>{0, 1});
    assert(std::abs(diag.loss - 0.8) < 1e-12);
    assert(diag.estimated_error < 1e-12);
    assert(!diag.naive_fallback_used);

    std::cout << "[PASS] test_subnetwork_solves_and_cut_time\n";
}

void test_state_shedding() {
    std::vector<std::vector<int>> states = {{0}, {1}, {2}};
    const std::vector<std::size_t> rows = {0, 0, 1, 1, 1, 2, 2};
    const std::vector<std::size_t> cols = {0, 1, 0, 1, 2, 1, 2};
    const std::vector<double> vals = {-2.0, 1.0, 1.0, -3.0, 1.0, 1.0, -2.0};
    auto R = else_sim::SparseMatrix<double>::from_triplets(3, 3, rows, cols, vals);

    std::vector<else_sim::BoundaryTransition<std::size_t, std::vector<int>, double>> boundary = {
        {0, {100}, 1.0},
        {2, {100}, 1.0},
    };

    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(states, std::move(R),
                                                                    std::move(boundary));
    std::vector<double> p0 = {1.0, 0.0, 0.0};
    auto u = sub.occupation(p0);

    std::vector<std::size_t> indices = {0, 1, 2};
    auto scores = else_sim::expected_visit_scores(sub, u, std::span<const std::size_t>(indices));
    std::vector<std::size_t> protected_indices = {0};
    auto shed = else_sim::lowest_scores<double, std::size_t>(scores, protected_indices, 1);
    assert(shed.size() == 1);
    assert(shed[0] != 0);

    std::cout << "[PASS] test_state_shedding\n";
}

void test_talbot_density() {
    std::vector<std::vector<int>> states = {{0}, {1}};
    const std::vector<std::size_t> rows = {0, 0, 1, 1};
    const std::vector<std::size_t> cols = {0, 1, 0, 1};
    const std::vector<double> vals = {-2.0, 1.0, 1.0, -2.0};
    auto R = else_sim::SparseMatrix<double>::from_triplets(2, 2, rows, cols, vals);

    std::vector<else_sim::BoundaryTransition<std::size_t, std::vector<int>, double>> boundary;
    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(states, std::move(R),
                                                                    std::move(boundary));

    std::vector<else_sim::Subnetwork<double, std::size_t, std::vector<int>>> chain;
    chain.push_back(std::move(sub));

    else_sim::LaplaceDensitySolver<double, std::size_t, std::vector<int>> solver(std::move(chain));
    auto sol = solver.solve({0}, 1.0, 16);
    assert(sol.probability.size() == 2);
    assert(sol.probability[0] > 0.0);
    assert(sol.probability[1] > 0.0);

    std::cout << "[PASS] test_talbot_density\n";
}

int main() {
    std::cout << "=== Running ELSE Tests ===\n";
    test_unpivoted_lu();
    test_block_tridiagonal_lu();
    test_block_tridiagonal_cholesky();
    test_woodbury_safe_add();
    test_subnetwork_solves_and_cut_time();
    test_state_shedding();
    test_talbot_density();
    std::cout << "=== All ELSE tests passed successfully! ===\n";
    return 0;
}
