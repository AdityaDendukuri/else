#include "else/else.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_unpivoted_and_pivoted_lu() {
    // 2x2 M-matrix: [2, -1; -1, 3]
    else_sim::Matrix<double> A(2, 2);
    A(0, 0) = 2.0; A(0, 1) = -1.0;
    A(1, 0) = -1.0; A(1, 1) = 3.0;

    // 1. Unpivoted LU
    auto lu_unpiv = else_sim::factorize_lu(A, false);
    assert(!lu_unpiv.pivoted);
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

    // 2. Pivoted LU
    auto lu_piv = else_sim::factorize_lu(A, true);
    assert(lu_piv.pivoted);
    std::vector<double> x_piv(2);
    else_sim::lu_solve(lu_piv, b, x_piv);
    assert(std::abs(x_piv[0] - 0.6) < 1e-15);
    assert(std::abs(x_piv[1] - 0.2) < 1e-15);

    std::cout << "[PASS] test_unpivoted_and_pivoted_lu\n";
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
    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(states, std::move(R), std::move(boundary), false);

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

    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(states, std::move(R), std::move(boundary));
    std::vector<double> p0 = {1.0, 0.0, 0.0};
    auto u = sub.occupation(p0);

    else_sim::SheddingOptions<std::size_t, double> opts{
        .method = else_sim::SheddingMethod::ExpectedVisits,
        .target_capacity = 2,
    };
    auto result = else_sim::shed_states(sub, u, {0}, opts);
    assert(result.kept_indices.size() == 2);
    assert(result.shed_indices.size() == 1);
    assert(result.kept_indices[0] == 0);

    std::cout << "[PASS] test_state_shedding\n";
}

void test_talbot_density() {
    std::vector<std::vector<int>> states = {{0}, {1}};
    const std::vector<std::size_t> rows = {0, 0, 1, 1};
    const std::vector<std::size_t> cols = {0, 1, 0, 1};
    const std::vector<double> vals = {-2.0, 1.0, 1.0, -2.0};
    auto R = else_sim::SparseMatrix<double>::from_triplets(2, 2, rows, cols, vals);

    std::vector<else_sim::BoundaryTransition<std::size_t, std::vector<int>, double>> boundary;
    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(states, std::move(R), std::move(boundary));

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
    std::cout << "=== Running ELSE Pure Template Library Tests (Zero Dependencies) ===\n";
    test_unpivoted_and_pivoted_lu();
    test_woodbury_safe_add();
    test_subnetwork_solves_and_cut_time();
    test_state_shedding();
    test_talbot_density();
    std::cout << "=== All ELSE tests passed successfully! ===\n";
    return 0;
}
