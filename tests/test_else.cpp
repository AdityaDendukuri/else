#include "else/else.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_woodbury_safe_add() {
    double x = 1.0;
    double err = 0.0;

    // Normal positive updates
    assert(else_sim::safe_add(x, err, -0.2, 1e-6));
    assert(std::abs(x - 0.8) < 1e-15);
    assert(err > 0.0);

    // Negative / non-positive result must fail
    assert(!else_sim::safe_add(x, err, -0.9, 1e-6));

    // Accumulated roundoff breach
    double y = 1.0;
    double y_err = 0.0;
    for (int i = 0; i < 1000; ++i) {
        else_sim::safe_add(y, y_err, 1e-3, 1e-6);
    }
    assert(y > 1.0);

    std::cout << "[PASS] test_woodbury_safe_add\n";
}

void test_subnetwork_solves_and_cut_time() {
    // 2-state test system:
    // -R = [2 -1; -1 3], det = 5
    // Z = (-R)^-1 = [0.6 0.2; 0.2 0.4]
    // u = Z * [1, 0]^T = [0.6, 0.2]
    // q = Z^T * 1 = [0.8, 0.6]
    // Single-state cut losses: Delta T = [0.8, 0.3]

    std::vector<std::vector<int>> states = {{0}, {1}};
    const std::vector<num::idx> rows = {0, 0, 1, 1};
    const std::vector<num::idx> cols = {0, 1, 0, 1};
    const std::vector<double> vals = {-2.0, 1.0, 1.0, -3.0};
    auto R = num::SparseMatrix::from_triplets(2, 2, rows, cols, vals);

    std::vector<else_sim::BoundaryTransition<std::size_t, std::vector<int>, double>> boundary = {
        {0, {100}, 1.0}, // State 0 exits at rate 1.0
        {1, {100}, 2.0}, // State 1 exits at rate 2.0
    };

    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(states, std::move(R), std::move(boundary));

    num::Vector p0(2, 0.0);
    p0[0] = 1.0;
    auto u = sub.occupation(p0);
    assert(std::abs(u[0] - 0.6) < 1e-12);
    assert(std::abs(u[1] - 0.2) < 1e-12);

    // Verify cut-time losses
    auto delta_t = sub.cut_time_losses(u);
    assert(std::abs(delta_t[0] - 0.8) < 1e-12);
    assert(std::abs(delta_t[1] - 0.3) < 1e-12);

    // Verify Woodbury vs Naive ground truth
    assert(std::abs(sub.naive_cut_time_loss(u, std::vector<std::size_t>{0}) - 0.8) < 1e-12);
    assert(std::abs(sub.naive_cut_time_loss(u, std::vector<std::size_t>{1}) - 0.3) < 1e-12);
    assert(std::abs(sub.naive_cut_time_loss(u, std::vector<std::size_t>{0, 1}) - 0.8) < 1e-12);

    // Verify backward residual diagnostics
    auto diag = sub.cut_time_loss_with_diagnostics(u, std::vector<std::size_t>{0, 1});
    assert(std::abs(diag.loss - 0.8) < 1e-12);
    assert(diag.estimated_error < 1e-12);
    assert(!diag.naive_fallback_used);

    std::cout << "[PASS] test_subnetwork_solves_and_cut_time\n";
}

void test_state_shedding() {
    std::vector<std::vector<int>> states = {{0}, {1}, {2}};
    const std::vector<num::idx> rows = {0, 0, 1, 1, 1, 2, 2};
    const std::vector<num::idx> cols = {0, 1, 0, 1, 2, 1, 2};
    const std::vector<double> vals = {-2.0, 1.0, 1.0, -3.0, 1.0, 1.0, -2.0};
    auto R = num::SparseMatrix::from_triplets(3, 3, rows, cols, vals);

    std::vector<else_sim::BoundaryTransition<std::size_t, std::vector<int>, double>> boundary = {
        {0, {100}, 1.0},
        {2, {100}, 1.0},
    };

    else_sim::Subnetwork<double, std::size_t, std::vector<int>> sub(states, std::move(R), std::move(boundary));
    num::Vector p0(3, 0.0);
    p0[0] = 1.0;
    auto u = sub.occupation(p0);

    // Shed down to capacity 2 with state 0 protected
    else_sim::SheddingOptions<std::size_t, double> opts{
        .method = else_sim::SheddingMethod::ExpectedVisits,
        .target_capacity = 2,
    };
    auto result = else_sim::shed_states(sub, u, {0}, opts);
    assert(result.kept_indices.size() == 2);
    assert(result.shed_indices.size() == 1);
    assert(result.kept_indices[0] == 0); // State 0 was kept

    std::cout << "[PASS] test_state_shedding\n";
}

int main() {
    std::cout << "=== Running ELSE Pure Template Library Tests ===\n";
    test_woodbury_safe_add();
    test_subnetwork_solves_and_cut_time();
    test_state_shedding();
    std::cout << "=== All ELSE tests passed successfully! ===\n";
    return 0;
}
