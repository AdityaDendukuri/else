#include "markovkit.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numerics.hpp>
#include <random>
#include <vector>

namespace {

num::vec compute_krylov_reference(const num::mat &Q, const num::vec &p0, double t) {
    num::operators::dense_op Q_op(Q);
    return num::expv(t, Q_op, p0, 50, 1e-15);
}

num::vec compute_talbot_density(const num::hessenberg_resolvent_solver &solver, const num::vec &p0,
                                double t, num::idx nodes) {
    const num::idx n = solver.size();
    num::array<num::cplx> density(n, num::cplx(0.0, 0.0));

    num::inverse_laplace_accumulate(t, nodes, [&](num::cplx shift, num::cplx weight) {
        auto sol = solver.solve(shift, p0);
        for (num::idx i = 0; i < n; ++i) {
            density[i] += weight * sol[i];
        }
    });

    num::vec p(n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        p[i] = std::max(0.0, density[i].real());
    }
    num::clip_and_normalize_nonnegative(p);
    return p;
}

std::pair<double, double> state_errors(const num::vec &p, const num::vec &p_ref) {
    double l_inf = 0.0;
    double l_1 = 0.0;
    for (num::idx i = 0; i < p.size(); ++i) {
        double diff = std::abs(p[i] - p_ref[i]);
        l_inf = std::max(l_inf, diff);
        l_1 += diff;
    }
    return {l_inf, l_1};
}

} // namespace

int main() {
    using namespace num;

    const idx N = 120;
    num::rng64 rng(12345);

    // Generate a connected random Markov generator.
    graph G = structures::erdos_renyi(N, 0.08, rng, true, 0.5, 2.0);
    mat Q = linear::dense_markov_generator(G, true);
    vec p0 = unit_vector(N, 0);

    hessenberg_resolvent_solver hess_solver(Q);

    const double t_target = 1.0;
    vec p_exact = compute_krylov_reference(Q, p0, t_target);

    const num::array<idx> node_counts = {4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32};
    num::array<double> node_list_dbl;
    num::array<double> err_inf_list;
    num::array<double> err_l1_list;

    for (idx M : node_counts) {
        vec p_talbot = compute_talbot_density(hess_solver, p0, t_target, M);
        auto [l_inf, l_1] = state_errors(p_talbot, p_exact);

        node_list_dbl.push_back(static_cast<double>(M));
        err_inf_list.push_back(std::max(1e-16, l_inf));
        err_l1_list.push_back(std::max(1e-16, l_1));
    }

    const auto time_grid = logspace(-2.0, 1.0, 25);
    num::array<double> err_t_M8, err_t_M14, err_t_M24;

    for (double t : time_grid) {
        vec p_ex = compute_krylov_reference(Q, p0, t);
        vec p_m8 = compute_talbot_density(hess_solver, p0, t, 8);
        vec p_m14 = compute_talbot_density(hess_solver, p0, t, 14);
        vec p_m24 = compute_talbot_density(hess_solver, p0, t, 24);

        err_t_M8.push_back(std::max(1e-16, state_errors(p_m8, p_ex).first));
        err_t_M14.push_back(std::max(1e-16, state_errors(p_m14, p_ex).first));
        err_t_M24.push_back(std::max(1e-16, state_errors(p_m24, p_ex).first));
    }

    plt::subplot(1, 3);

    // plot_panel 1: Error vs Talbot nodes M
    plt::plot(node_list_dbl, err_inf_list, "L_inf Error", "linespoints pt 7 lw 2 lc rgb '#1f77b4'");
    plt::plot(node_list_dbl, err_l1_list, "L_1 Error", "linespoints pt 5 lw 2 lc rgb '#ff7f0e'");
    plt::title("Talbot Convergence vs Nodes M (t = 1.0)");
    plt::xlabel("Nodes (M)");
    plt::ylabel("Error vs Krylov reference");
    plt::semilogy();
    plt::legend();
    plt::next();

    // plot_panel 2: Error vs Propagation time
    plt::plot(time_grid, err_t_M8, "M = 8", "lines lw 2 lc rgb '#d62728'");
    plt::plot(time_grid, err_t_M14, "M = 14", "lines lw 2 lc rgb '#2ca02c'");
    plt::plot(time_grid, err_t_M24, "M = 24", "lines lw 2 lc rgb '#9467bd'");
    plt::title("Error vs Time t in [0.01, 10.0]");
    plt::xlabel("Time t");
    plt::ylabel("L_inf Error");
    plt::loglog();
    plt::legend();
    plt::next();

    // plot_panel 3: Sorted state probability density profile
    vec p_m14_final = compute_talbot_density(hess_solver, p0, t_target, 14);

    num::array<idx> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](idx a, idx b) { return p_exact[a] > p_exact[b]; });

    num::array<double> rank_indices(N);
    num::array<double> exact_sorted(N);
    num::array<double> talbot_sorted(N);
    for (idx i = 0; i < N; ++i) {
        rank_indices[i] = static_cast<double>(i + 1);
        exact_sorted[i] = p_exact[order[i]];
        talbot_sorted[i] = p_m14_final[order[i]];
    }

    plt::plot(rank_indices, exact_sorted, "Krylov reference", "lines lw 3 lc rgb '#000000'");
    plt::plot(rank_indices, talbot_sorted, "Talbot (M=14)", "points pt 6 ps 0.8 lc rgb '#1f77b4'");
    plt::title("Rank-Ordered Density p(t=1.0)");
    plt::xlabel("State Rank");
    plt::ylabel("Probability p_{(i)}(t)");
    plt::legend();

    plt::savefig("talbot_vs_fsp_laplacian.png");

    return 0;
}
