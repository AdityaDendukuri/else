// Talbot inversion of the resolvent against a Krylov reference on a random
// Markov generator: error in the node count and in time, and the ranked density.
#include "markovkit.hpp"
#include <numerics.hpp>

namespace {

num::vec reference(const num::mat &Q, const num::vec &p0, double t) {
    return num::expv(t, num::operators::dense_op(Q), p0, 50, 1e-15);
}

num::vec talbot(const num::hessenberg_resolvent_solver &solver, const num::vec &p0, double t,
                num::idx nodes) {
    num::array<num::cplx> density(solver.size(), num::cplx(0.0, 0.0));
    num::inverse_laplace_accumulate(t, nodes, [&](num::cplx shift, num::cplx weight) {
        const auto solution = solver.solve(shift, p0);
        for (num::idx i = 0; i < density.size(); ++i)
            density[i] += weight * solution[i];
    });
    num::vec p(solver.size(), 0.0);
    for (num::idx i = 0; i < p.size(); ++i)
        p[i] = std::max(0.0, density[i].real());
    num::clip_and_normalize_nonnegative(p);
    return p;
}

double max_error(const num::vec &p, const num::vec &q) {
    double worst = 1e-16;
    for (num::idx i = 0; i < p.size(); ++i)
        worst = std::max(worst, std::abs(p[i] - q[i]));
    return worst;
}

double l1_error(const num::vec &p, const num::vec &q) {
    double total = 0.0;
    for (num::idx i = 0; i < p.size(); ++i)
        total += std::abs(p[i] - q[i]);
    return std::max(1e-16, total);
}

} // namespace

int main() {
    using namespace num;
    const idx n = 120;
    rng64 random(12345);
    const mat Q = linear::dense_markov_generator(
        structures::erdos_renyi(n, 0.08, random, true, 0.5, 2.0), true);
    const vec p0 = unit_vector(n, 0);
    const hessenberg_resolvent_solver solver(Q);
    const vec exact = reference(Q, p0, 1.0);

    array<double> nodes, error_inf, error_1;
    for (idx m : {4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32}) {
        const vec p = talbot(solver, p0, 1.0, m);
        num::append(nodes, static_cast<double>(m));
        num::append(error_inf, max_error(p, exact));
        num::append(error_1, l1_error(p, exact));
    }
    const auto times = logspace(-2.0, 1.0, 25);
    array<double> error_8, error_14, error_24;
    for (double t : times) {
        const vec p = reference(Q, p0, t);
        num::append(error_8, max_error(talbot(solver, p0, t, 8), p));
        num::append(error_14, max_error(talbot(solver, p0, t, 14), p));
        num::append(error_24, max_error(talbot(solver, p0, t, 24), p));
    }
    array<idx> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](idx a, idx b) { return exact[a] > exact[b]; });
    const vec p14 = talbot(solver, p0, 1.0, 14);
    array<double> rank(n), exact_sorted(n), talbot_sorted(n);
    for (idx i = 0; i < n; ++i) {
        rank[i] = static_cast<double>(i + 1);
        exact_sorted[i] = exact[order[i]];
        talbot_sorted[i] = p14[order[i]];
    }

    plt::subplot(1, 3);
    plt::plot(nodes, error_inf, "L_inf error", "linespoints pt 7 lw 2 lc rgb '#1f77b4'");
    plt::plot(nodes, error_1, "L_1 error", "linespoints pt 5 lw 2 lc rgb '#ff7f0e'");
    plt::title("Talbot convergence in the node count (t = 1)");
    plt::xlabel("nodes");
    plt::ylabel("error against the Krylov reference");
    plt::semilogy(), plt::legend(), plt::next();
    plt::plot(times, error_8, "M = 8", "lines lw 2 lc rgb '#d62728'");
    plt::plot(times, error_14, "M = 14", "lines lw 2 lc rgb '#2ca02c'");
    plt::plot(times, error_24, "M = 24", "lines lw 2 lc rgb '#9467bd'");
    plt::title("Error against time");
    plt::xlabel("time");
    plt::ylabel("L_inf error");
    plt::loglog(), plt::legend(), plt::next();
    plt::plot(rank, exact_sorted, "Krylov reference", "lines lw 3 lc rgb '#000000'");
    plt::plot(rank, talbot_sorted, "Talbot (M = 14)", "points pt 6 ps 0.8 lc rgb '#1f77b4'");
    plt::title("Rank-ordered density at t = 1");
    plt::xlabel("state rank");
    plt::ylabel("probability");
    plt::legend(), plt::savefig("talbot_vs_fsp_laplacian.png");
}
