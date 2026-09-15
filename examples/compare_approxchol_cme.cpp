#include "markovkit.hpp"
#include "stochastic/rng.hpp"
#include <approxchol/approxchol.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

class ApproxCholPreconditioner final {
  public:
    using domain_type = num::vec;
    using codomain_type = num::vec;
    using math_laws = num::math::type_list<num::law::spd>;

    explicit ApproxCholPreconditioner(approxchol::CholeskyFactor<double> factor)
        : factor_(std::move(factor)), n_(factor_.order.size()), scratch_(n_, 0.0) {}

    [[nodiscard]] num::idx rows() const noexcept { return n_; }
    [[nodiscard]] num::idx cols() const noexcept { return n_; }

    void apply(const num::vec &r, num::vec &z) const {
        if (z.size() != n_) {
            z = num::vec(n_, 0.0);
        }
        approxchol::solve(factor_, r.data(), z.data(), scratch_);
    }

  private:
    approxchol::CholeskyFactor<double> factor_;
    num::idx n_;
    mutable num::array<double> scratch_;
};

template <>
struct num::math::claims_of<ApproxCholPreconditioner> {
    using type = type_list<law::linear_map>;
};

int main() {
    const num::array<num::idx> sizes = {100, 500, 2000, 5000};

    for (num::idx n : sizes) {
        std::cout << "N = " << n << ":\n";

        // Build a 1D reversible birth-death Laplacian graph
        approxchol::Graph<double> G(n);
        const double birth = 10.0;
        const double death = 1.0;

        for (num::idx i = 0; i + 1 < n; ++i) {
            double w = std::sqrt((i + 1) * birth * death);
            approxchol::add_edge(G, i, i + 1, w);
        }
        G[0].push_back({0, 1.0, 1});
        G[n - 1].push_back({n - 1, 1.0, 1});

        num::mat a_dense(n, n, 0.0);
        for (num::idx u = 0; u < n; ++u) {
            double diag = 0.0;
            for (const auto &e : G[u]) {
                if (e.to != u) {
                    a_dense(u, e.to) -= e.weight;
                }
                diag += e.weight;
            }
            a_dense(u, u) = diag;
        }

        num::vec x_exact(n, 1.0);
        num::vec b(n, 0.0);
        for (num::idx i = 0; i < n; ++i) {
            for (num::idx j = 0; j < n; ++j) {
                b[i] += a_dense(i, j) * x_exact[j];
            }
        }

        num::operators::spd_op<num::operators::dense_op> A_op{num::operators::dense_op(a_dense)};

        if (n <= 2000) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto factor = num::cholesky(num::assume_spd(a_dense));
            num::vec x_dense(n, 0.0);
            num::cholesky_solve(factor, b, x_dense);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  Dense Cholesky : " << std::fixed << std::setprecision(2) << ms
                      << " ms\n";
        }

        {
            num::vec x_cg(n, 0.0);
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = num::cg(A_op, b, x_cg, 1e-8, 20000);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  Unprecond CG   : " << std::fixed << std::setprecision(2) << ms
                      << " ms (" << res.iterations << " iters)\n";
        }

        {
            auto jacobi = num::make_jacobi_preconditioner(a_dense);
            num::vec x_jacobi(n, 0.0);
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = num::pcg(A_op, jacobi, b, x_jacobi, 1e-8, 20000);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  Jacobi PCG     : " << std::fixed << std::setprecision(2) << ms
                      << " ms (" << res.iterations << " iters)\n";
        }

        {
            num::rng64 rng(42);
            auto t0 = std::chrono::high_resolution_clock::now();
            auto factor = approxchol::ac2(G, rng);
            ApproxCholPreconditioner prec(std::move(factor));
            num::vec x_ac(n, 0.0);
            auto res = num::pcg(A_op, prec, b, x_ac, 1e-8, 1000);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  ApproxChol PCG : " << std::fixed << std::setprecision(2) << ms
                      << " ms (" << res.iterations << " iters)\n";
        }
    }

    return 0;
}
