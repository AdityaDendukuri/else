#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "markovkit.hpp"
#include "linalg/sparse/sparse_op.hpp"
#include <approxchol/approxchol.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

class ApproxCholPreconditioner final {
  public:
    explicit ApproxCholPreconditioner(approxchol::CholeskyFactor<double> factor)
        : factor_(std::move(factor)), n_(factor_.order.size()), scratch_(n_, 0.0) {}

    [[nodiscard]] num::idx rows() const noexcept { return n_; }
    [[nodiscard]] num::idx cols() const noexcept { return n_; }

    void apply(const num::Vector &r, num::Vector &z) const {
        if (z.size() != n_) {
            z = num::Vector(n_, 0.0);
        }
        approxchol::solve(factor_, r.data(), z.data(), scratch_);
    }

  private:
    approxchol::CholeskyFactor<double> factor_;
    num::idx n_;
    mutable std::vector<double> scratch_;
};

approxchol::Graph<double> graph_from_sparse(const num::SparseMatrix &L) {
    const num::idx n = L.n_rows();
    approxchol::Graph<double> G(n);

    for (num::idx u = 0; u < n; ++u) {
        double off_diag_sum = 0.0;
        double diag_val = 0.0;
        for (num::idx k = L.row_ptr()[u]; k < L.row_ptr()[u + 1]; ++k) {
            num::idx v = L.col_idx()[k];
            double val = L.values()[k];
            if (u == v) {
                diag_val = val;
            } else if (u < v) {
                double w = -val;
                if (w > 0.0) {
                    approxchol::add_edge(G, u, v, w);
                }
            }
            if (u != v) {
                off_diag_sum += -val;
            }
        }
        double excess = diag_val - off_diag_sum;
        if (excess > 1e-12) {
            G[u].push_back({u, excess, 1});
        }
    }
    return G;
}

void run_benchmark_on_file(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "File not found: " << path << '\n';
        return;
    }

    const auto data = num::io::read_json(path);
    const auto &L_data = data.contains("L") ? data["L"] : data;
    const auto L_sparse = num::io::sparse_matrix(L_data);

    const num::idx n = L_sparse.n_rows();
    const num::idx nnz = L_sparse.nnz();
    std::cout << path << " (N=" << n << ", nnz=" << nnz << "):\n";

    auto G = graph_from_sparse(L_sparse);
    num::operators::SPDOp<num::operators::SparseOp> A_op{num::operators::SparseOp(L_sparse)};

    num::Vector x_true(n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        x_true[i] = std::sin(static_cast<double>(i) * 0.1);
    }
    num::Vector b(n, 0.0);
    A_op.apply(x_true, b);

    const double b_norm = num::norm(b);
    const double tol = std::max(1e-12, 1e-6 * b_norm);

    if (n <= 30000) {
        num::Vector x_cg(n, 0.0);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = num::cg(A_op, b, x_cg, tol, 50000);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  Unpreconditioned CG : " << std::fixed << std::setprecision(2) << ms
                  << " ms (" << res.iterations << " iters)\n";
    }

    if (n <= 30000) {
        num::Vector inv_diag(n, 0.0);
        for (num::idx i = 0; i < n; ++i) {
            for (num::idx k = L_sparse.row_ptr()[i]; k < L_sparse.row_ptr()[i + 1]; ++k) {
                if (L_sparse.col_idx()[k] == i && L_sparse.values()[k] > 0.0) {
                    inv_diag[i] = 1.0 / L_sparse.values()[k];
                }
            }
        }
        num::JacobiPreconditioner jacobi(std::move(inv_diag));
        num::Vector x_jacobi(n, 0.0);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = num::pcg(A_op, jacobi, b, x_jacobi, tol, 50000);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  Jacobi PCG          : " << std::fixed << std::setprecision(2) << ms
                  << " ms (" << res.iterations << " iters)\n";
    }

    {
        std::mt19937_64 rng(42);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto factor = approxchol::ac1(G, rng);
        ApproxCholPreconditioner prec(std::move(factor));
        num::Vector x_ac1(n, 0.0);
        auto res = num::pcg(A_op, prec, b, x_ac1, tol, 1000);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  approxchol (AC1) PCG: " << std::fixed << std::setprecision(2) << ms
                  << " ms (" << res.iterations << " iters)\n";
    }

    {
        std::mt19937_64 rng(42);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto factor = approxchol::ac2(G, rng);
        ApproxCholPreconditioner prec(std::move(factor));
        num::Vector x_ac2(n, 0.0);
        auto res = num::pcg(A_op, prec, b, x_ac2, tol, 1000);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  approxchol (AC2) PCG: " << std::fixed << std::setprecision(2) << ms
                  << " ms (" << res.iterations << " iters)\n";
    }
}

int main() {
    run_benchmark_on_file("laplacians/small-laplacian.json");
    run_benchmark_on_file("laplacians/medium-laplacian.json");
    run_benchmark_on_file("laplacians/large-laplacian.json");
    return 0;
}
