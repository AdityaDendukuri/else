// diag(Z) in the sparse regime by Gaussian probing through an ApproxChol
// preconditioner and a Lanczos matrix square root.
#pragma once

#include "subsweep/linear/factorization.hpp"
#include "linear/eigen/lanczos.hpp"
#include "linear/graph/randommat/preconditioner.hpp"
#include "linear/matrix_properties.hpp"
#include "operator/properties.hpp"
#include "stochastic/probe.hpp"
#include "stochastic/rng.hpp"
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>

namespace subsweep {

namespace detail {

[[nodiscard]] inline num::spmat symmetric_part(const num::spmat &matrix) {
    array<idx> rows, columns;
    array<real> values;
    for (idx row = 0; row < matrix.n_rows(); ++row)
        for (idx entry = matrix.row_ptr()[row]; entry < matrix.row_ptr()[row + 1]; ++entry) {
            const idx column = matrix.col_idx()[entry];
            const real half = 0.5 * matrix.values()[entry];
            num::append(rows, row);
            num::append(columns, column);
            num::append(values, half);
            num::append(rows, column);
            num::append(columns, row);
            num::append(values, half);
        }
    return num::spmat::from_triplets(matrix.n_rows(), matrix.n_cols(), rows, columns, values);
}

// W S W for W = diag(weights).
[[nodiscard]] inline num::spmat congruent(const num::spmat &symmetric, view<const real> weights) {
    array<idx> rows, columns;
    array<real> values;
    for (idx row = 0; row < symmetric.n_rows(); ++row)
        for (idx entry = symmetric.row_ptr()[row]; entry < symmetric.row_ptr()[row + 1]; ++entry) {
            const idx column = symmetric.col_idx()[entry];
            num::append(rows, row);
            num::append(columns, column);
            num::append(values, weights[row] * symmetric.values()[entry] * weights[column]);
        }
    return num::spmat::from_triplets(symmetric.n_rows(), symmetric.n_cols(), rows, columns, values);
}

// S_tilde = C^-1 M C^-T for the ApproxChol factor C of M.
class preconditioned_operator final {
  public:
    using math_laws = num::math::type_list<num::law::linear_map>;
    using domain_type = num::vec;
    using codomain_type = num::vec;
    preconditioned_operator(const num::spmat &matrix, const num::grounded_approx_chol_factor &c)
        : matrix_(matrix), factor_(c) {}
    void apply(const num::vec &input, num::vec &output) const {
        const num::vec upper = factor_.solve_upper(input);
        num::vec product(rows(), 0.0);
        num::sparse_matvec(matrix_, upper, product);
        output = factor_.solve_lower(product);
    }
    [[nodiscard]] idx rows() const noexcept { return matrix_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return matrix_.n_cols(); }

  private:
    const num::spmat &matrix_;
    const num::grounded_approx_chol_factor &factor_;
};

[[nodiscard]] inline num::mat gaussian_probe(idx n, idx count, unsigned seed) {
    num::rng generator(seed);
    std::normal_distribution<real> normal(0.0, 1.0);
    num::mat probe(n, count, 0.0);
    for (idx column = 0; column < count; ++column)
        for (idx row = 0; row < n; ++row)
            probe(row, column) = normal(generator);
    return probe;
}


} // namespace detail

// With R_tilde = D R_bar D^-1 and S its symmetric part, S = K S_tilde K^T
// for the ApproxChol factor K and S_tilde = K^-1 S K^-T, so
//   Z_jj = || delta_j^T R_tilde^-1 K S_tilde^{1/2} ||^2,
// estimated by the row mean square over Gaussian probes. Reversible:
// D = W = sqrt(pi) and R_tilde = S = L. Otherwise D = sqrt(r/q),
// W = sqrt(r q) for q = Z 1 and r = Z^T 1, and W S W = (G + G^T)/2 for
// G = diag(r) R_bar diag(q), a grounded Laplacian.
[[nodiscard]] inline num::vec probed_diagonal(const factorization &f) {
    const idx n = f.size();
    const probe_options &probes = f.probes();
    const bool reversible = f.sparse_weights().size() != 0;
    if (probes.count == 0)
        throw std::invalid_argument("the probe estimate needs at least one probe");
    const num::spmat &matrix = f.sparse_matrix();
    num::vec similarity, congruence;
    if (reversible) {
        similarity = f.sparse_weights();
        congruence = f.sparse_weights();
    } else {
        const num::vec ones(n, 1.0);
        const num::vec q = f.solve(ones), r = f.solve_transpose(ones);
        similarity = num::vec(n);
        congruence = num::vec(n);
        for (idx j = 0; j < n; ++j) {
            if (!(q[j] > 0.0) || !(r[j] > 0.0))
                throw std::runtime_error("the truncated rate matrix is not an M-matrix");
            similarity[j] = std::sqrt(r[j] / q[j]);
            congruence[j] = std::sqrt(r[j] * q[j]);
        }
    }
    const num::spmat scaled = detail::similarity_scaled(matrix, similarity.span());
    const num::spmat symmetric = reversible ? scaled : detail::symmetric_part(scaled);
    const num::spmat laplacian = detail::congruent(symmetric, congruence.span());
    const num::grounded_approx_chol_factor approximate =
        num::grounded_approxchol_factor(laplacian, 2, probes.seed ^ 0x9e3779b9U);
    const auto preconditioned =
        num::operators::assume_spd(detail::preconditioned_operator(laplacian, approximate));
    std::optional<num::auto_linear_solver> scaled_factor;
    if (!reversible)
        scaled_factor.emplace(scaled);

    const num::mat probe = detail::gaussian_probe(n, probes.count, probes.seed);
    num::mat probed(n, probes.count, 0.0);
    for (idx column = 0; column < probes.count; ++column) {
        num::vec direction(n, 0.0);
        for (idx row = 0; row < n; ++row)
            direction[row] = probe(row, column);
        num::vec value(n, 0.0);
        if (reversible) {
            const auto action = num::inverse_sqrt_lanczos(
                preconditioned, direction, probes.lanczos_tolerance, probes.lanczos_steps);
            const num::vec unscaled = approximate.solve_upper(action.value);
            for (idx row = 0; row < n; ++row)
                value[row] = similarity[row] * unscaled[row];
        } else {
            const auto action = num::sqrt_lanczos(preconditioned, direction,
                                                  probes.lanczos_tolerance,
                                                  probes.lanczos_steps);
            const num::vec factored = approximate.apply_lower(action.value);
            num::vec rhs(n, 0.0);
            for (idx row = 0; row < n; ++row)
                rhs[row] = factored[row] / congruence[row];
            value = num::solve(*scaled_factor, rhs);
        }
        for (idx row = 0; row < n; ++row)
            probed(row, column) = value[row];
    }
    return num::hutchinson_row_mean_square(probed);
}

} // namespace subsweep
