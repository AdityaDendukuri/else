/// @file tests/test_elsex_shedding.cpp
/// @brief The three shedding rules.
///
/// The exact rule is checked against its definition -- the cut time actually
/// lost when the state is removed and the reduced system re-solved -- rather
/// than only against the legacy implementation, because a shared derivation
/// error would pass an implementation-to-implementation comparison.

#include "check.hpp"

#include "else/core/subnetwork.hpp"
#include "else/else.hpp"
#include "else/quantities/shedding.hpp"

#include "container/matrix_expr.hpp"
#include "linear/factorization/inverse_diagonal.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"

#include <cmath>
#include <span>
#include <vector>

namespace {

using State = num::multi_index;
using num::idx;
using num::real;

constexpr real birth_rate = 1.3;
constexpr real death_coefficient = 0.45;

struct Chain {
    idx size = 0;
    num::array<State> states;
    num::array<idx> rows, cols;
    num::array<real> values;
    num::array<else_sim::BoundaryTransition<State>> boundary;
    num::array<real> stationary; // full-chain stationary weights, restricted
};

/// Birth-death chain on copy numbers 1..n with constant birth and linear death.
///
/// Detailed balance holds, so the restricted operator is reversible and the
/// stationary weights are the Poisson weights of the unrestricted chain.
Chain make_chain(idx n) {
    Chain chain;
    chain.size = n;
    for (idx i = 0; i < n; ++i) {
        chain.states.push_back(State{static_cast<int>(i + 1)});
    }

    for (idx i = 0; i < n; ++i) {
        const real copies = static_cast<real>(i + 1);
        const real death = death_coefficient * copies;

        if (i + 1 < n) {
            chain.rows.push_back(i);
            chain.cols.push_back(i + 1);
            chain.values.push_back(birth_rate);
        } else {
            chain.boundary.push_back({i, State{static_cast<int>(i + 2)}, birth_rate});
        }

        if (i > 0) {
            chain.rows.push_back(i);
            chain.cols.push_back(i - 1);
            chain.values.push_back(death);
        } else {
            chain.boundary.push_back({i, State{0}, death});
        }

        chain.rows.push_back(i);
        chain.cols.push_back(i);
        chain.values.push_back(-(birth_rate + death));
    }

    // pi_x proportional to (birth/death)^x / x! for x = 1..n.
    chain.stationary.assign(n, 0.0);
    real weight = 1.0;
    for (idx i = 0; i < n; ++i) {
        weight *= birth_rate / (death_coefficient * static_cast<real>(i + 1));
        chain.stationary[i] = weight;
    }
    return chain;
}

else_sim::Subnetwork<State> build(const Chain &chain) {
    return else_sim::make_subnetwork(
        chain.states,
        num::spmat::from_triplets(chain.size, chain.size, chain.rows, chain.cols, chain.values),
        chain.boundary);
}

/// Cut time with `removed` deleted: solve the reduced system directly.
real reduced_cut_time(const Chain &chain, const num::vec &entrance_mixture, idx removed) {
    const idx n = chain.size;
    num::array<idx> kept;
    num::array<idx> position(n, n);
    for (idx i = 0; i < n; ++i) {
        if (i != removed) {
            position[i] = kept.size();
            kept.push_back(i);
        }
    }

    const num::spmat full = num::spmat::from_triplets(n, n, chain.rows, chain.cols, chain.values);
    num::mat reduced(kept.size(), kept.size(), 0.0);
    for (idx a = 0; a < kept.size(); ++a) {
        for (idx b = 0; b < kept.size(); ++b) {
            reduced(a, b) = -full(kept[a], kept[b]); // M = -R restricted to the kept states
        }
    }

    const auto factor = num::lu(num::make_square(reduced));
    num::vec ones(kept.size(), 1.0);
    num::vec exit_time(kept.size(), 0.0);
    num::lu_solve(factor, ones, exit_time);

    real total = 0.0;
    for (idx a = 0; a < kept.size(); ++a) {
        total += entrance_mixture[kept[a]] * exit_time[a];
    }
    return total;
}

num::vec make_mixture(idx n) {
    num::vec mixture(n, 0.0);
    mixture[2] = 0.6;
    mixture[5] = 0.4;
    return mixture;
}

} // namespace

void test_exact_loss_matches_its_definition() {
    const Chain chain = make_chain(11);
    const auto subnetwork = build(chain);
    const num::vec mixture = make_mixture(chain.size);
    const auto state = else_sim::shedding_state(subnetwork, mixture);
    const num::vec losses = else_sim::exact_cut_time_losses(subnetwork, state);

    real full_cut_time = 0.0;
    for (idx i = 0; i < chain.size; ++i) {
        full_cut_time += mixture[i] * state.exit_time[i];
    }

    // The lemma requires rho_j = 0 for an eligible state, so skip the entrances.
    for (idx j = 0; j < chain.size; ++j) {
        if (mixture[j] != 0.0) {
            continue;
        }
        const real expected = full_cut_time - reduced_cut_time(chain, mixture, j);
        check::close(losses[j], expected, "exact loss vs reduced re-solve", 1e-11);
    }
    check::done("exact cut-time loss equals the time actually lost");
}

void test_stationary_normalization_preserves_inverse_diagonal() {
    const Chain chain = make_chain(11);
    const auto subnetwork = build(chain);

    num::array<real> weights(chain.size);
    for (idx j = 0; j < chain.size; ++j) {
        weights[j] = std::sqrt(chain.stationary[j]);
    }
    const num::spmat normalized =
        else_sim::detail::similarity_scaled(subnetwork.operator_matrix, weights);

    // Detailed balance makes the normalized operator symmetric.
    const num::mat dense_normalized = num::dense(normalized);
    for (idx i = 0; i < chain.size; ++i) {
        for (idx j = 0; j < chain.size; ++j) {
            check::close(dense_normalized(i, j), dense_normalized(j, i),
                         "normalized operator is symmetric");
        }
    }

    // Diagonal similarity preserves the inverse diagonal.
    const num::auto_linear_solver normalized_factor(normalized);
    const num::vec normalized_diagonal = num::inverse_diagonal(normalized_factor);
    const num::vec exact_diagonal = else_sim::inverse_diagonal(subnetwork);
    for (idx j = 0; j < chain.size; ++j) {
        check::close(normalized_diagonal[j], exact_diagonal[j], "diag(Mtilde^-1) equals diag(M^-1)",
                     1e-11);
    }
    check::done("stationary normalization preserves the inverse diagonal");
}

void test_probe_estimator_converges() {
    const Chain chain = make_chain(11);
    const auto subnetwork = build(chain);
    const num::vec exact = else_sim::inverse_diagonal(subnetwork);

    const auto mean_relative_error = [&](idx probes) {
        const num::vec estimate = else_sim::probed_inverse_diagonal(
            subnetwork, num::view<const real>(chain.stationary), probes, 20260828u);
        real total = 0.0;
        for (idx j = 0; j < chain.size; ++j) {
            total += std::abs(estimate[j] - exact[j]) / exact[j];
        }
        return total / static_cast<real>(chain.size);
    };

    // Hutchinson error falls like 1/sqrt(h), so this asserts a trend and a loose
    // bound at a fixed seed, not a tight numerical identity.
    const real few = mean_relative_error(64);
    const real many = mean_relative_error(8192);
    check::that(many < few, "probe error decreases with more probes");
    check::that(many < 0.05, "probe error is small at 8192 probes");
    std::printf("          (mean relative error: %.4f at 64 probes, %.4f at 8192)\n", few, many);
    check::done("probe estimator converges to the exact inverse diagonal");
}

void test_sparse_lanczos_estimator_on_large_laplacian() {
    constexpr idx n = 10000;
    constexpr real shift = 0.5;
    num::array<int> states(n);
    num::array<idx> rows;
    num::array<idx> columns;
    num::array<real> values;
    rows.reserve(3 * n - 2);
    columns.reserve(3 * n - 2);
    values.reserve(3 * n - 2);
    for (idx j = 0; j < n; ++j) {
        states[j] = static_cast<int>(j);
        rows.push_back(j);
        columns.push_back(j);
        values.push_back(-(2.0 + shift));
        if (j > 0) {
            rows.push_back(j);
            columns.push_back(j - 1);
            values.push_back(1.0);
        }
        if (j + 1 < n) {
            rows.push_back(j);
            columns.push_back(j + 1);
            values.push_back(1.0);
        }
    }

    const auto subnetwork = else_sim::make_subnetwork(
        std::move(states), num::spmat::from_triplets(n, n, rows, columns, values),
        num::array<else_sim::BoundaryTransition<int>>{}, num::vec(n, 1.0), num::view<const idx>{},
        false);
    const num::vec estimate = else_sim::probed_inverse_diagonal(
        subnetwork, subnetwork.stationary.span(), 64, 20260910u, 32, 1e-9);

    // Exact diagonal of the inverse of the shifted tridiagonal Laplacian from
    // its LDL^T recurrence, computed in O(n) without a dense reference matrix.
    num::vec pivots(n, 0.0);
    num::vec exact(n, 0.0);
    pivots[0] = 2.0 + shift;
    for (idx j = 1; j < n; ++j)
        pivots[j] = 2.0 + shift - 1.0 / pivots[j - 1];
    exact[n - 1] = 1.0 / pivots[n - 1];
    for (idx j = n - 1; j-- > 0;)
        exact[j] = 1.0 / pivots[j] + exact[j + 1] / (pivots[j] * pivots[j]);

    real mean_relative_error = 0.0;
    for (idx j = 0; j < n; ++j)
        mean_relative_error += std::abs(estimate[j] - exact[j]) / exact[j];
    mean_relative_error /= static_cast<real>(n);

    check::that(mean_relative_error < 0.1,
                "large sparse Lanczos estimate has small mean relative error");
    std::printf("          (n: %zu, probes: 64, mean relative error: %.4f)\n",
                static_cast<num::idx>(n), mean_relative_error);
    check::done("Lanczos probing remains sparse on a 10,000-state Laplacian");
}

void test_irreversible_probe_estimator() {
    constexpr idx n = 3;
    const num::array<int> states{0, 1, 2};
    const num::array<idx> rows{0, 0, 0, 1, 1, 1, 2, 2, 2};
    const num::array<idx> columns{0, 1, 2, 0, 1, 2, 0, 1, 2};
    // Principal block of the four-state clockwise/counterclockwise cycle with
    // rates 2 and 1.  Its stationary distribution is uniform, but it is not
    // reversible because the two cycle fluxes differ.
    const num::array<real> values{-3.0, 2.0, 0.0, 1.0, -3.0, 2.0, 0.0, 1.0, -3.0};
    const num::array<else_sim::BoundaryTransition<int>> boundary{{0, 3, 1.0}, {2, 3, 2.0}};
    const num::vec stationary(n, 1.0);
    const auto subnetwork =
        else_sim::make_subnetwork(states, num::spmat::from_triplets(n, n, rows, columns, values),
                                  boundary, stationary, num::view<const idx>{}, true, false);

    const num::vec exact = else_sim::inverse_diagonal(subnetwork);
    const num::vec estimate = else_sim::probed_inverse_diagonal(subnetwork, stationary.span(),
                                                                32768, 20260910u, n, 1e-12, false);
    for (idx j = 0; j < n; ++j)
        check::close(estimate[j], exact[j], "irreversible normalized estimator", 0.015);
    check::done("normalized and symmetrized probing handles irreversible rates");
}

void test_lowest_scores_selection() {
    const num::array<real> scores{5.0, 1.0, 3.0, 1.0, 4.0, 0.5};
    const num::array<idx> protected_states{5, 1};

    const auto chosen = else_sim::lowest_scores(num::view<const real>(scores),
                                                num::view<const idx>(protected_states), 3);
    check::that(chosen.size() == 3, "returns the requested count");
    check::that(chosen[0] == 3, "smallest eligible score first");
    check::that(chosen[1] == 2, "then the next smallest");
    check::that(chosen[2] == 4, "then the next");
    for (idx j : chosen) {
        check::that(j != 5 && j != 1, "protected states are never shed");
    }

    const auto clamped = else_sim::lowest_scores(num::view<const real>(scores),
                                                 num::view<const idx>(protected_states), 99);
    check::that(clamped.size() == 4, "count clamps to the eligible set");
    check::done("lowest_scores respects protection, order, and count");
}

int main() {
    std::printf("else shedding\n");
    test_exact_loss_matches_its_definition();
    test_stationary_normalization_preserves_inverse_diagonal();
    test_probe_estimator_converges();
    test_sparse_lanczos_estimator_on_large_laplacian();
    test_irreversible_probe_estimator();
    test_lowest_scores_selection();
    return check::report("else shedding");
}
