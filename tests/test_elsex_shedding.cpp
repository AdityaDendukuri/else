/// @file tests/test_elsex_shedding.cpp
/// @brief The three shedding rules.
///
/// The exact rule is checked against its definition -- the cut time actually
/// lost when the state is removed and the reduced system re-solved -- rather
/// than only against the legacy implementation, because a shared derivation
/// error would pass an implementation-to-implementation comparison.

#include "check.hpp"

#include "else/else.hpp"
#include "elsex/shedding.hpp"
#include "elsex/subnetwork.hpp"

#include "container/matrix_expr.hpp"
#include "linear/factorization/inverse_diagonal.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"

#include <cmath>
#include <span>
#include <vector>

namespace {

using State = std::vector<int>;
using num::idx;
using num::real;

constexpr real birth_rate = 1.3;
constexpr real death_coefficient = 0.45;

struct Chain {
    idx size = 0;
    std::vector<State> states;
    std::vector<idx> rows, cols;
    std::vector<real> values;
    std::vector<elsex::BoundaryTransition<State>> boundary;
    std::vector<real> stationary; // full-chain stationary weights, restricted
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

elsex::Subnetwork<State> build(const Chain &chain) {
    return elsex::Subnetwork<State>(
        chain.states,
        num::SparseMatrix::from_triplets(chain.size, chain.size, chain.rows, chain.cols,
                                         chain.values),
        chain.boundary);
}

/// Legacy tree, which stores the transpose.
else_sim::Subnetwork<double, std::size_t, State> build_legacy(const Chain &chain) {
    std::vector<std::size_t> rows(chain.cols.begin(), chain.cols.end());
    std::vector<std::size_t> cols(chain.rows.begin(), chain.rows.end());
    std::vector<else_sim::BoundaryTransition<std::size_t, State, double>> boundary;
    for (const auto &transition : chain.boundary) {
        boundary.push_back({transition.source, transition.destination, transition.rate});
    }
    return else_sim::Subnetwork<double, std::size_t, State>(
        chain.states,
        else_sim::SparseMatrix<double, std::size_t>::from_triplets(chain.size, chain.size, rows,
                                                                   cols, chain.values),
        std::move(boundary));
}

/// Cut time with `removed` deleted: solve the reduced system directly.
real reduced_cut_time(const Chain &chain, const num::Vector &entrance_mixture, idx removed) {
    const idx n = chain.size;
    std::vector<idx> kept;
    std::vector<idx> position(n, n);
    for (idx i = 0; i < n; ++i) {
        if (i != removed) {
            position[i] = kept.size();
            kept.push_back(i);
        }
    }

    const num::SparseMatrix full =
        num::SparseMatrix::from_triplets(n, n, chain.rows, chain.cols, chain.values);
    num::Matrix reduced(kept.size(), kept.size(), 0.0);
    for (idx a = 0; a < kept.size(); ++a) {
        for (idx b = 0; b < kept.size(); ++b) {
            reduced(a, b) = -full(kept[a], kept[b]); // M = -R restricted to the kept states
        }
    }

    const auto factor = num::lu(num::make_square(reduced));
    num::Vector ones(kept.size(), 1.0);
    num::Vector exit_time(kept.size(), 0.0);
    num::lu_solve(factor, ones, exit_time);

    real total = 0.0;
    for (idx a = 0; a < kept.size(); ++a) {
        total += entrance_mixture[kept[a]] * exit_time[a];
    }
    return total;
}

num::Vector make_mixture(idx n) {
    num::Vector mixture(n, 0.0);
    mixture[2] = 0.6;
    mixture[5] = 0.4;
    return mixture;
}

} // namespace

void test_exact_loss_matches_its_definition() {
    const Chain chain = make_chain(11);
    const auto subnetwork = build(chain);
    const num::Vector mixture = make_mixture(chain.size);
    const auto state = elsex::shedding_state(subnetwork, mixture);
    const num::Vector losses = elsex::exact_cut_time_losses(subnetwork, state);

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

void test_exact_loss_matches_legacy() {
    const Chain chain = make_chain(11);
    const auto subnetwork = build(chain);
    const auto legacy = build_legacy(chain);
    const num::Vector mixture = make_mixture(chain.size);
    const auto state = elsex::shedding_state(subnetwork, mixture);
    const num::Vector losses = elsex::exact_cut_time_losses(subnetwork, state);

    std::vector<double> legacy_occupation(state.occupation.data(),
                                          state.occupation.data() + state.occupation.size());
    const auto legacy_losses = legacy.cut_time_losses(legacy_occupation);
    for (idx j = 0; j < chain.size; ++j) {
        check::close(losses[j], legacy_losses[j], "exact loss vs legacy", 1e-11);
    }
    check::done("exact cut-time loss matches the legacy implementation");
}

void test_expected_entries_matches_legacy() {
    const Chain chain = make_chain(11);
    const auto subnetwork = build(chain);
    const auto legacy = build_legacy(chain);
    const num::Vector mixture = make_mixture(chain.size);
    const auto state = elsex::shedding_state(subnetwork, mixture);
    const num::Vector entries = elsex::expected_entries(subnetwork, state);

    std::vector<double> legacy_occupation(state.occupation.data(),
                                          state.occupation.data() + state.occupation.size());
    std::vector<std::size_t> all(chain.size);
    for (idx i = 0; i < chain.size; ++i) {
        all[i] = i;
    }
    const auto legacy_scores =
        else_sim::expected_visit_scores(legacy, legacy_occupation, std::span<const std::size_t>(all));

    // The legacy routine stops at (-R_jj) u_j; the paper subtracts the entrance
    // mass, which the legacy caller does separately at its call site.
    for (idx j = 0; j < chain.size; ++j) {
        check::close(entries[j] + mixture[j], legacy_scores[j], "expected entries vs legacy");
    }
    check::done("expected entries matches the legacy implementation");
}

void test_stationary_normalization_preserves_inverse_diagonal() {
    const Chain chain = make_chain(11);
    const auto subnetwork = build(chain);

    std::vector<real> weights(chain.size);
    for (idx j = 0; j < chain.size; ++j) {
        weights[j] = std::sqrt(chain.stationary[j]);
    }
    const num::SparseMatrix normalized =
        elsex::detail::similarity_scaled(subnetwork.operator_matrix(), weights);

    // Detailed balance makes the normalized operator symmetric.
    const num::Matrix dense_normalized = num::dense(normalized);
    for (idx i = 0; i < chain.size; ++i) {
        for (idx j = 0; j < chain.size; ++j) {
            check::close(dense_normalized(i, j), dense_normalized(j, i),
                         "normalized operator is symmetric");
        }
    }

    // Diagonal similarity preserves the inverse diagonal.
    const num::AutoLinearSolver normalized_factor(normalized);
    const num::Vector normalized_diagonal = num::inverse_diagonal(normalized_factor);
    const num::Vector exact_diagonal = num::inverse_diagonal(subnetwork.factor());
    for (idx j = 0; j < chain.size; ++j) {
        check::close(normalized_diagonal[j], exact_diagonal[j],
                     "diag(Mtilde^-1) equals diag(M^-1)", 1e-11);
    }
    check::done("stationary normalization preserves the inverse diagonal");
}

void test_probe_estimator_converges() {
    const Chain chain = make_chain(11);
    const auto subnetwork = build(chain);
    const num::Vector exact = num::inverse_diagonal(subnetwork.factor());

    const auto mean_relative_error = [&](idx probes) {
        const num::Vector estimate = elsex::probed_inverse_diagonal(
            subnetwork, std::span<const real>(chain.stationary), probes, 20260828u);
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

void test_lowest_scores_selection() {
    const std::vector<real> scores{5.0, 1.0, 3.0, 1.0, 4.0, 0.5};
    const std::vector<idx> protected_states{5, 1};

    const auto chosen = elsex::lowest_scores(std::span<const real>(scores),
                                             std::span<const idx>(protected_states), 3);
    check::that(chosen.size() == 3, "returns the requested count");
    check::that(chosen[0] == 3, "smallest eligible score first");
    check::that(chosen[1] == 2, "then the next smallest");
    check::that(chosen[2] == 4, "then the next");
    for (idx j : chosen) {
        check::that(j != 5 && j != 1, "protected states are never shed");
    }

    const auto clamped = elsex::lowest_scores(std::span<const real>(scores),
                                              std::span<const idx>(protected_states), 99);
    check::that(clamped.size() == 4, "count clamps to the eligible set");
    check::done("lowest_scores respects protection, order, and count");
}

int main() {
    std::printf("elsex shedding\n");
    test_exact_loss_matches_its_definition();
    test_exact_loss_matches_legacy();
    test_expected_entries_matches_legacy();
    test_stationary_normalization_preserves_inverse_diagonal();
    test_probe_estimator_converges();
    test_lowest_scores_selection();
    return check::report("elsex shedding");
}
