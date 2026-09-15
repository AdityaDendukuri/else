/// @file tests/test_elsex_law.cpp
/// @brief Parity harness for the numerics-based ELSE entrance law.
///
/// The legacy tree stores the generator in column convention and applies a plain
/// solve; the new tree stores it in the paper's row convention and applies a
/// transpose solve. Both therefore evaluate M^-T delta_i, and the law-level
/// quantities must agree to round-off. Trajectory sampling is deliberately not
/// compared here: a rewrite cannot reproduce the old RNG consumption order, so
/// that comparison belongs at the distribution level once the ensemble lands.

#include "check.hpp"

#include "else/core/subnetwork.hpp"
#include "else/else.hpp"
#include "else/quantities/law.hpp"

#include <cmath>
#include <cstdio>
#include <span>
#include <variant>
#include <vector>

namespace {

using State = num::multi_index;

constexpr double birth_rate = 1.7;
constexpr double death_coefficient = 0.35;

/// A birth-death chain restricted to copy numbers 1..n.
///
/// Death from copy number 1 and birth from copy number n both leave the set, so
/// there are two escape states and the escape distribution is non-trivial.
struct Chain {
    num::idx size = 0;
    num::array<State> states;
    num::array<num::idx> sources;      // interior transition source
    num::array<num::idx> destinations; // interior transition destination
    num::array<double> rates;
    num::array<double> diagonal; // -(total outgoing rate), including escapes
    num::array<num::idx> escape_sources;
    num::array<State> escape_destinations;
    num::array<double> escape_rates;
};

Chain make_chain(num::idx n) {
    Chain chain;
    chain.size = n;
    for (num::idx i = 0; i < n; ++i) {
        chain.states.push_back(State{static_cast<int>(i + 1)});
    }
    chain.diagonal.assign(n, 0.0);

    for (num::idx i = 0; i < n; ++i) {
        const double copies = static_cast<double>(i + 1);
        const double death = death_coefficient * copies;

        // Birth: i -> i+1, escaping at the top of the window.
        if (i + 1 < n) {
            chain.sources.push_back(i);
            chain.destinations.push_back(i + 1);
            chain.rates.push_back(birth_rate);
        } else {
            chain.escape_sources.push_back(i);
            chain.escape_destinations.push_back(State{static_cast<int>(i + 2)});
            chain.escape_rates.push_back(birth_rate);
        }

        // Death: i -> i-1, escaping at the bottom of the window.
        if (i > 0) {
            chain.sources.push_back(i);
            chain.destinations.push_back(i - 1);
            chain.rates.push_back(death);
        } else {
            chain.escape_sources.push_back(i);
            chain.escape_destinations.push_back(State{0});
            chain.escape_rates.push_back(death);
        }

        chain.diagonal[i] = -(birth_rate + death);
    }
    return chain;
}

/// Row convention: generator(i, j) is the rate from i to j.
else_sim::Subnetwork<State> build_new(const Chain &chain) {
    num::array<num::idx> rows, cols;
    num::array<num::real> vals;
    for (num::idx k = 0; k < chain.sources.size(); ++k) {
        rows.push_back(chain.sources[k]);
        cols.push_back(chain.destinations[k]);
        vals.push_back(chain.rates[k]);
    }
    for (num::idx i = 0; i < chain.size; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(chain.diagonal[i]);
    }

    num::array<else_sim::BoundaryTransition<State>> boundary;
    for (num::idx k = 0; k < chain.escape_sources.size(); ++k) {
        boundary.push_back(
            {chain.escape_sources[k], chain.escape_destinations[k], chain.escape_rates[k]});
    }
    return else_sim::make_subnetwork(
        chain.states, num::spmat::from_triplets(chain.size, chain.size, rows, cols, vals),
        std::move(boundary));
}

} // namespace

void test_escape_rates_agree_with_row_sums() {
    const Chain chain = make_chain(12);
    const auto subnetwork = build_new(chain);

    // Escape rates come from summing the boundary transitions. The identity
    // w = -(R 1) = M 1 must still hold, which checks that the generator's
    // diagonal accounts for exactly the declared escapes. It is a check, not the
    // definition: evaluated as a row sum it cancels, so interior states land on
    // signed round-off rather than exact zero.
    const num::vec ones(chain.size, 1.0);
    num::vec row_sums(chain.size, 0.0);
    num::sparse_matvec(subnetwork.operator_matrix, ones, row_sums);

    const auto &escape_rates = subnetwork.escape_rates;
    for (num::idx j = 0; j < chain.size; ++j) {
        check::close(escape_rates[j], row_sums[j], "escape rate vs -(R 1)", 1e-12);
    }

    // Only the two window edges can be escaped from; interior round-off must not
    // be mistaken for an escape.
    check::that(subnetwork.escape_states.size() == 2, "exactly two escape states");
    check::that(subnetwork.escape_states[0] == 0, "bottom edge escapes");
    check::that(subnetwork.escape_states[1] == chain.size - 1, "top edge escapes");

    check::done("escape rates agree with -(R 1), interior round-off excluded");
}

void test_cme_restriction_uses_no_pivot_lu() {
    const Chain chain = make_chain(12);
    const auto subnetwork = build_new(chain);
    check::that(subnetwork.factor.has_value(), "CME restriction is factorized");
    check::that(std::holds_alternative<num::no_pivot_lu>(*subnetwork.factor),
                "general CME restriction uses no-pivot LU");
    check::done("CME restriction selects no-pivot LU");
}

void test_first_exit_law_normalization() {
    const Chain chain = make_chain(12);
    const auto subnetwork = build_new(chain);

    // Independent of the legacy tree: the paper's first-exit law gives
    // sum_j w_j Z_ij = 1 for every entrance i.
    num::array<num::idx> all(chain.size);
    for (num::idx i = 0; i < chain.size; ++i) {
        all[i] = i;
    }
    const auto law = else_sim::entrance_law(subnetwork, num::view<const num::idx>(all));
    const auto &escape_rates = subnetwork.escape_rates;

    for (num::idx k = 0; k < chain.size; ++k) {
        double total = 0.0;
        for (num::idx j = 0; j < chain.size; ++j) {
            total += escape_rates[j] * law.occupation(k, j);
        }
        check::close(total, 1.0, "sum_j w_j Z_ij");
    }

    check::done("first-exit law normalization holds for every entrance");
}

int main() {
    std::printf("else law\n");
    test_escape_rates_agree_with_row_sums();
    test_cme_restriction_uses_no_pivot_lu();
    test_first_exit_law_normalization();
    return check::report("else law");
}
