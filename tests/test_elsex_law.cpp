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

#include "else/else.hpp"
#include "elsex/law.hpp"
#include "elsex/subnetwork.hpp"

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace {

using State = std::vector<int>;

constexpr double birth_rate = 1.7;
constexpr double death_coefficient = 0.35;

/// A birth-death chain restricted to copy numbers 1..n.
///
/// Death from copy number 1 and birth from copy number n both leave the set, so
/// there are two escape states and the escape distribution is non-trivial.
struct Chain {
    num::idx size = 0;
    std::vector<State> states;
    std::vector<num::idx> sources;      // interior transition source
    std::vector<num::idx> destinations; // interior transition destination
    std::vector<double> rates;
    std::vector<double> diagonal; // -(total outgoing rate), including escapes
    std::vector<num::idx> escape_sources;
    std::vector<State> escape_destinations;
    std::vector<double> escape_rates;
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
elsex::Subnetwork<State> build_new(const Chain &chain) {
    std::vector<num::idx> rows, cols;
    std::vector<num::real> vals;
    for (std::size_t k = 0; k < chain.sources.size(); ++k) {
        rows.push_back(chain.sources[k]);
        cols.push_back(chain.destinations[k]);
        vals.push_back(chain.rates[k]);
    }
    for (num::idx i = 0; i < chain.size; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(chain.diagonal[i]);
    }

    std::vector<elsex::BoundaryTransition<State>> boundary;
    for (std::size_t k = 0; k < chain.escape_sources.size(); ++k) {
        boundary.push_back({chain.escape_sources[k], chain.escape_destinations[k],
                            chain.escape_rates[k]});
    }
    return elsex::Subnetwork<State>(
        chain.states, num::SparseMatrix::from_triplets(chain.size, chain.size, rows, cols, vals),
        std::move(boundary));
}

/// Column convention: the legacy tree stores the transpose.
else_sim::Subnetwork<double, std::size_t, State> build_legacy(const Chain &chain) {
    std::vector<std::size_t> rows, cols;
    std::vector<double> vals;
    for (std::size_t k = 0; k < chain.sources.size(); ++k) {
        rows.push_back(chain.destinations[k]);
        cols.push_back(chain.sources[k]);
        vals.push_back(chain.rates[k]);
    }
    for (std::size_t i = 0; i < chain.size; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(chain.diagonal[i]);
    }

    std::vector<else_sim::BoundaryTransition<std::size_t, State, double>> boundary;
    for (std::size_t k = 0; k < chain.escape_sources.size(); ++k) {
        boundary.push_back({chain.escape_sources[k], chain.escape_destinations[k],
                            chain.escape_rates[k]});
    }
    return else_sim::Subnetwork<double, std::size_t, State>(
        chain.states,
        else_sim::SparseMatrix<double, std::size_t>::from_triplets(chain.size, chain.size, rows,
                                                                   cols, vals),
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
    const num::Vector ones(chain.size, 1.0);
    num::Vector row_sums(chain.size, 0.0);
    num::sparse_matvec(subnetwork.operator_matrix(), ones, row_sums);

    const auto escape_rates = subnetwork.escape_rates();
    for (num::idx j = 0; j < chain.size; ++j) {
        check::close(escape_rates[j], row_sums[j], "escape rate vs -(R 1)", 1e-12);
    }

    // Only the two window edges can be escaped from; interior round-off must not
    // be mistaken for an escape.
    check::that(subnetwork.escape_states().size() == 2, "exactly two escape states");
    check::that(subnetwork.escape_states()[0] == 0, "bottom edge escapes");
    check::that(subnetwork.escape_states()[1] == chain.size - 1, "top edge escapes");

    check::done("escape rates agree with -(R 1), interior round-off excluded");
}

void test_first_exit_law_normalization() {
    const Chain chain = make_chain(12);
    const auto subnetwork = build_new(chain);

    // Independent of the legacy tree: the paper's first-exit law gives
    // sum_j w_j Z_ij = 1 for every entrance i.
    std::vector<num::idx> all(chain.size);
    for (num::idx i = 0; i < chain.size; ++i) {
        all[i] = i;
    }
    const auto law = elsex::entrance_law(subnetwork, std::span<const num::idx>(all));
    const auto escape_rates = subnetwork.escape_rates();

    for (num::idx k = 0; k < chain.size; ++k) {
        double total = 0.0;
        for (num::idx j = 0; j < chain.size; ++j) {
            total += escape_rates[j] * law.occupation(k, j);
        }
        check::close(total, 1.0, "sum_j w_j Z_ij");
    }

    check::done("first-exit law normalization holds for every entrance");
}

void test_law_matches_legacy() {
    const Chain chain = make_chain(12);
    const auto subnetwork = build_new(chain);
    const auto legacy = build_legacy(chain);

    const std::vector<num::idx> entrances{0, 3, 7, chain.size - 1};
    const auto law = elsex::entrance_law(subnetwork, std::span<const num::idx>(entrances));

    else_sim::Matrix<double> legacy_entrances(chain.size, entrances.size(), 0.0);
    for (std::size_t k = 0; k < entrances.size(); ++k) {
        legacy_entrances(entrances[k], k) = 1.0;
    }
    const auto legacy_integrals = legacy.occupation_integrals(legacy_entrances);

    for (std::size_t k = 0; k < entrances.size(); ++k) {
        for (num::idx j = 0; j < chain.size; ++j) {
            check::close(law.occupation(k, j), legacy_integrals.occupation(j, k), "occupation U");
            check::close(law.second_occupation(k, j), legacy_integrals.time_weighted_occupation(j, k),
                  "second occupation V");
        }

        const auto distribution = elsex::escape_distribution(subnetwork, law, k);
        const auto legacy_distribution = legacy.exit_probabilities(legacy_integrals, k);
        check::that(distribution.size() == legacy_distribution.size(), "escape state count agrees");
        for (std::size_t b = 0; b < legacy_distribution.size(); ++b) {
            check::close(distribution[b], legacy_distribution[b], "escape distribution beta");
        }

        for (num::idx b : subnetwork.escape_states()) {
            check::close(elsex::conditional_escape_time(law, k, b),
                  legacy.conditional_exit_time(legacy_integrals, b, k),
                  "conditional escape time mu");
        }
    }

    check::done("entrance law matches the legacy implementation");
}

void test_mean_escape_time_matches_legacy() {
    const Chain chain = make_chain(12);
    const auto subnetwork = build_new(chain);
    const auto legacy = build_legacy(chain);

    const std::vector<num::idx> entrances{2, 9};
    const auto law = elsex::entrance_law(subnetwork, std::span<const num::idx>(entrances));

    for (std::size_t k = 0; k < entrances.size(); ++k) {
        std::vector<double> start(chain.size, 0.0);
        start[entrances[k]] = 1.0;
        const auto legacy_occupation = legacy.occupation(start);

        double legacy_total = 0.0;
        for (num::idx j = 0; j < chain.size; ++j) {
            legacy_total += legacy_occupation[j];
        }
        check::close(elsex::mean_escape_time(law, k), legacy_total, "mean escape time");
    }

    check::done("mean escape time matches the legacy implementation");
}

int main() {
    std::printf("elsex law\n");
    test_escape_rates_agree_with_row_sums();
    test_first_exit_law_normalization();
    test_law_matches_legacy();
    test_mean_escape_time_matches_legacy();
    return check::report("elsex law");
}
