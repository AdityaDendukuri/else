/// @file tests/test_elsex_factor_update.cpp
/// @brief Woodbury reuse of a stored factorization, and stable slots.
///
/// A corrected solve is an algebraic identity, not an approximation, so it is
/// checked against a full refactorization of the updated operator rather than
/// against a tolerance chosen by hand. Both directions are checked: ELSE reaches
/// for the transpose solve more often than the forward one.

#include "check.hpp"

#include "elsex/factor_update.hpp"
#include "elsex/restriction.hpp"
#include "elsex/state_graph.hpp"

#include "container/matrix_expr.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"

#include <cmath>
#include <random>
#include <span>
#include <vector>

namespace {

using num::idx;
using num::real;

constexpr idx order = 24;

/// A nonsymmetric, diagonally dominant CSR operator; `perturbed` slots get a
/// different row and column, as replacing a state would produce.
num::SparseMatrix build_operator(std::span<const idx> perturbed) {
    std::vector<bool> changed(order, false);
    for (idx j : perturbed) {
        changed[j] = true;
    }

    std::vector<idx> rows, cols;
    std::vector<real> values;
    for (idx i = 0; i < order; ++i) {
        const real bump = changed[i] ? 3.7 : 0.0;
        rows.push_back(i);
        cols.push_back(i);
        values.push_back(9.0 + 0.4 * static_cast<real>(i) + bump);

        if (i + 1 < order) {
            rows.push_back(i);
            cols.push_back(i + 1);
            values.push_back(-1.0 - (changed[i] ? 0.9 : 0.0));
        }
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            values.push_back(-0.6 - (changed[i - 1] ? 0.8 : 0.0));
        }
        if (i + 3 < order) {
            rows.push_back(i);
            cols.push_back(i + 3);
            values.push_back(-0.3 - (changed[i] ? 0.5 : 0.0));
        }
    }
    return num::SparseMatrix::from_triplets(order, order, rows, cols, values);
}

num::Vector make_rhs(unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> spread(-1.0, 1.0);
    num::Vector b(order, 0.0);
    for (idx i = 0; i < order; ++i) {
        b[i] = spread(generator);
    }
    return b;
}

} // namespace

void test_correction_matches_refactorization() {
    for (const std::vector<idx> &changed :
         std::vector<std::vector<idx>>{{5}, {2, 11}, {0, 7, 18, 23}}) {
        const auto base = build_operator({});
        const auto current = build_operator(std::span<const idx>(changed));

        const num::AutoLinearSolver base_factor(base);
        const num::AutoLinearSolver current_factor(current); // ground truth

        const auto delta = elsex::row_column_delta(base, current, std::span<const idx>(changed));
        const elsex::FactorUpdate update(base_factor, delta.left, delta.right);
        check::that(update.rank() == 2 * changed.size(), "rank is twice the changed slot count");

        const num::Vector b = make_rhs(11u + static_cast<unsigned>(changed.size()));

        const num::Vector corrected_transpose = update.solve_transpose(b);
        const num::Vector reference_transpose = num::solve_transpose(current_factor, b);
        for (idx i = 0; i < order; ++i) {
            check::close(corrected_transpose[i], reference_transpose[i],
                         "corrected transpose solve equals refactorization", 1e-11);
        }

        check::that(elsex::relative_residual(num::transpose(current), corrected_transpose, b) <
                        1e-12,
                    "the corrected solve has a small backward residual");
    }
    check::done("Woodbury correction reproduces a full refactorization");
}

void test_correction_handles_many_right_hand_sides() {
    const std::vector<idx> changed{3, 9};
    const auto base = build_operator({});
    const auto current = build_operator(std::span<const idx>(changed));

    const num::AutoLinearSolver base_factor(base);
    const num::AutoLinearSolver current_factor(current);
    const auto delta = elsex::row_column_delta(base, current, std::span<const idx>(changed));
    const elsex::FactorUpdate update(base_factor, delta.left, delta.right);

    num::Matrix rhs(order, 3, 0.0);
    for (idx i = 0; i < order; ++i) {
        for (idx c = 0; c < 3; ++c) {
            rhs(i, c) = std::sin(0.7 * static_cast<real>(i) + static_cast<real>(c));
        }
    }

    const num::Matrix corrected_transpose = update.solve_transpose(rhs);
    const num::Matrix reference_transpose = num::solve_transpose(current_factor, rhs);
    for (idx i = 0; i < order; ++i) {
        for (idx c = 0; c < 3; ++c) {
            check::close(corrected_transpose(i, c), reference_transpose(i, c),
                         "block corrected transpose solve", 1e-11);
        }
    }
    check::done("one correction serves every right-hand side");
}

void test_delta_is_exactly_low_rank() {
    using namespace num::ops;
    const std::vector<idx> changed{4, 15};
    const auto base = build_operator({});
    const auto current = build_operator(std::span<const idx>(changed));

    const auto delta = elsex::row_column_delta(base, current, std::span<const idx>(changed));
    const num::Matrix reconstructed = delta.left * num::transpose(delta.right);

    const num::Matrix dense_base = num::dense(base);
    const num::Matrix dense_current = num::dense(current);
    for (idx i = 0; i < order; ++i) {
        for (idx j = 0; j < order; ++j) {
            check::close(reconstructed(i, j), dense_current(i, j) - dense_base(i, j),
                         "U V^T reproduces the difference exactly");
        }
    }
    check::done("the row-and-column delta is exactly U V^T");
}

void test_slots_stay_stable_across_shedding() {
    elsex::ActiveSlots active;
    for (idx identity = 0; identity < 6; ++identity) {
        check::that(active.insert(identity) == identity, "slots are appended in order");
    }
    const std::vector<idx> before = active.identities();

    // Two newcomers arrive, then two incumbents are shed. The newcomers must
    // land in the vacated slots so every survivor keeps its row and column.
    active.insert(100);
    active.insert(101);
    std::vector<bool> remove(active.size(), false);
    remove[1] = true;
    remove[4] = true;
    active.retain(remove, 6);

    check::that(active.size() == 6, "capacity is preserved");
    for (idx slot : {idx(0), idx(2), idx(3), idx(5)}) {
        check::that(active.identities()[slot] == before[slot], "survivors keep their slot");
    }
    check::that(active.identities()[1] == 100, "a newcomer fills the first hole");
    check::that(active.identities()[4] == 101, "a newcomer fills the second hole");

    const auto changed = active.changed_since(before);
    check::that(changed.size() == 2, "exactly two slots changed");
    check::that(changed[0] == 1 && changed[1] == 4, "the changed slots are the vacated ones");

    check::that(active.find(100) == 1, "lookup follows the new occupant");
    check::that(active.find(1) == elsex::ActiveSlots::invalid(), "a shed state is gone");
    check::done("stable slots keep the operator difference low rank");
}

int main() {
    std::printf("elsex factor update\n");
    test_correction_matches_refactorization();
    test_correction_handles_many_right_hand_sides();
    test_delta_is_exactly_low_rank();
    test_slots_stay_stable_across_shedding();
    return check::report("elsex factor update");
}
