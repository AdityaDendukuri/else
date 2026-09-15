/// @file tests/test_elsex_factor_update.cpp
/// @brief Woodbury reuse of a stored factorization, and stable slots.
///
/// A corrected solve is an algebraic identity, not an approximation, so it is
/// checked against a full refactorization of the updated operator rather than
/// against a tolerance chosen by hand. Both directions are checked: ELSE reaches
/// for the transpose solve more often than the forward one.

#include "check.hpp"

#include "else/core/state_graph.hpp"
#include "else/quantities/factor_update.hpp"
#include "else/restriction/restriction.hpp"

#include "container/matrix_expr.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"

#include "stochastic/rng.hpp"
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
num::spmat build_operator(num::view<const idx> perturbed) {
    num::array<bool> changed(order, false);
    for (idx j : perturbed) {
        changed[j] = true;
    }

    num::array<idx> rows, cols;
    num::array<real> values;
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
    return num::spmat::from_triplets(order, order, rows, cols, values);
}

num::vec make_rhs(unsigned seed) {
    num::rng generator(seed);
    std::uniform_real_distribution<real> spread(-1.0, 1.0);
    num::vec b(order, 0.0);
    for (idx i = 0; i < order; ++i) {
        b[i] = spread(generator);
    }
    return b;
}

num::spmat build_spd_operator(num::view<const idx> perturbed) {
    num::array<bool> changed(order, false);
    for (idx j : perturbed)
        changed[j] = true;
    num::array<idx> rows, columns;
    num::array<real> values;
    for (idx i = 0; i < order; ++i) {
        rows.push_back(i);
        columns.push_back(i);
        values.push_back(4.0 + (changed[i] ? 0.7 : 0.0));
        if (i + 1 < order) {
            const real coupling = changed[i] || changed[i + 1] ? -0.55 : -0.4;
            rows.push_back(i);
            columns.push_back(i + 1);
            values.push_back(coupling);
            rows.push_back(i + 1);
            columns.push_back(i);
            values.push_back(coupling);
        }
    }
    return num::spmat::from_triplets(order, order, rows, columns, values);
}

} // namespace

void test_correction_matches_refactorization() {
    for (const num::array<idx> &changed :
         num::array<num::array<idx>>{{5}, {2, 11}, {0, 7, 18, 23}}) {
        const auto base = build_operator({});
        const auto current = build_operator(num::view<const idx>(changed));

        const num::auto_linear_solver base_factor(base);
        const num::auto_linear_solver current_factor(current); // ground truth

        const auto delta = else_sim::row_column_delta(base, current, num::view<const idx>(changed));
        const auto update = else_sim::make_factor_update(base_factor, delta.left, delta.right);
        check::that(else_sim::rank(update) == 2 * changed.size(),
                    "rank is twice the changed slot count");

        const num::vec b = make_rhs(11u + static_cast<unsigned>(changed.size()));

        const num::vec corrected_transpose = else_sim::solve_transpose(update, b);
        const num::vec reference_transpose = num::solve_transpose(current_factor, b);
        for (idx i = 0; i < order; ++i) {
            check::close(corrected_transpose[i], reference_transpose[i],
                         "corrected transpose solve equals refactorization", 1e-11);
        }

        check::that(else_sim::relative_residual(num::transpose(current), corrected_transpose, b) <
                        1e-12,
                    "the corrected solve has a small backward residual");
    }
    check::done("Woodbury correction reproduces a full refactorization");
}

void test_correction_handles_many_right_hand_sides() {
    const num::array<idx> changed{3, 9};
    const auto base = build_operator({});
    const auto current = build_operator(num::view<const idx>(changed));

    const num::auto_linear_solver base_factor(base);
    const num::auto_linear_solver current_factor(current);
    const auto delta = else_sim::row_column_delta(base, current, num::view<const idx>(changed));
    const auto update = else_sim::make_factor_update(base_factor, delta.left, delta.right);

    num::mat rhs(order, 3, 0.0);
    for (idx i = 0; i < order; ++i) {
        for (idx c = 0; c < 3; ++c) {
            rhs(i, c) = std::sin(0.7 * static_cast<real>(i) + static_cast<real>(c));
        }
    }

    const num::mat corrected_transpose = else_sim::solve_transpose(update, rhs);
    const num::mat reference_transpose = num::solve_transpose(current_factor, rhs);
    for (idx i = 0; i < order; ++i) {
        for (idx c = 0; c < 3; ++c) {
            check::close(corrected_transpose(i, c), reference_transpose(i, c),
                         "block corrected transpose solve", 1e-11);
        }
    }
    check::done("one correction serves every right-hand side");
}

void test_correction_uses_block_factor() {
    const num::array<idx> changed{9};
    const auto base = build_operator({});
    const auto current = build_operator(num::view<const idx>(changed));
    num::array<idx> levels(order);
    for (idx i = 0; i < order; ++i)
        levels[i] = i / 3;

    const auto base_factor = num::factor_block_lu(base, num::view<const idx>(levels));
    const auto current_factor = num::factor_block_lu(current, num::view<const idx>(levels));
    const auto delta = else_sim::row_column_delta(base, current, num::view<const idx>(changed));
    const auto update = else_sim::make_factor_update(base_factor, delta.left, delta.right);
    const num::vec right_hand_side = make_rhs(73);
    const num::vec corrected = else_sim::solve_transpose(update, right_hand_side);
    num::vec reference;
    num::solve_transpose(current_factor, right_hand_side, reference);
    for (idx i = 0; i < order; ++i)
        check::close(corrected[i], reference[i], "block Woodbury transpose solve", 1e-11);
    check::done("Woodbury correction reuses block-tridiagonal factors");
}

void test_block_lu_suffix_matches_refactorization() {
    const num::array<idx> changed{10, 11};
    const auto base = build_operator({});
    const auto current = build_operator(num::view<const idx>(changed));
    num::array<idx> levels(order);
    for (idx i = 0; i < order; ++i)
        levels[i] = i / 3;
    const auto base_factor = num::factor_block_lu(base, num::view<const idx>(levels));
    const auto reference_factor = num::factor_block_lu(current, num::view<const idx>(levels));
    const auto updated = else_sim::refactor_block_suffix(
        base_factor, current, num::view<const idx>(levels), num::view<const idx>(changed));
    check::that(updated.has_value(), "unchanged block layout accepts an LU suffix update");
    const num::vec right_hand_side = make_rhs(91);
    num::vec reference, reused;
    num::solve_transpose(reference_factor, right_hand_side, reference);
    num::solve_transpose(*updated, right_hand_side, reused);
    for (idx i = 0; i < order; ++i)
        check::close(reused[i], reference[i], "suffix-updated block LU solve", 1e-11);
    check::done("block LU suffix update matches full refactorization");
}

void test_block_cholesky_suffix_matches_refactorization() {
    const num::array<idx> changed{16};
    const auto base = build_spd_operator({});
    const auto current = build_spd_operator(num::view<const idx>(changed));
    num::array<idx> levels(order);
    for (idx i = 0; i < order; ++i)
        levels[i] = i / 3;
    const auto base_factor = num::factor_block_cholesky(base, num::view<const idx>(levels));
    const auto reference_factor =
        num::factor_block_cholesky(current, num::view<const idx>(levels));
    const auto updated = else_sim::refactor_block_suffix(
        base_factor, current, num::view<const idx>(levels), num::view<const idx>(changed));
    check::that(updated.has_value(), "unchanged block layout accepts a Cholesky suffix update");
    const num::vec right_hand_side = make_rhs(109);
    num::vec reference, reused;
    num::solve(reference_factor, right_hand_side, reference);
    num::solve(*updated, right_hand_side, reused);
    for (idx i = 0; i < order; ++i)
        check::close(reused[i], reference[i], "suffix-updated block Cholesky solve", 1e-11);
    check::done("block Cholesky suffix update matches full refactorization");
}

void test_delta_is_exactly_low_rank() {
    using namespace num::ops;
    const num::array<idx> changed{4, 15};
    const auto base = build_operator({});
    const auto current = build_operator(num::view<const idx>(changed));

    const auto delta = else_sim::row_column_delta(base, current, num::view<const idx>(changed));
    const num::mat reconstructed = delta.left * num::transpose(delta.right);

    const num::mat dense_base = num::dense(base);
    const num::mat dense_current = num::dense(current);
    for (idx i = 0; i < order; ++i) {
        for (idx j = 0; j < order; ++j) {
            check::close(reconstructed(i, j), dense_current(i, j) - dense_base(i, j),
                         "U V^T reproduces the difference exactly");
        }
    }
    check::done("the row-and-column delta is exactly U V^T");
}

void test_slots_stay_stable_across_shedding() {
    else_sim::ActiveSlots active;
    for (idx identity = 0; identity < 6; ++identity) {
        check::that(else_sim::insert(active, identity) == identity, "slots are appended in order");
    }
    const num::array<idx> before = active.state_ids;

    // Two newcomers arrive, then two incumbents are shed. The newcomers must
    // land in the vacated slots so every survivor keeps its row and column.
    else_sim::insert(active, 100);
    else_sim::insert(active, 101);
    num::array<bool> remove(else_sim::size(active), false);
    remove[1] = true;
    remove[4] = true;
    else_sim::retain(active, remove, 6);

    check::that(else_sim::size(active) == 6, "capacity is preserved");
    for (idx slot : {idx(0), idx(2), idx(3), idx(5)}) {
        check::that(active.state_ids[slot] == before[slot], "survivors keep their slot");
    }
    check::that(active.state_ids[1] == 100, "a newcomer fills the first hole");
    check::that(active.state_ids[4] == 101, "a newcomer fills the second hole");

    const auto changed = else_sim::changed_since(active, before);
    check::that(changed.size() == 2, "exactly two slots changed");
    check::that(changed[0] == 1 && changed[1] == 4, "the changed slots are the vacated ones");

    check::that(else_sim::find(active, 100) == 1, "lookup follows the new occupant");
    check::that(else_sim::find(active, 1) == else_sim::invalid_index, "a shed state is gone");
    check::done("stable slots keep the operator difference low rank");
}

int main() {
    std::printf("else factor update\n");
    test_correction_matches_refactorization();
    test_correction_handles_many_right_hand_sides();
    test_correction_uses_block_factor();
    test_block_lu_suffix_matches_refactorization();
    test_block_cholesky_suffix_matches_refactorization();
    test_delta_is_exactly_low_rank();
    test_slots_stay_stable_across_shedding();
    return check::report("else factor update");
}
