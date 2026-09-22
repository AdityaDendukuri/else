// The three regimes agree on Z, and each reuse mechanism reproduces a
// fresh factorization of the changed matrix.
#include "check.hpp"
#include "fixtures.hpp"
#include <cstdio>

namespace {

using namespace subsweep;
using fixtures::with_regime;
constexpr idx order = 24;

template <typename A, typename B>
void expect_same_solves(const A &a, const B &b, const char *what, real tolerance = 1e-11) {
    const idx n = a.size();
    const num::vec rhs = fixtures::random_vector(n, 7);
    const num::vec x = a.solve(rhs), y = b.solve(rhs);
    const num::vec xt = a.solve_transpose(rhs), yt = b.solve_transpose(rhs);
    for (idx i = 0; i < n; ++i) {
        check::close(x[i], y[i], std::string(what) + ": Z b", tolerance);
        check::close(xt[i], yt[i], std::string(what) + ": Z^T b", tolerance);
    }
    num::mat block(n, 3, 0.0);
    for (idx i = 0; i < n; ++i)
        for (idx k = 0; k < 3; ++k)
            block(i, k) = rhs[(i + 5 * k) % n];
    const num::mat xb = a.solve(block), yb = b.solve(block);
    const num::mat xbt = a.solve_transpose(block), ybt = b.solve_transpose(block);
    for (idx i = 0; i < n; ++i)
        for (idx k = 0; k < 3; ++k) {
            check::close(xb(i, k), yb(i, k), std::string(what) + ": Z B", tolerance);
            check::close(xbt(i, k), ybt(i, k), std::string(what) + ": Z^T B", tolerance);
        }
}

factorization factor_of(const subnetwork &sn) {
    return sn.factor();
}

} // namespace

void test_exit_rates_are_the_row_sums() {
    const subnetwork sn = fixtures::birth_death(12, {});
    const num::vec ones(12, 1.0), w = sn.exit_rates();
    num::vec row_sums(12, 0.0);
    num::sparse_matvec(sn.matrix(), ones, row_sums);
    for (idx j = 0; j < 12; ++j)
        check::close(w[j], row_sums[j], "w = R_bar 1", 1e-12);
    check::that(w[0] > 0.0 && w[11] > 0.0 && w[5] == 0.0, "only the window edges exit");
    check::done("exit rates are the row sums of R_bar");
}

void test_regimes_agree() {
    const factorization dense = factor_of(fixtures::birth_death(order, with_regime(solve_regime::dense)));
    const factorization block = factor_of(fixtures::birth_death(order, with_regime(solve_regime::block)));
    const factorization sparse = factor_of(fixtures::birth_death(order, with_regime(solve_regime::sparse)));
    expect_same_solves(dense, block, "dense vs block");
    expect_same_solves(dense, sparse, "dense vs sparse");
    const num::vec dense_diagonal = dense.diagonal(), block_diagonal = block.diagonal();
    for (idx j = 0; j < order; ++j)
        check::close(dense_diagonal[j], block_diagonal[j], "diag(Z) dense vs block", 1e-11);
    check::done("dense, block, and sparse regimes agree");
}

void test_reversible_regimes_agree_with_lu() {
    const factorization dense = factor_of(fixtures::reversible_chain(order, with_regime(solve_regime::dense)));
    const factorization block = factor_of(fixtures::reversible_chain(order, with_regime(solve_regime::block)));
    const factorization sparse = factor_of(fixtures::reversible_chain(order, with_regime(solve_regime::sparse)));
    check::that(dense.reversible() && block.reversible(), "stationary weights select Cholesky");
    subnetwork plain = fixtures::reversible_chain(order, with_regime(solve_regime::dense));
    const factorization lu(plain.matrix(), plain.levels(), {}, plain.options());
    check::that(!lu.reversible(), "no weights select LU");
    expect_same_solves(lu, dense, "LU vs dense Cholesky");
    expect_same_solves(lu, block, "LU vs block Cholesky");
    expect_same_solves(lu, sparse, "LU vs sparse");
    check::done("the reversible similarity reproduces the LU solves");
}

void test_woodbury_matches_refactorization() {
    using namespace num::ops;
    for (const array<idx> &changed : array<array<idx>>{{5}, {2, 11}, {0, 7, 18, 23}}) {
        const sweep_options options = with_regime(solve_regime::dense);
        const subnetwork base = fixtures::birth_death(order, options);
        const subnetwork current = fixtures::birth_death(order, options, changed);
        const factorization base_factor = base.factor(), fresh = current.factor();
        const low_rank_delta delta = row_column_delta(base.matrix(), current.matrix(), changed);
        const num::mat difference = num::dense(current.matrix()) - num::dense(base.matrix());
        const num::mat product = delta.left * num::transpose(delta.right);
        for (idx i = 0; i < order; ++i)
            for (idx j = 0; j < order; ++j)
                check::close(product(i, j), difference(i, j), "P Q^T is the difference");
        const woodbury corrected(base_factor, delta);
        check::that(corrected.rank() == 2 * changed.size(), "rank is twice the changed slots");
        expect_same_solves(fresh, corrected, "Woodbury vs fresh LU");
        const num::vec fresh_diagonal = fresh.diagonal();
        const num::vec corrected_diagonal = corrected.diagonal(base_factor.diagonal().span());
        for (idx j = 0; j < order; ++j)
            check::close(fresh_diagonal[j], corrected_diagonal[j], "diag(Z) by Woodbury", 1e-11);
    }
    check::done("the Woodbury correction reproduces a fresh factorization");
}

void test_block_suffix_matches_refactorization() {
    const array<idx> changed{17};
    {
        const sweep_options options = with_regime(solve_regime::block);
        const subnetwork base = fixtures::birth_death(order, options);
        const subnetwork current = fixtures::birth_death(order, options, changed);
        block_suffix_info info;
        const auto updated = base.factor().update_suffix(current.matrix(), current.levels(), {},
                                                         changed, &info);
        check::that(updated.has_value(), "the LU suffix update is accepted");
        check::that(info.reused_prefix_blocks == 17 && info.reused_prefix_states == 17,
                    "levels before the changed state are reused");
        expect_same_solves(current.factor(), *updated, "block LU suffix vs fresh");
    }
    {
        const sweep_options options = with_regime(solve_regime::block);
        const subnetwork base = fixtures::reversible_chain(order, options);
        const subnetwork current = fixtures::reversible_chain(order, options, changed);
        block_suffix_info info;
        const auto updated = base.factor().update_suffix(current.matrix(), current.levels(),
                                                         current.stationary(), changed, &info);
        check::that(updated.has_value(), "the Cholesky suffix update is accepted");
        check::that(info.reused_prefix_blocks == 5, "levels before the changed level are reused");
        expect_same_solves(current.factor(), *updated, "block Cholesky suffix vs fresh");
    }
    check::done("the block suffix update reproduces a fresh factorization");
}

void test_probed_diagonal_converges() {
    sweep_options sparse = with_regime(solve_regime::sparse);
    sparse.probes.count = 4000;
    sparse.probes.seed = 11;
    for (const bool reversible : {true, false}) {
        const num::vec exact =
            (reversible ? fixtures::reversible_chain(order, {}) : fixtures::birth_death(order, {}))
                .factor()
                .diagonal();
        const num::vec probed = (reversible ? fixtures::reversible_chain(order, sparse)
                                            : fixtures::birth_death(order, sparse))
                                    .factor()
                                    .diagonal();
        for (idx j = 0; j < order; ++j)
            check::close(probed[j] / exact[j], 1.0,
                         reversible ? "reversible probe" : "intrinsic-scaling probe", 0.1);
    }
    check::done("the probed diagonal estimates diag(Z) with and without a stationary measure");
}

int main() {
    std::printf("subsweep factorization\n");
    test_exit_rates_are_the_row_sums();
    test_regimes_agree();
    test_reversible_regimes_agree_with_lu();
    test_woodbury_matches_refactorization();
    test_block_suffix_matches_refactorization();
    test_probed_diagonal_converges();
    return check::report("subsweep factorization");
}
