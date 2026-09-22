// A factorization of R_bar in one of the paper's three regimes: dense
// (no-pivot LU, or Cholesky of H R_bar H^-1 when stationary weights are
// given), block (block LU or Cholesky in the level ordering), sparse (sparse
// LU). It applies Z = R_bar^-1 and Z^T in place, produces diag(Z), and
// refactors from the first changed level in the block regime.
#pragma once

#include "subsweep/linear/reversible.hpp"
#include "subsweep/types.hpp"
#include "kernel/factor.hpp"
#include "linear/factorization/block_tridiagonal.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu_no_pivot.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>

namespace subsweep {

class factorization;
[[nodiscard]] num::vec probed_diagonal(const factorization &f);

namespace detail {

// The first level touched by a changed slot in either layout, or no_slot
// when the two layouts disagree on the prefix before it.
template <typename Factor>
[[nodiscard]] idx first_changed_block(const Factor &base, const num::detail::block_layout &current,
                                      view<const idx> changed) {
    array<idx> base_position(base.size);
    for (idx position = 0; position < base.size; ++position)
        base_position[base.order[position]] = position;
    idx first = current.offsets.size() - 1;
    for (idx state : changed) {
        first = std::min(first, current.block_of[state]);
        const auto boundary =
            std::upper_bound(base.offsets.begin(), base.offsets.end(), base_position[state]);
        first = std::min(first, static_cast<idx>(boundary - base.offsets.begin() - 1));
    }
    if (first > base.blocks() || first + 1 > current.offsets.size())
        return static_cast<idx>(-1);
    for (idx k = 0; k <= first; ++k)
        if (base.offsets[k] != current.offsets[k])
            return static_cast<idx>(-1);
    for (idx position = 0; position < current.offsets[first]; ++position)
        if (base.order[position] != current.order[position])
            return static_cast<idx>(-1);
    return first;
}

} // namespace detail


struct block_suffix_info {
    idx block_count = 0;
    idx reused_prefix_blocks = 0;
    idx reused_prefix_states = 0;
};

// A factorization of R_bar. Dense: no-pivot LU, or Cholesky of the
// symmetrized H R_bar H^-1 when stationary weights are given. Block: block
// LU or block Cholesky in the level ordering. Sparse: sparse LU, with
// diag(Z) estimated by Gaussian probing.
class factorization {
  public:
    factorization(const num::spmat &matrix, view<const idx> levels, view<const real> stationary,
                  const sweep_options &options)
        : kind_(options.regime), n_(matrix.n_rows()), probes_(options.probes) {
        const bool reversible = stationary.size() != 0;
        if (kind_ == solve_regime::block && levels.size() != n_)
            throw std::invalid_argument("the block regime needs one level per state");
        // The sparse regime factors R_bar itself; the weights only serve its probe.
        if (reversible && kind_ != solve_regime::sparse)
            weights_ = detail::similarity_weights(stationary, n_);
        switch (kind_) {
        case solve_regime::dense:
            if (reversible) {
                auto factor = num::cholesky(num::assume_spd(
                    num::dense(detail::similarity_scaled(matrix, weights_.span()))));
                if (!factor.success)
                    throw std::runtime_error("the symmetrized rate matrix is not positive definite");
                factor_ = std::move(factor);
            } else {
                auto factor = num::factor_no_pivot(num::assume_square(num::dense(matrix)));
                if (factor.singular)
                    throw std::runtime_error("the truncated rate matrix is singular");
                factor_ = std::move(factor);
            }
            break;
        case solve_regime::block:
            if (reversible)
                factor_ = num::factor_block_cholesky(
                    detail::similarity_scaled(matrix, weights_.span()), levels);
            else
                factor_ = num::factor_block_lu(matrix, levels);
            break;
        case solve_regime::sparse:
            factor_ = std::make_shared<num::auto_linear_solver>(matrix);
            sparse_matrix_ = matrix;
            if (reversible)
                sparse_weights_ = detail::similarity_weights(stationary, n_);
            break;
        }
    }

    [[nodiscard]] idx size() const { return n_; }
    [[nodiscard]] solve_regime kind() const { return kind_; }
    [[nodiscard]] bool reversible() const { return weights_.size() != 0; }

    [[nodiscard]] num::vec solve(const num::vec &rhs) const { return apply(rhs, false); }
    [[nodiscard]] num::mat solve(const num::mat &rhs) const { return apply(rhs, false); }
    [[nodiscard]] num::vec solve_transpose(const num::vec &rhs) const { return apply(rhs, true); }
    [[nodiscard]] num::mat solve_transpose(const num::mat &rhs) const { return apply(rhs, true); }

    // The same solves into `out`, which may be `rhs` itself; nothing is
    // allocated once `out` and the internal workspace have the right size.
    void solve(const num::vec &rhs, num::vec &out) const { apply(rhs, out, false); }
    void solve(const num::mat &rhs, num::mat &out) const { apply(rhs, out, false); }
    void solve_transpose(const num::vec &rhs, num::vec &out) const { apply(rhs, out, true); }
    void solve_transpose(const num::mat &rhs, num::mat &out) const { apply(rhs, out, true); }

    // diag(Z): by basis-vector solves in the dense and block regimes, by the
    // randomized estimator in the sparse regime.
    [[nodiscard]] num::vec diagonal() const {
        return kind_ == solve_regime::sparse ? probed_diagonal(*this) : exact_diagonal();
    }

    // What the probe needs of a sparse factorization: R_bar, h = sqrt(pi)
    // when the chain is reversible (empty otherwise), and the options.
    [[nodiscard]] const num::spmat &sparse_matrix() const { return *sparse_matrix_; }
    [[nodiscard]] const num::vec &sparse_weights() const { return sparse_weights_; }
    [[nodiscard]] const probe_options &probes() const { return probes_; }

    // Block regime: refactor from the first level touched by `changed`,
    // copying the factors of the levels before it. Nullopt when the prefix
    // layouts differ or the regime has no suffix structure.
    [[nodiscard]] std::optional<factorization>
    update_suffix(const num::spmat &matrix, view<const idx> levels, view<const real> stationary,
                  view<const idx> changed, block_suffix_info *info = nullptr) const {
        if (kind_ != solve_regime::block || matrix.n_rows() != n_ || levels.size() != n_)
            return std::nullopt;
        const num::detail::block_layout layout = num::detail::build_block_order(levels);
        try {
            factorization updated(*this);
            if (const auto *lu = std::get_if<num::block_lu_factor>(&factor_)) {
                const idx first = detail::first_changed_block(*lu, layout, changed);
                if (first == static_cast<idx>(-1))
                    return std::nullopt;
                record(info, layout, first);
                updated.factor_ = num::refactor_block_lu_suffix(matrix, levels, *lu, first);
            } else {
                const auto &ch = std::get<num::block_cholesky_factor>(factor_);
                const idx first = detail::first_changed_block(ch, layout, changed);
                if (first == static_cast<idx>(-1))
                    return std::nullopt;
                record(info, layout, first);
                updated.weights_ = detail::similarity_weights(stationary, n_);
                updated.factor_ = num::refactor_block_cholesky_suffix(
                    detail::similarity_scaled(matrix, updated.weights_.span()), levels, ch,
                    first);
            }
            return updated;
        } catch (const std::exception &) {
            return std::nullopt;
        }
    }

  private:
    template <typename RightHandSide>
    [[nodiscard]] RightHandSide apply(const RightHandSide &rhs, bool transpose) const {
        RightHandSide output;
        apply(rhs, output, transpose);
        return output;
    }

    // Reversible: R_bar = H^-1 L H, so Z b = H^-1 L^-1 H b and Z^T b = H L^-1 H^-1 b.
    // The dense and block factors solve in place on `out`; the sparse solver
    // writes `out` from `rhs`.
    template <typename RightHandSide>
    void apply(const RightHandSide &rhs, RightHandSide &out, bool transpose) const {
        constexpr bool is_vector = std::is_same_v<RightHandSide, num::vec>;
        const idx columns = is_vector ? 1 : rhs_columns(rhs);
        if (kind_ == solve_regime::sparse) {
            const auto &f = std::get<std::shared_ptr<num::auto_linear_solver>>(factor_);
            if (transpose)
                f->solve_transpose(rhs, out);
            else
                f->solve(rhs, out);
            return;
        }
        if (&rhs != &out) {
            if constexpr (is_vector) {
                if (out.size() != rhs.size())
                    out = num::vec(rhs.size(), 0.0);
            } else {
                if (out.rows() != rhs.rows() || out.cols() != rhs.cols())
                    out = num::mat(rhs.rows(), rhs.cols(), 0.0);
            }
            std::copy_n(rhs.data(), n_ * columns, out.data());
        }
        if (reversible())
            detail::scale_rows(out, weights_, transpose);
        real *x = out.data();
        std::visit(
            [&](const auto &f) {
                using F = std::decay_t<decltype(f)>;
                if constexpr (std::is_same_v<F, num::no_pivot_lu>) {
                    if (transpose)
                        num::kernel::lu_no_pivot_solve_transpose_multiple(x, f.packed.data(), n_,
                                                                          columns);
                    else
                        num::kernel::lu_no_pivot_solve_multiple(x, f.packed.data(), n_, columns);
                } else if constexpr (std::is_same_v<F, num::cholesky_result>) {
                    num::kernel::trsm_lower_inplace(x, columns, f.L.data(), n_, columns);
                    num::kernel::trsm_lower_transpose_inplace(x, columns, f.L.data(), n_, columns);
                } else if constexpr (std::is_same_v<F, num::block_lu_factor>) {
                    if (work_.size() < n_ * columns)
                        work_ = num::vec(n_ * columns, 0.0);
                    if (transpose)
                        num::solve_transpose_in_place(f, x, columns, work_.data());
                    else
                        num::solve_in_place(f, x, columns, work_.data());
                } else if constexpr (std::is_same_v<F, num::block_cholesky_factor>) {
                    if (work_.size() < n_ * columns)
                        work_ = num::vec(n_ * columns, 0.0);
                    num::solve_in_place(f, x, columns, work_.data());
                } else {
                    throw std::logic_error("empty factorization");
                }
            },
            factor_);
        if (reversible())
            detail::scale_rows(out, weights_, !transpose);
    }

    [[nodiscard]] static idx rhs_columns(const num::mat &rhs) { return rhs.cols(); }
    [[nodiscard]] static idx rhs_columns(const num::vec &) { return 1; }

    [[nodiscard]] num::vec exact_diagonal() const {
        num::vec diagonal(n_, 0.0);
        constexpr idx columns_per_solve = 64;
        for (idx first = 0; first < n_; first += columns_per_solve) {
            const idx count = std::min(columns_per_solve, n_ - first);
            num::mat basis(n_, count, 0.0);
            for (idx column = 0; column < count; ++column)
                basis(first + column, column) = 1.0;
            const num::mat columns = solve(basis);
            for (idx column = 0; column < count; ++column)
                diagonal[first + column] = columns(first + column, column);
        }
        return diagonal;
    }

    static void record(block_suffix_info *info, const num::detail::block_layout &layout,
                       idx first) {
        if (info)
            *info = {.block_count = static_cast<idx>(layout.offsets.size() - 1),
                     .reused_prefix_blocks = first,
                     .reused_prefix_states = layout.offsets[first]};
    }

    solve_regime kind_;
    idx n_;
    probe_options probes_;
    num::vec weights_; // h = sqrt(pi); empty unless reversible
    mutable num::vec work_; // block solves
    std::variant<std::monostate, num::no_pivot_lu, num::cholesky_result, num::block_lu_factor,
                 num::block_cholesky_factor, std::shared_ptr<num::auto_linear_solver>>
        factor_;
    std::optional<num::spmat> sparse_matrix_; // for the probe
    num::vec sparse_weights_;
};

// ---------------------------------------------------------------------------

} // namespace subsweep

#include "subsweep/linear/probe.hpp"
