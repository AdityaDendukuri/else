// The three state-shedding rules.
#pragma once

#include "container/matrix_expr.hpp"
#include "elsex/subnetwork.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/inverse_diagonal.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace elsex {

/// The quantities all three shedding rules share.
///
/// \f$\rho = \sum_k (c_k/m_{\rm active})\delta_{i_k}\f$ is the entrance mixture,
/// \f$u = Z^T\rho\f$ the expected occupation of each state under it, and
/// \f$q = Z\mathbf 1\f$ the expected time to escape starting from each state.
struct SheddingState {
    num::Vector entrance_mixture; ///< \f$\rho\f$
    num::Vector occupation;       ///< \f$u = Z^T\rho\f$
    num::Vector exit_time;        ///< \f$q = Z\mathbf 1\f$
};

/// Solve \f$M^Tu = \rho\f$ and \f$Mq = \mathbf 1\f$ against the stored factorization.
template <typename State>
[[nodiscard]] SheddingState shedding_state(const Subnetwork<State> &subnetwork,
                                           num::Vector entrance_mixture) {
    if (entrance_mixture.size() != subnetwork.size()) {
        throw std::invalid_argument("entrance mixture size must match the subnetwork");
    }
    num::Vector occupation = subnetwork.solve_transpose(entrance_mixture);
    num::Vector exit_time = subnetwork.solve(num::Vector(subnetwork.size(), 1.0));
    return {std::move(entrance_mixture), std::move(occupation), std::move(exit_time)};
}

/// Exact cut-time loss \f$\ell_j = u_jq_j/Z_{jj}\f$.
///
/// This is the time the subnetwork loses by removing `j`, and it is exact rather
/// than a heuristic: the principal downdate of \f$M^{-1}\f$ telescopes to this
/// single ratio.
template <typename State>
[[nodiscard]] num::Vector exact_cut_time_losses(const Subnetwork<State> &subnetwork,
                                                const SheddingState &state) {
    const num::Vector diagonal_of_inverse = num::inverse_diagonal(subnetwork.factor());
    num::Vector losses(subnetwork.size(), 0.0);
    for (idx j = 0; j < subnetwork.size(); ++j) {
        losses[j] = diagonal_of_inverse[j] > 0.0
                        ? state.occupation[j] * state.exit_time[j] / diagonal_of_inverse[j]
                        : 0.0;
    }
    return losses;
}

/// Expected number of entries \f$s_j = (-R_{jj})u_j - \rho_j\f$.
///
/// The cheapest of the three: one transpose solve and the operator diagonal, no
/// inverse diagonal at all.
template <typename State>
[[nodiscard]] num::Vector expected_entries(const Subnetwork<State> &subnetwork,
                                           const SheddingState &state) {
    const num::Vector operator_diagonal = num::diagonal(subnetwork.operator_matrix());
    num::Vector entries(subnetwork.size(), 0.0);
    for (idx j = 0; j < subnetwork.size(); ++j) {
        entries[j] = operator_diagonal[j] * state.occupation[j] - state.entrance_mixture[j];
    }
    return entries;
}

/// Joint cut-time loss for removing a whole set at once, with its backward residual.
struct JointCutTime {
    real loss = 0.0;              ///< \f$\Delta\mathcal T = q_J^T Z_{JJ}^{-1}u_J\f$
    real backward_residual = 0.0; ///< \f$\lVert Z_{JJ}c-u_J\rVert_\infty/\lVert u_J\rVert_\infty\f$
};

/// Cut time lost by removing `removed` all together.
///
/// The single-state lemma generalizes through the principal block of \f$Z\f$ on
/// the removed set. Substituting the block downdate
/// \f$M_{II}^{-1} = Z_{II} - Z_{IJ}Z_{JJ}^{-1}Z_{JI}\f$ into
/// \f$\mathcal T - \mathcal T_{-J}\f$ and using \f$\rho_J = 0\f$ telescopes to
/// \f[
///   \Delta\mathcal T = u_J^T Z_{JJ}^{-1} q_J,
/// \f]
/// which reduces to \f$u_jq_j/Z_{jj}\f$ for a single state.
///
/// The order matters: \f$Z_{JJ}\f$ is a principal block of a generally
/// nonsymmetric \f$Z\f$, so \f$q_J^TZ_{JJ}^{-1}u_J\f$ is a different number. The
/// two agree only for a one-by-one block, which is why a single-state check
/// cannot catch the transposed form.
///
/// The reported residual is the *actual* backward error of the block solve, not
/// an accumulated epsilon bound. An epsilon accumulator over `r` terms grows like
/// `r * eps` and so measures the loop length rather than the conditioning.
template <typename State>
[[nodiscard]] JointCutTime joint_cut_time_loss(const Subnetwork<State> &subnetwork,
                                               const SheddingState &state,
                                               std::span<const idx> removed) {
    const idx count = removed.size();
    if (count == 0) {
        return {};
    }
    for (idx j : removed) {
        if (j >= subnetwork.size()) {
            throw std::out_of_range("removed state is outside the subnetwork");
        }
    }

    const num::Matrix block = num::inverse_principal_block(subnetwork.factor(), removed);
    num::Vector occupation(count, 0.0);
    num::Vector exit_time(count, 0.0);
    for (idx i = 0; i < count; ++i) {
        occupation[i] = state.occupation[removed[i]];
        exit_time[i] = state.exit_time[removed[i]];
    }

    // c = Z_JJ^-1 q_J, then the loss is u_J . c.
    const auto factor = num::lu(num::make_square(block));
    num::Vector corrected(count, 0.0);
    num::lu_solve(factor, exit_time, corrected);

    real loss = 0.0;
    for (idx i = 0; i < count; ++i) {
        loss += occupation[i] * corrected[i];
    }

    const num::Vector reconstructed = num::matvec(block, corrected);
    real residual = 0.0;
    real scale = 0.0;
    for (idx i = 0; i < count; ++i) {
        residual = std::max(residual, std::abs(reconstructed[i] - exit_time[i]));
        scale = std::max(scale, std::abs(exit_time[i]));
    }
    return {loss, residual / std::max(scale, 1.0)};
}

namespace detail {

/// \f$HMH^{-1}\f$ with \f$H = \mathrm{diag}(weights)\f$, preserving the sparsity pattern.
[[nodiscard]] inline num::SparseMatrix similarity_scaled(const num::SparseMatrix &matrix,
                                                         std::span<const real> weights) {
    std::vector<real> values(matrix.nnz());
    std::vector<idx> columns(matrix.col_idx(), matrix.col_idx() + matrix.nnz());
    std::vector<idx> row_offsets(matrix.row_ptr(), matrix.row_ptr() + matrix.n_rows() + 1);
    for (idx i = 0; i < matrix.n_rows(); ++i) {
        for (idx k = matrix.row_ptr()[i]; k < matrix.row_ptr()[i + 1]; ++k) {
            values[k] = matrix.values()[k] * weights[i] / weights[matrix.col_idx()[k]];
        }
    }
    return num::SparseMatrix(matrix.n_rows(), matrix.n_cols(), std::move(values),
                             std::move(columns), std::move(row_offsets));
}

} // namespace detail

/// Hutchinson probe estimate of \f$\mathrm{diag}(Z)\f$ after stationary normalization.
///
/// Diagonal similarity preserves the inverse diagonal, so with
/// \f$H = \mathrm{diag}(\sqrt{\pi_S})\f$, \f$\tilde M = HMH^{-1}\f$ and
/// \f$KK^T = (\tilde M + \tilde M^T)/2\f$, the rows of \f$\Phi = \tilde M^{-1}K\f$
/// satisfy \f$Z_{jj} = \lVert\Phi_{j,:}\rVert^2\f$. Probing with random signs
/// \f$\Xi\f$ estimates that in `probes` solves rather than `n`.
///
/// Symmetrization only supplies the SPD factor \f$K\f$; the rate matrix itself
/// stays nonsymmetric, and \f$S^{-1} \ne \tilde M^{-1}\f$ in general.
template <typename State>
[[nodiscard]] num::Vector probed_inverse_diagonal(const Subnetwork<State> &subnetwork,
                                                  std::span<const real> stationary, idx probes,
                                                  unsigned seed = 42) {
    const idx n = subnetwork.size();
    if (stationary.size() != n) {
        throw std::invalid_argument("stationary weights must have one entry per state");
    }
    if (probes == 0) {
        throw std::invalid_argument("probe estimate requires at least one probe");
    }

    std::vector<real> weights(n);
    for (idx j = 0; j < n; ++j) {
        if (!(stationary[j] > 0.0)) {
            throw std::invalid_argument("stationary weights must be positive");
        }
        weights[j] = std::sqrt(stationary[j]);
    }

    const num::SparseMatrix normalized =
        detail::similarity_scaled(subnetwork.operator_matrix(), weights);
    const num::Matrix dense_normalized = num::dense(normalized);
    const num::Matrix symmetric_part =
        num::scaled(num::add(dense_normalized, num::transpose(dense_normalized)), 0.5);

    // The paper proves this symmetric part is positive definite for an
    // irreducible chain restricted to a proper subset, so the invariant is
    // established by theorem rather than re-verified on every shedding call.
    const auto symmetric_factor = num::cholesky(num::assume_spd(symmetric_part));
    if (!symmetric_factor.success) {
        throw std::runtime_error("stationary-normalized operator is not positive definite");
    }

    std::mt19937 generator(seed);
    std::bernoulli_distribution sign(0.5);
    num::Matrix probe(n, probes, 0.0);
    for (idx j = 0; j < n; ++j) {
        for (idx p = 0; p < probes; ++p) {
            probe(j, p) = sign(generator) ? 1.0 : -1.0;
        }
    }

    const num::AutoLinearSolver normalized_factor(normalized);
    const num::Matrix probed =
        num::solve(normalized_factor, num::matmul(symmetric_factor.L, probe));

    num::Vector estimate(n, 0.0);
    for (idx j = 0; j < n; ++j) {
        real total = 0.0;
        for (idx p = 0; p < probes; ++p) {
            total += probed(j, p) * probed(j, p);
        }
        estimate[j] = total / static_cast<real>(probes);
    }
    return estimate;
}

/// Cut-time loss using the probed inverse diagonal in place of the exact one.
template <typename State>
[[nodiscard]] num::Vector
probed_cut_time_losses(const Subnetwork<State> &subnetwork, const SheddingState &state,
                       std::span<const real> stationary, idx probes, unsigned seed = 42) {
    const num::Vector estimate = probed_inverse_diagonal(subnetwork, stationary, probes, seed);
    num::Vector losses(subnetwork.size(), 0.0);
    for (idx j = 0; j < subnetwork.size(); ++j) {
        losses[j] =
            estimate[j] > 0.0 ? state.occupation[j] * state.exit_time[j] / estimate[j] : 0.0;
    }
    return losses;
}

/// The `count` eligible states with the smallest score.
///
/// Protected states are never returned. Ties break by index so the selection is
/// reproducible.
[[nodiscard]] inline std::vector<idx>
lowest_scores(std::span<const real> scores, std::span<const idx> protected_states, idx count) {
    std::vector<bool> ineligible(scores.size(), false);
    for (idx j : protected_states) {
        if (j < scores.size()) {
            ineligible[j] = true;
        }
    }

    std::vector<idx> eligible;
    eligible.reserve(scores.size());
    for (idx j = 0; j < scores.size(); ++j) {
        if (!ineligible[j]) {
            eligible.push_back(j);
        }
    }

    count = std::min(count, static_cast<idx>(eligible.size()));
    std::partial_sort(
        eligible.begin(), eligible.begin() + count, eligible.end(),
        [scores](idx a, idx b) { return scores[a] != scores[b] ? scores[a] < scores[b] : a < b; });
    eligible.resize(count);
    return eligible;
}

} // namespace elsex
