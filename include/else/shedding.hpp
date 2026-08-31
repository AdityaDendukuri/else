#pragma once

#include "else/linalg.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace else_sim {

template <typename Float, typename Index, typename State>
class Subnetwork;

/// Expected departures w_j u_j; subtract initial mass rho_j to obtain expected later entries.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
inline std::vector<Float> expected_visit_scores(const Subnetwork<Float, Index, State> &subnetwork,
                                                const std::vector<Float> &occupancy,
                                                std::span<const Index> candidate_indices) {
    std::vector<Float> losses;
    losses.reserve(candidate_indices.size());
    for (Index idx : candidate_indices) {
        if (idx >= subnetwork.size()) {
            throw std::out_of_range("candidate state index out of range");
        }
        const Float tau_i = static_cast<Float>(occupancy[idx]);
        const Float q_ii = -subnetwork.generator()(idx, idx);
        const Float loss = (q_ii > static_cast<Float>(0)) ? (tau_i * q_ii) : tau_i;
        losses.push_back(loss);
    }
    return losses;
}

/// Dirichlet scores after the stationary scaling phi_i = u_i/sqrt(pi_i).
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
inline std::vector<Float> symmetrized_scores(const Subnetwork<Float, Index, State> &subnetwork,
                                             const std::vector<Float> &occupancy,
                                             std::span<const Index> candidate_indices) {
    if (!subnetwork.is_reversible()) {
        throw std::invalid_argument("Symmetrized shedding requires a reversible Markov chain");
    }
    const auto weights = subnetwork.stationary_sqrt();
    std::vector<Float> losses;
    losses.reserve(candidate_indices.size());

    const auto &R = subnetwork.generator();
    for (Index idx : candidate_indices) {
        if (idx >= subnetwork.size()) {
            throw std::out_of_range("candidate state index out of range");
        }
        const Float u_i = static_cast<Float>(occupancy[idx]);
        const Float h_i = weights[idx];
        const Float phi_i = (h_i > static_cast<Float>(0)) ? (u_i / h_i) : static_cast<Float>(0);

        Float s_ii = -R(idx, idx);
        Float dirichlet_sum = s_ii * (phi_i * phi_i);

        const auto row_start = R.row_ptr[idx];
        const auto row_stop = R.row_ptr[idx + 1];
        for (auto k = row_start; k < row_stop; ++k) {
            const Index j = R.col_idx[k];
            if (j != idx && j < subnetwork.size()) {
                const Float u_j = static_cast<Float>(occupancy[j]);
                const Float h_j = weights[j];
                const Float phi_j =
                    (h_j > static_cast<Float>(0)) ? (u_j / h_j) : static_cast<Float>(0);
                const Float s_ij = -R.values[k] * (h_j / h_i);
                const Float diff = phi_i - phi_j;
                dirichlet_sum += s_ij * (diff * diff);
            }
        }
        losses.push_back(std::max(static_cast<Float>(0), dirichlet_sum));
    }
    return losses;
}

/// Exact one-state Schur loss u_j q_j/Z_jj in the notation of the paper.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
inline std::vector<Float> exact_cut_time_scores(const Subnetwork<Float, Index, State> &subnetwork,
                                                const std::vector<Float> &occupancy,
                                                std::span<const Index> candidate_indices) {
    return subnetwork.cut_time_losses(occupancy, candidate_indices);
}

/// Indices of the `count` smallest unprotected scores.
template <typename Float = double, typename Index = std::size_t>
inline std::vector<Index> lowest_scores(std::span<const Float> scores,
                                        std::span<const Index> protected_indices, Index count) {
    std::vector<bool> protected_state(scores.size(), false);
    for (Index i : protected_indices)
        protected_state[i] = true;

    std::vector<std::pair<Float, Index>> ranked;
    for (Index i = 0; i < scores.size(); ++i)
        if (!protected_state[i])
            ranked.push_back({scores[i], i});

    std::sort(ranked.begin(), ranked.end());
    count = std::min(count, static_cast<Index>(ranked.size()));
    std::vector<Index> selected(count);
    for (Index i = 0; i < count; ++i)
        selected[i] = ranked[i].second;
    return selected;
}

} // namespace else_sim
