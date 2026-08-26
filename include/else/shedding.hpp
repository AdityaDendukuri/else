#pragma once

#include "else/linalg.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace else_sim {

template <typename Float, typename Index, typename State>
class Subnetwork;

/// @brief Method 1: Expected Visits / Residence Time Shedding
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>, typename Vec>
inline std::vector<Float>
compute_expected_visits_losses(const Subnetwork<Float, Index, State> &subnetwork,
                               const Vec &occupancy,
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

/// @brief Method 2: Symmetrized Dirichlet Form Upper-Bound Shedding
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>, typename Vec>
inline std::vector<Float>
compute_symmetrized_losses(const Subnetwork<Float, Index, State> &subnetwork,
                           const Vec &occupancy,
                           std::span<const Index> candidate_indices) {
    if (!subnetwork.is_reversible()) {
        throw std::invalid_argument("Symmetrized shedding requires a reversible Markov chain");
    }
    const auto weights = subnetwork.stationary_weights();
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
                const Float phi_j = (h_j > static_cast<Float>(0)) ? (u_j / h_j) : static_cast<Float>(0);
                const Float s_ij = -R.values[k] * (h_j / h_i);
                const Float diff = phi_i - phi_j;
                dirichlet_sum += s_ij * (diff * diff);
            }
        }
        losses.push_back(std::max(static_cast<Float>(0), dirichlet_sum));
    }
    return losses;
}

/// @brief Method 3: Woodbury Resolvent Inverse Update
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>, typename Vec>
inline std::vector<Float>
compute_woodbury_losses(const Subnetwork<Float, Index, State> &subnetwork,
                        const Vec &occupancy,
                        std::span<const Index> candidate_indices) {
    return subnetwork.cut_time_losses(occupancy, candidate_indices);
}

/// @brief Unified shedding loss dispatcher across Methods 1, 2, and 3.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>, typename Vec>
inline std::vector<Float>
compute_shedding_losses(const Subnetwork<Float, Index, State> &subnetwork,
                        const Vec &occupancy,
                        std::span<const Index> candidate_indices,
                        SheddingMethod method = SheddingMethod::CholeskyWoodbury) {
    switch (method) {
        case SheddingMethod::ExpectedVisits:
            return compute_expected_visits_losses<Float, Index, State>(subnetwork, occupancy, candidate_indices);
        case SheddingMethod::NormalizedSymmetrized:
            return compute_symmetrized_losses<Float, Index, State>(subnetwork, occupancy, candidate_indices);
        case SheddingMethod::CholeskyWoodbury:
        case SheddingMethod::Auto:
            return compute_woodbury_losses<Float, Index, State>(subnetwork, occupancy, candidate_indices);
        default:
            throw std::invalid_argument("unknown SheddingMethod specified");
    }
}

/// @brief Shed states based on SheddingOptions and protected state indices.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>, typename Vec>
inline SheddingResult<Index, Float>
shed_states(const Subnetwork<Float, Index, State> &subnetwork,
            const Vec &occupancy,
            std::span<const Index> protected_indices,
            const SheddingOptions<Index, Float> &options = {}) {
    const Index n = subnetwork.size();
    std::unordered_set<Index> prot_set(protected_indices.begin(), protected_indices.end());

    std::vector<Index> candidates;
    for (Index i = 0; i < n; ++i) {
        if (!prot_set.count(i)) candidates.push_back(i);
    }

    std::vector<Index> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), static_cast<Index>(0));
    const auto losses = compute_shedding_losses(subnetwork, occupancy, std::span<const Index>(all_indices), options.method);

    Index target = options.target_capacity > 0 ? options.target_capacity : n;
    if (target < protected_indices.size()) target = protected_indices.size();

    const Index needed_removals = (n > target) ? (n - target) : 0;

    std::vector<std::pair<Float, Index>> ranked_candidates;
    for (Index c : candidates) {
        ranked_candidates.push_back({losses[c], c});
    }
    std::sort(ranked_candidates.begin(), ranked_candidates.end());

    std::unordered_set<Index> to_remove;
    for (Index i = 0; i < needed_removals && i < ranked_candidates.size(); ++i) {
        to_remove.insert(ranked_candidates[i].second);
    }

    SheddingResult<Index, Float> result;
    result.state_losses = losses;
    for (Index i = 0; i < n; ++i) {
        if (to_remove.count(i)) {
            result.shed_indices.push_back(i);
        } else {
            result.kept_indices.push_back(i);
        }
    }
    result.diagnostics.initial_states = n;
    result.diagnostics.shed_states = result.shed_indices.size();
    result.diagnostics.remaining_states = result.kept_indices.size();
    return result;
}

template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>, typename Vec, typename ProtIndex = Index>
inline SheddingResult<Index, Float>
shed_states(const Subnetwork<Float, Index, State> &subnetwork,
            const Vec &occupancy,
            std::initializer_list<ProtIndex> protected_indices,
            const SheddingOptions<Index, Float> &options = {}) {
    std::vector<Index> converted;
    converted.reserve(protected_indices.size());
    for (auto p : protected_indices) converted.push_back(static_cast<Index>(p));
    return shed_states(subnetwork, occupancy, std::span<const Index>(converted.data(), converted.size()), options);
}

} // namespace else_sim
