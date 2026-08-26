#pragma once

#include "else/linalg.hpp"
#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace else_sim {

/// @brief Method 3: Flow-balanced expected visits loss approximation (O(n) speed).
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
inline std::vector<Float>
compute_expected_visits_losses(const Subnetwork<Float, Index, State> &subnetwork,
                               const std::vector<Float> &occupation) {
    const Index n = subnetwork.size();
    const auto &column_sums = subnetwork.inverse_column_sums();
    const auto &R = subnetwork.generator();

    std::vector<Float> losses(n, static_cast<Float>(0));
    for (Index i = 0; i < n; ++i) {
        const Float rate = -R(i, i);
        const Float w_i = (i < column_sums.size()) ? column_sums[i] : static_cast<Float>(1);
        losses[i] = std::max(static_cast<Float>(0), occupation[i] * rate * w_i);
    }
    return losses;
}

/// @brief Method 2: Steady-State Normalized Symmetrized Surrogate Operator.
/// \widetilde{S} = 0.5 * (H (-R) H^-1 + H^-1 (-R)^T H) with H = diag(sqrt(pi)).
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
inline std::vector<Float>
compute_symmetrized_losses(const Subnetwork<Float, Index, State> &subnetwork,
                           const std::vector<Float> &occupation) {
    const Index n = subnetwork.size();
    const auto &column_sums = subnetwork.inverse_column_sums();
    const auto &R = subnetwork.generator();
    const auto weights = subnetwork.stationary_weights();

    Matrix<Float> S(n, n, static_cast<Float>(0));
    for (Index i = 0; i < n; ++i) {
        const Float h_i = (!weights.empty() && i < weights.size()) ? weights[i] : static_cast<Float>(1);
        for (Index j = 0; j < n; ++j) {
            const Float h_j = (!weights.empty() && j < weights.size()) ? weights[j] : static_cast<Float>(1);
            const Float r_ij = (h_i / std::max(static_cast<Float>(1e-12), h_j)) * (-R(i, j));
            const Float r_ji = (h_j / std::max(static_cast<Float>(1e-12), h_i)) * (-R(j, i));
            S(i, j) = static_cast<Float>(0.5) * (r_ij + r_ji);
        }
    }
    for (Index i = 0; i < n; ++i) {
        Float row_sum = static_cast<Float>(0);
        for (Index j = 0; j < n; ++j) {
            if (i != j) {
                row_sum += std::abs(S(i, j));
            }
        }
        S(i, i) = std::max(S(i, i), row_sum + static_cast<Float>(1e-6));
    }

    auto chol = factorize_cholesky(std::move(S));
    if (!chol.success) {
        return compute_expected_visits_losses(subnetwork, occupation);
    }

    std::vector<Float> losses(n, static_cast<Float>(0));
    std::vector<Float> e(n, static_cast<Float>(0));
    std::vector<Float> z(n, static_cast<Float>(0));
    for (Index i = 0; i < n; ++i) {
        e[i] = static_cast<Float>(1);
        cholesky_solve(chol, e, z);
        e[i] = static_cast<Float>(0);
        const Float diag = std::max(z[i], static_cast<Float>(1e-12));
        const Float w_i = (i < column_sums.size()) ? column_sums[i] : static_cast<Float>(1);
        losses[i] = std::max(static_cast<Float>(0), occupation[i] * (w_i / diag));
    }
    return losses;
}

/// @brief Evaluates shedding losses across all candidates under the selected algorithmic method.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
inline std::vector<Float>
compute_shedding_losses(const Subnetwork<Float, Index, State> &subnetwork,
                        const std::vector<Float> &occupation,
                        const SheddingOptions<Index, Float> &options = {},
                        SheddingDiagnostics<Index, Float> *diagnostics = nullptr) {
    const Index n = subnetwork.size();
    if (occupation.size() != n) {
        throw std::invalid_argument("occupation size must match subnetwork size");
    }

    SheddingMethod effective_method = options.method;
    if (effective_method == SheddingMethod::Auto) {
        effective_method = subnetwork.is_reversible() ? SheddingMethod::CholeskyWoodbury
                                                      : SheddingMethod::ExpectedVisits;
    }

    std::vector<Float> losses;
    switch (effective_method) {
    case SheddingMethod::CholeskyWoodbury:
        losses = subnetwork.cut_time_losses(occupation);
        break;
    case SheddingMethod::NormalizedSymmetrized:
        losses = compute_symmetrized_losses(subnetwork, occupation);
        break;
    case SheddingMethod::ExpectedVisits:
        losses = compute_expected_visits_losses(subnetwork, occupation);
        break;
    case SheddingMethod::Auto:
        break;
    }

    if (diagnostics) {
        diagnostics->requested_method = options.method;
        diagnostics->effective_method = effective_method;
        diagnostics->initial_states = n;
    }
    return losses;
}

/// @brief Performs state space shedding under capacity constraints with protected states.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
inline SheddingResult<Index, Float>
shed_states(const Subnetwork<Float, Index, State> &subnetwork,
            const std::vector<Float> &occupation,
            const std::vector<Index> &protected_states = {},
            const SheddingOptions<Index, Float> &options = {}) {
    const Index n = subnetwork.size();
    SheddingDiagnostics<Index, Float> diag;
    const auto losses = compute_shedding_losses(subnetwork, occupation, options, &diag);

    std::vector<bool> is_protected(n, false);
    for (Index p : protected_states) {
        if (p < n) is_protected[p] = true;
    }

    std::vector<std::pair<Float, Index>> ranking;
    for (Index i = 0; i < n; ++i) {
        if (!is_protected[i]) {
            ranking.emplace_back(losses[i], i);
        }
    }
    std::sort(ranking.begin(), ranking.end());

    Index target = options.target_capacity > 0 ? options.target_capacity : n;
    Index remove_count = (n > target) ? (n - target) : 0;
    remove_count = std::min(remove_count, static_cast<Index>(ranking.size()));

    std::vector<bool> is_shed(n, false);
    std::vector<Index> shed_indices;
    shed_indices.reserve(remove_count);
    for (Index i = 0; i < remove_count; ++i) {
        const Index idx = ranking[i].second;
        is_shed[idx] = true;
        shed_indices.push_back(idx);
    }

    std::vector<Index> kept_indices;
    kept_indices.reserve(n - remove_count);
    for (Index i = 0; i < n; ++i) {
        if (!is_shed[i]) {
            kept_indices.push_back(i);
        }
    }

    diag.shed_states = shed_indices.size();
    diag.remaining_states = kept_indices.size();
    return {
        .kept_indices = std::move(kept_indices),
        .shed_indices = std::move(shed_indices),
        .state_losses = losses,
        .diagnostics = diag,
    };
}

} // namespace else_sim
