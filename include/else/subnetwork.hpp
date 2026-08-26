#pragma once

#include "else/types.hpp"
#include "else/woodbury.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <numerics.hpp>

namespace else_sim {

/// @brief Pure template representation of a discrete subnetwork with exact exit-time solvers.
template <typename Float = double, typename Index = std::size_t,
          typename State = std::vector<int>>
class Subnetwork {
  public:
    using MatrixType = num::Matrix;
    using VectorType = num::Vector;
    using SparseMatrixType = num::SparseMatrix;

    Subnetwork() = default;

    /// @brief Construct a general (non-reversible) subnetwork.
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary)
        : states_(std::move(states)), R_(std::move(R)), boundary_(std::move(boundary)) {
        initialize();
        solver_ = std::make_unique<SolverImpl>(num::scaled(R_, -1.0));
    }

    /// @brief Construct a reversible subnetwork with stationary scaling weights h = sqrt(pi).
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary,
               std::vector<Float> stationary_sqrt)
        : states_(std::move(states)), R_(std::move(R)), boundary_(std::move(boundary)),
          stationary_weights_(std::move(stationary_sqrt)) {
        initialize();
        std::vector<double> weights(stationary_weights_.begin(), stationary_weights_.end());
        auto symmetric = num::diagonal_similarity(num::scaled(R_, -1.0), weights);
        solver_ = std::make_unique<SolverImpl>(std::move(symmetric), std::move(weights));
    }

    [[nodiscard]] Index size() const { return states_.size(); }
    [[nodiscard]] const std::vector<State> &states() const { return states_; }
    [[nodiscard]] const std::vector<Index> &boundary_states() const { return boundary_states_; }
    [[nodiscard]] const std::vector<BoundaryTransition<Index, State, Float>> &boundary() const {
        return boundary_;
    }
    [[nodiscard]] const SparseMatrixType &generator() const { return R_; }
    [[nodiscard]] bool is_reversible() const { return solver_ && solver_->cholesky_factor.has_value(); }
    [[nodiscard]] std::span<const Float> stationary_weights() const { return stationary_weights_; }

    [[nodiscard]] int find(const State &x) const {
        auto it = index_.find(x);
        return it != index_.end() ? it->second : -1;
    }

    /// @brief Solves (-R) u = p0 for the transient occupation time profile.
    [[nodiscard]] VectorType occupation(const VectorType &start) const {
        VectorType u(size(), 0.0);
        solver_->solve(start, u);
        return u;
    }

    /// @brief Computes first and second time-weighted occupation moments.
    [[nodiscard]] OccupationIntegrals<MatrixType> occupation_integrals(const MatrixType &starts) const {
        OccupationIntegrals<MatrixType> integrals;
        integrals.occupation = MatrixType(size(), starts.cols(), 0.0);
        integrals.time_weighted_occupation = MatrixType(size(), starts.cols(), 0.0);
        solver_->solve(starts, integrals.occupation);
        solver_->solve(integrals.occupation, integrals.time_weighted_occupation);
        return integrals;
    }

    /// @brief Computes exit probability distribution across boundary states.
    [[nodiscard]] std::vector<Float> exit_probabilities(const VectorType &occupation) const {
        std::vector<Float> probs;
        probs.reserve(boundary_states_.size());
        for (Index b : boundary_states_) {
            probs.push_back(static_cast<Float>(exit_rates_[b] * std::max(0.0, occupation[b])));
        }
        num::clip_and_normalize_nonnegative(probs);
        return probs;
    }

    [[nodiscard]] std::vector<Float> exit_probabilities(const OccupationIntegrals<MatrixType> &integrals,
                                                        Index column = 0) const {
        std::vector<Float> probs;
        probs.reserve(boundary_states_.size());
        for (Index b : boundary_states_) {
            probs.push_back(static_cast<Float>(exit_rates_[b] * std::max(0.0, integrals.occupation(b, column))));
        }
        num::clip_and_normalize_nonnegative(probs);
        return probs;
    }

    /// @brief Computes conditional mean exit time \tau(b) = (u_t)_b / u_b.
    [[nodiscard]] Float conditional_exit_time(const OccupationIntegrals<MatrixType> &integrals,
                                              Index boundary_state, Index column = 0) const {
        const double u_occ = integrals.occupation(boundary_state, column);
        if (u_occ <= 1e-15) return static_cast<Float>(0);
        const double u_time = integrals.time_weighted_occupation(boundary_state, column);
        return static_cast<Float>(std::max(0.0, u_time / u_occ));
    }

    /// @brief Computes flow-balanced expected visits s_j = max(0, w_j u_j - p_j).
    [[nodiscard]] std::vector<Float> expected_entries(const VectorType &occupancy,
                                                      const VectorType &start) const {
        std::vector<Float> counts(size(), static_cast<Float>(0));
        for (Index j = 0; j < size(); ++j) {
            counts[j] = static_cast<Float>(std::max(0.0, (total_rates_[j] * occupancy[j]) - start[j]));
        }
        return counts;
    }

    [[nodiscard]] const VectorType &inverse_column_sums() const {
        return solver_->inverse_column_sums();
    }

    /// @brief Exact singleton cut-time losses \ell_j = u_j q_j / Z_jj.
    [[nodiscard]] std::vector<Float> cut_time_losses(const VectorType &occupancy) const {
        std::vector<Index> candidates(size());
        std::iota(candidates.begin(), candidates.end(), Index{0});
        return cut_time_losses(occupancy, candidates);
    }

    [[nodiscard]] std::vector<Float> cut_time_losses(const VectorType &occupancy,
                                                     std::span<const Index> candidates) const {
        if (occupancy.size() != size()) {
            throw std::invalid_argument("cut-time occupation size must match the subnetwork");
        }
        const auto &column_sums = solver_->inverse_column_sums();
        std::vector<double> diagonal(candidates.size(), 0.0);
        solver_->inverse_diagonal(candidates, diagonal);

        std::vector<Float> losses(candidates.size(), static_cast<Float>(0));
        for (Index index = 0; index < candidates.size(); ++index) {
            const Index candidate = candidates[index];
            if (diagonal[index] > 1e-15) {
                losses[index] = static_cast<Float>(occupancy[candidate] * (column_sums[candidate] / diagonal[index]));
            }
        }
        return losses;
    }

    /// @brief Direct ground-truth naive method: refactorizes reduced generator from scratch.
    [[nodiscard]] Float naive_cut_time_loss(const VectorType &occupancy,
                                            std::span<const Index> removed_states) const {
        const Index n = size();
        if (removed_states.empty()) return static_cast<Float>(0);
        if (removed_states.size() >= n) {
            return static_cast<Float>(std::accumulate(occupancy.begin(), occupancy.end(), 0.0));
        }

        std::vector<bool> removed(n, false);
        for (Index s : removed_states) {
            if (s < n) removed[s] = true;
        }

        std::vector<Index> kept;
        kept.reserve(n - removed_states.size());
        for (Index i = 0; i < n; ++i) {
            if (!removed[i]) kept.push_back(i);
        }
        const Index m = kept.size();

        MatrixType R_red(m, m, 0.0);
        VectorType p_red(m, 0.0);
        for (Index i = 0; i < m; ++i) {
            for (Index j = 0; j < m; ++j) {
                R_red(i, j) = -R_(kept[i], kept[j]);
            }
            for (Index j = 0; j < n; ++j) {
                p_red[i] += -R_(kept[i], j) * occupancy[j];
            }
        }

        VectorType u_red(m, 0.0);
        auto lu_red = num::lu(num::assume_square(std::move(R_red)));
        if (lu_red.singular) return static_cast<Float>(0);
        num::lu_solve(lu_red, p_red, u_red);

        const double tau_orig = std::accumulate(occupancy.begin(), occupancy.end(), 0.0);
        const double tau_red = std::accumulate(u_red.begin(), u_red.end(), 0.0);
        return static_cast<Float>(std::max(0.0, tau_orig - tau_red));
    }

    /// @brief Detailed cut-time loss evaluation returning loss, floating-point error, and fallback diagnostics.
    [[nodiscard]] CutTimeLossResult<Float>
    cut_time_loss_with_diagnostics(const VectorType &occupancy,
                                   std::span<const Index> removed_states,
                                   Float tolerance = static_cast<Float>(1e-6)) const {
        if (occupancy.size() != size()) {
            throw std::invalid_argument("cut-time occupation size must match subnetwork");
        }
        if (removed_states.empty()) {
            return {.loss = static_cast<Float>(0), .estimated_error = static_cast<Float>(0), .naive_fallback_used = false};
        }
        if (removed_states.size() == 1) {
            return {.loss = cut_time_losses(occupancy, removed_states).front(),
                    .estimated_error = static_cast<Float>(0),
                    .naive_fallback_used = false};
        }

        const Index r = removed_states.size();
        VectorType u_S(r, 0.0);
        for (Index i = 0; i < r; ++i) {
            u_S[i] = occupancy[removed_states[i]];
        }

        VectorType c(r, 0.0);
        MatrixType Z_SS;
        try {
            solver_->solve_principal_inverse(removed_states, u_S, c, &Z_SS);

            // Backward residual: || Z_SS * c - u_S ||_inf / ||u_S||_inf
            VectorType reconstructed(r, 0.0);
            num::matvec(Z_SS, c, reconstructed);
            double err = 0.0, scale = 0.0;
            for (Index i = 0; i < r; ++i) {
                err = std::max(err, std::abs(reconstructed[i] - u_S[i]));
                scale = std::max(scale, std::abs(u_S[i]));
            }
            const double residual = err / std::max(1e-12, scale);

            if (residual <= tolerance && std::isfinite(residual)) {
                const auto &q = solver_->inverse_column_sums();
                double loss = 0.0;
                for (Index i = 0; i < r; ++i) {
                    loss += q[removed_states[i]] * c[i];
                }
                if (std::isfinite(loss) && loss >= 0.0) {
                    return {.loss = static_cast<Float>(loss),
                            .estimated_error = static_cast<Float>(residual),
                            .naive_fallback_used = false};
                }
            }
        } catch (...) {}

        return {.loss = naive_cut_time_loss(occupancy, removed_states),
                .estimated_error = static_cast<Float>(1.0),
                .naive_fallback_used = true};
    }

    /// @brief Exact combined loss from removing a state block via the Woodbury formula.
    [[nodiscard]] Float cut_time_loss(const VectorType &occupancy,
                                      std::span<const Index> removed_states,
                                      Float tolerance = static_cast<Float>(1e-6)) const {
        return cut_time_loss_with_diagnostics(occupancy, removed_states, tolerance).loss;
    }

  private:
    void initialize() {
        total_rates_.assign(states_.size(), 0.0);
        exit_rates_.assign(states_.size(), 0.0);
        for (Index i = 0; i < states_.size(); ++i) {
            index_[states_[i]] = static_cast<int>(i);
            total_rates_[i] = -R_(i, i);
        }
        for (const auto &t : boundary_) {
            exit_rates_[t.source] += t.rate;
        }
        for (Index i = 0; i < states_.size(); ++i) {
            if (exit_rates_[i] > 1e-15) {
                boundary_states_.push_back(i);
            }
        }
    }

    struct SolverImpl {
        std::unique_ptr<num::AutoLinearSolver> general_solver;
        std::optional<num::CholeskyResult> cholesky_factor;
        std::vector<double> weights;
        mutable num::InverseDiagonalWorkspace inverse_workspace;
        mutable std::optional<num::Vector> inverse_column_sums_cache;
        mutable std::vector<double> inverse_diagonal_cache;

        explicit SolverImpl(const num::SparseMatrix &operator_matrix)
            : general_solver(std::make_unique<num::AutoLinearSolver>(operator_matrix)) {}

        SolverImpl(num::Matrix symmetric, std::vector<double> scaling_weights)
            : weights(std::move(scaling_weights)) {
            cholesky_factor = num::cholesky(num::linalg::assume_spd(std::move(symmetric)));
            if (!cholesky_factor->success) {
                throw std::runtime_error("subnetwork operator is not positive definite");
            }
        }

        void solve(const VectorType &b, VectorType &x) const {
            if (cholesky_factor) {
                VectorType transformed_b = b;
                num::divide_elements(transformed_b, weights);
                VectorType transformed_x(b.size(), 0.0);
                num::cholesky_solve(*cholesky_factor, transformed_b, transformed_x);
                x = std::move(transformed_x);
                num::scale_elements(x, weights);
                return;
            }
            general_solver->solve(b, x);
        }

        void solve(const MatrixType &right_hand_side, MatrixType &solution) const {
            if (cholesky_factor) {
                MatrixType transformed = right_hand_side;
                num::divide_rows(transformed, weights);
                num::cholesky_solve(*cholesky_factor, transformed, solution);
                num::scale_rows(solution, weights);
                return;
            }
            general_solver->solve(right_hand_side, solution);
        }

        void solve_transpose(const VectorType &b, VectorType &x) const {
            if (cholesky_factor) {
                VectorType transformed_b = b;
                num::scale_elements(transformed_b, weights);
                VectorType transformed_x(b.size(), 0.0);
                num::cholesky_solve(*cholesky_factor, transformed_b, transformed_x);
                x = std::move(transformed_x);
                num::divide_elements(x, weights);
                return;
            }
            general_solver->solve_transpose(b, x);
        }

        const VectorType &inverse_column_sums() const {
            if (!inverse_column_sums_cache) {
                VectorType ones(size(), 1.0);
                VectorType column_sums(size(), 0.0);
                solve_transpose(ones, column_sums);
                inverse_column_sums_cache = std::move(column_sums);
            }
            return *inverse_column_sums_cache;
        }

        void inverse_diagonal(std::span<const Index> candidates, std::vector<double> &diagonal) const {
            if (inverse_diagonal_cache.empty()) {
                inverse_diagonal_cache.assign(size(), 0.0);
            }
            std::vector<Index> missing;
            for (Index candidate : candidates) {
                if (inverse_diagonal_cache[candidate] == 0.0) {
                    missing.push_back(candidate);
                }
            }
            if (!missing.empty()) {
                std::vector<num::idx> missing_idx(missing.begin(), missing.end());
                num::Vector fetched(missing.size(), 0.0);
                if (cholesky_factor) {
                    num::selected_inverse(*cholesky_factor, missing_idx, missing_idx, fetched, inverse_workspace);
                } else {
                    num::selected_inverse(*general_solver, missing_idx, missing_idx, fetched, inverse_workspace);
                }
                for (Index index = 0; index < missing.size(); ++index) {
                    inverse_diagonal_cache[missing[index]] = fetched[index];
                }
            }
            for (Index index = 0; index < candidates.size(); ++index) {
                diagonal[index] = inverse_diagonal_cache[candidates[index]];
            }
        }

        void solve_principal_inverse(std::span<const Index> indices,
                                     const VectorType &right_hand_side,
                                     VectorType &solution,
                                     MatrixType *extracted_block = nullptr) const {
            std::vector<num::idx> idx_span(indices.begin(), indices.end());
            MatrixType block;
            if (cholesky_factor) {
                num::inverse_principal_block(*cholesky_factor, idx_span, block, inverse_workspace);
                if (extracted_block) *extracted_block = block;
                const auto block_factor = num::cholesky(num::linalg::assume_spd(std::move(block)));
                if (!block_factor.success) {
                    throw std::runtime_error("cut-time inverse block is not positive definite");
                }
                VectorType transformed = right_hand_side;
                for (Index i = 0; i < indices.size(); ++i) transformed[i] /= weights[indices[i]];
                num::cholesky_solve(block_factor, transformed, solution);
                for (Index i = 0; i < indices.size(); ++i) solution[i] *= weights[indices[i]];
                return;
            }
            num::inverse_principal_block(*general_solver, idx_span, block, inverse_workspace);
            if (extracted_block) *extracted_block = block;
            const auto block_factor = num::lu(num::assume_square(std::move(block)));
            if (block_factor.singular) {
                throw std::runtime_error("cut-time inverse block is singular");
            }
            num::lu_solve(block_factor, right_hand_side, solution);
        }

        [[nodiscard]] num::idx size() const {
            return cholesky_factor ? weights.size() : general_solver->size();
        }
    };

    std::vector<State> states_;
    std::unordered_map<State, int, StateHash<State>> index_;
    SparseMatrixType R_{0, 0, {}, {}, {0}};
    std::vector<double> total_rates_;
    std::vector<double> exit_rates_;
    std::vector<Index> boundary_states_;
    std::vector<BoundaryTransition<Index, State, Float>> boundary_;
    std::vector<Float> stationary_weights_;

    std::unique_ptr<SolverImpl> solver_;
};

} // namespace else_sim
