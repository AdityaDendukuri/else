#pragma once

#include "else/block_tridiagonal.hpp"
#include "else/linalg.hpp"
#include "else/types.hpp"
#include "else/woodbury.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace else_sim {

/// @brief Expanding Local Subnetwork Enumeration (ELSE) container and solver.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
class Subnetwork {
  public:
    using MatrixType = Matrix<Float>;
    using SparseMatrixType = SparseMatrix<Float, Index>;
    using VectorType = std::vector<Float>;

    /// @brief Primary constructor for general (non-reversible) subnetwork.
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary)
        : Subnetwork(std::move(states), std::move(R), std::move(boundary), std::vector<Index>{}) {}

    /// @brief General subnetwork with an explicit block-tridiagonal ordering.
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary,
               std::vector<Index> levels, bool factorize = true)
        : states_(std::move(states)), R_(std::move(R)), boundary_(std::move(boundary)) {
        initialize();
        if (factorize)
            solver_.emplace(R_, std::move(levels));
    }

    /// @brief Primary constructor for reversible subnetwork with stationary weights.
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary,
               std::vector<Float> stationary_sqrt)
        : Subnetwork(std::move(states), std::move(R), std::move(boundary),
                     std::move(stationary_sqrt), std::vector<Index>{}) {}

    /// @brief Reversible subnetwork with an explicit block-tridiagonal ordering.
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary,
               std::vector<Float> stationary_sqrt, std::vector<Index> levels)
        : states_(std::move(states)), R_(std::move(R)), boundary_(std::move(boundary)),
          stationary_sqrt_(std::move(stationary_sqrt)) {
        initialize();
        // S_ij = -R_ij sqrt(pi_j/pi_i) is symmetric positive definite.
        auto symmetric = R_;
        for (Index i = 0; i < states_.size(); ++i)
            for (Index k = symmetric.row_ptr[i]; k < symmetric.row_ptr[i + 1]; ++k) {
                const Index j = symmetric.col_idx[k];
                symmetric.values[k] *= -stationary_sqrt_[j] / stationary_sqrt_[i];
            }
        solver_.emplace(std::move(symmetric), stationary_sqrt_, std::move(levels));
    }

    [[nodiscard]] Index size() const { return states_.size(); }
    [[nodiscard]] const std::vector<State> &states() const { return states_; }
    [[nodiscard]] const std::vector<Index> &boundary_states() const { return boundary_states_; }
    [[nodiscard]] const std::vector<BoundaryTransition<Index, State, Float>> &boundary() const {
        return boundary_;
    }
    [[nodiscard]] const SparseMatrixType &generator() const { return R_; }
    [[nodiscard]] bool is_reversible() const { return solver_ && solver_->is_cholesky(); }
    [[nodiscard]] bool uses_block_solver() const {
        return solver_ && (solver_->block_factor || solver_->block_cholesky_factor);
    }
    [[nodiscard]] std::span<const Float> stationary_sqrt() const { return stationary_sqrt_; }

    [[nodiscard]] Index find(const State &x) const {
        auto it = index_.find(x);
        return it != index_.end() ? it->second : size();
    }

    /// @brief Solves (-R) u = p0 for transient occupation.
    [[nodiscard]] VectorType occupation(std::span<const Float> start) const {
        if (start.size() != size())
            throw std::invalid_argument("occupation vector size must match subnetwork");
        VectorType u;
        solver_->solve(VectorType(start.begin(), start.end()), u);
        return u;
    }

    /// For entrance columns E, compute U = Z E and V = Z U = Z^2 E.
    [[nodiscard]] OccupationIntegrals<MatrixType>
    occupation_integrals(const MatrixType &starts) const {
        OccupationIntegrals<MatrixType> integrals;
        solver_->solve_multiple(starts, integrals.occupation);
        solver_->solve_multiple(integrals.occupation, integrals.time_weighted_occupation);
        return integrals;
    }

    [[nodiscard]] std::vector<Float>
    exit_probabilities(const OccupationIntegrals<MatrixType> &integrals, Index column = 0) const {
        // P(exit through b) = Delta_b Z_ba; normalization removes accumulated roundoff.
        std::vector<Float> probs;
        probs.reserve(boundary_states_.size());
        Float sum = static_cast<Float>(0);
        for (Index b : boundary_states_) {
            Float p = static_cast<Float>(
                exit_rates_[b] * std::max(static_cast<Float>(0),
                                          static_cast<Float>(integrals.occupation(b, column))));
            probs.push_back(p);
            sum += p;
        }
        if (sum > static_cast<Float>(0)) {
            for (auto &p : probs)
                p /= sum;
        }
        return probs;
    }

    /// @brief Computes mean waiting time conditioned on exit via local boundary state.
    [[nodiscard]] Float conditional_exit_time(const OccupationIntegrals<MatrixType> &integrals,
                                              Index boundary_state, Index column = 0) const {
        const Float occ = integrals.occupation(boundary_state, column);
        if (!(occ > static_cast<Float>(0)))
            return static_cast<Float>(0);
        const Float tw = integrals.time_weighted_occupation(boundary_state, column);
        return tw / occ;
    }

    /// @brief Evaluates cut-time losses for candidate states.
    [[nodiscard]] std::vector<Float> cut_time_losses(std::span<const Float> occupancy,
                                                     std::span<const Index> candidates) const {
        std::vector<Float> losses;
        losses.reserve(candidates.size());
        for (Index c : candidates) {
            if (c >= size()) {
                throw std::out_of_range("candidate state is out of range");
            }
            const Float occ_c = static_cast<Float>(occupancy[c]);
            const Float inv_diag = solver_->inverse_diagonal(c);
            losses.push_back(occ_c * solver_->inverse_column_sum(c) / inv_diag);
        }
        return losses;
    }

    /// @brief Evaluates cut-time losses for all states in the subnetwork.
    [[nodiscard]] std::vector<Float> cut_time_losses(std::span<const Float> occupancy) const {
        std::vector<Index> all(size());
        std::iota(all.begin(), all.end(), static_cast<Index>(0));
        return cut_time_losses(occupancy, all);
    }

    /// @brief Naive cut-time loss via reduced subnetwork solve.
    [[nodiscard]] Float naive_cut_time_loss(std::span<const Float> occupancy,
                                            std::span<const Index> removed_states) const {
        const Index n = size();
        if (removed_states.size() >= n) {
            return std::accumulate(occupancy.begin(), occupancy.end(), static_cast<Float>(0));
        }

        std::vector<bool> removed(n, false);
        for (Index s : removed_states) {
            if (s < n)
                removed[s] = true;
        }

        std::vector<Index> kept;
        std::vector<Index> new_index(n, n);
        kept.reserve(n - removed_states.size());
        for (Index i = 0; i < n; ++i) {
            if (!removed[i]) {
                new_index[i] = kept.size();
                kept.push_back(i);
            }
        }
        const Index m = kept.size();

        MatrixType R_red(m, m, static_cast<Float>(0));
        VectorType p_red(m, static_cast<Float>(0));
        for (Index i = 0; i < m; ++i) {
            for (Index k = R_.row_ptr[kept[i]]; k < R_.row_ptr[kept[i] + 1]; ++k) {
                const Index j_orig = R_.col_idx[k];
                if (!removed[j_orig])
                    R_red(i, new_index[j_orig]) = -R_.values[k];
                p_red[i] += -R_.values[k] * static_cast<Float>(occupancy[j_orig]);
            }
        }

        VectorType u_red(m, static_cast<Float>(0));
        auto lu_red = factorize_lu<Float>(std::move(R_red));
        if (lu_red.singular)
            return static_cast<Float>(0);
        lu_solve(lu_red, p_red, u_red);

        const Float tau_orig =
            std::accumulate(occupancy.begin(), occupancy.end(), static_cast<Float>(0));
        const Float tau_red = std::accumulate(u_red.begin(), u_red.end(), static_cast<Float>(0));
        return std::max(static_cast<Float>(0), tau_orig - tau_red);
    }

    /// @brief Evaluates cut-time loss for a set of removed states.
    [[nodiscard]] Float cut_time_loss(std::span<const Float> occupancy,
                                      std::span<const Index> removed_states,
                                      Float tolerance = static_cast<Float>(1e-6)) const {
        return cut_time_loss_with_diagnostics(occupancy, removed_states, tolerance).loss;
    }

    /// @brief Detailed cut-time loss evaluation returning loss, floating-point error, and fallback
    /// diagnostics.
    [[nodiscard]] CutTimeLossResult<Float>
    cut_time_loss_with_diagnostics(std::span<const Float> occupancy,
                                   std::span<const Index> removed_states,
                                   Float tolerance = static_cast<Float>(1e-6)) const {
        if (occupancy.size() != size()) {
            throw std::invalid_argument("cut-time occupation size must match subnetwork");
        }
        if (removed_states.empty()) {
            return {.loss = static_cast<Float>(0),
                    .estimated_error = static_cast<Float>(0),
                    .naive_fallback_used = false};
        }
        if (removed_states.size() == 1) {
            return {.loss = cut_time_losses(occupancy, removed_states).front(),
                    .estimated_error = static_cast<Float>(std::numeric_limits<Float>::epsilon()),
                    .naive_fallback_used = false};
        }

        const Index r = removed_states.size();
        VectorType y(r);
        for (Index i = 0; i < r; ++i)
            y[i] = static_cast<Float>(occupancy[removed_states[i]]);

        VectorType y_corr(r);
        solver_->solve_block_inverse(removed_states, y, y_corr);

        Float delta_tau = static_cast<Float>(0);
        Float accumulated_err = static_cast<Float>(0);
        bool precision_acceptable = true;

        // Delta tau = q_J^T Z_JJ^{-1} u_J, where q = Z^T 1.
        for (Index i = 0; i < r; ++i) {
            const Float contribution = solver_->inverse_column_sum(removed_states[i]) * y_corr[i];
            if (!safe_add(delta_tau, accumulated_err, contribution, tolerance)) {
                precision_acceptable = false;
            }
        }

        if (precision_acceptable && delta_tau > static_cast<Float>(0)) {
            return {.loss = delta_tau,
                    .estimated_error = accumulated_err,
                    .naive_fallback_used = false};
        }

        Float naive_loss = naive_cut_time_loss(occupancy, removed_states);
        return {
            .loss = naive_loss, .estimated_error = accumulated_err, .naive_fallback_used = true};
    }

  private:
    void initialize() {
        index_.clear();
        for (std::size_t i = 0; i < states_.size(); ++i) {
            index_[states_[i]] = static_cast<Index>(i);
        }
        const Index n = states_.size();
        exit_rates_.assign(n, static_cast<Float>(0));
        boundary_states_.clear();

        for (const auto &b : boundary_) {
            exit_rates_[b.source] += b.rate;
        }
        for (Index i = 0; i < n; ++i) {
            if (exit_rates_[i] > static_cast<Float>(0)) {
                boundary_states_.push_back(i);
            }
        }
    }

    struct SolverImpl {
        std::optional<LUFactor<Float>> lu_factor;
        std::optional<BlockLUFactor<Float, Index>> block_factor;
        std::optional<BlockCholeskyFactor<Float, Index>> block_cholesky_factor;
        std::optional<CholeskyFactor<Float>> cholesky_factor;
        std::vector<Float> stationary_sqrt;
        mutable std::vector<Float> inv_diag_cache;
        mutable std::vector<Float> inv_column_sum_cache;

        static constexpr Index dense_limit = 32;

        SolverImpl(const SparseMatrixType &R, std::vector<Index> levels) {
            factor_lu(R, std::move(levels));
        }

        SolverImpl(MatrixType S, std::vector<Float> h) : stationary_sqrt(std::move(h)) {
            factor_dense_cholesky(S);
        }

        SolverImpl(SparseMatrixType S, std::vector<Float> w, std::vector<Index> levels)
            : stationary_sqrt(std::move(w)) {
            factor_cholesky(S, std::move(levels));
        }

        [[nodiscard]] bool is_cholesky() const {
            return cholesky_factor.has_value() || block_cholesky_factor.has_value();
        }

        void solve(const VectorType &b, VectorType &x) const {
            if (is_cholesky())
                solve_cholesky(b, x);
            else
                solve_lu(b, x);
        }

        void solve_multiple(const MatrixType &b, MatrixType &x) const {
            if (is_cholesky())
                solve_multiple_cholesky(b, x);
            else if (block_factor)
                block_solve(*block_factor, b, x);
            else
                lu_solve(*lu_factor, b, x);
        }

        Float inverse_diagonal(Index j) const {
            if (inv_diag_cache.empty()) {
                // The exact cut-time rule eventually needs every Z_jj, so cache them together.
                const Index n = size();
                inv_diag_cache.resize(n, static_cast<Float>(0));
                inv_column_sum_cache.resize(n, static_cast<Float>(0));
                VectorType ej(n, static_cast<Float>(0));
                VectorType x(n);
                for (Index k = 0; k < n; ++k) {
                    ej[k] = static_cast<Float>(1);
                    solve(ej, x);
                    inv_diag_cache[k] = x[k];
                    inv_column_sum_cache[k] =
                        std::accumulate(x.begin(), x.end(), static_cast<Float>(0));
                    ej[k] = static_cast<Float>(0);
                }
            }
            return inv_diag_cache[j];
        }

        Float inverse_column_sum(Index j) const {
            if (inv_column_sum_cache.empty())
                inverse_diagonal(j);
            return inv_column_sum_cache[j];
        }

        void solve_block_inverse(std::span<const Index> indices, const VectorType &right_hand_side,
                                 VectorType &solution) const {
            MatrixType block = inverse_block(indices);
            if (is_cholesky())
                solve_cholesky_block(std::move(block), indices, right_hand_side, solution);
            else
                solve_lu_block(std::move(block), right_hand_side, solution);
        }

        [[nodiscard]] Index size() const {
            if (is_cholesky())
                return stationary_sqrt.size();
            if (block_factor)
                return block_factor->size;
            return lu_factor->LU.rows();
        }

      private:
        static MatrixType dense_copy(const SparseMatrixType &A, Float scale) {
            MatrixType dense(A.rows, A.cols, static_cast<Float>(0));
            for (Index i = 0; i < A.rows; ++i)
                for (Index k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
                    dense(i, A.col_idx[k]) += scale * A.values[k];
            return dense;
        }

        void factor_lu(const SparseMatrixType &R, std::vector<Index> levels) {
            if (R.rows > dense_limit && !levels.empty()) {
                block_factor = factorize_block_lu(R, levels, static_cast<Float>(-1));
                return;
            }
            lu_factor = factorize_lu<Float>(dense_copy(R, static_cast<Float>(-1)));
            if (lu_factor->singular)
                throw std::runtime_error("subnetwork operator is singular");
        }

        void factor_dense_cholesky(const MatrixType &S) {
            cholesky_factor = factorize_cholesky(S);
            if (!cholesky_factor->success)
                throw std::runtime_error("reversible subnetwork operator is not positive definite");
        }

        void factor_cholesky(const SparseMatrixType &S, std::vector<Index> levels) {
            if (S.rows > dense_limit && !levels.empty()) {
                block_cholesky_factor = factorize_block_cholesky(S, levels);
                return;
            }
            factor_dense_cholesky(dense_copy(S, static_cast<Float>(1)));
        }

        void solve_lu(const VectorType &b, VectorType &x) const {
            if (block_factor)
                block_solve(*block_factor, b, x);
            else
                lu_solve(*lu_factor, b, x);
        }

        void solve_cholesky(const VectorType &b, VectorType &x) const {
            // If S = -D^{-1} R D, then (-R)^{-1}b = D S^{-1}D^{-1}b.
            VectorType scaled_b = b;
            kernel::raw::scale_rows(scaled_b.data(), stationary_sqrt.data(), scaled_b.size(), 1,
                                    true);

            if (block_cholesky_factor)
                block_solve(*block_cholesky_factor, scaled_b, x);
            else
                cholesky_solve(*cholesky_factor, scaled_b, x);

            kernel::raw::scale_rows(x.data(), stationary_sqrt.data(), x.size(), 1, false);
        }

        void solve_multiple_cholesky(const MatrixType &b, MatrixType &x) const {
            MatrixType scaled_b = b;
            kernel::raw::scale_rows(scaled_b.data(), stationary_sqrt.data(), scaled_b.rows(),
                                    scaled_b.cols(), true);

            if (block_cholesky_factor)
                block_solve(*block_cholesky_factor, scaled_b, x);
            else
                cholesky_solve(*cholesky_factor, scaled_b, x);

            kernel::raw::scale_rows(x.data(), stationary_sqrt.data(), x.rows(), x.cols(), false);
        }

        MatrixType inverse_block(std::span<const Index> indices) const {
            const Index r = indices.size();
            MatrixType block(r, r, static_cast<Float>(0));
            MatrixType basis(size(), r, static_cast<Float>(0));
            for (Index j = 0; j < r; ++j)
                basis(indices[j], j) = static_cast<Float>(1);

            // Restrict Z E to the removed states to obtain Z_JJ in one block solve.
            MatrixType columns;
            solve_multiple(basis, columns);
            for (Index j = 0; j < r; ++j)
                for (Index i = 0; i < r; ++i)
                    block(i, j) = columns(indices[i], j);
            return block;
        }

        void solve_lu_block(MatrixType block, const VectorType &right_hand_side,
                            VectorType &solution) const {
            auto factor = factorize_lu<Float>(std::move(block));
            if (factor.singular)
                throw std::runtime_error("cut-time block is singular");
            lu_solve(factor, right_hand_side, solution);
        }

        void solve_cholesky_block(MatrixType block, std::span<const Index> indices,
                                  const VectorType &right_hand_side, VectorType &solution) const {
            const Index r = indices.size();
            // Z_JJ = D_J (S^{-1})_JJ D_J^{-1}; recover the symmetric block by similarity.
            for (Index i = 0; i < r; ++i)
                for (Index j = 0; j < r; ++j)
                    block(i, j) *= stationary_sqrt[indices[j]] / stationary_sqrt[indices[i]];

            auto factor = factorize_cholesky(block);
            if (!factor.success)
                throw std::runtime_error("cut-time Cholesky block is not positive definite");

            VectorType scaled_rhs(r);
            for (Index i = 0; i < r; ++i)
                scaled_rhs[i] = right_hand_side[i] / stationary_sqrt[indices[i]];
            cholesky_solve(factor, scaled_rhs, solution);
            for (Index i = 0; i < r; ++i)
                solution[i] *= stationary_sqrt[indices[i]];
        }
    };

    std::vector<State> states_;
    std::unordered_map<State, Index, StateHash<State>> index_;
    SparseMatrixType R_;
    std::vector<Float> exit_rates_;
    std::vector<Index> boundary_states_;
    std::vector<BoundaryTransition<Index, State, Float>> boundary_;
    std::vector<Float> stationary_sqrt_;

    std::optional<SolverImpl> solver_;
};

} // namespace else_sim
