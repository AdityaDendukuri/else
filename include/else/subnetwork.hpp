#pragma once

#include "else/linalg.hpp"
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

namespace else_sim {

/// @brief Pure template discrete subnetwork with unpivoted/pivoted LU and Cholesky solvers.
template <typename Float = double, typename Index = std::size_t,
          typename State = std::vector<int>>
class Subnetwork {
  public:
    using MatrixType = Matrix<Float>;
    using VectorType = std::vector<Float>;
    using SparseMatrixType = SparseMatrix<Float, Index>;

    Subnetwork() = default;

    /// @brief Construct a general (non-reversible) subnetwork using LU (default: unpivoted for M-matrices).
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary,
               bool pivot = false)
        : states_(std::move(states)), R_(std::move(R)), boundary_(std::move(boundary)) {
        initialize();
        MatrixType M(states_.size(), states_.size(), static_cast<Float>(0));
        for (Index i = 0; i < states_.size(); ++i) {
            for (Index j = 0; j < states_.size(); ++j) {
                M(i, j) = -R_(i, j);
            }
        }
        solver_ = std::make_unique<SolverImpl>(std::move(M), pivot);
    }

    /// @brief Construct a reversible subnetwork with stationary scaling weights h = sqrt(pi).
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary,
               std::vector<Float> stationary_sqrt)
        : states_(std::move(states)), R_(std::move(R)), boundary_(std::move(boundary)),
          stationary_weights_(std::move(stationary_sqrt)) {
        initialize();
        // S = H^-1 (-R) H (i.e. S_ij = -R_ij * h_j / h_i)
        MatrixType S(states_.size(), states_.size(), static_cast<Float>(0));
        for (Index i = 0; i < states_.size(); ++i) {
            for (Index j = 0; j < states_.size(); ++j) {
                if (i == j) {
                    S(i, i) = -R_(i, i);
                } else {
                    const Float s_ij = -R_(i, j) * (stationary_weights_[j] / stationary_weights_[i]);
                    const Float s_ji = -R_(j, i) * (stationary_weights_[i] / stationary_weights_[j]);
                    S(i, j) = static_cast<Float>(0.5) * (s_ij + s_ji);
                }
            }
        }
        solver_ = std::make_unique<SolverImpl>(std::move(S), stationary_weights_);
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

    /// @brief Solves (-R) u = p0 for transient occupation.
    [[nodiscard]] VectorType occupation(const VectorType &start) const {
        VectorType u(size(), static_cast<Float>(0));
        solver_->solve(start, u);
        return u;
    }

    /// @brief Computes first and second time-weighted occupation moments.
    [[nodiscard]] OccupationIntegrals<MatrixType> occupation_integrals(const MatrixType &starts) const {
        OccupationIntegrals<MatrixType> integrals;
        const Index n = size();
        const Index cols = starts.cols();
        integrals.occupation = MatrixType(n, cols, static_cast<Float>(0));
        integrals.time_weighted_occupation = MatrixType(n, cols, static_cast<Float>(0));

        VectorType col_in(n), col_occ(n), col_time(n);
        for (Index c = 0; c < cols; ++c) {
            for (Index r = 0; r < n; ++r) col_in[r] = starts(r, c);
            solver_->solve(col_in, col_occ);
            solver_->solve(col_occ, col_time);
            for (Index r = 0; r < n; ++r) {
                integrals.occupation(r, c) = col_occ[r];
                integrals.time_weighted_occupation(r, c) = col_time[r];
            }
        }
        return integrals;
    }

    /// @brief Computes exit probability distribution across boundary states.
    [[nodiscard]] std::vector<Float> exit_probabilities(const VectorType &occupation) const {
        std::vector<Float> probs;
        probs.reserve(boundary_states_.size());
        Float sum = static_cast<Float>(0);
        for (Index b : boundary_states_) {
            Float p = static_cast<Float>(exit_rates_[b] * std::max(static_cast<Float>(0), occupation[b]));
            probs.push_back(p);
            sum += p;
        }
        if (sum > static_cast<Float>(0)) {
            for (auto &p : probs) p /= sum;
        }
        return probs;
    }

    [[nodiscard]] std::vector<Float> exit_probabilities(const OccupationIntegrals<MatrixType> &integrals,
                                                        Index column = 0) const {
        std::vector<Float> probs;
        probs.reserve(boundary_states_.size());
        Float sum = static_cast<Float>(0);
        for (Index b : boundary_states_) {
            Float p = static_cast<Float>(exit_rates_[b] * std::max(static_cast<Float>(0), integrals.occupation(b, column)));
            probs.push_back(p);
            sum += p;
        }
        if (sum > static_cast<Float>(0)) {
            for (auto &p : probs) p /= sum;
        }
        return probs;
    }

    /// @brief Computes conditional mean exit time \tau(b) = (u_t)_b / u_b.
    [[nodiscard]] Float conditional_exit_time(const OccupationIntegrals<MatrixType> &integrals,
                                              Index boundary_state, Index column = 0) const {
        const Float u_occ = integrals.occupation(boundary_state, column);
        if (u_occ <= static_cast<Float>(1e-15)) return static_cast<Float>(0);
        const Float u_time = integrals.time_weighted_occupation(boundary_state, column);
        return std::max(static_cast<Float>(0), u_time / u_occ);
    }

    /// @brief Computes flow-balanced expected visits s_j = max(0, w_j u_j - p_j).
    [[nodiscard]] std::vector<Float> expected_entries(const VectorType &occupancy,
                                                      const VectorType &start) const {
        std::vector<Float> counts(size(), static_cast<Float>(0));
        for (Index j = 0; j < size(); ++j) {
            counts[j] = std::max(static_cast<Float>(0), (total_rates_[j] * occupancy[j]) - start[j]);
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
        std::vector<Float> diagonal(candidates.size(), static_cast<Float>(0));
        solver_->inverse_diagonal(candidates, diagonal);

        std::vector<Float> losses(candidates.size(), static_cast<Float>(0));
        for (Index index = 0; index < candidates.size(); ++index) {
            const Index candidate = candidates[index];
            if (diagonal[index] > static_cast<Float>(1e-15)) {
                losses[index] = occupancy[candidate] * (column_sums[candidate] / diagonal[index]);
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
            return std::accumulate(occupancy.begin(), occupancy.end(), static_cast<Float>(0));
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

        MatrixType R_red(m, m, static_cast<Float>(0));
        VectorType p_red(m, static_cast<Float>(0));
        for (Index i = 0; i < m; ++i) {
            for (Index j = 0; j < m; ++j) {
                R_red(i, j) = -R_(kept[i], kept[j]);
            }
            for (Index j = 0; j < n; ++j) {
                p_red[i] += -R_(kept[i], j) * occupancy[j];
            }
        }

        VectorType u_red(m, static_cast<Float>(0));
        auto lu_red = factorize_lu(std::move(R_red), false);
        if (lu_red.singular) return static_cast<Float>(0);
        lu_solve(lu_red, p_red, u_red);

        const Float tau_orig = std::accumulate(occupancy.begin(), occupancy.end(), static_cast<Float>(0));
        const Float tau_red = std::accumulate(u_red.begin(), u_red.end(), static_cast<Float>(0));
        return std::max(static_cast<Float>(0), tau_orig - tau_red);
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
        VectorType u_S(r, static_cast<Float>(0));
        for (Index i = 0; i < r; ++i) {
            u_S[i] = occupancy[removed_states[i]];
        }

        VectorType c(r, static_cast<Float>(0));
        MatrixType Z_SS;
        try {
            solver_->solve_principal_inverse(removed_states, u_S, c, &Z_SS);

            // Backward residual: || Z_SS * c - u_S ||_inf / ||u_S||_inf
            VectorType reconstructed(r, static_cast<Float>(0));
            matvec(Z_SS, c, reconstructed);
            Float err = static_cast<Float>(0), scale = static_cast<Float>(0);
            for (Index i = 0; i < r; ++i) {
                err = std::max(err, std::abs(reconstructed[i] - u_S[i]));
                scale = std::max(scale, std::abs(u_S[i]));
            }
            const Float residual = err / std::max(static_cast<Float>(1e-12), scale);

            if (residual <= tolerance && std::isfinite(residual)) {
                const auto &q = solver_->inverse_column_sums();
                Float loss = static_cast<Float>(0);
                for (Index i = 0; i < r; ++i) {
                    loss += q[removed_states[i]] * c[i];
                }
                if (std::isfinite(loss) && loss >= static_cast<Float>(0)) {
                    return {.loss = loss,
                            .estimated_error = residual,
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
        total_rates_.assign(states_.size(), static_cast<Float>(0));
        exit_rates_.assign(states_.size(), static_cast<Float>(0));
        for (Index i = 0; i < states_.size(); ++i) {
            index_[states_[i]] = static_cast<int>(i);
            total_rates_[i] = -R_(i, i);
        }
        for (const auto &t : boundary_) {
            exit_rates_[t.source] += t.rate;
        }
        for (Index i = 0; i < states_.size(); ++i) {
            if (exit_rates_[i] > static_cast<Float>(1e-15)) {
                boundary_states_.push_back(i);
            }
        }
    }

    struct SolverImpl {
        std::optional<LUFactor<Float, Index>> lu_factor;
        std::optional<CholeskyFactor<Float>> cholesky_factor;
        VectorType weights;
        mutable std::optional<VectorType> inverse_column_sums_cache;
        mutable std::vector<Float> inverse_diagonal_cache;

        explicit SolverImpl(MatrixType operator_matrix, bool pivot = false) {
            lu_factor = factorize_lu(std::move(operator_matrix), pivot);
            if (lu_factor->singular) {
                throw std::runtime_error("subnetwork operator is singular");
            }
        }

        SolverImpl(MatrixType symmetric, VectorType scaling_weights)
            : weights(std::move(scaling_weights)) {
            cholesky_factor = factorize_cholesky(std::move(symmetric));
            if (!cholesky_factor->success) {
                throw std::runtime_error("reversible subnetwork operator is not positive definite");
            }
        }

        void solve(const VectorType &b, VectorType &x) const {
            if (cholesky_factor) {
                VectorType transformed_b = b;
                for (std::size_t i = 0; i < b.size(); ++i) transformed_b[i] /= weights[i];
                cholesky_solve(*cholesky_factor, transformed_b, x);
                for (std::size_t i = 0; i < x.size(); ++i) x[i] *= weights[i];
                return;
            }
            lu_solve(*lu_factor, b, x);
        }

        void solve_transpose(const VectorType &b, VectorType &x) const {
            if (cholesky_factor) {
                VectorType transformed_b = b;
                for (std::size_t i = 0; i < b.size(); ++i) transformed_b[i] *= weights[i];
                cholesky_solve(*cholesky_factor, transformed_b, x);
                for (std::size_t i = 0; i < x.size(); ++i) x[i] /= weights[i];
                return;
            }
            lu_solve_transpose(*lu_factor, b, x);
        }

        const VectorType &inverse_column_sums() const {
            if (!inverse_column_sums_cache) {
                VectorType ones(size(), static_cast<Float>(1));
                VectorType column_sums(size(), static_cast<Float>(0));
                solve_transpose(ones, column_sums);
                inverse_column_sums_cache = std::move(column_sums);
            }
            return *inverse_column_sums_cache;
        }

        void inverse_diagonal(std::span<const Index> candidates, std::vector<Float> &diagonal) const {
            if (inverse_diagonal_cache.empty()) {
                inverse_diagonal_cache.assign(size(), static_cast<Float>(0));
            }
            VectorType e(size(), static_cast<Float>(0));
            VectorType z(size(), static_cast<Float>(0));

            for (Index i = 0; i < candidates.size(); ++i) {
                const Index c = candidates[i];
                if (inverse_diagonal_cache[c] == static_cast<Float>(0)) {
                    e[c] = static_cast<Float>(1);
                    solve(e, z);
                    e[c] = static_cast<Float>(0);
                    inverse_diagonal_cache[c] = z[c];
                }
                diagonal[i] = inverse_diagonal_cache[c];
            }
        }

        void solve_principal_inverse(std::span<const Index> indices,
                                     const VectorType &right_hand_side,
                                     VectorType &solution,
                                     MatrixType *extracted_block = nullptr) const {
            const Index r = indices.size();
            MatrixType block(r, r, static_cast<Float>(0));
            VectorType e(size(), static_cast<Float>(0));
            VectorType z(size(), static_cast<Float>(0));

            for (Index j = 0; j < r; ++j) {
                const Index col = indices[j];
                e[col] = static_cast<Float>(1);
                solve(e, z);
                e[col] = static_cast<Float>(0);
                for (Index i = 0; i < r; ++i) {
                    block(i, j) = z[indices[i]];
                }
            }

            if (extracted_block) *extracted_block = block;

            if (cholesky_factor) {
                // Symmetric coordinate solve
                MatrixType sym_block(r, r, static_cast<Float>(0));
                for (Index i = 0; i < r; ++i) {
                    for (Index j = 0; j < r; ++j) {
                        sym_block(i, j) = block(i, j) / (weights[indices[i]] * weights[indices[j]]);
                    }
                }
                auto block_fac = factorize_cholesky(std::move(sym_block));
                if (!block_fac.success) {
                    throw std::runtime_error("cut-time block is not positive definite");
                }
                VectorType transformed = right_hand_side;
                for (Index i = 0; i < r; ++i) transformed[i] /= weights[indices[i]];
                cholesky_solve(block_fac, transformed, solution);
                for (Index i = 0; i < r; ++i) solution[i] *= weights[indices[i]];
                return;
            }

            auto block_fac = factorize_lu(std::move(block), false);
            if (block_fac.singular) {
                throw std::runtime_error("cut-time block is singular");
            }
            lu_solve(block_fac, right_hand_side, solution);
        }

        [[nodiscard]] Index size() const {
            return cholesky_factor ? weights.size() : lu_factor->LU.rows();
        }
    };

    std::vector<State> states_;
    std::unordered_map<State, int, StateHash<State>> index_;
    SparseMatrixType R_;
    std::vector<Float> total_rates_;
    std::vector<Float> exit_rates_;
    std::vector<Index> boundary_states_;
    std::vector<BoundaryTransition<Index, State, Float>> boundary_;
    std::vector<Float> stationary_weights_;

    std::unique_ptr<SolverImpl> solver_;
};

} // namespace else_sim
