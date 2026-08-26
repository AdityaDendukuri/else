#pragma once

#include "else/linalg.hpp"
#include "else/shedding.hpp"
#include "else/types.hpp"
#include "else/woodbury.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace else_sim {

/// @brief Expanding Local Subnetwork Enumeration (ELSE) container and solver.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
class Subnetwork {
  public:
    using FloatType = Float;
    using IndexType = Index;
    using StateType = State;
    using MatrixType = Matrix<Float>;
    using SparseMatrixType = SparseMatrix<Float, Index>;
    using VectorType = std::vector<Float>;

    Subnetwork() = default;

    /// @brief Primary constructor for general (non-reversible) subnetwork.
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary,
               bool pivot = false)
        : states_(std::move(states)), R_(std::move(R)), boundary_(std::move(boundary)) {
        initialize();
        MatrixType M(states_.size(), states_.size(), static_cast<Float>(0));
        for (Index i = 0; i < states_.size(); ++i) {
            for (Index k = R_.row_ptr[i]; k < R_.row_ptr[i + 1]; ++k) {
                const Index j = R_.col_idx[k];
                M(i, j) = -R_.values[k];
            }
        }
        solver_ = std::make_unique<SolverImpl>(std::move(M), pivot);
    }

    /// @brief Primary constructor for reversible subnetwork with stationary weights.
    Subnetwork(std::vector<State> states, SparseMatrixType R,
               std::vector<BoundaryTransition<Index, State, Float>> boundary,
               std::vector<Float> stationary_sqrt)
        : states_(std::move(states)), R_(std::move(R)), boundary_(std::move(boundary)),
          stationary_weights_(std::move(stationary_sqrt)) {
        initialize();
        MatrixType S(states_.size(), states_.size(), static_cast<Float>(0));
        for (Index i = 0; i < states_.size(); ++i) {
            for (Index k = R_.row_ptr[i]; k < R_.row_ptr[i + 1]; ++k) {
                const Index j = R_.col_idx[k];
                S(i, j) = -R_.values[k] * (stationary_weights_[j] / stationary_weights_[i]);
            }
        }
        solver_ = std::make_unique<SolverImpl>(std::move(S), stationary_weights_);
    }

    /// @brief Construct from a RestrictedGenerator object (such as cme::RestrictedGenerator).
    template <typename RestrictedGen>
    requires requires(RestrictedGen r) { r.states; r.generator; r.boundary; }
    explicit Subnetwork(RestrictedGen &&gen, bool pivot = false)
        : states_(std::forward<RestrictedGen>(gen).states) {
        const auto &g = gen.generator;
        R_.rows = static_cast<Index>(g.n_rows());
        R_.cols = static_cast<Index>(g.n_cols());
        R_.row_ptr.assign(g.row_ptr(), g.row_ptr() + g.n_rows() + 1);
        R_.col_idx.assign(g.col_idx(), g.col_idx() + g.nnz());
        R_.values.assign(g.values(), g.values() + g.nnz());
        boundary_.reserve(gen.boundary.size());
        for (const auto &b : gen.boundary) {
            boundary_.push_back({static_cast<Index>(b.source), b.destination, static_cast<Float>(b.rate)});
        }
        initialize();
        MatrixType M(states_.size(), states_.size(), static_cast<Float>(0));
        for (Index i = 0; i < states_.size(); ++i) {
            for (Index k = R_.row_ptr[i]; k < R_.row_ptr[i + 1]; ++k) {
                const Index j = R_.col_idx[k];
                M(i, j) = -R_.values[k];
            }
        }
        solver_ = std::make_unique<SolverImpl>(std::move(M), pivot);
    }

    /// @brief Construct from a ReversibleGenerator object (such as cme::ReversibleGenerator).
    template <typename ReversibleGen>
    requires requires(ReversibleGen r) { r.restricted; r.stationary_sqrt; }
    explicit Subnetwork(ReversibleGen &&gen)
        : states_(std::forward<ReversibleGen>(gen).restricted.states),
          stationary_weights_(std::forward<ReversibleGen>(gen).stationary_sqrt) {
        const auto &g = gen.restricted.generator;
        R_.rows = static_cast<Index>(g.n_rows());
        R_.cols = static_cast<Index>(g.n_cols());
        R_.row_ptr.assign(g.row_ptr(), g.row_ptr() + g.n_rows() + 1);
        R_.col_idx.assign(g.col_idx(), g.col_idx() + g.nnz());
        R_.values.assign(g.values(), g.values() + g.nnz());
        boundary_.reserve(gen.restricted.boundary.size());
        for (const auto &b : gen.restricted.boundary) {
            boundary_.push_back({static_cast<Index>(b.source), b.destination, static_cast<Float>(b.rate)});
        }
        initialize();
        MatrixType S(states_.size(), states_.size(), static_cast<Float>(0));
        for (Index i = 0; i < states_.size(); ++i) {
            for (Index k = R_.row_ptr[i]; k < R_.row_ptr[i + 1]; ++k) {
                const Index j = R_.col_idx[k];
                S(i, j) = -R_.values[k] * (stationary_weights_[j] / stationary_weights_[i]);
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
    template <typename Vec>
    [[nodiscard]] VectorType occupation(const Vec &start) const {
        VectorType in(size(), static_cast<Float>(0));
        for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(size()), static_cast<std::size_t>(start.size())); ++i) {
            in[i] = static_cast<Float>(start[i]);
        }
        VectorType u(size(), static_cast<Float>(0));
        solver_->solve(in, u);
        return u;
    }

    /// @brief Computes first and second time-weighted occupation moments.
    template <typename Mat>
    [[nodiscard]] OccupationIntegrals<MatrixType> occupation_integrals(const Mat &starts) const {
        OccupationIntegrals<MatrixType> integrals;
        const Index n = size();
        const Index cols = starts.cols();
        integrals.occupation = MatrixType(n, cols, static_cast<Float>(0));
        integrals.time_weighted_occupation = MatrixType(n, cols, static_cast<Float>(0));

        VectorType rhs(n), occ(n), tw_occ(n);
        for (Index c = 0; c < cols; ++c) {
            for (Index r = 0; r < n; ++r) rhs[r] = starts(r, c);
            solver_->solve(rhs, occ);
            for (Index r = 0; r < n; ++r) integrals.occupation(r, c) = occ[r];
            solver_->solve(occ, tw_occ);
            for (Index r = 0; r < n; ++r) integrals.time_weighted_occupation(r, c) = tw_occ[r];
        }
        return integrals;
    }

    /// @brief Computes exit probability distribution across boundary states.
    template <typename Vec>
    requires (!requires(Vec v) { v.occupation; })
    [[nodiscard]] std::vector<Float> exit_probabilities(const Vec &occupation) const {
        std::vector<Float> probs;
        probs.reserve(boundary_states_.size());
        Float sum = static_cast<Float>(0);
        for (Index b : boundary_states_) {
            Float p = static_cast<Float>(exit_rates_[b] * std::max(static_cast<Float>(0), static_cast<Float>(occupation[b])));
            probs.push_back(p);
            sum += p;
        }
        if (sum > static_cast<Float>(0)) {
            for (auto &p : probs) p /= sum;
        }
        return probs;
    }

    template <typename IntegralsType>
    requires requires(IntegralsType v) { v.occupation; }
    [[nodiscard]] std::vector<Float> exit_probabilities(const IntegralsType &integrals,
                                                        Index column = 0) const {
        std::vector<Float> probs;
        probs.reserve(boundary_states_.size());
        Float sum = static_cast<Float>(0);
        for (Index b : boundary_states_) {
            Float p = static_cast<Float>(exit_rates_[b] * std::max(static_cast<Float>(0), static_cast<Float>(integrals.occupation(b, column))));
            probs.push_back(p);
            sum += p;
        }
        if (sum > static_cast<Float>(0)) {
            for (auto &p : probs) p /= sum;
        }
        return probs;
    }

    /// @brief Computes mean waiting time conditioned on exit via local boundary state.
    template <typename IntegralsType>
    [[nodiscard]] Float conditional_exit_time(const IntegralsType &integrals,
                                              Index boundary_state, Index column = 0) const {
        const Float occ = integrals.occupation(boundary_state, column);
        if (!(occ > static_cast<Float>(0))) return static_cast<Float>(0);
        const Float tw = integrals.time_weighted_occupation(boundary_state, column);
        return tw / occ;
    }

    /// @brief Evaluates cut-time losses for candidate states.
    template <typename Vec>
    [[nodiscard]] std::vector<Float> cut_time_losses(const Vec &occupancy,
                                                     std::span<const Index> candidates) const {
        std::vector<Float> losses;
        losses.reserve(candidates.size());
        for (Index c : candidates) {
            if (c >= size()) {
                throw std::out_of_range("candidate state is out of range");
            }
            const Float occ_c = static_cast<Float>(occupancy[c]);
            const Float inv_diag = solver_->inverse_diagonal(c);
            losses.push_back(occ_c / inv_diag);
        }
        return losses;
    }

    /// @brief Evaluates cut-time losses for all states in the subnetwork.
    template <typename Vec>
    [[nodiscard]] std::vector<Float> cut_time_losses(const Vec &occupancy) const {
        std::vector<Index> all(size());
        std::iota(all.begin(), all.end(), static_cast<Index>(0));
        return cut_time_losses(occupancy, all);
    }

    /// @brief Naive cut-time loss via reduced subnetwork solve.
    template <typename Vec>
    [[nodiscard]] Float naive_cut_time_loss(const Vec &occupancy,
                                            std::span<const Index> removed_states) const {
        const Index n = size();
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
            for (Index k = R_.row_ptr[kept[i]]; k < R_.row_ptr[kept[i] + 1]; ++k) {
                const Index j_orig = R_.col_idx[k];
                if (!removed[j_orig]) {
                    auto it = std::lower_bound(kept.begin(), kept.end(), j_orig);
                    if (it != kept.end() && *it == j_orig) {
                        const Index j_red = static_cast<Index>(std::distance(kept.begin(), it));
                        R_red(i, j_red) = -R_.values[k];
                    }
                }
                p_red[i] += -R_.values[k] * static_cast<Float>(occupancy[j_orig]);
            }
        }

        VectorType u_red(m, static_cast<Float>(0));
        auto lu_red = factorize_lu<Float, Index>(std::move(R_red), false);
        if (lu_red.singular) return static_cast<Float>(0);
        lu_solve(lu_red, p_red, u_red);

        const Float tau_orig = std::accumulate(occupancy.begin(), occupancy.end(), static_cast<Float>(0));
        const Float tau_red = std::accumulate(u_red.begin(), u_red.end(), static_cast<Float>(0));
        return std::max(static_cast<Float>(0), tau_orig - tau_red);
    }

    /// @brief Evaluates cut-time loss for a set of removed states.
    template <typename Vec>
    [[nodiscard]] Float cut_time_loss(const Vec &occupancy,
                                      std::span<const Index> removed_states,
                                      Float tolerance = static_cast<Float>(1e-6)) const {
        return cut_time_loss_with_diagnostics(occupancy, removed_states, tolerance).loss;
    }

    /// @brief Detailed cut-time loss evaluation returning loss, floating-point error, and fallback diagnostics.
    template <typename Vec>
    [[nodiscard]] CutTimeLossResult<Float>
    cut_time_loss_with_diagnostics(const Vec &occupancy,
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
                    .estimated_error = static_cast<Float>(std::numeric_limits<Float>::epsilon()),
                    .naive_fallback_used = false};
        }

        const Index r = removed_states.size();
        VectorType y(r);
        for (Index i = 0; i < r; ++i) y[i] = static_cast<Float>(occupancy[removed_states[i]]);

        VectorType y_corr(r);
        solver_->solve_block_inverse(removed_states, y, y_corr);

        Float delta_tau = static_cast<Float>(0);
        Float accumulated_err = static_cast<Float>(0);
        bool precision_acceptable = true;

        for (Index i = 0; i < r; ++i) {
            if (!safe_add(delta_tau, accumulated_err, y_corr[i], tolerance)) {
                precision_acceptable = false;
            }
        }

        if (precision_acceptable && delta_tau > static_cast<Float>(0)) {
            return {.loss = delta_tau, .estimated_error = accumulated_err, .naive_fallback_used = false};
        }

        Float naive_loss = naive_cut_time_loss(occupancy, removed_states);
        return {.loss = naive_loss, .estimated_error = accumulated_err, .naive_fallback_used = true};
    }

  private:
    void initialize() {
        index_.clear();
        for (std::size_t i = 0; i < states_.size(); ++i) {
            index_[states_[i]] = static_cast<int>(i);
        }
        const Index n = states_.size();
        total_rates_.assign(n, static_cast<Float>(0));
        exit_rates_.assign(n, static_cast<Float>(0));
        boundary_states_.clear();

        for (Index i = 0; i < n; ++i) {
            Float sum_off = static_cast<Float>(0);
            for (Index k = R_.row_ptr[i]; k < R_.row_ptr[i + 1]; ++k) {
                if (R_.col_idx[k] != i) sum_off += R_.values[k];
            }
            total_rates_[i] = sum_off;
        }

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
        std::optional<LUFactor<Float, Index>> lu_factor;
        std::optional<CholeskyFactor<Float>> cholesky_factor;
        std::vector<Float> weights;
        mutable std::vector<Float> inv_diag_cache;

        SolverImpl(MatrixType M, bool pivot)
            : lu_factor(factorize_lu<Float, Index>(std::move(M), pivot)) {
            if (lu_factor->singular) {
                throw std::runtime_error("subnetwork operator is singular");
            }
        }

        SolverImpl(MatrixType S, std::vector<Float> w)
            : cholesky_factor(factorize_cholesky(S)), weights(std::move(w)) {
            if (!cholesky_factor->success) {
                throw std::runtime_error("reversible subnetwork operator is not positive definite");
            }
        }

        void solve(const VectorType &b, VectorType &x) const {
            if (cholesky_factor) {
                const Index n = weights.size();
                VectorType scaled_b(n);
                for (Index i = 0; i < n; ++i) scaled_b[i] = b[i] / weights[i];
                cholesky_solve(*cholesky_factor, scaled_b, x);
                for (Index i = 0; i < n; ++i) x[i] *= weights[i];
            } else {
                lu_solve(*lu_factor, b, x);
            }
        }

        Float inverse_diagonal(Index j) const {
            if (inv_diag_cache.empty()) {
                const Index n = size();
                inv_diag_cache.resize(n, static_cast<Float>(0));
                VectorType ej(n, static_cast<Float>(0));
                VectorType x(n);
                for (Index k = 0; k < n; ++k) {
                    ej[k] = static_cast<Float>(1);
                    solve(ej, x);
                    inv_diag_cache[k] = x[k];
                    ej[k] = static_cast<Float>(0);
                }
            }
            return inv_diag_cache[j];
        }

        void solve_block_inverse(std::span<const Index> indices,
                                 const VectorType &right_hand_side,
                                 VectorType &solution) const {
            const Index r = indices.size();
            const Index n = size();
            MatrixType block(r, r, static_cast<Float>(0));

            VectorType e(n, static_cast<Float>(0));
            VectorType col(n);
            for (Index j = 0; j < r; ++j) {
                e[indices[j]] = static_cast<Float>(1);
                solve(e, col);
                for (Index i = 0; i < r; ++i) {
                    block(i, j) = col[indices[i]];
                }
                e[indices[j]] = static_cast<Float>(0);
            }

            if (cholesky_factor) {
                for (Index i = 0; i < r; ++i) {
                    for (Index j = 0; j < r; ++j) {
                        block(i, j) = block(i, j) / (weights[indices[i]] * weights[indices[j]]);
                    }
                }
                auto chol_block = factorize_cholesky(block);
                if (!chol_block.success) {
                    throw std::runtime_error("cut-time Cholesky block is not positive definite");
                }
                VectorType sc_rhs(r);
                for (Index i = 0; i < r; ++i) sc_rhs[i] = right_hand_side[i] / weights[indices[i]];
                cholesky_solve(chol_block, sc_rhs, solution);
                for (Index i = 0; i < r; ++i) solution[i] *= weights[indices[i]];
                return;
            }

            auto block_fac = factorize_lu<Float, Index>(std::move(block), false);
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

// Explicit deduction guides
template <typename State, typename SparseMatrixType, typename Index, typename Float>
Subnetwork(std::vector<State>, SparseMatrixType, std::vector<BoundaryTransition<Index, State, Float>>, bool)
    -> Subnetwork<Float, Index, State>;

template <typename State, typename SparseMatrixType, typename Index, typename Float>
Subnetwork(std::vector<State>, SparseMatrixType, std::vector<BoundaryTransition<Index, State, Float>>, std::vector<Float>)
    -> Subnetwork<Float, Index, State>;

template <typename RestrictedGen>
requires requires(RestrictedGen r) { r.states; }
Subnetwork(RestrictedGen &&)
    -> Subnetwork<double, std::size_t, typename std::decay_t<decltype(std::declval<RestrictedGen>().states)>::value_type>;

template <typename ReversibleGen>
requires requires(ReversibleGen r) { r.restricted; }
Subnetwork(ReversibleGen &&)
    -> Subnetwork<double, std::size_t, typename std::decay_t<decltype(std::declval<ReversibleGen>().restricted.states)>::value_type>;

} // namespace else_sim
