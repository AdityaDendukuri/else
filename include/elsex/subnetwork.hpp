// Finite state set with its restricted generator and factorization.
#pragma once

#include "elsex/types.hpp"
#include "linear/factorization/block_tridiagonal.hpp"
#include "linear/factorization/lu_no_pivot.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace elsex {

/// A finite state set \f$S\f$ carrying the restricted generator
/// \f$R = \bar R_{S,S}\f$ and a reusable factorization of \f$M = -R\f$.
///
/// Row convention, matching the paper: `R(i, j)` is the rate from state `i` to
/// state `j`. The diagonal retains the rates of transitions leaving \f$S\f$, so
/// the row sums are negative by exactly the escaping rate and
/// \f$w = -(R\mathbf 1) = M\mathbf 1\f$ is the escape rate.
///
/// Under the paper's exit assumption \f$M\f$ is a nonsingular M-matrix and
/// \f$Z = M^{-1}\f$ has \f$Z_{ij}\f$ equal to the expected time spent in `j`
/// before escape when starting from `i`.
template <typename State = std::vector<int>>
class Subnetwork {
  public:
    Subnetwork(std::vector<State> states, num::SparseMatrix generator,
               std::vector<BoundaryTransition<State>> boundary, std::vector<real> stationary = {},
               std::span<const idx> levels = {}, bool factorize = true)
        : states_(std::move(states)), generator_(std::move(generator)),
          boundary_(std::move(boundary)), stationary_(std::move(stationary)),
          operator_(num::scaled(generator_, -1.0)), escape_rates_(states_.size(), 0.0) {
        if (!stationary_.empty() && stationary_.size() != states_.size()) {
            throw std::invalid_argument("stationary weights must have one entry per state");
        }
        if (generator_.n_rows() != states_.size() || generator_.n_cols() != states_.size()) {
            throw std::invalid_argument("generator must be square with one row per state");
        }
        constexpr idx block_threshold = 128;
        if (factorize && !levels.empty() && states_.size() > block_threshold) {
            if (levels.size() != states_.size())
                throw std::invalid_argument("one block level is required per state");
            factor_.emplace(num::factor_block_lu(operator_, levels));
        } else if (factorize && !levels.empty()) {
            num::Matrix dense = num::dense(operator_);
            auto factor = num::factor_no_pivot(num::assume_square(dense));
            if (factor.singular)
                throw std::runtime_error("subnetwork operator is singular");
            factor_.emplace(std::move(factor));
        } else if (factorize) {
            factor_.emplace(std::in_place_type<num::AutoLinearSolver>, operator_,
                            num::AutoLinearOptions{.dense_limit = block_threshold});
        }

        index_.reserve(states_.size());
        for (idx i = 0; i < states_.size(); ++i) {
            index_.emplace(states_[i], i);
        }

        // w_j = sum_c r_jc, summed from the individual boundary transitions.
        //
        // Mathematically w = -(R 1), but evaluating that row sum cancels the
        // diagonal against the interior rates, so an interior state lands on a
        // signed round-off residual rather than exactly zero and a `> 0` test
        // would admit it as an escape state. The boundary rates are exact input
        // data and sum without cancellation.
        for (const auto &transition : boundary_) {
            if (transition.source >= states_.size()) {
                throw std::out_of_range("boundary transition source is outside the subnetwork");
            }
            escape_rates_[transition.source] += transition.rate;
        }
        for (idx j = 0; j < states_.size(); ++j) {
            if (escape_rates_[j] > 0.0) {
                escape_states_.push_back(j);
            }
        }
    }

    [[nodiscard]] idx size() const { return states_.size(); }
    [[nodiscard]] const std::vector<State> &states() const { return states_; }

    /// Position of `state`, or `size()` when it is outside the subnetwork.
    [[nodiscard]] idx find(const State &state) const {
        const auto found = index_.find(state);
        return found == index_.end() ? size() : found->second;
    }

    /// The restricted generator \f$R\f$, in row convention.
    [[nodiscard]] const num::SparseMatrix &generator() const { return generator_; }
    /// \f$M = -R\f$, the matrix that was factored.
    [[nodiscard]] const num::SparseMatrix &operator_matrix() const { return operator_; }
    /// Reusable factorization of \f$M\f$.
    [[nodiscard]] const num::AutoLinearSolver &factor() const {
        if (!factor_)
            throw std::logic_error("subnetwork has no factorization");
        if (!std::holds_alternative<num::AutoLinearSolver>(*factor_))
            throw std::logic_error("this operation requires the general factorization");
        return std::get<num::AutoLinearSolver>(*factor_);
    }

    [[nodiscard]] num::Matrix solve_transpose(const num::Matrix &rhs) const {
        num::Matrix result;
        require_factor();
        std::visit(
            [&](const auto &factor) {
                using Factor = std::decay_t<decltype(factor)>;
                if constexpr (std::is_same_v<Factor, num::AutoLinearSolver>)
                    factor.solve_transpose(rhs, result);
                else
                    num::solve_transpose(factor, rhs, result);
            },
            *factor_);
        return result;
    }
    [[nodiscard]] num::Vector solve_transpose(const num::Vector &rhs) const {
        num::Vector result;
        require_factor();
        std::visit(
            [&](const auto &factor) {
                using Factor = std::decay_t<decltype(factor)>;
                if constexpr (std::is_same_v<Factor, num::AutoLinearSolver>)
                    factor.solve_transpose(rhs, result);
                else
                    num::solve_transpose(factor, rhs, result);
            },
            *factor_);
        return result;
    }
    [[nodiscard]] num::Vector solve(const num::Vector &rhs) const {
        num::Vector result;
        require_factor();
        std::visit(
            [&](const auto &factor) {
                using Factor = std::decay_t<decltype(factor)>;
                if constexpr (std::is_same_v<Factor, num::AutoLinearSolver>)
                    factor.solve(rhs, result);
                else
                    num::solve(factor, rhs, result);
            },
            *factor_);
        return result;
    }
    [[nodiscard]] num::Matrix solve(const num::Matrix &rhs) const {
        num::Matrix result;
        require_factor();
        std::visit(
            [&](const auto &factor) {
                using Factor = std::decay_t<decltype(factor)>;
                if constexpr (std::is_same_v<Factor, num::AutoLinearSolver>)
                    factor.solve(rhs, result);
                else
                    num::solve(factor, rhs, result);
            },
            *factor_);
        return result;
    }
    /// Escape rates \f$w_j\f$.
    [[nodiscard]] std::span<const real> escape_rates() const {
        return {escape_rates_.data(), escape_rates_.size()};
    }
    /// States that can be escaped from, i.e. those with \f$w_j > 0\f$.
    [[nodiscard]] const std::vector<idx> &escape_states() const { return escape_states_; }
    /// Individual transitions leaving the subnetwork.
    [[nodiscard]] const std::vector<BoundaryTransition<State>> &boundary() const {
        return boundary_;
    }
    /// Stationary weights \f$\pi_S\f$, empty unless the caller supplied them.
    ///
    /// Only the probe estimator needs these; the solve path is the same either way.
    [[nodiscard]] std::span<const real> stationary() const {
        return {stationary_.data(), stationary_.size()};
    }
    [[nodiscard]] bool is_reversible() const { return !stationary_.empty(); }

  private:
    void require_factor() const {
        if (!factor_)
            throw std::logic_error("subnetwork has no factorization");
    }
    std::vector<State> states_;
    std::unordered_map<State, idx, StateHash<State>> index_;
    num::SparseMatrix generator_;
    std::vector<BoundaryTransition<State>> boundary_;
    std::vector<real> stationary_;
    num::SparseMatrix operator_;
    std::optional<std::variant<num::AutoLinearSolver, num::NoPivotLU, num::BlockLUFactor>> factor_;
    num::Vector escape_rates_;
    std::vector<idx> escape_states_;
};

} // namespace elsex
