#pragma once

#include "else/linalg.hpp"
#include "else/restriction.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if __has_include(<linear/solvers/auto_resolvent.hpp>)
#include <linear/solvers/auto_resolvent.hpp>
#define ELSE_HAS_AUTO_RESOLVENT 1
#else
#define ELSE_HAS_AUTO_RESOLVENT 0
#endif

namespace else_sim {

/// @brief Invert the Laplace-domain density propagated through an ELSE subnetwork chain
/// via shifted resolvent linear solves on a modified Talbot contour.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
class LaplaceDensitySolver {
    struct BoundaryTransfer {
        Index source;
        Index target;
        Float rate;
    };

  public:
    explicit LaplaceDensitySolver(std::vector<Subnetwork<Float, Index, State>> subnetworks)
        : subnetworks_(std::move(subnetworks)) {
        if (subnetworks_.empty()) {
            throw std::invalid_argument("Talbot density requires at least one ELSE subnetwork");
        }
        for (const auto &sub : subnetworks_) {
            for (const auto &s : sub.states()) {
                if (!position_.count(s)) {
                    position_[s] = states_.size();
                    states_.push_back(s);
                }
            }
#if ELSE_HAS_AUTO_RESOLVENT
            const auto &R = sub.generator();
            std::vector<num::idx> r_rows(R.row_ptr.begin(), R.row_ptr.end());
            std::vector<num::idx> c_idx(R.col_idx.begin(), R.col_idx.end());
            std::vector<double> vals(R.values.begin(), R.values.end());
            num::SparseMatrix sp_R(R.rows, R.cols, std::move(vals), std::move(c_idx),
                                   std::move(r_rows));
            solvers_.emplace_back(sp_R, num::AutoResolventOptions{.symmetric_pattern = true});
#endif
        }

        // Build boundary transfers between successive subnetworks
        transfers_.resize(subnetworks_.size());
        for (std::size_t s = 0; s + 1 < subnetworks_.size(); ++s) {
            const auto &next_sub = subnetworks_[s + 1];
            for (const auto &edge : subnetworks_[s].boundary()) {
                int pos = next_sub.find(edge.destination);
                if (pos >= 0) {
                    transfers_[s].push_back(
                        {edge.source, static_cast<Index>(pos), edge.rate});
                }
            }
        }
    }

    [[nodiscard]] const std::vector<State> &states() const { return states_; }

    [[nodiscard]] DensitySolution<State, Float> solve(const State &initial, Float time,
                                                      Index nodes = 14) {
        std::map<State, Float> init_map;
        init_map[initial] = static_cast<Float>(1.0);
        return solve(init_map, time, nodes);
    }

    [[nodiscard]] DensitySolution<State, Float> solve(const std::map<State, Float> &initial,
                                                      Float time, Index nodes = 14) {
        if (!(time > static_cast<Float>(0))) {
            throw std::invalid_argument("Talbot density time must be positive");
        }
        const auto contour = talbot_contour<Float>(time, nodes);
        std::vector<Float> probability(states_.size(), static_cast<Float>(0));

        // Buffers for subnetwork arrivals and occupations
        std::vector<std::vector<std::complex<Float>>> arrivals(subnetworks_.size());
        std::vector<std::vector<std::complex<Float>>> occupations(subnetworks_.size());
        for (std::size_t s = 0; s < subnetworks_.size(); ++s) {
            arrivals[s].assign(subnetworks_[s].size(), std::complex<Float>(0, 0));
            occupations[s].assign(subnetworks_[s].size(), std::complex<Float>(0, 0));
        }

        for (std::size_t k = 0; k < contour.size(); ++k) {
            const auto &node = contour[k];
            for (std::size_t s = 0; s < subnetworks_.size(); ++s) {
                std::fill(arrivals[s].begin(), arrivals[s].end(), std::complex<Float>(0, 0));
            }

            // Initial condition enters subnetwork 0
            for (const auto &[s, val] : initial) {
                int pos = subnetworks_[0].find(s);
                if (pos >= 0) {
                    arrivals[0][static_cast<std::size_t>(pos)] += std::complex<Float>(val, 0);
                }
            }

            // Propagate through subnetwork chain
            for (std::size_t s = 0; s < subnetworks_.size(); ++s) {
                solve_resolvent(s, node.shift, arrivals[s], occupations[s]);
                add_probability(s, node.weight, occupations[s], probability);
                if (s + 1 < subnetworks_.size())
                    transfer(s, occupations[s], arrivals[s + 1]);
            }
        }

        return DensitySolution<State, Float>{
            .states = states_,
            .probability = normalize(std::move(probability)),
        };
    }

  private:
    using Complex = std::complex<Float>;

    void solve_resolvent(std::size_t s, Complex shift,
                         const std::vector<Complex> &arrival,
                         std::vector<Complex> &occupation) const {
#if ELSE_HAS_AUTO_RESOLVENT
        solvers_[s].factorize(shift);
        solvers_[s].solve(arrival, occupation);
#else
        const auto &R = subnetworks_[s].generator();
        Matrix<Complex> shifted(R.rows, R.cols, Complex(0, 0));
        for (Index i = 0; i < R.rows; ++i) {
            shifted(i, i) = shift;
            for (Index k = R.row_ptr[i]; k < R.row_ptr[i + 1]; ++k)
                shifted(i, R.col_idx[k]) -= Complex(R.values[k], 0);
        }
        auto factor = factorize_lu<Complex, Index>(std::move(shifted));
        lu_solve(factor, arrival, occupation);
#endif
    }

    void add_probability(std::size_t s, Complex weight,
                         const std::vector<Complex> &occupation,
                         std::vector<Float> &probability) const {
        const auto &sub = subnetworks_[s];
        for (Index i = 0; i < sub.size(); ++i)
            probability[position_.at(sub.states()[i])] += (weight * occupation[i]).real();
    }

    void transfer(std::size_t s, const std::vector<Complex> &occupation,
                  std::vector<Complex> &next_arrival) const {
        for (const auto &edge : transfers_[s])
            next_arrival[edge.target] += edge.rate * occupation[edge.source];
    }

    static std::vector<Float> normalize(std::vector<Float> probability) {
        Float total = static_cast<Float>(0);
        for (auto &value : probability) {
            value = std::max(static_cast<Float>(0), value);
            total += value;
        }
        if (total > static_cast<Float>(0))
            for (auto &value : probability) value /= total;
        return probability;
    }

    std::vector<Subnetwork<Float, Index, State>> subnetworks_;
    std::vector<State> states_;
    std::map<State, std::size_t> position_;
    std::vector<std::vector<BoundaryTransfer>> transfers_;
#if ELSE_HAS_AUTO_RESOLVENT
    mutable std::vector<num::AutoResolventSolver> solvers_;
#endif
};

template <typename ReactionSystem, typename Rates, typename State = std::vector<int>,
          typename Float = double, typename Index = std::size_t,
          typename LevelFunction = SingleBlockLevel>
inline std::vector<Subnetwork<Float, Index, State>>
density_subnetworks(const ReactionSystem &model, const Rates &rates, const State &initial,
                    int count, ELSEOptions<Index, Float> options = {}, LevelFunction level = {}) {
    if (count < 1) {
        throw std::invalid_argument("ELSE density requires at least one subnetwork");
    }

    std::unordered_set<State, StateHash<State>> workspace_set;
    std::vector<State> workspace_states;
    std::unordered_map<State, Float, StateHash<State>> density{{initial, static_cast<Float>(1)}};
    std::vector<Subnetwork<Float, Index, State>> result;
    result.reserve(static_cast<std::size_t>(count));

    for (int step = 0; step < count; ++step) {
        Float mass = static_cast<Float>(0);
        for (const auto &[s, w] : density)
            mass += w;
        if (!(mass > options.tolerance))
            break;
        for (auto &[s, w] : density)
            w /= mass;

        for (const auto &[s, w] : density) {
            if (workspace_set.insert(s).second) {
                workspace_states.push_back(s);
            }
        }

        std::size_t start_expand = 0;
        while (workspace_states.size() < options.capacity &&
               start_expand < workspace_states.size()) {
            const auto current = workspace_states[start_expand++];
            for (std::size_t rx = 0; rx < model.changes.size(); ++rx) {
                if (model.propensities[rx](current, rates, static_cast<Float>(0)) >
                    static_cast<Float>(0)) {
                    State next_st = current;
                    for (std::size_t d = 0; d < model.changes[rx].size(); ++d) {
                        next_st[d] += model.changes[rx][d];
                    }
                    if (workspace_set.insert(next_st).second) {
                        workspace_states.push_back(next_st);
                        if (workspace_states.size() >= options.capacity)
                            break;
                    }
                }
            }
        }

        result.push_back(cme_subnetwork<ReactionSystem, Rates, State, Float, Index>(
            model, rates, workspace_states, level));

        if (result.back().boundary().empty())
            break;

        std::vector<Float> entrance(result.back().size(), static_cast<Float>(0));
        for (const auto &[s, w] : density) {
            int pos = result.back().find(s);
            if (pos >= 0)
                entrance[static_cast<std::size_t>(pos)] = w;
        }
        const auto occ = result.back().occupation(entrance);

        std::unordered_map<State, Float, StateHash<State>> next_density;
        for (const auto &trans : result.back().boundary()) {
            next_density[trans.destination] += trans.rate * occ[trans.source];
        }
        density = std::move(next_density);
    }
    return result;
}

} // namespace else_sim
