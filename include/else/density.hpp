#pragma once

#include "else/linalg.hpp"
#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if __has_include(<linalg/solvers/auto_resolvent.hpp>)
#include <linalg/solvers/auto_resolvent.hpp>
#define ELSE_HAS_AUTO_RESOLVENT 1
#else
#define ELSE_HAS_AUTO_RESOLVENT 0
#endif

namespace else_sim {

template <typename Float = double>
struct ContourNodeStatusTemplate {
    Float time = static_cast<Float>(0);
    std::size_t node_index = 0;
    std::size_t total_nodes = 0;
    std::complex<Float> shift{0, 0};
};

using ContourNodeStatus = ContourNodeStatusTemplate<double>;

template <typename Float = double>
using ContourObserver = std::function<void(const ContourNodeStatusTemplate<Float> &)>;

template <typename State, typename Arg>
inline State to_state(const Arg &arg) {
    if constexpr (std::is_same_v<std::decay_t<Arg>, State>) {
        return arg;
    } else {
        State s;
        for (std::size_t i = 0; i < arg.size(); ++i) {
            s.push_back(static_cast<int>(arg[i]));
        }
        return s;
    }
}

struct BoundaryTransfer {
    std::size_t source;
    std::size_t target_local;
    double rate;
};

/// @brief Invert the Laplace-domain density propagated through an ELSE subnetwork chain
/// via shifted resolvent linear solves on a modified Talbot contour.
template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
class LaplaceDensitySolver {
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
            num::SparseMatrix sp_R(R.rows, R.cols, std::move(vals), std::move(c_idx), std::move(r_rows));
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
                    transfers_[s].push_back({edge.source, static_cast<std::size_t>(pos), static_cast<double>(edge.rate)});
                }
            }
        }
    }

    [[nodiscard]] const std::vector<State> &states() const { return states_; }

    [[nodiscard]] DensitySolution<State, Float> solve(const State &initial, Float time,
                                                      Index nodes = 14,
                                                      ContourObserver<Float> observer = nullptr) {
        std::map<State, Float> init_map;
        init_map[initial] = static_cast<Float>(1.0);
        return solve(init_map, time, nodes, observer);
    }

    template <typename StateArg>
    requires (!std::is_same_v<std::decay_t<StateArg>, State> &&
              !std::is_same_v<std::decay_t<StateArg>, std::map<State, Float>>)
    [[nodiscard]] DensitySolution<State, Float> solve(const StateArg &initial, Float time,
                                                      Index nodes = 14,
                                                      ContourObserver<Float> observer = nullptr) {
        State converted = to_state<State>(initial);
        std::map<State, Float> init_map;
        init_map[std::move(converted)] = static_cast<Float>(1.0);
        return solve(init_map, time, nodes, observer);
    }

    [[nodiscard]] DensitySolution<State, Float> solve(const std::map<State, Float> &initial,
                                                      Float time, Index nodes = 14,
                                                      ContourObserver<Float> observer = nullptr) {
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
            if (observer) {
                observer(ContourNodeStatusTemplate<Float>{
                    .time = time,
                    .node_index = k,
                    .total_nodes = contour.size(),
                    .shift = node.shift,
                });
            }

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
                const auto &sub = subnetworks_[s];
                const Index n_sub = sub.size();

#if ELSE_HAS_AUTO_RESOLVENT
                solvers_[s].factorize(node.shift);
                solvers_[s].solve(arrivals[s], occupations[s]);
#else
                const auto &R = sub.generator();
                Matrix<std::complex<Float>> shifted_op(n_sub, n_sub, std::complex<Float>(0, 0));
                for (Index i = 0; i < n_sub; ++i) {
                    shifted_op(i, i) = node.shift;
                    const auto row_start = R.row_ptr[i];
                    const auto row_stop = R.row_ptr[i + 1];
                    for (auto idx = row_start; idx < row_stop; ++idx) {
                        const Index j = R.col_idx[idx];
                        shifted_op(i, j) -= std::complex<Float>(R.values[idx], 0);
                    }
                }
                auto shifted_lu = factorize_lu<std::complex<Float>, Index>(std::move(shifted_op), false);
                lu_solve(shifted_lu, arrivals[s], occupations[s]);
#endif

                // Accumulate density contribution at this node
                for (Index i = 0; i < n_sub; ++i) {
                    const auto global_idx = position_[sub.states()[i]];
                    const auto term = node.weight * occupations[s][i];
                    probability[global_idx] += term.real();
                }

                // Transfer boundary flux into next subnetwork
                if (s + 1 < subnetworks_.size()) {
                    for (const auto &tf : transfers_[s]) {
                        arrivals[s + 1][tf.target_local] += std::complex<Float>(tf.rate, 0) * occupations[s][tf.source];
                    }
                }
            }
        }

        std::vector<Float> result(states_.size(), static_cast<Float>(0));
        Float total = static_cast<Float>(0);
        for (std::size_t i = 0; i < states_.size(); ++i) {
            result[i] = std::max(static_cast<Float>(0), probability[i]);
            total += result[i];
        }
        if (total > static_cast<Float>(0)) {
            for (auto &v : result) v /= total;
        }

        return DensitySolution<State, Float>{
            .states = states_,
            .probability = std::move(result),
        };
    }

  private:
    std::vector<Subnetwork<Float, Index, State>> subnetworks_;
    std::vector<State> states_;
    std::map<State, std::size_t> position_;
    std::vector<std::vector<BoundaryTransfer>> transfers_;
#if ELSE_HAS_AUTO_RESOLVENT
    mutable std::vector<num::AutoResolventSolver> solvers_;
#endif
};

template <typename Float = double, typename Index = std::size_t, typename State = std::vector<int>>
using TalbotDensitySolver = LaplaceDensitySolver<Float, Index, State>;

template <typename ReactionSystem, typename Rates, typename State = std::vector<int>, typename Float = double, typename Index = std::size_t>
inline std::vector<Subnetwork<Float, Index, State>>
density_subnetworks(const ReactionSystem &model, const Rates &rates, const State &initial,
                    int count, ELSEOptions<Index, Float> options = {}) {
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
        for (const auto &[s, w] : density) mass += w;
        if (!(mass > options.tolerance)) break;
        for (auto &[s, w] : density) w /= mass;

        for (const auto &[s, w] : density) {
            if (workspace_set.insert(s).second) {
                workspace_states.push_back(s);
            }
        }

        std::size_t start_expand = 0;
        while (workspace_states.size() < options.capacity && start_expand < workspace_states.size()) {
            const auto current = workspace_states[start_expand++];
            for (std::size_t rx = 0; rx < model.changes.size(); ++rx) {
                if (model.propensities[rx](current, rates, static_cast<Float>(0)) > static_cast<Float>(0)) {
                    State next_st = current;
                    for (std::size_t d = 0; d < model.changes[rx].size(); ++d) {
                        next_st[d] += model.changes[rx][d];
                    }
                    if (workspace_set.insert(next_st).second) {
                        workspace_states.push_back(next_st);
                        if (workspace_states.size() >= options.capacity) break;
                    }
                }
            }
        }

        const Index n_ws = workspace_states.size();
        std::unordered_map<State, Index, StateHash<State>> local_index;
        for (Index i = 0; i < n_ws; ++i) local_index[workspace_states[i]] = i;

        std::vector<Index> r_rows, r_cols;
        std::vector<Float> r_vals;
        std::vector<BoundaryTransition<Index, State, Float>> boundary;

        for (Index j = 0; j < n_ws; ++j) {
            const auto &from_state = workspace_states[j];
            Float col_sum = static_cast<Float>(0);

            for (std::size_t rx = 0; rx < model.changes.size(); ++rx) {
                const Float rate = static_cast<Float>(model.propensities[rx](from_state, rates, static_cast<Float>(0)));
                if (rate <= static_cast<Float>(0)) continue;

                State to_state = from_state;
                for (std::size_t d = 0; d < model.changes[rx].size(); ++d) {
                    to_state[d] += model.changes[rx][d];
                }

                auto it = local_index.find(to_state);
                if (it != local_index.end()) {
                    r_rows.push_back(it->second);
                    r_cols.push_back(j);
                    r_vals.push_back(rate);
                } else {
                    boundary.push_back({j, to_state, rate});
                }
                col_sum += rate;
            }
            r_rows.push_back(j);
            r_cols.push_back(j);
            r_vals.push_back(-col_sum);
        }

        auto R = SparseMatrix<Float, Index>::from_triplets(n_ws, n_ws, r_rows, r_cols, r_vals);
        result.emplace_back(workspace_states, std::move(R), std::move(boundary));

        if (result.back().boundary().empty()) break;

        std::vector<Float> entrance(result.back().size(), static_cast<Float>(0));
        for (const auto &[s, w] : density) {
            int pos = result.back().find(s);
            if (pos >= 0) entrance[static_cast<std::size_t>(pos)] = w;
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
