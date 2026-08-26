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
#include <vector>

namespace else_sim {

template <typename Float = double>
struct ContourNodeStatus {
    Float time = static_cast<Float>(0);
    std::size_t node_index = 0;
    std::size_t total_nodes = 0;
    std::complex<Float> shift{0, 0};
};

template <typename Float = double>
using ContourObserver = std::function<void(const ContourNodeStatus<Float> &)>;

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

    [[nodiscard]] DensitySolution<State, Float> solve(const std::map<State, Float> &initial,
                                                      Float time, Index nodes = 14,
                                                      ContourObserver<Float> observer = nullptr) {
        if (!(time > static_cast<Float>(0))) {
            throw std::invalid_argument("Talbot density time must be positive");
        }
        const auto contour = talbot_contour<Float>(time, nodes);
        std::vector<Float> probability(states_.size(), static_cast<Float>(0));

        for (std::size_t k = 0; k < contour.size(); ++k) {
            const auto &node = contour[k];
            if (observer) {
                observer(ContourNodeStatus<Float>{
                    .time = time,
                    .node_index = k,
                    .total_nodes = contour.size(),
                    .shift = node.shift,
                });
            }

            // Propagate through subnetwork chain
            std::vector<std::complex<Float>> current_in;
            for (std::size_t sub_idx = 0; sub_idx < subnetworks_.size(); ++sub_idx) {
                const auto &sub = subnetworks_[sub_idx];
                const Index n_sub = sub.size();
                std::vector<std::complex<Float>> rhs(n_sub, std::complex<Float>(0, 0));

                if (sub_idx == 0) {
                    for (const auto &[s, val] : initial) {
                        int pos = sub.find(s);
                        if (pos >= 0) {
                            rhs[static_cast<std::size_t>(pos)] += std::complex<Float>(val, 0);
                        }
                    }
                } else {
                    rhs = current_in;
                }

                // Solve (shift * I - R) sol = rhs using unpivoted complex LU
                Matrix<std::complex<Float>> shifted_op(n_sub, n_sub, std::complex<Float>(0, 0));
                const auto &R = sub.generator();
                for (Index i = 0; i < n_sub; ++i) {
                    shifted_op(i, i) = node.shift - std::complex<Float>(R(i, i), 0);
                    for (Index j = 0; j < n_sub; ++j) {
                        if (i != j) shifted_op(i, j) = std::complex<Float>(-R(i, j), 0);
                    }
                }

                auto shifted_lu = factorize_lu(std::move(shifted_op), false);
                std::vector<std::complex<Float>> sol(n_sub, std::complex<Float>(0, 0));
                lu_solve(shifted_lu, rhs, sol);

                // Accumulate probability contribution at this node
                for (Index i = 0; i < n_sub; ++i) {
                    const auto global_idx = position_[sub.states()[i]];
                    const auto term = node.weight * sol[i];
                    probability[global_idx] += term.imag();
                }
            }
        }

        const Float factor = static_cast<Float>(-1.0 / M_PI);
        std::vector<Float> result(states_.size(), static_cast<Float>(0));
        for (std::size_t i = 0; i < states_.size(); ++i) {
            result[i] = std::max(static_cast<Float>(0), probability[i] * factor);
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
};

} // namespace else_sim
