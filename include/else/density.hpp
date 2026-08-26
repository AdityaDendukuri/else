#pragma once

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

struct ContourNodeStatus {
    double time;
    num::idx node_index;
    num::idx total_nodes;
    num::cplx shift;
};

using ContourObserver = std::function<void(const ContourNodeStatus &)>;

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
            solvers_.emplace_back(
                sub.generator(),
                num::AutoResolventOptions{.symmetric_pattern = sub.is_reversible()});
        }
    }

    [[nodiscard]] const std::vector<State> &states() const { return states_; }

    [[nodiscard]] DensitySolution<State, Float> solve(const State &initial, Float time,
                                                      Index nodes = 14,
                                                      ContourObserver observer = nullptr) {
        std::map<State, Float> init_map;
        init_map[initial] = static_cast<Float>(1.0);
        return solve(init_map, time, nodes, observer);
    }

    [[nodiscard]] DensitySolution<State, Float> solve(const std::map<State, Float> &initial,
                                                      Float time, Index nodes = 14,
                                                      ContourObserver observer = nullptr) {
        if (!(time > static_cast<Float>(0))) {
            throw std::invalid_argument("Talbot density time must be positive");
        }
        const auto contour = num::TalbotQuadrature(nodes);
        std::vector<double> probability(states_.size(), 0.0);

        num::idx current_node = 0;
        contour.accumulate(static_cast<double>(time), [&](num::cplx shift, num::cplx weight) {
            if (observer) {
                observer(ContourNodeStatus{
                    .time = static_cast<double>(time),
                    .node_index = current_node++,
                    .total_nodes = contour.modes,
                    .shift = shift,
                });
            }

            // Propagate through subnetwork chain
            std::vector<num::cplx> current_in;
            for (std::size_t sub_idx = 0; sub_idx < subnetworks_.size(); ++sub_idx) {
                const auto &sub = subnetworks_[sub_idx];
                std::vector<num::cplx> rhs(sub.size(), num::cplx(0.0, 0.0));

                if (sub_idx == 0) {
                    for (const auto &[s, val] : initial) {
                        int pos = sub.find(s);
                        if (pos >= 0) {
                            rhs[static_cast<std::size_t>(pos)] += val;
                        }
                    }
                } else {
                    rhs = current_in;
                }

                solvers_[sub_idx].factorize(shift);
                std::vector<num::cplx> sol(sub.size(), num::cplx(0.0, 0.0));
                solvers_[sub_idx].solve(rhs, sol);

                // Accumulate probability contribution at this node
                for (Index i = 0; i < sub.size(); ++i) {
                    const auto global_idx = position_[sub.states()[i]];
                    const auto term = weight * sol[i];
                    probability[global_idx] += term.imag();
                }
            }
        });

        // Divide by -pi
        const double factor = -1.0 / M_PI;
        std::vector<Float> result(states_.size(), static_cast<Float>(0));
        for (std::size_t i = 0; i < states_.size(); ++i) {
            result[i] = static_cast<Float>(std::max(0.0, probability[i] * factor));
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
    std::vector<num::AutoResolventSolver> solvers_;
};

} // namespace else_sim
