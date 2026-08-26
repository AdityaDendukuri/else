#pragma once

#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace else_sim {

/// @brief Connected neighborhood grown in descending stationary weight order.
template <typename Float = double, typename Index = std::size_t, typename SparseMatrix = num::SparseMatrix>
inline std::vector<Index>
laplacian_neighborhood(const SparseMatrix &laplacian,
                       const std::vector<Float> &stationary_sqrt,
                       Index initial, Index capacity) {
    if (initial >= laplacian.n_rows() || capacity == 0 || capacity > laplacian.n_rows()) {
        throw std::invalid_argument("invalid Laplacian neighborhood request");
    }

    std::vector<bool> selected(laplacian.n_rows(), false);
    num::MaxIndexedPQ<double> frontier(laplacian.n_rows());
    std::vector<Index> states;
    states.reserve(capacity);

    const auto expose = [&](Index state) {
        for (std::size_t k = laplacian.row_ptr()[state]; k < laplacian.row_ptr()[state + 1]; ++k) {
            const Index neighbor = laplacian.col_idx()[k];
            if (neighbor != state && !selected[neighbor] && !frontier.contains(neighbor)) {
                frontier.push(neighbor, stationary_sqrt[neighbor]);
            }
        }
    };

    selected[initial] = true;
    states.push_back(initial);
    expose(initial);

    while (states.size() < capacity) {
        if (frontier.empty()) {
            throw std::invalid_argument("initial component is smaller than the requested capacity");
        }
        const Index state = frontier.top_index();
        frontier.pop();
        selected[state] = true;
        states.push_back(state);
        expose(state);
    }
    return states;
}

} // namespace else_sim
