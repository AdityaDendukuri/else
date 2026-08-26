#pragma once

#include "else/linalg.hpp"
#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace else_sim {

/// @brief Connected neighborhood grown in descending stationary weight order.
template <typename Float = double, typename Index = std::size_t>
inline std::vector<Index>
laplacian_neighborhood(const SparseMatrix<Float, Index> &laplacian,
                       const std::vector<Float> &stationary_sqrt,
                       Index initial, Index capacity) {
    if (initial >= laplacian.rows || capacity == 0 || capacity > laplacian.rows) {
        throw std::invalid_argument("invalid Laplacian neighborhood request");
    }

    std::vector<bool> selected(laplacian.rows, false);
    std::vector<bool> in_frontier(laplacian.rows, false);
    std::priority_queue<std::pair<Float, Index>> frontier;
    std::vector<Index> states;
    states.reserve(capacity);

    const auto expose = [&](Index state) {
        for (Index k = laplacian.row_ptr[state]; k < laplacian.row_ptr[state + 1]; ++k) {
            const Index neighbor = laplacian.col_idx[k];
            if (neighbor != state && !selected[neighbor] && !in_frontier[neighbor]) {
                in_frontier[neighbor] = true;
                frontier.push({stationary_sqrt[neighbor], neighbor});
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
        const Index state = frontier.top().second;
        frontier.pop();
        if (!selected[state]) {
            selected[state] = true;
            states.push_back(state);
            expose(state);
        }
    }
    return states;
}

} // namespace else_sim
