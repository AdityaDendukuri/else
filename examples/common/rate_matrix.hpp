// Rate matrices as row callbacks for the paper's problems: reaction models,
// sparse matrices, graph Laplacians, and absorbing targets.
#pragma once

#include "subsweep/subsweep.hpp"
#include "structures/containers/indexed_priority_queue.hpp"
#include <functional>
#include <utility>

namespace subsweep::examples {

// R from a reaction model: `changes[r]` and `propensities[r](state, rates, t)`.
template <typename State = num::multi_index, typename Model, typename Rates>
[[nodiscard]] auto reaction_rate_matrix(const Model &model, const Rates &rates) {
    return [&model, &rates](const State &state, auto &&visit) {
        for (idx r = 0; r < model.changes.size(); ++r) {
            const real rate = static_cast<real>(model.propensities[r](state, rates, real(0)));
            if (!(rate > 0.0))
                continue;
            State destination = state;
            bool representable = true;
            for (idx i = 0; i < model.changes[r].size(); ++i) {
                destination[i] += model.changes[r][i];
                representable = representable && destination[i] >= 0;
            }
            if (representable)
                visit(std::move(destination), rate);
        }
    };
}

// R from a sparse matrix: entry (i, j), i != j, is the rate from i to j.
template <typename State = num::multi_index>
[[nodiscard]] auto sparse_rate_matrix(const num::spmat &entries) {
    return [&entries](const State &state, auto &&visit) {
        const idx i = static_cast<idx>(state[0]);
        for (auto k = entries.row_ptr()[i]; k < entries.row_ptr()[i + 1]; ++k) {
            const idx j = static_cast<idx>(entries.col_idx()[k]);
            const real rate = static_cast<real>(entries.values()[k]);
            if (j != i && rate > 0.0)
                visit(State{static_cast<int>(j)}, rate);
        }
    };
}

// A reversible R from a symmetric graph Laplacian L and h = sqrt(pi): the
// rate from a to b is -L(a,b) h_b / h_a.
template <typename State = num::multi_index>
[[nodiscard]] auto laplacian_rate_matrix(const num::spmat &laplacian, view<const real> h) {
    return [&laplacian, h](const State &state, auto &&visit) {
        const idx a = static_cast<idx>(state[0]);
        for (auto k = laplacian.row_ptr()[a]; k < laplacian.row_ptr()[a + 1]; ++k) {
            const idx b = static_cast<idx>(laplacian.col_idx()[k]);
            const real rate = -static_cast<real>(laplacian.values()[k]) * h[b] / h[a];
            if (b != a && rate > 0.0)
                visit(State{static_cast<int>(b)}, rate);
        }
    };
}

template <typename State = num::multi_index>
[[nodiscard]] auto laplacian_stationary(view<const real> h) {
    return [h](const State &state) {
        const real value = h[static_cast<idx>(state[0])];
        return value * value;
    };
}

// R with the states satisfying `predicate` made absorbing.
template <typename R, typename Predicate>
[[nodiscard]] auto absorbing(R r, Predicate predicate) {
    return [r = std::move(r), predicate = std::move(predicate)](const auto &state, auto &&visit) {
        if (!predicate(state))
            r(state, visit);
    };
}

// The `capacity` states nearest `origin` by stationary weight, grown from
// the origin one neighbor at a time.
[[nodiscard]] inline array<idx> laplacian_neighborhood(const num::spmat &laplacian,
                                                       view<const real> h, idx origin,
                                                       idx capacity) {
    const idx n = laplacian.n_rows();
    array<bool> selected(n, false);
    num::indexed_priority_queue<real, idx, std::greater<real>> frontier(n);
    array<idx> states;
    const auto expose = [&](idx state) {
        for (auto k = laplacian.row_ptr()[state]; k < laplacian.row_ptr()[state + 1]; ++k) {
            const idx neighbor = static_cast<idx>(laplacian.col_idx()[k]);
            if (neighbor != state && !selected[neighbor] && !frontier.contains(neighbor))
                frontier.push(neighbor, h[neighbor]);
        }
    };
    selected[origin] = true;
    num::append(states, origin);
    expose(origin);
    while (states.size() < capacity && !frontier.empty()) {
        const idx state = frontier.top_index();
        frontier.pop();
        selected[state] = true;
        num::append(states, state);
        expose(state);
    }
    return states;
}

} // namespace subsweep::examples
