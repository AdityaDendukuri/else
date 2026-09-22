// Shedding: the two scores of the paper and the choice of slots to drop.
#pragma once

#include "subsweep/sweep/solution.hpp"
#include "subsweep/types.hpp"
#include <algorithm>
#include <stdexcept>

namespace subsweep {

// Expected number of jumps into j before exit: R_jj u_j - rho_j.
[[nodiscard]] inline num::vec expected_visits(view<const real> total_rates,
                                              const sweep_solution &solution) {
    const num::vec u = solution.mean_occupation();
    num::vec result(u.size(), 0.0);
    for (idx j = 0; j < u.size(); ++j)
        result[j] = total_rates[j] * u[j] - solution.rho[j];
    return result;
}

// Exact cut-time loss of removing j alone: u_j q_j / Z_jj.
[[nodiscard]] inline num::vec cut_time_losses(const sweep_solution &solution) {
    if (!solution.exit_time || !solution.diagonal)
        throw std::logic_error("the cut-time score needs q = Z 1 and diag(Z)");
    const num::vec u = solution.mean_occupation();
    num::vec result(u.size(), 0.0);
    for (idx j = 0; j < u.size(); ++j)
        result[j] = (*solution.diagonal)[j] > 0.0
                        ? u[j] * (*solution.exit_time)[j] / (*solution.diagonal)[j]
                        : 0.0;
    return result;
}

[[nodiscard]] inline num::vec scores(shedding_rule rule, view<const real> total_rates,
                                     const sweep_solution &solution) {
    return rule == shedding_rule::expected_visits ? expected_visits(total_rates, solution)
                                                  : cut_time_losses(solution);
}

// The `count` lowest scores among the unprotected slots. Ties, in practice
// the slots whose score underflows to zero, break toward the highest level
// and then by index, so that the block prefix retained across sweeps is as
// long as possible.
[[nodiscard]] inline array<idx> lowest_scores(view<const real> scores,
                                              view<const idx> protected_slots, idx count,
                                              view<const idx> levels = {}) {
    array<bool> ineligible(scores.size(), false);
    for (idx j : protected_slots)
        if (j < scores.size())
            ineligible[j] = true;
    array<idx> eligible;
    for (idx j = 0; j < scores.size(); ++j)
        if (!ineligible[j])
            num::append(eligible, j);
    count = std::min(count, static_cast<idx>(eligible.size()));
    std::partial_sort(eligible.begin(), eligible.begin() + count, eligible.end(),
                      [&](idx a, idx b) {
                          if (scores[a] != scores[b])
                              return scores[a] < scores[b];
                          if (!levels.empty() && levels[a] != levels[b])
                              return levels[a] > levels[b];
                          return a < b;
                      });
    eligible.resize(count);
    return eligible;
}

} // namespace subsweep
