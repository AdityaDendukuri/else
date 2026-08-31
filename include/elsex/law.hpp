// Entrance laws: occupation quantities shared by trajectories.
#pragma once

#include "elsex/subnetwork.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/auto_linear.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>

namespace elsex {

// U = E Z and V = E Z^2. Each row is one entrance law.
///
/// Row `k` corresponds to the `k`th requested entrance, so
/// \f$U_{kj} = Z_{i_k j}\f$ is the expected time spent in `j` before escape when
/// starting from entrance \f$i_k\f$. Every trajectory sitting at the same
/// entrance shares this row.
struct EntranceLaw {
    num::Matrix occupation;        ///< \f$U\f$, with `occupation(k, j)`.
    num::Matrix second_occupation; ///< \f$V\f$, with `second_occupation(k, j)`.
};

// Solve U M = E and then V M = U using transpose solves.
///
/// Both solves use transpose M; columns are entrance indicators.
// Columns are entrance indicators.
template <typename State, typename Solver>
[[nodiscard]] EntranceLaw entrance_law(const Subnetwork<State> &subnetwork,
                                       std::span<const idx> entrances,
                                       const Solver &solver) {
    if (entrances.empty()) {
        throw std::invalid_argument("entrance_law requires at least one entrance");
    }

    num::Matrix indicators(subnetwork.size(), entrances.size(), 0.0);
    for (idx k = 0; k < entrances.size(); ++k) {
        if (entrances[k] >= subnetwork.size()) {
            throw std::out_of_range("entrance state is outside the subnetwork");
        }
        indicators(entrances[k], k) = 1.0;
    }

    const num::Matrix occupation = solver.solve_transpose(indicators);
    const num::Matrix second_occupation = solver.solve_transpose(occupation);
    return {num::transpose(occupation), num::transpose(second_occupation)};
}

template <typename State>
[[nodiscard]] EntranceLaw entrance_law(const Subnetwork<State> &subnetwork,
                                       std::span<const idx> entrances) {
    return entrance_law(subnetwork, entrances, subnetwork);
}

// beta_j = w_j U_kj over escape states, normalized.
///
/// Normalize w_j U_kj over the escape states.
// Normalization removes roundoff; entries follow the escape-state list.
/// position within `subnetwork.escape_states()`.
template <typename State>
[[nodiscard]] num::Vector escape_distribution(const Subnetwork<State> &subnetwork,
                                              const EntranceLaw &law, idx entrance_row) {
    const auto &escape_states = subnetwork.escape_states();
    const auto escape_rates = subnetwork.escape_rates();

    num::Vector distribution(escape_states.size(), 0.0);
    real mass = 0.0;
    for (idx position = 0; position < escape_states.size(); ++position) {
        const idx j = escape_states[position];
        distribution[position] = std::max(0.0, escape_rates[j] * law.occupation(entrance_row, j));
        mass += distribution[position];
    }
    if (!(mass > 0.0)) {
        throw std::runtime_error("subnetwork has no reachable escape from this entrance");
    }
    for (real &value : distribution) {
        value /= mass;
    }
    return distribution;
}

/// \f$\mu_{kj} = V_{kj}/U_{kj}\f$, the mean escape time conditioned on escaping via `j`.
[[nodiscard]] inline real conditional_escape_time(const EntranceLaw &law, idx entrance_row,
                                                  idx escape_state) {
    const real occupation = law.occupation(entrance_row, escape_state);
    if (!(occupation > 0.0)) {
        return 0.0;
    }
    return law.second_occupation(entrance_row, escape_state) / occupation;
}

/// Mean escape time from entrance `k`, i.e. \f$\sum_j U_{kj}\f$.
[[nodiscard]] inline real mean_escape_time(const EntranceLaw &law, idx entrance_row) {
    real total = 0.0;
    for (idx j = 0; j < law.occupation.cols(); ++j) {
        total += law.occupation(entrance_row, j);
    }
    return total;
}

} // namespace elsex
