// The first-exit quantities of one sweep for the current slots i_1..i_d:
// U = EZ, V = EZ^2, the exit rates w, and what follows from them.
#pragma once

#include "subsweep/types.hpp"
#include "container/matrix.hpp"
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace subsweep {

// The first-exit quantities of one sweep for the current slots i_1..i_d.
struct sweep_solution {
    array<idx> current; // i_k
    num::vec rho;       // current distribution over [0, n)
    num::mat occupation;        // U = EZ, one row per current slot
    num::mat second_occupation; // V = EZ^2
    num::vec exit_rates;        // w = R_bar 1
    std::optional<num::vec> exit_time; // q = Z 1, cut-time rule
    std::optional<num::vec> diagonal;  // diag Z, cut-time rule
    sweep_diagnostics diagnostics;

    [[nodiscard]] idx size() const { return exit_rates.size(); }

    // beta_k(j) = w_j U(k,j), normalized: the law of the pre-exit slot.
    [[nodiscard]] num::vec exit_law(idx k) const {
        num::vec law(size(), 0.0);
        real mass = 0.0;
        for (idx j = 0; j < size(); ++j) {
            law[j] = std::max(0.0, exit_rates[j] * occupation(k, j));
            mass += law[j];
        }
        if (!(mass > 0.0))
            throw std::runtime_error("no exit is reachable from this current slot");
        for (real &value : law)
            value /= mass;
        return law;
    }

    // E[tau | X_0 = i_k, X_{tau-} = j] = V(k,j) / U(k,j).
    [[nodiscard]] real duration(idx k, idx j) const {
        const real u = occupation(k, j);
        return u > 0.0 ? second_occupation(k, j) / u : 0.0;
    }

    // u = Z^T rho = sum_k rho_k U(k, .).
    [[nodiscard]] num::vec mean_occupation() const {
        num::vec u(size(), 0.0);
        for (idx k = 0; k < current.size(); ++k)
            for (idx j = 0; j < size(); ++j)
                u[j] += rho[current[k]] * occupation(k, j);
        return u;
    }

    // w_j u_j: the exit flux through each pre-exit slot under rho.
    [[nodiscard]] num::vec exit_flux() const {
        num::vec flux(size(), 0.0);
        const num::vec u = mean_occupation();
        for (idx j = 0; j < size(); ++j)
            flux[j] = std::max(0.0, exit_rates[j] * u[j]);
        return flux;
    }
};

} // namespace subsweep
