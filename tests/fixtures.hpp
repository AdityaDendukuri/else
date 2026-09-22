// Small subnetworks with known structure for the library tests.
#pragma once

#include "subsweep/subsweep.hpp"
#include <cmath>
#include <random>

namespace fixtures {

using namespace subsweep;

// Birth-death on copy numbers 1..n with exits at both ends; the level of a
// slot is its copy number. `perturbed` slots get different rates, as
// replacing a state would produce.
inline subnetwork birth_death(idx n, const sweep_options &options, view<const idx> perturbed = {}) {
    array<bool> changed(n, false);
    for (idx j : perturbed)
        changed[j] = true;
    subnetwork sn(options);
    for (idx i = 0; i < n; ++i) {
        const real birth = 1.7 + (changed[i] ? 0.9 : 0.0);
        const real death = 0.35 * static_cast<real>(i + 1) + (changed[i] ? 0.8 : 0.0);
        array<slot_rate> row, column;
        if (i > 0) {
            num::append(row, i - 1, death);
            const real previous_birth = 1.7 + (changed[i - 1] ? 0.9 : 0.0);
            num::append(column, i - 1, previous_birth);
        }
        sn.add(birth + death, row, column, i);
    }
    return sn;
}

// A reversible chain: stationary weights pi_i and symmetric conductances,
// three slots per level, exits of rate 0.4 at both ends.
inline subnetwork reversible_chain(idx n, const sweep_options &options,
                                   view<const idx> perturbed = {}) {
    array<bool> changed(n, false);
    for (idx j : perturbed)
        changed[j] = true;
    const auto pi = [](idx i) { return 1.0 + 0.3 * std::sin(0.7 * static_cast<real>(i)); };
    const auto conductance = [&](idx i, idx j) {
        return 0.5 + (changed[i] || changed[j] ? 0.35 : 0.0);
    };
    subnetwork sn(options);
    for (idx i = 0; i < n; ++i) {
        real total = (i == 0 || i + 1 == n) ? 0.4 : 0.0;
        array<slot_rate> row, column;
        if (i > 0) {
            num::append(row, i - 1, conductance(i, i - 1) / pi(i));
            num::append(column, i - 1, conductance(i - 1, i) / pi(i - 1));
            total += conductance(i, i - 1) / pi(i);
        }
        if (i + 1 < n)
            total += conductance(i, i + 1) / pi(i);
        sn.add(total, row, column, i / 3, pi(i));
    }
    return sn;
}

inline num::vec random_vector(idx n, unsigned seed) {
    num::rng generator(seed);
    std::uniform_real_distribution<real> spread(-1.0, 1.0);
    num::vec b(n, 0.0);
    for (idx i = 0; i < n; ++i)
        b[i] = spread(generator);
    return b;
}

inline sweep_options with_regime(solve_regime regime, shedding_rule rule = shedding_rule::expected_visits) {
    sweep_options options;
    options.regime = regime;
    options.rule = rule;
    return options;
}

} // namespace fixtures
