// A system on the concept directly: a walker on the 2D lattice drifting
// toward the origin, absorbed at |x| + |y| = radius.
#pragma once

#include "subsweep/subsweep.hpp"
#include <cmath>

namespace subsweep::examples {

using site = num::multi_index; // (x, y)
inline int norm(const site &s) { return std::abs(s[0]) + std::abs(s[1]); }
struct lattice_system {
    int radius;
    real drift;
    num::rng random{7};
    array<site> states;
    table<site, idx> slot_of;
    // Unit steps, the inward ones faster by `drift`; boundary sites are absorbing.
    template <typename Visit> void rates(site s, Visit &&visit) const {
        for (site t : {site{s[0] + 1, s[1]}, site{s[0] - 1, s[1]}, site{s[0], s[1] + 1}, site{s[0], s[1] - 1}})
            if (norm(s) < radius)
                visit(t, norm(t) < norm(s) ? 1.0 + drift : 1.0);
    }
    // The slot for `s`, new unless present, with its rates to and from the slots present.
    grown_state place(site s) {
        grown_state g;
        if (const auto found = slot_of.find(s); found != slot_of.end())
            return g.slot = found->second, g;
        g.slot = states.size(), g.fresh = true, g.level = norm(s);
        rates(s, [&](site t, real rate) {
            g.total_rate += rate;
            if (const auto found = slot_of.find(t); found != slot_of.end()) {
                num::append(g.row, found->second, rate);
                rates(t, [&](site back, real r) { if (back == s) num::append(g.column, found->second, r); });
            }
        });
        return slot_of.emplace(s, g.slot), num::append(states, s), g;
    }
    grown_state grow(idx, idx from, idx n) { // one exit of `from` outside slots [0, n)
        array<site> exits;
        array<real> weights;
        rates(states[from], [&](site t, real rate) {
            if (const auto found = slot_of.find(t); found == slot_of.end() || found->second >= n)
                num::append(exits, t), num::append(weights, rate);
        });
        return place(exits[num::sample_categorical(view<const real>(weights), random)]);
    }
    void discard(idx j) { // the last slot moves into j
        slot_of.erase(states[j]);
        if (j + 1 != states.size()) states[j] = states.back(), slot_of[states[j]] = j;
        states.pop_back();
    }
};
} // namespace subsweep::examples
