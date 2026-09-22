// What the sweep asks of a problem. The sweep never sees a state: it works
// on slots [0, n) of the subnetwork, and the system owns the states behind
// them, its random numbers, and whatever it records. Three operations
// connect the two.
//
//   grow(walker, from, n)   The walker leaves the subnetwork through slot
//                           `from`. Sample one of its transitions whose
//                           destination lies outside slots [0, n), the
//                           subnetwork the exit law was computed on, and
//                           return that destination as a grown_state. A
//                           destination added since the solve is not fresh.
//   discard(j)              Forget slot j. The last slot moves into j, so
//                           the slots stay [0, n - 1); the subnetwork does
//                           the same on its side.
//   expand(slot, visit)     Optional. Place every exterior neighbor of
//                           `slot` and call visit on each: the layer
//                           expansion of the paper.
//
// A grown_state carries what the subnetwork needs to append a slot: the
// total outgoing rate (the diagonal of R_bar), the rates from the state to
// the slots already present (its row) and from them to it (its column),
// and the level and stationary weight when the block or reversible regime
// is in use. row_system implements all of this from a row callback.
#pragma once

#include "subsweep/sweep/subnetwork.hpp"
#include "subsweep/types.hpp"
#include <concepts>

namespace subsweep {

struct grown_state {
    idx slot = 0;
    bool fresh = false;
    real total_rate = 0.0;
    array<slot_rate> row, column;
    idx level = 0;
    real stationary = 0.0;
};

template <typename S>
concept system = requires(S &s, idx walker, idx slot) {
    { s.grow(walker, slot, slot) } -> std::same_as<grown_state>;
    { s.discard(slot) };
};

template <typename S>
concept expandable_system =
    system<S> && requires(S &s, idx slot, void (*visit)(const grown_state &)) {
        s.expand(slot, visit);
    };

} // namespace subsweep
