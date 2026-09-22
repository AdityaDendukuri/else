// Unordered reconstruction: p(t) by Talbot inversion of
//   p_hat(s) = sum_r a_r(s)^T (sI + R_bar_r^T)^-1,   a_{r+1}(s) = B_r^T x_r(s),
// over the chain of subnetworks a distribution was swept through.
#pragma once

#include "linear/solvers/auto_resolvent.hpp"
#include "linear/sparse/sparse.hpp"
#include "quadrature/talbot.hpp"
#include "subsweep/types.hpp"
#include <algorithm>
#include <stdexcept>

namespace subsweep {

// One link of the chain: R_bar_r, the label of each slot, the coupling
// B_r restricted to the next link's slots, and the empirical-to-exact
// ratio of each slot's arrival mass when the support was resampled.
struct chain_link {
    num::spmat matrix;
    array<idx> label;
    array<coupling> exits; // source slot here, target slot in the next link
    num::vec arrival_scale; // empty means one
};

// With `normalize` the density is scaled to unit mass over the chain; without
// it the mass is the probability of not having been absorbed or lost.
[[nodiscard]] inline table<idx, real> reconstruct(const array<chain_link> &chain, idx start,
                                                   real time, idx modes = 14,
                                                   bool normalize = true) {
    if (chain.empty())
        throw std::invalid_argument("reconstruction needs at least one link");
    if (!(time > 0.0))
        throw std::invalid_argument("the reconstruction time must be positive");
    array<num::auto_resolvent_solver> resolvents;
    for (const chain_link &link : chain)
        resolvents.emplace_back(num::scaled(num::transpose(link.matrix), -1.0));

    table<idx, real> density;
    num::inverse_laplace_accumulate(time, modes, [&](num::cplx shift, num::cplx weight) {
        array<num::cplx> arrival(chain.front().matrix.n_rows(), num::cplx(0.0, 0.0));
        arrival[start] = num::cplx(1.0, 0.0);
        for (idx r = 0; r < chain.size(); ++r) {
            const chain_link &link = chain[r];
            resolvents[r].factorize(shift);
            array<num::cplx> local;
            resolvents[r].solve(arrival, local);
            for (idx j = 0; j < local.size(); ++j)
                density[link.label[j]] += (weight * local[j]).real();
            if (r + 1 == chain.size())
                break;
            arrival.assign(chain[r + 1].matrix.n_rows(), num::cplx(0.0, 0.0));
            for (const coupling &c : link.exits)
                arrival[c.target] += c.rate * local[c.source];
            const num::vec &scale = chain[r + 1].arrival_scale;
            for (idx j = 0; j < scale.size(); ++j)
                arrival[j] *= scale[j];
        }
    });

    real total = 0.0;
    for (auto &[label, value] : density) {
        value = std::max(0.0, value);
        total += value;
    }
    if (normalize && total > 0.0)
        for (auto &[label, value] : density)
            value /= total;
    return density;
}

} // namespace subsweep
