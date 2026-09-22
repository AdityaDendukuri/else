// The reaction systems of the paper's experiments, on markovkit's model type.
#pragma once

#include "markovkit/reaction_system.hpp"
#include "common/rate_matrix.hpp"

namespace subsweep::examples {

using markovkit::state;

struct reaction_model {
    markovkit::reaction_system system;
    array<real> rates;
    state initial;
};

// Oregonator at y1 = 500, y2 = 1000, y3 = 2000; the level is half the total count.
inline reaction_model oregonator() {
    const real y1 = 500.0, y2 = 1000.0, y3 = 2000.0, mu1 = 2000.0, mu2 = 50000.0;
    return {{.changes = {{1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-2, 0, 0}, {0, 1, -1}},
             .propensities =
                 {
                     [](const state &x, const auto &r, double) { return r[0] * x[1]; },
                     [](const state &x, const auto &r, double) { return r[1] * x[0] * x[1]; },
                     [](const state &x, const auto &r, double) { return r[2] * x[0]; },
                     [](const state &x, const auto &r, double) {
                         return x[0] < 2 ? 0.0 : 0.5 * r[3] * x[0] * (x[0] - 1);
                     },
                     [](const state &x, const auto &r, double) { return r[4] * x[2]; },
                 }},
            {mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1, 2.0 * mu1 / (y1 * y1), (mu1 + mu2) / y3},
            {500, 1000, 2000}};
}

inline idx oregonator_level(const state &x) { return static_cast<idx>((x[0] + x[1] + x[2]) / 2); }

// Genetic toggle switch with Hill repression at `scale`; the level is the U count.
inline reaction_model toggle_switch(real scale, state initial) {
    const real alpha = 20.0 * scale, beta = 400.0 * scale, K3 = std::pow(100.0 * scale, 3);
    return {{.changes = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}},
             .propensities =
                 {
                     [](const state &x, const auto &r, double) {
                         return r[0] + r[1] * r[2] / (r[2] + std::pow(real(x[1]), 3));
                     },
                     [](const state &x, const auto &r, double) { return r[3] * x[0]; },
                     [](const state &x, const auto &r, double) {
                         return r[4] + r[5] * r[6] / (r[6] + std::pow(real(x[0]), 3));
                     },
                     [](const state &x, const auto &r, double) { return r[7] * x[1]; },
                 }},
            {alpha, beta, K3, 1.0 + 0.1 / 1.1, alpha, beta, K3, 1.0},
            initial};
}

inline idx toggle_level(const state &x) { return static_cast<idx>(x[0]); }

// Substrate and enzyme bind, unbind, and form product.
inline reaction_model michaelis_menten() {
    return {{.changes = {{-1, -1, 1, 0}, {1, 1, -1, 0}, {0, 1, -1, 1}},
             .propensities =
                 {
                     [](const state &x, const auto &r, double) { return r[0] * x[0] * x[1]; },
                     [](const state &x, const auto &r, double) { return r[1] * x[2]; },
                     [](const state &x, const auto &r, double) { return r[2] * x[2]; },
                 }},
            {0.01, 0.1, 0.1},
            {50, 10, 0, 0}};
}

} // namespace subsweep::examples
