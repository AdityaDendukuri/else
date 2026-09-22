// The reversible chain of a graph Laplacian, R = H^-1 L H with H = diag(sqrt pi),
// observed through its committors: SSA, ordered, and unordered subsweep
// estimates of E[C(X_t)] from each basin, and the panels that show them.
#pragma once

#include "common/rate_matrix.hpp"
#include "container/util/math.hpp"
#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "linear/eigen/jacobi_eig.hpp"
#include "plot/plot.hpp"
#include "stats/selection.hpp"
#include <cmath>
#include <random>
#include <string>

namespace subsweep::examples {

constexpr idx committor_count = 5;
using curves = std::array<array<real>, committor_count>;

struct curve_summary {
    curves mean, standard_error;
};

struct laplacian_problem {
    num::spmat laplacian;
    array<real> h;
    array<array<real>> committors;
    std::array<idx, committor_count> starts; // the state of largest committor per basin
};

inline laplacian_problem load_laplacian(const std::string &path) {
    const auto data = num::io::read_json(path);
    laplacian_problem p{num::io::sparse_matrix(data.at("L")), num::io::json_vector<real>(data.at("h")),
                        num::io::json_matrix<real>(data.at("C")), {}};
    for (idx basin = 0; basin < committor_count; ++basin)
        p.starts[basin] = num::argmax(p.laplacian.n_rows(),
                                      [&](idx state) { return p.committors[state][basin]; });
    return p;
}

namespace detail {

// Row `row` of exp(-t L) from the eigendecomposition of L.
inline num::vec semigroup_row(const num::eigen_result &eigen, real time, idx row) {
    const idx n = eigen.values.size();
    num::vec result(n, 0.0);
    for (idx mode = 0; mode < n; ++mode) {
        const real coefficient = eigen.vectors(row, mode) * std::exp(-time * eigen.values[mode]);
        for (idx column = 0; column < n; ++column)
            result[column] += coefficient * eigen.vectors(column, mode);
    }
    return result;
}

// P(tau > t | X_0 = origin, X_tau- = exit) from the modes of L.
inline real conditional_survival(const num::eigen_result &eigen, idx origin, idx exit, real time) {
    real numerator = 0.0, denominator = 0.0;
    for (idx mode = 0; mode < eigen.values.size(); ++mode) {
        const real c = eigen.vectors(origin, mode) * eigen.vectors(exit, mode) / eigen.values[mode];
        denominator += c;
        numerator += c * std::exp(-time * eigen.values[mode]);
    }
    return std::clamp(numerator / denominator, 0.0, 1.0);
}

inline real sample_exit_time(const num::eigen_result &eigen, idx origin, idx exit, num::rng &random) {
    const real target = 1.0 - std::uniform_real_distribution<real>(0.0, 1.0)(random);
    real lower = 0.0, upper = 1e-16;
    while (conditional_survival(eigen, origin, exit, upper) > target)
        upper *= 2.0;
    for (int iteration = 0; iteration < 56; ++iteration) {
        const real midpoint = 0.5 * (lower + upper);
        (conditional_survival(eigen, origin, exit, midpoint) > target ? lower : upper) = midpoint;
    }
    return 0.5 * (lower + upper);
}

// Mean and standard error over per-sample curves.
inline curve_summary reduce(const array<curves> &samples, idx grid) {
    curve_summary out;
    for (idx j = 0; j < committor_count; ++j) {
        out.mean[j].assign(grid, 0.0);
        out.standard_error[j].assign(grid, 0.0);
        for (idx k = 0; k < grid; ++k) {
            real sum = 0.0, square = 0.0;
            for (const curves &sample : samples) {
                sum += sample[j][k];
                square += sample[j][k] * sample[j][k];
            }
            const real n = static_cast<real>(samples.size());
            out.mean[j][k] = sum / n;
            if (samples.size() > 1)
                out.standard_error[j][k] =
                    std::sqrt(std::max(0.0, square - sum * sum / n) / (n - 1) / n);
        }
    }
    return out;
}

} // namespace detail

// One direct (SSA) path per sample, observed at `times`.
inline curve_summary ssa_committor_means(const laplacian_problem &p, idx initial,
                                         const array<real> &times, idx samples, unsigned seed) {
    const auto r = laplacian_rate_matrix(p.laplacian, p.h);
    array<curves> per_sample(samples);
#pragma omp parallel for schedule(static)
    for (idx sample = 0; sample < samples; ++sample) {
        num::rng random(seed + static_cast<unsigned>(sample));
        num::multi_index state{static_cast<int>(initial)};
        real time = 0.0;
        for (idx k = 0; k < times.size();) {
            array<num::multi_index> destinations;
            array<real> rates;
            real total = 0.0;
            r(state, [&](num::multi_index destination, real rate) {
                num::append(destinations, destination);
                num::append(rates, rate);
                total += rate;
            });
            const real next = time + std::exponential_distribution<real>(total)(random);
            for (; k < times.size() && times[k] < next; ++k)
                for (idx j = 0; j < committor_count; ++j)
                    num::append(per_sample[sample][j], p.committors[state[0]][j]);
            state = destinations[num::sample_categorical(view<const real>(rates), random)];
            time = next;
        }
    }
    return detail::reduce(per_sample, times.size());
}

// One ordered walker per sample on windows of `capacity` states around its
// current state; within a sweep the state at an observation time is drawn
// from the bridge between entry and sampled exit.
inline curve_summary ordered_committor_means(const laplacian_problem &p, idx initial,
                                             const array<real> &times, idx samples,
                                             unsigned seed, idx capacity) {
    const auto r = laplacian_rate_matrix(p.laplacian, p.h);
    const auto pi = laplacian_stationary(p.h);
    array<curves> per_sample(samples);
#pragma omp parallel for schedule(dynamic)
    for (idx sample = 0; sample < samples; ++sample) {
        num::rng random(seed + static_cast<unsigned>(sample));
        idx current = initial;
        real time = 0.0;
        for (idx k = 0; k < times.size();) {
            const array<idx> window = laplacian_neighborhood(p.laplacian, p.h, current, capacity);
            array<num::multi_index> states;
            for (idx state : window)
                num::append(states, num::multi_index{static_cast<int>(state)});
            const subnetwork sn = restriction(r, states, {}, nullptr, pi);
            const factorization f = sn.factor();
            num::vec indicator(sn.size(), 0.0);
            indicator[0] = 1.0;
            const num::vec u = f.solve_transpose(indicator), w = sn.exit_rates();
            num::vec law(sn.size(), 0.0);
            for (idx j = 0; j < sn.size(); ++j)
                law[j] = std::max(0.0, w[j] * u[j]);
            const idx exit = num::sample_categorical(view<const real>(law), random);
            num::mat symmetric = num::dense(sn.matrix());
            for (idx i = 0; i < sn.size(); ++i)
                for (idx j = 0; j < sn.size(); ++j)
                    symmetric(i, j) *= p.h[window[i]] / p.h[window[j]];
            const num::eigen_result eigen = num::eig_sym(num::assume_symmetric(symmetric));
            const real waiting = detail::sample_exit_time(eigen, 0, exit, random);
            for (; k < times.size() && times[k] < time + waiting; ++k) {
                const num::vec left = detail::semigroup_row(eigen, times[k] - time, 0);
                const num::vec right = detail::semigroup_row(eigen, time + waiting - times[k], exit);
                num::vec bridge(sn.size(), 0.0);
                for (idx s = 0; s < sn.size(); ++s)
                    bridge[s] = std::max(0.0, left[s] * right[s]);
                const idx at = window[num::sample_categorical(view<const real>(bridge), random)];
                for (idx j = 0; j < committor_count; ++j)
                    num::append(per_sample[sample][j], p.committors[at][j]);
            }
            array<idx> destinations;
            array<real> rates;
            r(states[exit], [&](const num::multi_index &destination, real rate) {
                if (!std::count(window.begin(), window.end(), idx(destination[0]))) {
                    num::append(destinations, idx(destination[0]));
                    num::append(rates, rate);
                }
            });
            current = destinations[num::sample_categorical(view<const real>(rates), random)];
            time += waiting;
        }
    }
    return detail::reduce(per_sample, times.size());
}

// The unordered chain from `initial` on a first window of `capacity` states
// nearest by stationary weight, with multinomial particle resampling, and
// its reconstructed committor means at `times`.
inline curves unordered_committor_means(const laplacian_problem &p, idx initial,
                                        const array<real> &times, int count, idx capacity,
                                        unsigned seed) {
    sweep_options options;
    options.capacity = capacity;
    options.expansion_depth = 0;
    options.regime = solve_regime::sparse;
    options.resampling = support_resampling::multinomial_particles;
    array<num::multi_index> window;
    for (idx state : laplacian_neighborhood(p.laplacian, p.h, initial, capacity))
        num::append(window, num::multi_index{static_cast<int>(state)});
    const auto chain = unordered_subsweep(laplacian_rate_matrix(p.laplacian, p.h),
                                          num::multi_index{static_cast<int>(initial)}, count,
                                          options, seed, nullptr, laplacian_stationary(p.h), window);
    curves result;
    for (real time : times) {
        const auto density = reconstruct_distribution(chain, time);
        for (idx j = 0; j < committor_count; ++j) {
            real mean = 0.0;
            for (idx k = 0; k < density.states.size(); ++k)
                mean += density.probability[k] * p.committors[density.states[k][0]][j];
            num::append(result[j], mean);
        }
    }
    return result;
}

inline void plot_curves(const array<real> &times, const curves &c, const std::string &style,
                        bool labeled) {
    for (idx j = 0; j < committor_count; ++j)
        num::plt::plot(times, c[j], labeled ? "C" + std::to_string(j + 1) : std::string{},
                       style + " lc " + std::to_string(j + 1));
}

inline void finish_panel(idx basin) {
    num::plt::title("start in basin " + std::to_string(basin + 1));
    num::plt::xlabel("time");
    num::plt::ylabel("committor expectation");
    num::plt::semilogx();
    num::plt::ylim(-0.02, 1.02);
    if (basin == 0)
        num::plt::legend("bottom left");
    if (basin + 1 < committor_count)
        num::plt::next();
}

inline real maximum_discrepancy(const curves &a, const curves &b) {
    real worst = 0.0;
    for (idx j = 0; j < committor_count; ++j)
        for (idx k = 0; k < a[j].size(); ++k)
            worst = std::max(worst, std::abs(a[j][k] - b[j][k]));
    return worst;
}

} // namespace subsweep::examples
