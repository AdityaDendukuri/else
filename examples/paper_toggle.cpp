#include "else/algorithms/ensemble.hpp"
#include "markovkit/reaction_system.hpp"
#include "ssa/ssa.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr num::idx paths_per_batch = 100;
constexpr num::idx batches = 5;
constexpr num::idx paths = paths_per_batch * batches;
constexpr num::idx maximum_steps = 200000;
constexpr num::idx samples = 251;
constexpr double final_time = 500.0;

struct ToggleSystem {
    markovkit::ReactionSystem model;
    num::array<double> rates;
    markovkit::State initial;
};

ToggleSystem make_toggle(double scale, markovkit::State initial) {
    const double alpha = 20.0 * scale;
    const double beta = 400.0 * scale;
    const double K = 100.0 * scale;
    const double K3 = K * K * K;
    ToggleSystem system;
    system.rates = {alpha, beta, K3, 1.0 + (0.1 / 1.1), alpha, beta, K3, 1.0};
    system.initial = std::move(initial);
    system.model.changes = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    system.model.propensities = {
        [](const markovkit::State &x, const auto &r, double) {
            const double v3 = static_cast<double>(x[1]) * x[1] * x[1];
            return r[0] + (r[1] * r[2] / (r[2] + v3));
        },
        [](const markovkit::State &x, const auto &r, double) { return r[3] * x[0]; },
        [](const markovkit::State &x, const auto &r, double) {
            const double u3 = static_cast<double>(x[0]) * x[0] * x[0];
            return r[4] + (r[5] * r[6] / (r[6] + u3));
        },
        [](const markovkit::State &x, const auto &r, double) { return r[7] * x[1]; },
    };
    return system;
}

template <typename Trajectories>
num::array<double> v_basin_probability(const Trajectories &trajectories,
                                       const num::array<double> &times) {
    num::array<double> result(times.size(), 0.0);
    for (num::idx k = 0; k < times.size(); ++k) {
        for (const auto &path : trajectories) {
            const auto next = std::upper_bound(path.times.begin(), path.times.end(), times[k]);
            const num::idx state = next == path.times.begin()
                                       ? 0
                                       : static_cast<num::idx>(next - path.times.begin() - 1);
            result[k] += path.states[state][1] > path.states[state][0] ? 1.0 : 0.0;
        }
        result[k] /= static_cast<double>(trajectories.size());
    }
    return result;
}

void confidence_band(const num::array<double> &times, const num::array<double> &probability,
                     const std::string &color) {
    num::series polygon;
    polygon.reserve(2 * times.size());
    for (num::idx k = 0; k < times.size(); ++k) {
        const double half = 1.96 * std::sqrt(probability[k] * (1.0 - probability[k]) / paths);
        polygon.emplace_back(times[k], std::min(1.0, probability[k] + half));
    }
    for (num::idx k = times.size(); k-- > 0;) {
        const double half = 1.96 * std::sqrt(probability[k] * (1.0 - probability[k]) / paths);
        polygon.emplace_back(times[k], std::max(0.0, probability[k] - half));
    }
    num::plt::plot(polygon, std::string{},
                   "filledcurves closed fs transparent solid 0.13 noborder lc rgb '" + color + "'");
}

struct Result {
    num::array<double> ib_probability;
    num::array<double> ssa_probability;
    else_sim::EnsembleDiagnostics diagnostics;
};

Result run(const ToggleSystem &system, num::idx capacity, unsigned seed,
           const num::array<double> &times) {
    const auto level = [](const markovkit::State &x) { return x[0]; };
    Result result;
    result.ib_probability.assign(times.size(), 0.0);
    result.ssa_probability.assign(times.size(), 0.0);
    for (num::idx batch = 0; batch < batches; ++batch) {
        else_sim::EnsembleDiagnostics batch_diagnostics;
        const unsigned batch_seed = seed + static_cast<unsigned>(10000 * batch);
        const auto ib = else_sim::else_ensemble(system.model, system.rates, system.initial,
                                                paths_per_batch, 0.0, final_time,
                                                {.capacity = capacity,
                                                 .expansion_depth = 1,
                                                 .tolerance = 1e-12,
                                                 .maximum_steps = maximum_steps},
                                                batch_seed, level, &batch_diagnostics);
        num::array<markovkit::Trajectory> direct(paths_per_batch);
#pragma omp parallel for schedule(dynamic)
        for (int p = 0; p < static_cast<int>(paths_per_batch); ++p)
            direct[static_cast<num::idx>(p)] =
                ssa::gillespie(system.model, system.rates, system.initial, 0.0, final_time,
                               batch_seed + static_cast<unsigned>(p), maximum_steps);
        const auto ib_probability = v_basin_probability(ib, times);
        const auto ssa_probability = v_basin_probability(direct, times);
        for (num::idx k = 0; k < times.size(); ++k) {
            result.ib_probability[k] += ib_probability[k] / static_cast<double>(batches);
            result.ssa_probability[k] += ssa_probability[k] / static_cast<double>(batches);
        }
        result.diagnostics.steps.insert(result.diagnostics.steps.end(),
                                        batch_diagnostics.steps.begin(),
                                        batch_diagnostics.steps.end());
    }
    return result;
}

double rmse(const num::array<double> &a, const num::array<double> &b) {
    double value = 0.0;
    for (num::idx i = 0; i < a.size(); ++i)
        value += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(value / static_cast<double>(a.size()));
}

void probability_panel(const num::array<double> &times, const Result &result,
                       const std::string &title, double upper_limit) {
    confidence_band(times, result.ib_probability, "#1f77b4");
    confidence_band(times, result.ssa_probability, "#ff7f0e");
    num::plt::plot(times, result.ib_probability, "IB", "lines lw 2 lc rgb '#1f77b4'");
    num::plt::plot(times, result.ssa_probability, "SSA", "lines lw 2 lc rgb '#ff7f0e'");
    num::plt::title(
        title + ": RMSE = " + std::to_string(rmse(result.ib_probability, result.ssa_probability)));
    num::plt::xlabel("physical time");
    num::plt::ylabel("Pr(V > U)");
    num::plt::xlim(0.0, final_time);
    num::plt::ylim(0.0, upper_limit);
    num::plt::legend();
}

void diagnostics_panel(const else_sim::EnsembleDiagnostics &diagnostics) {
    num::array<double> update;
    num::array<double> distinct_fraction;
    const num::idx stride = std::max<num::idx>(1, diagnostics.steps.size() / 1000);
    for (num::idx k = 0; k < diagnostics.steps.size(); k += stride) {
        const auto &step = diagnostics.steps[k];
        update.push_back(static_cast<double>(k));
        distinct_fraction.push_back(static_cast<double>(step.distinct_entrances) /
                                    std::max<double>(1.0, step.active_trajectories));
    }
    num::plt::plot(update, distinct_fraction, "distinct / active", "lines lw 1.5 lc rgb '#2ca02c'");
    num::plt::title("Sharing within the low-barrier ensemble");
    num::plt::xlabel("IB update");
    num::plt::ylabel("distinct entrance fraction");
    num::plt::ylim(0.0, 1.0);
    num::plt::legend();
}

} // namespace

int main() {
    num::array<double> times(samples);
    for (num::idx k = 0; k < samples; ++k)
        times[k] = final_time * static_cast<double>(k) / static_cast<double>(samples - 1);

    const Result high = run(make_toggle(0.20, {17, 1}), 220, 41, times);
    const Result low = run(make_toggle(0.08, {7, 1}), 180, 142, times);

    num::plt::subplot(1, 3);
    probability_panel(times, high, "High barrier", 0.08);
    num::plt::next();
    probability_panel(times, low, "Low barrier", 0.75);
    num::plt::next();
    diagnostics_panel(low.diagnostics);
    num::plt::savefig("toggle_validation.png");

    const auto report = [](const char *name, const Result &result) {
        num::idx attempts = 0;
        num::idx accepted = 0;
        double mean_distinct_fraction = 0.0;
        for (const auto &step : result.diagnostics.steps) {
            attempts += step.reuse_attempted ? 1 : 0;
            accepted += step.reuse_accepted ? 1 : 0;
            mean_distinct_fraction += static_cast<double>(step.distinct_entrances) /
                                      std::max<double>(1.0, step.active_trajectories);
        }
        mean_distinct_fraction /= std::max<num::idx>(1, result.diagnostics.steps.size());
        std::cout << name << ": updates=" << result.diagnostics.steps.size()
                  << ", probability RMSE=" << rmse(result.ib_probability, result.ssa_probability)
                  << ", mean distinct fraction=" << mean_distinct_fraction << ", reuse=" << accepted
                  << "/" << attempts << "\n";
    };
    report("high barrier", high);
    report("low barrier", low);
}
