#include "else/algorithms/ensemble.hpp"
#include "markovkit/reaction_system.hpp"
#include "ssa/ssa.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr num::idx paths = 100;
constexpr num::idx maximum_steps = 600000;
constexpr num::idx capacity = 120;
constexpr double final_time = 1.50;
constexpr num::idx samples = 601;

struct Summary {
    num::array<double> mean;
    num::array<double> lower;
    num::array<double> upper;
};

template <typename Trajectories>
Summary summarize(const Trajectories &trajectories, const num::array<double> &times,
                  num::idx component) {
    Summary result{num::array<double>(times.size()), num::array<double>(times.size()),
                   num::array<double>(times.size())};
    num::array<double> values(trajectories.size());
    for (num::idx k = 0; k < times.size(); ++k) {
        for (num::idx p = 0; p < trajectories.size(); ++p) {
            const auto &path = trajectories[p];
            const auto next = std::upper_bound(path.times.begin(), path.times.end(), times[k]);
            const num::idx state = next == path.times.begin()
                                       ? 0
                                       : static_cast<num::idx>(next - path.times.begin() - 1);
            values[p] = static_cast<double>(path.states[state][component]);
        }
        std::sort(values.begin(), values.end());
        result.mean[k] = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        result.lower[k] = values[static_cast<num::idx>(0.10 * (values.size() - 1))];
        result.upper[k] = values[static_cast<num::idx>(0.90 * (values.size() - 1))];
    }
    return result;
}

void band(const num::array<double> &times, const Summary &summary, const std::string &color) {
    num::series polygon;
    polygon.reserve(2 * times.size());
    for (num::idx k = 0; k < times.size(); ++k)
        polygon.emplace_back(times[k], summary.upper[k]);
    for (num::idx k = times.size(); k-- > 0;)
        polygon.emplace_back(times[k], summary.lower[k]);
    num::plt::plot(polygon, std::string{},
                   "filledcurves closed fs transparent solid 0.16 noborder lc rgb '" + color + "'");
}

double relative_rmse(const num::array<double> &estimate, const num::array<double> &reference) {
    double error = 0.0;
    double scale = 0.0;
    for (num::idx k = 0; k < estimate.size(); ++k) {
        error += std::pow(estimate[k] - reference[k], 2);
        scale += std::pow(reference[k], 2);
    }
    return std::sqrt(error / std::max(scale, 1.0));
}

} // namespace

int main() {
    const double y1 = 500.0;
    const double y2 = 1000.0;
    const double y3 = 2000.0;
    const double mu1 = 2000.0;
    const double mu2 = 50000.0;
    const num::array<double> rates = {mu1 / y2, mu2 / (y1 * y2), (mu1 + mu2) / y1,
                                      2.0 * mu1 / (y1 * y1), (mu1 + mu2) / y3};
    markovkit::ReactionSystem model{
        .changes = {{1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-2, 0, 0}, {0, 1, -1}},
        .propensities = {
            [](const markovkit::State &x, const auto &r, double) { return r[0] * x[1]; },
            [](const markovkit::State &x, const auto &r, double) { return r[1] * x[0] * x[1]; },
            [](const markovkit::State &x, const auto &r, double) { return r[2] * x[0]; },
            [](const markovkit::State &x, const auto &r, double) {
                return x[0] < 2 ? 0.0 : 0.5 * r[3] * x[0] * (x[0] - 1);
            },
            [](const markovkit::State &x, const auto &r, double) { return r[4] * x[2]; },
        }};
    const markovkit::State initial{500, 1000, 2000};
    const auto level = [](const markovkit::State &x) { return (x[0] + x[1] + x[2]) / 2; };

    else_sim::EnsembleDiagnostics diagnostics;
    const auto ib = else_sim::else_ensemble(model, rates, initial, paths, 0.0, final_time,
                                            {.capacity = capacity, .maximum_steps = maximum_steps},
                                            42, level, &diagnostics);
    num::array<markovkit::Trajectory> ssa(paths);
#pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < static_cast<int>(paths); ++p)
        ssa[static_cast<num::idx>(p)] =
            ssa::gillespie(model, rates, initial, 0.0, final_time, 42 + p, maximum_steps);

    const auto reaches_horizon = [](const auto &path) {
        return !path.times.empty() && path.times.back() >= final_time;
    };
    const auto ib_complete =
        static_cast<num::idx>(std::count_if(ib.begin(), ib.end(), reaches_horizon));
    const auto ssa_complete =
        static_cast<num::idx>(std::count_if(ssa.begin(), ssa.end(), reaches_horizon));
    if (ib_complete != paths || ssa_complete != paths) {
        std::cerr << "Incomplete matched-horizon ensemble: subsweep=" << ib_complete << "/" << paths
                  << ", SSA=" << ssa_complete << "/" << paths << '\n';
        return 1;
    }

    num::idx ib_boundary_updates = 0;
    for (const auto &path : ib) {
        for (num::idx k = 1; k < path.states.size(); ++k)
            ib_boundary_updates += path.states[k] != path.states[k - 1] ? 1 : 0;
    }
    const num::idx ssa_reaction_events = std::accumulate(
        ssa.begin(), ssa.end(), num::idx{0}, [](num::idx total, const markovkit::Trajectory &path) {
            return total + path.reactions.size();
        });

    num::array<double> times(samples);
    for (num::idx k = 0; k < samples; ++k)
        times[k] = final_time * static_cast<double>(k) / static_cast<double>(samples - 1);

    constexpr std::array<const char *, 3> labels = {"X", "Y", "Z"};
    num::plt::subplot(1, 3);
    for (num::idx component = 0; component < 3; ++component) {
        const Summary ib_summary = summarize(ib, times, component);
        const Summary ssa_summary = summarize(ssa, times, component);
        band(times, ib_summary, "#1f77b4");
        band(times, ssa_summary, "#ff7f0e");
        num::plt::plot(times, ib_summary.mean, "Labeled subsweep mean",
                       "lines lw 2 lc rgb '#1f77b4'");
        num::plt::plot(times, ssa_summary.mean, "SSA mean", "lines lw 2 lc rgb '#ff7f0e'");
        num::plt::title(std::string(labels[component]) + ": rel. RMSE = " +
                        std::to_string(relative_rmse(ib_summary.mean, ssa_summary.mean)));
        num::plt::xlabel("physical time");
        num::plt::ylabel("molecule count");
        num::plt::xlim(0.0, final_time);
        num::plt::legend();
        if (component + 1 < 3)
            num::plt::next();
    }
    num::plt::savefig("oregonator_ensemble_validation.png");

    num::idx reused = 0;
    for (const auto &step : diagnostics.steps)
        reused += step.reuse_accepted ? 1 : 0;
    std::cout << "Oregonator: trajectories=" << paths << ", horizon=" << final_time
              << ", capacity=" << capacity << ", macrosteps=" << diagnostics.steps.size()
              << ", reused=" << reused << ", completed=" << ib_complete << "/" << paths
              << " subsweep and " << ssa_complete << "/" << paths << " SSA"
              << ", subsweep boundary updates=" << ib_boundary_updates
              << ", SSA reaction events=" << ssa_reaction_events << ", event-reduction="
              << static_cast<double>(ssa_reaction_events) / static_cast<double>(ib_boundary_updates)
              << "x\n";
}
