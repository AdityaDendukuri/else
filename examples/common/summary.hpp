// Ensemble means with 95% bands on a time grid, for trajectories of
// molecule counts, and the plot helpers the figures share.
#pragma once

#include "common/rate_matrix.hpp"
#include "plot/plot.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

namespace subsweep::examples {

struct summary {
    array<real> mean, lower, upper, standard_error;
};

// The mean of component `component` over trajectories [first, last) at `times`.
template <typename Trajectories>
summary summarize(const Trajectories &trajectories, const array<real> &times, idx component,
                  idx first = 0, idx last = static_cast<idx>(-1)) {
    last = std::min(last, static_cast<idx>(trajectories.size()));
    summary result{array<real>(times.size()), array<real>(times.size()),
                   array<real>(times.size()), array<real>(times.size())};
    array<real> values(last - first);
    for (idx k = 0; k < times.size(); ++k) {
        for (idx p = first; p < last; ++p) {
            const auto &path = trajectories[p];
            const auto next = std::upper_bound(path.times.begin(), path.times.end(), times[k]);
            const idx at = next == path.times.begin() ? 0 : idx(next - path.times.begin() - 1);
            values[p - first] = static_cast<real>(path.states[at][component]);
        }
        result.mean[k] = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        real centered = 0.0;
        for (real value : values)
            centered += std::pow(value - result.mean[k], 2);
        result.standard_error[k] = std::sqrt(centered / real(values.size() * (values.size() - 1)));
        result.lower[k] = result.mean[k] - 1.96 * result.standard_error[k];
        result.upper[k] = result.mean[k] + 1.96 * result.standard_error[k];
    }
    return result;
}

inline void band(const array<real> &times, const summary &s, const std::string &color) {
    num::series polygon;
    for (idx k = 0; k < times.size(); ++k)
        num::append(polygon, times[k], s.upper[k]);
    for (idx k = times.size(); k-- > 0;)
        num::append(polygon, times[k], s.lower[k]);
    num::plt::plot(polygon, std::string{},
                   "filledcurves closed fs transparent solid 0.16 noborder lc rgb '" + color + "'");
}

inline real relative_rmse(const array<real> &estimate, const array<real> &reference) {
    real error = 0.0, scale = 0.0;
    for (idx k = 0; k < estimate.size(); ++k) {
        error += std::pow(estimate[k] - reference[k], 2);
        scale += std::pow(reference[k], 2);
    }
    return std::sqrt(error / std::max(scale, 1.0));
}

// RMS of the mean differences in units of their combined standard error.
inline real standardized_rms(const summary &left, const summary &right) {
    real square_sum = 0.0;
    idx count = 0;
    for (idx k = 0; k < left.mean.size(); ++k) {
        const real se = std::hypot(left.standard_error[k], right.standard_error[k]);
        if (se > 1e-14) {
            square_sum += std::pow((left.mean[k] - right.mean[k]) / se, 2);
            ++count;
        }
    }
    return std::sqrt(square_sum / static_cast<real>(count));
}

// One phase-plane panel of (u, v) paths with the start and basin markers.
template <typename Paths>
void phase_panel(const Paths &paths, idx count, const char *label, const char *color,
                 const state &start, real u_basin, real v_basin, real plot_max, const char *title) {
    for (idx i = 0; i < count && i < paths.size(); ++i) {
        array<real> u, v;
        for (const auto &x : paths[i].states) {
            num::append(u, static_cast<real>(x[0]));
            num::append(v, static_cast<real>(x[1]));
        }
        num::plt::plot(u, v, i == 0 ? label : "", std::string("lines lw 1.2 lc rgb '") + color + "'");
    }
    num::plt::plot(array<real>{real(start[0])}, array<real>{real(start[1])}, "Start",
                   "points pt 9 ps 1.8 lc rgb '#000000'");
    num::plt::plot(array<real>{u_basin}, array<real>{0.06 * v_basin}, "U-high basin",
                   "points pt 7 ps 2.0 lc rgb '#2980b9'");
    num::plt::plot(array<real>{0.06 * u_basin}, array<real>{v_basin}, "V-high basin",
                   "points pt 7 ps 2.0 lc rgb '#d35400'");
    num::plt::title(title);
    num::plt::xlabel("U");
    num::plt::ylabel("V");
    num::plt::xlim(0.0, plot_max);
    num::plt::ylim(0.0, plot_max);
    num::plt::legend();
}

} // namespace subsweep::examples
