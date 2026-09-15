#include "else/quantities/shedding.hpp"
#include "else/restriction/restriction.hpp"
#include "markovkit/reaction_system.hpp"
#include "plot/plot.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

int main() {
    constexpr num::idx n = 40;
    constexpr double birth_rate = 12.0;
    constexpr double death_rate = 1.0;
    const markovkit::ReactionSystem model{
        .changes = {{1}, {-1}},
        .propensities = {
            [](const markovkit::State &, const auto &rates, double) { return rates[0]; },
            [](const markovkit::State &state, const auto &rates, double) {
                return rates[1] * state[0];
            },
        }};
    const num::array<double> rates{birth_rate, death_rate};

    num::array<markovkit::State> states;
    num::array<double> h(n, 1.0);
    for (num::idx j = 0; j < n; ++j) {
        states.push_back({static_cast<int>(j)});
        if (j > 0)
            h[j] = h[j - 1] * std::sqrt(birth_rate / (death_rate * j));
    }

    const auto subnetwork =
        else_sim::reversible_restriction(model, rates, states, num::view<const double>(h));
    num::vec entrance(n, 0.0);
    entrance[0] = 1.0;
    const auto shedding = else_sim::shedding_state(subnetwork, entrance);
    const double full_time =
        std::accumulate(shedding.occupation.begin(), shedding.occupation.end(), 0.0);

    const auto direct_loss = [&](num::view<const num::idx> removed) {
        num::array<bool> drop(n, false);
        for (num::idx j : removed)
            drop[j] = true;
        num::array<markovkit::State> kept;
        num::array<double> kept_h;
        num::idx entrance_row = n;
        for (num::idx j = 0; j < n; ++j) {
            if (drop[j])
                continue;
            if (j == 0)
                entrance_row = kept.size();
            kept.push_back(states[j]);
            kept_h.push_back(h[j]);
        }
        if (entrance_row == n)
            return full_time;
        const auto reduced = else_sim::reversible_restriction(model, rates, std::move(kept),
                                                              num::view<const double>(kept_h));
        num::vec reduced_entrance(else_sim::size(reduced), 0.0);
        reduced_entrance[entrance_row] = 1.0;
        const num::vec occupation = else_sim::solve_transpose(reduced, reduced_entrance);
        return full_time - std::accumulate(occupation.begin(), occupation.end(), 0.0);
    };

    num::array<double> indices, relative_errors, residuals;
    for (num::idx j = 1; j < n; ++j) {
        const num::array<num::idx> removed{j};
        const auto update = else_sim::joint_cut_time_loss(subnetwork, shedding, removed);
        const double direct = direct_loss(removed);
        indices.push_back(static_cast<double>(j));
        relative_errors.push_back(
            std::max(1e-18, std::abs(update.loss - direct) / std::max(1e-12, direct)));
        residuals.push_back(std::max(1e-18, update.backward_residual));
    }

    num::plt::plot(indices, relative_errors, "Woodbury--direct discrepancy", "lines lw 2");
    num::plt::plot(indices, residuals, "small-system backward residual", "lines dt 2 lw 2");
    num::plt::title("Cut-time update floating-point check");
    num::plt::xlabel("removed state");
    num::plt::ylabel("relative error");
    num::plt::semilogy();
    num::plt::legend();
    num::plt::savefig("else_shedding_comparison.png");

    const num::array<num::idx> block_sizes{1, 2, 4, 8, 12, 16, 20};
    num::array<double> sizes, update_times, direct_times, block_residuals;
    constexpr int repetitions = 100;
    for (num::idx count : block_sizes) {
        num::array<num::idx> removed(count);
        std::iota(removed.begin(), removed.end(), num::idx(1));
        const auto update = else_sim::joint_cut_time_loss(subnetwork, shedding, removed);
        const auto start_update = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < repetitions; ++repeat)
            (void)else_sim::joint_cut_time_loss(subnetwork, shedding, removed);
        const auto stop_update = std::chrono::steady_clock::now();
        const auto start_direct = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < repetitions; ++repeat)
            (void)direct_loss(removed);
        const auto stop_direct = std::chrono::steady_clock::now();

        sizes.push_back(static_cast<double>(count));
        update_times.push_back(
            std::chrono::duration<double, std::micro>(stop_update - start_update).count() /
            repetitions);
        direct_times.push_back(
            std::chrono::duration<double, std::micro>(stop_direct - start_direct).count() /
            repetitions);
        block_residuals.push_back(std::max(1e-18, update.backward_residual));
    }

    num::plt::plot(sizes, block_residuals, "backward residual", "linespoints lw 2");
    num::plt::title("Floating-point check for block shedding");
    num::plt::xlabel("removed block size");
    num::plt::ylabel("relative residual");
    num::plt::semilogy();
    num::plt::savefig("else_safe_add_accumulation.png");

    std::cout << std::left << std::setw(8) << "states" << std::setw(18) << "Woodbury (us)"
              << "refactor (us)\n";
    for (num::idx i = 0; i < sizes.size(); ++i)
        std::cout << std::setw(8) << sizes[i] << std::setw(18) << update_times[i] << direct_times[i]
                  << '\n';
}
