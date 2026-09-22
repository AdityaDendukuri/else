// The subnetwork: R_bar on the slots [0, n), grown one state at a time and
// shed by swap-remove, with the first-exit quantities on it and factor
// reuse across sweeps. It knows nothing about states; the caller maps its
// own states to slots.
#pragma once

#include "subsweep/sweep/reuse.hpp"
#include "subsweep/sweep/shedding.hpp"
#include "subsweep/sweep/solution.hpp"
#include "subsweep/types.hpp"
#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <utility>

namespace subsweep {

struct slot_rate {
    idx slot;
    real rate;
};

// R_bar maintained from appended slots, with the retained factorization and
// what its reuse needs. `generation[j]` changes when slot j is refilled.
class subnetwork {
  public:
    explicit subnetwork(sweep_options options = {}) : options_(std::move(options)) {}

    [[nodiscard]] idx size() const { return total_rate_.size(); }
    [[nodiscard]] real total_rate(idx slot) const { return total_rate_[slot]; }
    [[nodiscard]] const sweep_options &options() const { return options_; }
    [[nodiscard]] view<const idx> levels() const { return level_; }

    // Append slot n from its total rate, its rates into existing slots, the
    // existing slots' rates into it, and its level and stationary weight.
    idx add(real total, view<const slot_rate> row, view<const slot_rate> column, idx level = 0,
            real stationary = 0.0) {
        const idx slot = size();
        num::append(total_rate_, total);
        num::append(internal_, table<idx, real>{});
        num::append(level_, level);
        num::append(stationary_, stationary);
        num::append(generation_, next_generation_++);
        for (const slot_rate &entry : row)
            internal_[slot][entry.slot] += entry.rate;
        for (const slot_rate &entry : column)
            internal_[entry.slot][slot] += entry.rate;
        reversible_ = reversible_ || stationary > 0.0;
        return slot;
    }

    // Remove slot j: the last slot moves into j.
    void discard(idx j) {
        const idx last = size() - 1;
        for (auto &row : internal_)
            row.erase(j);
        if (j != last) {
            total_rate_[j] = total_rate_[last];
            internal_[j] = std::move(internal_[last]);
            level_[j] = level_[last];
            stationary_[j] = stationary_[last];
            generation_[j] = generation_[last];
            for (auto &row : internal_)
                if (const auto found = row.find(last); found != row.end()) {
                    row[j] = found->second;
                    row.erase(found);
                }
        }
        total_rate_.pop_back();
        internal_.pop_back();
        level_.pop_back();
        stationary_.pop_back();
        generation_.pop_back();
    }

    // R_bar in the M-matrix convention.
    [[nodiscard]] num::spmat matrix() const {
        const idx n = size();
        array<idx> rows, columns;
        array<real> values;
        for (idx i = 0; i < n; ++i) {
            for (const auto &[j, rate] : internal_[i]) {
                num::append(rows, i);
                num::append(columns, j);
                num::append(values, -rate);
            }
            num::append(rows, i);
            num::append(columns, i);
            num::append(values, total_rate_[i]);
        }
        return num::spmat::from_triplets(n, n, rows, columns, values);
    }

    // w = R_bar 1, with round-off below the total rate taken as zero.
    [[nodiscard]] num::vec exit_rates() const {
        num::vec w;
        exit_rates(w);
        return w;
    }

    void exit_rates(num::vec &w) const {
        detail::shape(w, size());
        for (idx i = 0; i < size(); ++i) {
            real retained = 0.0;
            for (const auto &[j, rate] : internal_[i])
                retained += rate;
            const real exit = total_rate_[i] - retained;
            w[i] = exit > 1e-12 * total_rate_[i] ? exit : 0.0;
        }
    }

    [[nodiscard]] view<const real> stationary() const {
        return reversible_ ? view<const real>(stationary_) : view<const real>{};
    }

    [[nodiscard]] factorization factor() const {
        return factorization(matrix(), level_, stationary(), options_);
    }

    // The first-exit quantities for the current slots with multiplicities
    // `counts`, reusing the retained factorization by the regime's policy:
    // block refactors from the first changed level and updates cached rows
    // and diagonal by Woodbury; dense keeps its factor and corrects the
    // solves by Woodbury while few slots changed; sparse refactors.
    [[nodiscard]] sweep_solution solve(view<const idx> current, view<const idx> counts) {
        sweep_solution out;
        solve(current, counts, out);
        return out;
    }

    [[nodiscard]] sweep_solution solve(view<const idx> current, num::vec rho) {
        sweep_solution out;
        out.rho = std::move(rho);
        solve(current, out);
        return out;
    }

    // Into a solution kept across sweeps: nothing is allocated once its
    // buffers and the workspace have the sizes of the sweep.
    void solve(view<const idx> current, view<const idx> counts, sweep_solution &out) {
        idx total = 0;
        for (idx count : counts)
            total += count;
        detail::shape(out.rho, size());
        std::fill(out.rho.begin(), out.rho.end(), 0.0);
        for (idx k = 0; k < current.size(); ++k)
            out.rho[current[k]] = static_cast<real>(counts[k]) / static_cast<real>(total);
        solve(current, out);
    }

    void solve(view<const idx> current, sweep_solution &out) {
        using clock = std::chrono::steady_clock;
        const auto seconds = [](clock::time_point start) {
            return std::chrono::duration<double>(clock::now() - start).count();
        };
        out.current.assign(current.begin(), current.end());
        out.diagnostics = sweep_diagnostics{};
        out.diagnostics.subnetwork_size = size();
        out.diagnostics.distinct_current_states = current.size();

        const bool reusable =
            options_.reuse_factors && base_ && base_->generation.size() == size();
        const array<idx> changed = reusable ? changed_slots(*base_, generation_) : array<idx>{};
        out.diagnostics.changed_state_slots = changed.size();

        const auto matrix_start = clock::now();
        const num::spmat R = matrix();
        exit_rates(out.exit_rates);
        out.diagnostics.restriction_and_factor_seconds = seconds(matrix_start);

        const bool cut_time = options_.rule == shedding_rule::cut_time;
        detail::shape(work_.ones, size());
        std::fill(work_.ones.begin(), work_.ones.end(), 1.0);
        const num::vec &ones = work_.ones;
        const auto rows_start = clock::now();
        std::optional<factorization> keep;
        bool accepted = false;
        double fresh_seconds = 0.0;

        const auto solve_all = [&](const factorization &solver) {
            current_rows(solver, current, out.occupation, out.second_occupation, work_);
            if (cut_time) {
                if (!out.exit_time)
                    out.exit_time.emplace();
                solver.solve(ones, *out.exit_time);
                out.diagonal = solver.diagonal();
            }
        };
        const auto fresh = [&] {
            const auto start = clock::now();
            keep.emplace(factorization(R, level_, stationary(), options_));
            fresh_seconds = seconds(start);
            solve_all(*keep);
        };

        if (reusable && changed.empty()) {
            out.diagnostics.reuse_attempted = accepted = true;
            current_rows(base_->factor, current, out.occupation, out.second_occupation, work_);
            if (cut_time) {
                if (!out.exit_time)
                    out.exit_time.emplace();
                base_->factor.solve(ones, *out.exit_time);
                out.diagonal = base_->diagonal ? base_->diagonal : base_->factor.diagonal();
            }
        } else if (!reusable) {
            fresh();
        } else if (base_->factor.kind() == solve_regime::block) {
            out.diagnostics.reuse_attempted = true;
            const auto update_start = clock::now();
            block_suffix_info info;
            std::optional<factorization> updated =
                base_->factor.update_suffix(R, level_, stationary(), changed, &info);
            out.diagnostics.factor_update_seconds = seconds(update_start);
            out.diagnostics.block_count = info.block_count;
            out.diagnostics.reused_prefix_blocks = info.reused_prefix_blocks;
            out.diagnostics.reused_prefix_states = info.reused_prefix_states;
            if (!updated) {
                fresh();
            } else {
                accepted = true;
                const bool diagonal_by_woodbury = cut_time && base_->diagonal.has_value();
                const bool rows_by_woodbury = row_reuse_is_profitable(
                    *base_, generation_, current, changed.size(), diagonal_by_woodbury);
                std::optional<woodbury> wb;
                if (diagonal_by_woodbury || rows_by_woodbury)
                    wb.emplace(base_->factor, row_column_delta(base_->matrix, R, changed));
                bool reused = false;
                if (rows_by_woodbury)
                    reused = reuse_cached_rows(*base_, generation_, *wb, current, *updated, R,
                                               options_.reuse_residual, work_, out);
                if (!reused)
                    current_rows(*updated, current, out.occupation, out.second_occupation, work_);
                if (cut_time) {
                    out.exit_time = wb ? wb->solve(ones) : updated->solve(ones);
                    out.diagonal = diagonal_by_woodbury ? wb->diagonal(base_->diagonal->span())
                                                        : updated->diagonal();
                }
                keep = std::move(updated);
            }
        } else if (base_->factor.kind() == solve_regime::dense) {
            if (changed.size() <= options_.woodbury_cutoff)
                try {
                    out.diagnostics.reuse_attempted = true;
                    const woodbury wb(base_->factor, row_column_delta(base_->matrix, R, changed));
                    num::mat u, v;
                    current_rows(wb, current, u, v);
                    num::vec &indicator = work_.indicator, &first_row = work_.row;
                    detail::shape(indicator, size());
                    detail::shape(first_row, size());
                    std::fill(indicator.begin(), indicator.end(), 0.0);
                    indicator[current.front()] = 1.0;
                    std::copy_n(u.data(), size(), first_row.data());
                    if (relative_residual(num::transpose(R), first_row, indicator) <=
                        options_.reuse_residual) {
                        accepted = true;
                        out.occupation = std::move(u);
                        out.second_occupation = std::move(v);
                        if (cut_time) {
                            out.exit_time = wb.solve(ones);
                            out.diagonal = base_->diagonal ? wb.diagonal(base_->diagonal->span())
                                                           : wb.diagonal(base_->factor.diagonal().span());
                        }
                    }
                } catch (const std::runtime_error &) {
                    accepted = false;
                }
            if (!accepted)
                fresh();
        } else {
            fresh();
        }
        out.diagnostics.reuse_accepted = accepted;
        out.diagnostics.fresh_factorization = !accepted;
        out.diagnostics.restriction_and_factor_seconds += fresh_seconds;
        out.diagnostics.current_rows_seconds =
            seconds(rows_start) - out.diagnostics.factor_update_seconds - fresh_seconds;

        // A fresh or exactly updated factorization becomes the base; the block
        // regime also keeps the current-state rows for the cached update.
        if (options_.reuse_factors) {
            if (keep) {
                std::optional<cached_rows> rows = base_ ? std::move(base_->rows) : std::nullopt;
                base_.emplace(reuse_base{R, generation_, std::move(*keep), std::move(rows),
                                         out.diagonal});
                if (base_->factor.kind() == solve_regime::block)
                    cache_rows(*base_, out);
                else
                    base_->rows.reset();
            } else if (base_ && accepted && changed.empty() &&
                       base_->factor.kind() == solve_regime::block) {
                cache_rows(*base_, out);
            }
        }
    }

    // Shedding scores under the options' rule from a solution of this sweep.
    [[nodiscard]] num::vec scores(const sweep_solution &solution) const {
        return subsweep::scores(options_.rule, total_rate_, solution);
    }

    // Scores for a distribution, from one fresh factorization.
    [[nodiscard]] num::vec scores(num::vec rho) const {
        const factorization f = factor();
        sweep_solution solution;
        solution.rho = std::move(rho);
        solution.exit_rates = exit_rates();
        array<idx> support;
        for (idx j = 0; j < size(); ++j)
            if (solution.rho[j] != 0.0)
                num::append(support, j);
        solution.current = support;
        num::mat indicators(size(), support.size(), 0.0);
        for (idx k = 0; k < support.size(); ++k)
            indicators(support[k], k) = 1.0;
        solution.occupation = num::transpose(f.solve_transpose(indicators));
        if (options_.rule == shedding_rule::cut_time) {
            solution.exit_time = f.solve(num::vec(size(), 1.0));
            solution.diagonal = f.diagonal();
        }
        return scores(solution);
    }

  private:
    sweep_options options_;
    sweep_workspace work_;
    array<real> total_rate_;
    array<table<idx, real>> internal_;
    array<idx> level_;
    array<real> stationary_;
    array<idx> generation_;
    idx next_generation_ = 1;
    bool reversible_ = false;
    std::optional<reuse_base> base_;
};

} // namespace subsweep
