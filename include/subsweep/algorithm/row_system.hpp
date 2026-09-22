// A system from a rate matrix given as a row callback `r(state, visit)`,
// with `visit(destination, rate)` per transition out of `state`. States
// are their own keys; each walker records its path.
#pragma once

#include "subsweep/algorithm/system.hpp"
#include "stochastic/categorical.hpp"
#include "stochastic/rng.hpp"
#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

namespace subsweep {

template <typename R, typename State>
concept rate_matrix = std::invocable<const R &, const State &, void (*)(State, real)>;

template <typename State = num::multi_index>
struct trajectory {
    array<real> times;
    array<State> states;
};

template <typename State, rate_matrix<State> R, typename Level = std::nullptr_t,
          typename Stationary = std::nullptr_t>
class row_system {
  public:
    using state_type = State;

    row_system(R r, State initial, idx walkers, unsigned seed = 42, Level level = nullptr,
               Stationary stationary = nullptr)
        : r_(std::move(r)), level_(level), stationary_(stationary), paths_(walkers) {
        for (idx t = 0; t < walkers; ++t) {
            generators_.emplace_back(seed + static_cast<unsigned>(t));
            num::append(paths_[t], initial);
        }
        initial_ = place(initial);
    }

    [[nodiscard]] const grown_state &initial() const { return initial_; }

    // One transition of `walker` out of slot `from` whose destination lies
    // outside the slots [0, n), the subnetwork the exit law was computed on.
    [[nodiscard]] grown_state grow(idx walker, idx from, idx n) {
        array<State> destinations;
        array<real> weights;
        r_(states_[from], [&](State destination, real rate) {
            const auto found = slot_of_.find(destination);
            if (found == slot_of_.end() || found->second >= n) {
                num::append(destinations, std::move(destination));
                num::append(weights, rate);
            }
        });
        if (destinations.empty())
            throw std::runtime_error("grow was asked for an exit of a state that has none");
        const idx choice = num::sample_categorical(view<const real>(weights), generators_[walker]);
        num::append(paths_[walker], destinations[choice]);
        return place(std::move(destinations[choice]));
    }

    // Every exterior neighbor of `from`, each placed and passed to `visit`.
    template <typename Visit> void expand(idx from, Visit &&visit) {
        array<State> destinations;
        r_(states_[from], [&](State destination, real) {
            if (!slot_of_.contains(destination))
                num::append(destinations, std::move(destination));
        });
        for (State &destination : destinations)
            if (!slot_of_.contains(destination))
                visit(place(std::move(destination)));
    }

    void discard(idx j) {
        const idx last = states_.size() - 1;
        slot_of_.erase(states_[j]);
        if (j != last) {
            states_[j] = std::move(states_[last]);
            slot_of_[states_[j]] = j;
        }
        states_.pop_back();
    }

    [[nodiscard]] const State &state(idx slot) const { return states_[slot]; }
    [[nodiscard]] idx size() const { return states_.size(); }
    [[nodiscard]] const array<array<State>> &paths() const { return paths_; }

    // The recorded paths with the loop's jump times, closed at `final_time`.
    [[nodiscard]] array<trajectory<State>>
    trajectories(real initial_time, const array<array<real>> &jumps, real final_time) const {
        array<trajectory<State>> result(paths_.size());
        for (idx t = 0; t < paths_.size(); ++t) {
            num::append(result[t].times, initial_time);
            for (real time : jumps[t])
                num::append(result[t].times, time);
            result[t].states = paths_[t];
            if (final_time < std::numeric_limits<real>::infinity() &&
                result[t].times.back() < final_time) {
                num::append(result[t].times, final_time);
                num::append(result[t].states, result[t].states.back());
            }
        }
        return result;
    }

  private:
    [[nodiscard]] grown_state place(State state) {
        grown_state g;
        if (const auto found = slot_of_.find(state); found != slot_of_.end()) {
            g.slot = found->second;
            return g;
        }
        g.slot = states_.size();
        g.fresh = true;
        r_(state, [&](State destination, real rate) {
            g.total_rate += rate;
            if (const auto found = slot_of_.find(destination); found != slot_of_.end())
                num::append(g.row, found->second, rate);
        });
        for (idx slot = 0; slot < states_.size(); ++slot)
            r_(states_[slot], [&](const State &destination, real rate) {
                if (destination == state)
                    num::append(g.column, slot, rate);
            });
        if constexpr (!std::is_same_v<Level, std::nullptr_t>)
            g.level = static_cast<idx>(level_(state));
        if constexpr (!std::is_same_v<Stationary, std::nullptr_t>)
            g.stationary = stationary_(state);
        slot_of_.emplace(state, g.slot);
        num::append(states_, std::move(state));
        return g;
    }

    R r_;
    Level level_;
    Stationary stationary_;
    array<num::rng> generators_;
    array<State> states_;
    table<State, idx> slot_of_;
    array<array<State>> paths_;
    grown_state initial_;
};

template <typename State, typename R, typename Level = std::nullptr_t,
          typename Stationary = std::nullptr_t>
[[nodiscard]] auto rows_system(R r, State initial, idx walkers, unsigned seed = 42,
                               Level level = nullptr, Stationary stationary = nullptr) {
    return row_system<State, R, Level, Stationary>(std::move(r), std::move(initial), walkers, seed,
                                                   level, stationary);
}

// A subnetwork on a fixed list of states, filled in one pass over their rows.
template <typename State, rate_matrix<State> R, typename Level = std::nullptr_t,
          typename Stationary = std::nullptr_t>
[[nodiscard]] subnetwork restriction(const R &r, const array<State> &states,
                                     const sweep_options &options, Level level = nullptr,
                                     Stationary stationary = nullptr) {
    table<State, idx> position;
    for (idx i = 0; i < states.size(); ++i)
        position.emplace(states[i], i);
    array<real> total(states.size(), 0.0);
    array<array<slot_rate>> row(states.size()), column(states.size());
    for (idx i = 0; i < states.size(); ++i)
        r(states[i], [&](const State &destination, real rate) {
            total[i] += rate;
            const auto found = position.find(destination);
            if (found == position.end())
                return;
            if (found->second < i)
                num::append(row[i], found->second, rate);
            else if (found->second > i)
                num::append(column[found->second], i, rate);
        });
    subnetwork sn(options);
    for (idx i = 0; i < states.size(); ++i) {
        real pi = 0.0;
        idx lv = 0;
        if constexpr (!std::is_same_v<Level, std::nullptr_t>)
            lv = static_cast<idx>(level(states[i]));
        if constexpr (!std::is_same_v<Stationary, std::nullptr_t>)
            pi = stationary(states[i]);
        sn.add(total[i], row[i], column[i], lv, pi);
    }
    return sn;
}

} // namespace subsweep
