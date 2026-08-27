#pragma once

#include "else/linalg.hpp"
#include "else/subnetwork.hpp"
#include "else/types.hpp"
#include <algorithm>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace else_sim {

template <typename Mat> inline std::size_t matrix_rows_helper(const Mat &m) {
  if constexpr (requires { m.rows; })
    return m.rows;
  else
    return m.n_rows();
}

/// @brief Connected neighborhood grown in descending stationary weight order.
template <typename SparseMat, typename Float = double,
          typename Index1 = std::size_t, typename Index2 = std::size_t>
inline std::vector<std::size_t>
laplacian_neighborhood(const SparseMat &laplacian,
                       const std::vector<Float> &stationary_sqrt,
                       Index1 initial, Index2 capacity) {
  const std::size_t n_rows = matrix_rows_helper(laplacian);
  const std::size_t init_idx = static_cast<std::size_t>(initial);
  const std::size_t cap = static_cast<std::size_t>(capacity);
  if (init_idx >= n_rows || cap == 0 || cap > n_rows) {
    throw std::invalid_argument("invalid Laplacian neighborhood request");
  }

  std::vector<bool> selected(n_rows, false);
  std::vector<bool> in_frontier(n_rows, false);
  std::priority_queue<std::pair<Float, std::size_t>> frontier;
  std::vector<std::size_t> states;
  states.reserve(cap);

  const auto expose = [&](std::size_t state) {
    const auto row_begin = laplacian.row_ptr()[state];
    const auto row_end = laplacian.row_ptr()[state + 1];
    for (auto k = row_begin; k < row_end; ++k) {
      const std::size_t neighbor =
          static_cast<std::size_t>(laplacian.col_idx()[k]);
      if (neighbor != state && !selected[neighbor] && !in_frontier[neighbor]) {
        in_frontier[neighbor] = true;
        frontier.push({stationary_sqrt[neighbor], neighbor});
      }
    }
  };

  selected[init_idx] = true;
  states.push_back(init_idx);
  expose(init_idx);

  while (states.size() < cap) {
    if (frontier.empty()) {
      break;
    }
    const std::size_t state = frontier.top().second;
    frontier.pop();
    if (!selected[state]) {
      selected[state] = true;
      states.push_back(state);
      expose(state);
    }
  }
  return states;
}

/// Restrict a symmetric Laplacian to the selected global state indices.
template <typename SparseMat, typename Float = double,
          typename State = std::vector<int>>
inline Subnetwork<Float, std::size_t, State>
laplacian_subnetwork(const SparseMat &laplacian,
                     const std::vector<Float> &stationary_sqrt,
                     const std::vector<std::size_t> &states) {
  const std::size_t n = states.size();
  std::unordered_map<std::size_t, std::size_t> local_index;
  std::vector<State> local_states;
  std::vector<Float> local_weights(n);
  local_states.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    local_index[states[i]] = i;
    local_states.push_back(State{static_cast<int>(states[i])});
    local_weights[i] = stationary_sqrt[states[i]];
  }

  std::vector<std::size_t> rows, columns;
  std::vector<Float> values;
  std::vector<BoundaryTransition<std::size_t, State, Float>> boundary;
  for (std::size_t column = 0; column < n; ++column) {
    const std::size_t global_column = states[column];
    for (auto entry = laplacian.row_ptr()[global_column];
         entry < laplacian.row_ptr()[global_column + 1]; ++entry) {
      const std::size_t global_row =
          static_cast<std::size_t>(laplacian.col_idx()[entry]);
      if (global_row == global_column) {
        rows.push_back(column);
        columns.push_back(column);
        values.push_back(-static_cast<Float>(laplacian.values()[entry]));
        continue;
      }

      const Float rate = -static_cast<Float>(laplacian.values()[entry]) *
                         stationary_sqrt[global_row] /
                         stationary_sqrt[global_column];
      if (rate <= static_cast<Float>(0)) continue;
      auto found = local_index.find(global_row);
      if (found == local_index.end()) {
        boundary.push_back({column, State{static_cast<int>(global_row)}, rate});
      } else {
        rows.push_back(found->second);
        columns.push_back(column);
        values.push_back(rate);
      }
    }
  }

  auto generator = SparseMatrix<Float, std::size_t>::from_triplets(
      n, n, rows, columns, values);
  return {std::move(local_states), std::move(generator), std::move(boundary),
          std::move(local_weights)};
}

/// @brief Build successive restrictions of a symmetric Laplacian.
template <typename SparseMat, typename Float = double,
          typename Index1 = std::size_t, typename Index2 = std::size_t,
          typename State = std::vector<int>>
inline std::vector<Subnetwork<Float, std::size_t, State>>
laplacian_restrictions(const SparseMat &laplacian,
                       const std::vector<Float> &stationary_sqrt,
                       Index1 initial, Index2 capacity, int count) {
  if (count < 1) {
    throw std::invalid_argument("Laplacian restrictions require count >= 1");
  }

  std::vector<std::size_t> current_states =
      laplacian_neighborhood(laplacian, stationary_sqrt, initial, capacity);
  std::unordered_set<std::size_t> all_included(current_states.begin(),
                                               current_states.end());

  std::vector<Subnetwork<Float, std::size_t, State>> subnetworks;
  subnetworks.reserve(static_cast<std::size_t>(count));

  for (int step = 0; step < count; ++step) {
    subnetworks.push_back(
        laplacian_subnetwork<SparseMat, Float, State>(
            laplacian, stationary_sqrt, current_states));

    if (step + 1 == count || subnetworks.back().boundary().empty())
      break;

    // Build next layer starting from boundary states
    std::vector<std::size_t> next_layer_states;
    for (const auto &trans : subnetworks.back().boundary()) {
      const std::size_t g_dest = static_cast<std::size_t>(trans.destination[0]);
      if (all_included.insert(g_dest).second) {
        next_layer_states.push_back(g_dest);
      }
    }
    if (next_layer_states.empty())
      break;
    current_states = std::move(next_layer_states);
  }
  return subnetworks;
}

template <typename ReversibleLaplacianType, typename State = std::vector<int>,
          typename Float = double, typename Index = std::size_t>
inline Trajectory<State, Float> laplacian_else_trajectory(
    const ReversibleLaplacianType &generator, Index initial, Float initial_time,
    Float final_time, ELSEOptions<Index, Float> options = {}, int seed = 42) {
  const auto &laplacian = generator.matrix();
  const auto &stationary_sqrt = generator.stationary_sqrt();
  const std::size_t n_rows = matrix_rows_helper(laplacian);
  if (initial >= n_rows || initial_time > final_time || options.capacity == 0 ||
      options.capacity > n_rows) {
    throw std::invalid_argument("invalid Laplacian trajectory request");
  }

  std::mt19937 random(seed);
  Trajectory<State, Float> trajectory;
  trajectory.times.push_back(initial_time);
  trajectory.states.push_back(State{static_cast<int>(initial)});

  Index current = initial;
  Float time = initial_time;
  std::size_t completed_steps = 0;

  for (; completed_steps < options.maximum_steps && time < final_time;
       ++completed_steps) {
    const auto states = laplacian_neighborhood(laplacian, stationary_sqrt,
                                               current, options.capacity);
    auto subnetwork = laplacian_subnetwork<decltype(laplacian), Float, State>(
        laplacian, stationary_sqrt, states);
    if (subnetwork.boundary_states().empty())
      break;

    Matrix<Float> source(subnetwork.size(), 1, static_cast<Float>(0));
    int pos = subnetwork.find(State{static_cast<int>(current)});
    if (pos >= 0)
      source(static_cast<std::size_t>(pos), 0) = static_cast<Float>(1);
    const auto integrals = subnetwork.occupation_integrals(source);
    const auto exit_probs = subnetwork.exit_probabilities(integrals);

    std::discrete_distribution<std::size_t> dist(exit_probs.begin(),
                                                 exit_probs.end());
    const std::size_t local_exit = subnetwork.boundary_states()[dist(random)];
    const Float waiting_time =
        subnetwork.conditional_exit_time(integrals, local_exit);

    std::vector<Index> destinations;
    std::vector<Float> rates;
    for (const auto &transition : subnetwork.boundary()) {
      if (transition.source == local_exit) {
        destinations.push_back(static_cast<Index>(transition.destination[0]));
        rates.push_back(transition.rate);
      }
    }
    if (!(waiting_time > static_cast<Float>(0)) || destinations.empty() ||
        time + waiting_time >= final_time) {
      break;
    }

    time += waiting_time;
    std::discrete_distribution<std::size_t> rate_dist(rates.begin(),
                                                      rates.end());
    current = destinations[rate_dist(random)];
    trajectory.times.push_back(time);
    trajectory.states.push_back(State{static_cast<int>(current)});
  }

  if (completed_steps < options.maximum_steps && time < final_time &&
      final_time < std::numeric_limits<Float>::infinity()) {
    trajectory.times.push_back(final_time);
    trajectory.states.push_back(State{static_cast<int>(current)});
  }
  return trajectory;
}

} // namespace else_sim
