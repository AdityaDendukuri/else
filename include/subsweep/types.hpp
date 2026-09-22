// Shared value types for subsweep.
#pragma once

#include "container/multi_index.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include <cstddef>

namespace subsweep {

using num::array;
using num::idx;
using num::key_set;
using num::real;
using num::table;
using num::view;

// A rate between the slots of two subnetworks.
struct coupling {
    idx source = 0;
    idx target = 0;
    real rate = 0.0;
};

// The cut-time loss uses diag(Z): exact in the dense and block regimes, probed in the sparse one.
enum class shedding_rule {
    expected_visits, // (R_jj) u_j - rho_j
    cut_time,        // u_j q_j / Z_jj
};

enum class support_resampling {
    weighted_support,      // distinct states without replacement, original masses kept
    multinomial_particles, // m draws with replacement, repeated states combined
};

// The three solve regimes of the paper. A factorization is reversible,
// Cholesky rather than LU, whenever stationary weights are supplied.
enum class solve_regime {
    dense,  // no-pivot LU; Woodbury reuse
    block,  // block tridiagonal by level; suffix reuse
    sparse, // sparse LU; probed diagonal; no reuse
};

// Gaussian probing of diag(Z) in the sparse regime.
struct probe_options {
    idx count = 40;         // probe vectors h
    idx lanczos_steps = 64; // Krylov steps for the matrix square root
    real lanczos_tolerance = 1e-8;
    unsigned seed = 42;
};

struct sweep_options {
    idx capacity = 30;       // largest subnetwork to factor
    int expansion_depth = 1; // reaction layers to grow outward from the current states
    real tolerance = 1e-6;   // total propensity below which a state is not admitted
    idx maximum_sweeps = 100000;
    shedding_rule rule = shedding_rule::expected_visits;
    solve_regime regime = solve_regime::dense;
    probe_options probes;
    bool reuse_factors = true;
    idx woodbury_cutoff = 3;
    real reuse_residual = 1e-8;
    support_resampling resampling = support_resampling::weighted_support;
    idx particles = 0; // zero uses `capacity`
};

struct sweep_diagnostics {
    idx active_walkers = 0;
    idx distinct_current_states = 0;
    idx subnetwork_size = 0;
    idx changed_state_slots = 0;
    bool reuse_attempted = false;
    bool reuse_accepted = false;
    bool fresh_factorization = false;
    idx block_count = 0;
    idx reused_prefix_blocks = 0;
    idx reused_prefix_states = 0;
    idx reused_current_rows = 0;
    double selection_seconds = 0.0;
    double restriction_and_factor_seconds = 0.0;
    double factor_update_seconds = 0.0;
    double current_rows_seconds = 0.0;
    double score_seconds = 0.0;
    double exit_setup_seconds = 0.0;
    double sampling_seconds = 0.0;
};

struct run_diagnostics {
    array<sweep_diagnostics> sweeps;
};

} // namespace subsweep
