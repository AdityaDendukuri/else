# subsweep

Subnetwork sweeping: first-exit simulation of a continuous-time Markov chain through a sequence of finite subnetworks.
This is the code behind the paper.
Each sweep restricts the chain to a subnetwork containing the current states, solves with the truncated rate matrix `R̄` for the first-exit law, and samples the next exterior state.

## Layout

```
include/subsweep/
  subsweep.hpp             umbrella
  types.hpp                options (capacity, regime, shedding rule, reuse) and diagnostics
  algorithm/               the paper's algorithms on a system
    system.hpp             grown_state; the system and expandable_system concepts
    row_system.hpp         a system from a row callback r(state, visit) with states as keys; restriction
    ordered.hpp            Algorithm 1: ordered_subsweep on a system, ordered_paths on a row_system
    unordered.hpp          Algorithm 2: expansion, shedding, support resampling, the chain
    reconstruct.hpp        p(t) by Talbot inversion over the chain
  sweep/                   one sweep on a subnetwork, states as slots
    subnetwork.hpp         R̄ on slots [0, n): add / discard / matrix / solve / scores
    solution.hpp           sweep_solution: U = EZ, V = EZ², w, exit law, durations, flux
    shedding.hpp           expected visits, cut-time loss, lowest_scores with the level tie-break
    reuse.hpp              the retained base, the workspace, cached-row update and checks
  linear/                  linear algebra of one R̄
    factorization.hpp      dense, block, or sparse factorization; in-place solves; suffix update
    reversible.hpp         the Π^{1/2} similarity
    probe.hpp              probed diag(Z) for the sparse regime
    woodbury.hpp           low-rank delta, Woodbury correction, cached-row update
examples/common/
  rate_matrix.hpp          row callbacks: reaction models, sparse matrices, Laplacians, absorbing targets
  models.hpp, summary.hpp, laplacian.hpp   the paper's models, ensemble statistics, Laplacian experiments
```

The way in for a model of your own is the `system` concept: two functions on your state type, `grow(walker, from, n)` to sample one exit of a slot into the exterior and place the destination, and `discard(j)` to drop a slot (the last slot moves into `j`), optionally `expand(slot, visit)` for the layer expansion.
`ordered_subsweep` runs Algorithm 1 on any system; the library addresses states only by slot in `[0, n)`.
If the model is a rate matrix you can enumerate row by row, `row_system` implements the concept from a callback `r(state, visit)` with `visit(destination, rate)`:

```cpp
auto r = [&](const my_state &x, auto &&visit) { for (auto &[y, rate] : transitions(x)) visit(y, rate); };
auto system = subsweep::rows_system(r, initial, walkers, seed, level);   // level optional, enables the block regime
auto paths = subsweep::ordered_paths(system, options, 0.0, T, seed);      // trajectory<my_state> per walker

auto chain = subsweep::unordered_subsweep(r, initial, sweeps, options);
auto density = subsweep::reconstruct_distribution(chain, t);              // p(t) over the visited states
```

Underneath, a `subnetwork` holds `R̄` on the slots: `add` appends a slot from its total rate and its rates to and from existing slots, `discard(j)` moves the last slot into `j`, `solve(current, counts, out)` returns the rows, exit law, and durations for the current slots, and `scores` ranks slots for shedding.
The solve reuses the retained factorization by the regime in the options: Woodbury in the dense regime, the suffix refactorization with cached rows in the block regime, none in the sparse regime.

## Experiments

| Paper | Program |
| --- | --- |
| Oregonator mean time | `examples/oregonator_mean_time` |
| Computational scaling, panels (a) and (b) | `benchmarks/computational_scaling` |
| Cached-row crossover, panel (c) | `benchmarks/block_law_reuse` |
| Toggle switch | `examples/toggle_switch` |
| Michaelis–Menten density | `examples/michaelis_menten` |
| Laplacian density | `examples/density_laplacian` |
| Laplacian trajectories | `examples/laplacian_trajectories` |
| Talbot inversion | `examples/talbot_vs_fsp_laplacian` |

## Build

```bash
cmake -S . -B build -DSUBSWEEP_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build
```

The library is header-only on the adjacent numerics repository (`../numerics`).
Examples and benchmarks also use the adjacent markovkit repository (`../markovkit`).
