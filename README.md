# ELSE: Exact Low-Rank Subnetwork Embedding

A modern, high-performance, header-only C++20 template library for Exact Low-Rank Subnetwork Embedding (ELSE), state space shedding, renewal cut-time evaluation, and transient density computation.

## Mathematical Architecture

- **Subnetwork Inversion & Moments**: Reusable $\mathcal{O}(n^3)$ base factorizations (LU / Cholesky) evaluated across arbitrary right-hand side ensembles in $\mathcal{O}(k \cdot n^2)$ with zero refactoring.
- **Woodbury State Shedding**: Exact block cut-time losses $\ell_S = q_S^\top Z_{SS}^{-1} u_S$ computed in $\mathcal{O}(|S|^3 + |S| n^2)$.
- **Floating-Point Error & Precision Certification**: Linear backward residual tracking $\varepsilon_{\text{res}} = \|Z_{SS} c - u_S\|_\infty / \|u_S\|_\infty$ and lightweight `safe_add` error accumulation with automatic certified naive fallback.
- **Laplace Density Propagation**: Modified Talbot complex contour integration for transient probability densities.

## Quick Start

```cpp
#include <else/else.hpp>

// Create a subnetwork with states, generator R, and boundary exit rates
else_sim::Subnetwork subnetwork(states, R, boundary);

// Solve occupation integrals
auto occupation = subnetwork.occupation(p0);

// Evaluate exact Woodbury cut-time loss
auto result = subnetwork.cut_time_loss_with_diagnostics(occupation, removed_states);
```

## Build & Test

```bash
mkdir build && cd build
cmake ..
cmake --build .
ctest --output-on-failure
```
