// What reusing a factorization across sweeps needs: the retained base
// (matrix, factor, per-slot generation, cached current-state rows), the
// workspace the solves run in, and the cached-row update with its checks.
#pragma once

#include "subsweep/linear/factorization.hpp"
#include "subsweep/linear/woodbury.hpp"
#include "subsweep/sweep/solution.hpp"
#include "subsweep/types.hpp"
#include <algorithm>
#include <optional>

namespace subsweep {

namespace detail {

// Give `m` the shape without reallocating when it already has it.
inline void shape(num::mat &m, idx rows, idx cols) {
    if (m.rows() != rows || m.cols() != cols)
        m = num::mat(rows, cols, 0.0);
}

inline void shape(num::vec &v, idx n) {
    if (v.size() != n)
        v = num::vec(n, 0.0);
}

inline void copy_into(num::mat &dst, const num::mat &src) {
    shape(dst, src.rows(), src.cols());
    std::copy_n(src.data(), src.rows() * src.cols(), dst.data());
}


} // namespace detail

struct cached_rows {
    array<idx> slots;
    num::mat occupation, second_occupation;
};

struct reuse_base {
    num::spmat matrix;
    array<idx> generation;
    factorization factor;
    std::optional<cached_rows> rows;
    std::optional<num::vec> diagonal;
};

struct sweep_workspace {
    num::mat u, v;                        // cached rows under update
    num::mat indicators, columns, uf, vf; // basis solves for current rows
    num::vec ones, indicator, row, second;
    row_workspace rows;
};

// U = E Z and V = E Z^2 for `slots`, as rows, from column solves
//   Z^T E^T and Z^T (Z^T E^T) transposed into place.
inline void current_rows(const factorization &solver, view<const idx> slots, num::mat &u,
                         num::mat &v, sweep_workspace &work) {
    const idx n = solver.size(), d = slots.size();
    num::mat &indicators = work.indicators, &columns = work.columns;
    detail::shape(indicators, n, d);
    std::fill(indicators.data(), indicators.data() + n * d, 0.0);
    for (idx k = 0; k < d; ++k)
        indicators(slots[k], k) = 1.0;
    detail::shape(u, d, n);
    detail::shape(v, d, n);
    solver.solve_transpose(indicators, columns);
    for (idx j = 0; j < n; ++j)
        for (idx k = 0; k < d; ++k)
            u(k, j) = columns(j, k);
    solver.solve_transpose(columns, columns);
    for (idx j = 0; j < n; ++j)
        for (idx k = 0; k < d; ++k)
            v(k, j) = columns(j, k);
}

inline void current_rows(const woodbury &solver, view<const idx> slots, num::mat &u,
                         num::mat &v) {
    num::mat indicators(solver.size(), slots.size(), 0.0);
    for (idx k = 0; k < slots.size(); ++k)
        indicators(slots[k], k) = 1.0;
    const num::mat ut = solver.solve_transpose(indicators);
    u = num::transpose(ut);
    v = num::transpose(solver.solve_transpose(ut));
}

// The slots refilled since the base was factored.
[[nodiscard]] inline array<idx> changed_slots(const reuse_base &base,
                                              view<const idx> generation) {
    array<idx> changed;
    for (idx j = 0; j < generation.size(); ++j)
        if (base.generation[j] != generation[j])
            num::append(changed, j);
    return changed;
}

// A cached row applies to a slot only if it still holds the base's state.
[[nodiscard]] inline std::optional<idx> cached_row_of(const reuse_base &base,
                                                      view<const idx> generation, idx slot) {
    if (!base.rows || base.generation[slot] != generation[slot])
        return std::nullopt;
    for (idx k = 0; k < base.rows->slots.size(); ++k)
        if (base.rows->slots[k] == slot)
            return k;
    return std::nullopt;
}

// Use cached rows when p_r < d_c, or p_r < 2 d_c if the Woodbury data already exist.
[[nodiscard]] inline bool row_reuse_is_profitable(const reuse_base &base,
                                                  view<const idx> generation,
                                                  view<const idx> current, idx changed,
                                                  bool woodbury_already_formed) {
    idx matched = 0;
    for (idx slot : current)
        matched += cached_row_of(base, generation, slot) ? 1 : 0;
    const idx rank = 2 * changed;
    return matched != 0 && (woodbury_already_formed ? rank < 2 * matched : rank < matched);
}

// Cached rows updated by Woodbury for the current slots that persist, solved
// fresh for the others; every updated row is checked against R_bar. False
// when a residual exceeds the tolerance, leaving `out` untouched.
[[nodiscard]] inline bool reuse_cached_rows(const reuse_base &base, view<const idx> generation,
                                            const woodbury &wb, view<const idx> current,
                                            const factorization &updated, const num::spmat &R,
                                            real residual_tolerance, sweep_workspace &work,
                                            sweep_solution &out) {
    const idx n = updated.size();
    array<idx> matched, matched_cached, fresh, fresh_slots;
    for (idx k = 0; k < current.size(); ++k) {
        if (const auto row = cached_row_of(base, generation, current[k])) {
            num::append(matched, k);
            num::append(matched_cached, *row);
        } else {
            num::append(fresh, k);
            num::append(fresh_slots, current[k]);
        }
    }
    num::mat &ur = work.u, &vr = work.v;
    detail::shape(ur, matched.size(), n);
    detail::shape(vr, matched.size(), n);
    for (idx row = 0; row < matched.size(); ++row)
        for (idx j = 0; j < n; ++j) {
            ur(row, j) = base.rows->occupation(matched_cached[row], j);
            vr(row, j) = base.rows->second_occupation(matched_cached[row], j);
        }
    update_rows(wb, ur, vr, work.rows);

    num::mat &uf = work.uf, &vf = work.vf;
    if (!fresh_slots.empty())
        current_rows(updated, fresh_slots, uf, vf, work);

    num::mat &u = work.indicators, &v = work.columns; // free after current_rows
    detail::shape(u, current.size(), n);
    detail::shape(v, current.size(), n);
    for (idx row = 0; row < matched.size(); ++row)
        for (idx j = 0; j < n; ++j) {
            u(matched[row], j) = ur(row, j);
            v(matched[row], j) = vr(row, j);
        }
    for (idx row = 0; row < fresh.size(); ++row)
        for (idx j = 0; j < n; ++j) {
            u(fresh[row], j) = uf(row, j);
            v(fresh[row], j) = vf(row, j);
        }

    const num::spmat transposed = num::transpose(R);
    num::vec &indicator = work.indicator, &occupation = work.row, &second = work.second;
    detail::shape(indicator, n);
    detail::shape(occupation, n);
    detail::shape(second, n);
    for (idx k : matched) {
        std::fill(indicator.begin(), indicator.end(), 0.0);
        indicator[current[k]] = 1.0;
        std::copy_n(u.data() + k * n, n, occupation.data());
        std::copy_n(v.data() + k * n, n, second.data());
        if (relative_residual(transposed, occupation, indicator) > residual_tolerance ||
            relative_residual(transposed, second, occupation) > residual_tolerance)
            return false;
    }
    detail::copy_into(out.occupation, u);
    detail::copy_into(out.second_occupation, v);
    out.diagnostics.reused_current_rows = matched.size();
    return true;
}

inline void cache_rows(reuse_base &base, const sweep_solution &out) {
    if (!base.rows)
        base.rows.emplace();
    base.rows->slots = out.current;
    detail::copy_into(base.rows->occupation, out.occupation);
    detail::copy_into(base.rows->second_occupation, out.second_occupation);
}

} // namespace subsweep
