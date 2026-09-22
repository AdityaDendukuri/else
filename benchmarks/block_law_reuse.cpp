// Cost of solving the current-state rows U = EZ and V = EZ^2 afresh versus
// updating cached rows by the Woodbury identity, for a rank-2 change.
#include "subsweep/subsweep.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {

using namespace subsweep;
volatile double sink = 0.0;

// A block tridiagonal M-matrix of `blocks` blocks of `width`; `changed`
// perturbs the last diagonal entry.
num::spmat block_matrix(idx width, idx blocks, bool changed) {
    const idx n = width * blocks;
    array<idx> rows, columns;
    array<real> values;
    for (idx i = 0; i < n; ++i) {
        real row_sum = 0.0;
        for (idx j = 0; j < n; ++j) {
            if (i == j || std::abs(long(i / width) - long(j / width)) > 1)
                continue;
            const real value = -0.002 * (1.0 + static_cast<real>((11 * i + 7 * j) % 5));
            num::append(rows, i);
            num::append(columns, j);
            num::append(values, value);
            row_sum -= value;
        }
        num::append(rows, i);
        num::append(columns, i);
        num::append(values, row_sum + 1.0 + (changed && i + 1 == n ? 0.25 : 0.0));
    }
    return num::spmat::from_triplets(n, n, rows, columns, values);
}

template <typename Function> double median_milliseconds(Function &&function) {
    array<double> samples;
    function();
    for (int repetition = 0; repetition < 31; ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        function();
        num::append(samples, std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// U = EZ and V = EZ^2 for `slots`, through the in-place solves and a
// caller-held workspace, as the subnetwork computes them.
void rows_of(const factorization &f, view<const idx> slots, num::mat &u, num::mat &v,
             num::mat &columns) {
    const idx n = f.size(), d = slots.size();
    std::fill(columns.data(), columns.data() + n * d, 0.0);
    for (idx k = 0; k < d; ++k)
        columns(slots[k], k) = 1.0;
    f.solve_transpose(columns, columns);
    for (idx j = 0; j < n; ++j)
        for (idx k = 0; k < d; ++k)
            u(k, j) = columns(j, k);
    f.solve_transpose(columns, columns);
    for (idx j = 0; j < n; ++j)
        for (idx k = 0; k < d; ++k)
            v(k, j) = columns(j, k);
}

} // namespace

int main(int argc, char **argv) {
    const idx width = argc > 1 ? std::stoul(argv[1]) : 30;
    const idx blocks = argc > 2 ? std::stoul(argv[2]) : 20;
    const idx n = width * blocks;
    array<idx> levels(n, 0);
    for (idx i = 0; i < n; ++i)
        levels[i] = i / width;
    sweep_options options;
    options.regime = solve_regime::block;
    const num::spmat base_matrix = block_matrix(width, blocks, false);
    const num::spmat current_matrix = block_matrix(width, blocks, true);
    const array<idx> changed{n - 1};
    const factorization base(base_matrix, levels, {}, options);
    const auto updated = base.update_suffix(current_matrix, levels, {}, changed);
    if (!updated)
        throw std::runtime_error("suffix update was rejected");
    const low_rank_delta delta = row_column_delta(base_matrix, current_matrix, changed);

    std::ofstream csv(argc > 3 ? argv[3] : "block_law_reuse.csv");
    csv << "current_states,fresh_ms,reuse_ms,speedup\n";
    std::cout << "current  fresh_ms  reuse_ms  speedup\n";
    for (idx d : {1, 2, 4, 8, 16, 32, 64, 128}) {
        array<idx> positions(d);
        for (idx k = 0; k < d; ++k)
            positions[k] = (13 * k + 7) % (n - 1);
        num::mat ub(d, n, 0.0), vb(d, n, 0.0), u(d, n, 0.0), v(d, n, 0.0), columns(n, d, 0.0);
        rows_of(base, positions, ub, vb, columns);
        row_workspace work;
        const double fresh = median_milliseconds([&] {
            rows_of(*updated, positions, u, v, columns);
            sink += v(0, 0);
        });
        const double reuse = median_milliseconds([&] {
            const woodbury wb(base, delta);
            std::copy_n(ub.data(), ub.rows() * ub.cols(), u.data());
            std::copy_n(vb.data(), vb.rows() * vb.cols(), v.data());
            update_rows(wb, u, v, work);
            sink += v(0, 0);
        });
        csv << d << ',' << std::setprecision(10) << fresh << ',' << reuse << ',' << fresh / reuse
            << '\n';
        std::cout << std::setw(9) << d << std::fixed << std::setprecision(3) << std::setw(10)
                  << fresh << std::setw(10) << reuse << std::setw(9) << fresh / reuse << '\n';
    }
    return sink == -1.0 ? 1 : 0;
}
