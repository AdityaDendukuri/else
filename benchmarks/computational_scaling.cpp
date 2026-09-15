/// Kernel timings against the work terms used in the subsweep complexity claims.
#include "container/matrix.hpp"
#include "linear/eigen/lanczos.hpp"
#include "linear/factorization/block_tridiagonal.hpp"
#include "linear/factorization/lu_no_pivot.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/sparse/sparse.hpp"
#include "plot/plot.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using clock_type = std::chrono::steady_clock;
volatile double sink = 0.0;

template <typename Function>
double median_time(Function &&function, int repetitions = 9, int batch = 1) {
    num::array<double> samples;
    function();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto start = clock_type::now();
        for (int item = 0; item < batch; ++item)
            function();
        samples.push_back(std::chrono::duration<double>(clock_type::now() - start).count() / batch);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

num::mat dense_m_matrix(num::idx n) {
    num::mat matrix(n, n, 0.0);
    for (num::idx i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (num::idx j = 0; j < n; ++j) {
            if (i == j)
                continue;
            const double value = -0.01 * (1.0 + static_cast<double>((17 * i + 13 * j) % 7));
            matrix(i, j) = value;
            row_sum -= value;
        }
        matrix(i, i) = row_sum + 1.0;
    }
    return matrix;
}

struct BlockProblem {
    num::spmat matrix;
    num::array<num::idx> levels;
    double work = 0.0;
};

BlockProblem block_m_matrix(num::idx width, num::idx block_count = 8) {
    const num::idx n = width * block_count;
    num::array<num::idx> rows, columns, levels(n, 0);
    num::array<double> values, row_sum(n, 0.0);
    for (num::idx block = 0; block < block_count; ++block)
        for (num::idx local = 0; local < width; ++local)
            levels[block * width + local] = block;
    for (num::idx i = 0; i < n; ++i) {
        const num::idx block_i = i / width;
        for (num::idx j = 0; j < n; ++j) {
            const num::idx block_j = j / width;
            if (i == j || std::abs(static_cast<long>(block_i) - static_cast<long>(block_j)) > 1)
                continue;
            const double value = -0.01 * (1.0 + static_cast<double>((11 * i + 7 * j) % 5));
            rows.push_back(i);
            columns.push_back(j);
            values.push_back(value);
            row_sum[i] -= value;
        }
        rows.push_back(i);
        columns.push_back(i);
        values.push_back(row_sum[i] + 1.0);
    }
    double work = 0.0;
    for (num::idx block = 0; block < block_count; ++block) {
        work += static_cast<double>(width) * width * width;
        if (block + 1 < block_count)
            work += 2.0 * static_cast<double>(width) * width * width;
    }
    return {num::spmat::from_triplets(n, n, rows, columns, values), std::move(levels), work};
}

num::spmat shifted_path_laplacian(num::idx n) {
    num::array<num::idx> rows, columns;
    num::array<double> values;
    for (num::idx i = 0; i < n; ++i) {
        rows.push_back(i);
        columns.push_back(i);
        values.push_back(2.5);
        if (i > 0) {
            rows.push_back(i);
            columns.push_back(i - 1);
            values.push_back(-1.0);
        }
        if (i + 1 < n) {
            rows.push_back(i);
            columns.push_back(i + 1);
            values.push_back(-1.0);
        }
    }
    return num::spmat::from_triplets(n, n, rows, columns, values);
}

num::array<double> normalized(const num::array<double> &values) {
    num::array<double> result(values.size());
    for (num::idx i = 0; i < values.size(); ++i)
        result[i] = values[i] / values.front();
    return result;
}

void panel(const num::array<double> &sizes, const num::array<double> &times,
           const num::array<double> &work, const std::string &title,
           const std::string &work_label) {
    const auto relative_sizes = normalized(sizes);
    num::plt::plot(relative_sizes, normalized(times), "measured kernel time",
                   "linespoints lw 2 pt 7");
    num::plt::plot(relative_sizes, normalized(work), work_label,
                   "linespoints dt 2 lw 2 pt 9");
    num::plt::xlabel("dimension relative to smallest tested");
    num::plt::ylabel("relative time or operation count");
    num::plt::ylim(0.0, 140.0);
    num::plt::title(title);
    num::plt::legend();
}

} // namespace

int main(int argc, char **argv) {
    const std::string prefix = argc > 1 ? argv[1] : "computational_scaling";
    std::ofstream csv(prefix + ".csv");
    csv << "regime,n,work_proxy,seconds,lanczos_steps\n";

    num::array<double> dense_n, dense_time, dense_work;
    for (num::idx n : num::array<num::idx>{80, 120, 180, 260, 360}) {
        const num::mat matrix = dense_m_matrix(n);
        const double elapsed = median_time([&] {
            const auto factor = num::factor_no_pivot(num::assume_square(matrix));
            sink += factor.packed(0, 0);
        }, 9, 20);
        dense_n.push_back(static_cast<double>(n));
        dense_time.push_back(elapsed);
        dense_work.push_back(static_cast<double>(n) * n * n);
        csv << "dense," << n << ',' << dense_work.back() << ',' << std::setprecision(10)
            << elapsed << ",0\n";
    }

    num::array<double> block_n, block_time, block_work;
    for (num::idx width : num::array<num::idx>{64, 96, 144, 208, 288}) {
        const BlockProblem problem = block_m_matrix(width);
        const double elapsed = median_time([&] {
            const auto factor = num::factor_block_lu(problem.matrix, problem.levels);
            sink += factor.diagonal.front().packed(0, 0);
        }, 9, 10);
        block_n.push_back(static_cast<double>(problem.matrix.n_rows()));
        block_time.push_back(elapsed);
        block_work.push_back(problem.work);
        csv << "block," << problem.matrix.n_rows() << ',' << problem.work << ','
            << std::setprecision(10) << elapsed << ",0\n";
    }

    num::array<double> sparse_n, sparse_time, sparse_work;
    for (num::idx n : num::array<num::idx>{4000, 8000, 16000, 32000, 64000}) {
        const num::spmat matrix = shifted_path_laplacian(n);
        num::vec direction(n, 1.0);
        const auto op = num::operators::assume_spd(num::operators::sparse_op(matrix));
        const auto warm = num::inverse_sqrt_lanczos(op, direction, 1e-8, 64);
        num::idx steps = warm.steps;
        const double elapsed = median_time([&] {
            const auto result = num::inverse_sqrt_lanczos(op, direction, 1e-8, 64);
            steps = result.steps;
            sink += result.value[0];
        }, 9, 5);
        const double work = static_cast<double>(n) * steps * steps +
                            static_cast<double>(steps) * matrix.nnz();
        sparse_n.push_back(static_cast<double>(n));
        sparse_time.push_back(elapsed);
        sparse_work.push_back(work);
        csv << "sparse," << n << ',' << work << ',' << std::setprecision(10) << elapsed << ','
            << steps << '\n';
    }

    num::plt::subplot(1, 3);
    panel(dense_n, dense_time, dense_work, "Unstructured dense", "n^3 operation count");
    num::plt::next();
    panel(block_n, block_time, block_work, "Structured dense", "block LU operation count");
    num::plt::next();
    panel(sparse_n, sparse_time, sparse_work, "Unstructured sparse",
          "n l^2 + l nnz(S)");
    num::plt::savefig(prefix + ".png");
    std::cout << "wrote " << prefix << ".csv and " << prefix << ".png\n";
    return sink == -1.0 ? 1 : 0;
}
