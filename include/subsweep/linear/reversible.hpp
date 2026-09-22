// The reversible similarity: with H = diag(sqrt pi), H R_bar H^-1 is
// symmetric, so a symmetric factor of it applies Z through row scalings.
#pragma once

#include "subsweep/types.hpp"
#include "linear/sparse/sparse.hpp"
#include <cmath>
#include <type_traits>

namespace subsweep::detail {


// h = sqrt(pi), the diagonal of H = Pi^{1/2}.
[[nodiscard]] inline num::vec similarity_weights(view<const real> stationary, idx n) {
    if (stationary.size() != n)
        throw std::invalid_argument("a reversible factorization needs one weight per state");
    num::vec weights(n);
    for (idx i = 0; i < n; ++i) {
        if (!(stationary[i] > 0.0))
            throw std::invalid_argument("stationary weights must be positive");
        weights[i] = std::sqrt(stationary[i]);
    }
    return weights;
}

// H M H^-1 with H = diag(weights), same sparsity pattern.
[[nodiscard]] inline num::spmat similarity_scaled(const num::spmat &matrix,
                                                  view<const real> weights) {
    array<real> values(matrix.nnz());
    array<idx> columns(matrix.col_idx(), matrix.col_idx() + matrix.nnz());
    array<idx> rows(matrix.row_ptr(), matrix.row_ptr() + matrix.n_rows() + 1);
    for (idx i = 0; i < matrix.n_rows(); ++i)
        for (idx k = matrix.row_ptr()[i]; k < matrix.row_ptr()[i + 1]; ++k)
            values[k] = matrix.values()[k] * weights[i] / weights[matrix.col_idx()[k]];
    return num::spmat(matrix.n_rows(), matrix.n_cols(), std::move(values), std::move(columns),
                      std::move(rows));
}

template <typename RightHandSide>
void scale_rows(RightHandSide &value, const num::vec &weights, bool invert) {
    if constexpr (std::is_same_v<RightHandSide, num::vec>) {
        for (idx i = 0; i < value.size(); ++i)
            value[i] *= invert ? 1.0 / weights[i] : weights[i];
    } else {
        for (idx i = 0; i < value.rows(); ++i)
            for (idx j = 0; j < value.cols(); ++j)
                value(i, j) *= invert ? 1.0 / weights[i] : weights[i];
    }
}


} // namespace subsweep::detail
