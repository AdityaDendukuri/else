/// @file tests/test_elsex_laplacian.cpp
/// @brief Reversible Laplacian subnetworks and the joint cut-time loss.
///
/// The similarity R = -H^-1 L H is checked structurally: detailed balance must
/// hold with pi = h^2, and the resulting operator must reproduce the same
/// entrance law as the reaction-system path would. The joint cut-time loss is
/// checked against its definition -- the time actually lost when the whole set
/// is removed and the reduced system re-solved.

#include "check.hpp"

#include "elsex/laplacian.hpp"
#include "elsex/shedding.hpp"

#include "container/matrix_expr.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"

#include <cmath>
#include <span>
#include <vector>

namespace {

using State = std::vector<int>;
using num::idx;
using num::real;

constexpr idx graph_size = 12;

/// Path graph Laplacian with unequal edge weights, grounded at both ends so a
/// restriction has somewhere to escape to.
num::SparseMatrix path_laplacian() {
    std::vector<idx> rows, cols;
    std::vector<real> values;
    std::vector<real> degree(graph_size, 0.0);

    for (idx i = 0; i + 1 < graph_size; ++i) {
        const real weight = 0.5 + 0.25 * static_cast<real>(i % 3);
        rows.push_back(i);
        cols.push_back(i + 1);
        values.push_back(-weight);
        rows.push_back(i + 1);
        cols.push_back(i);
        values.push_back(-weight);
        degree[i] += weight;
        degree[i + 1] += weight;
    }
    for (idx i = 0; i < graph_size; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        values.push_back(degree[i]);
    }
    return num::SparseMatrix::from_triplets(graph_size, graph_size, rows, cols, values);
}

/// Uniform h makes L h = 0, so the unrestricted chain is conservative.
std::vector<real> uniform_weights() { return std::vector<real>(graph_size, 1.0); }

/// Non-uniform positive weights, to exercise the similarity scaling.
std::vector<real> graded_weights() {
    std::vector<real> h(graph_size, 0.0);
    for (idx i = 0; i < graph_size; ++i) {
        h[i] = std::sqrt(1.0 + 0.4 * static_cast<real>(i));
    }
    return h;
}

} // namespace

void test_neighborhood_is_connected_and_ordered() {
    const auto laplacian = path_laplacian();
    const auto h = graded_weights();

    const auto window =
        elsex::laplacian_neighborhood(laplacian, std::span<const real>(h), 5, 5);
    check::that(window.size() == 5, "the neighborhood reaches the requested capacity");
    check::that(window.front() == 5, "the origin comes first");

    // On a path graph the neighborhood must be a contiguous run containing 5.
    std::vector<idx> sorted = window;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t k = 1; k < sorted.size(); ++k) {
        check::that(sorted[k] == sorted[k - 1] + 1, "the neighborhood is contiguous");
    }
    check::that(sorted.front() <= 5 && 5 <= sorted.back(), "the origin is inside the run");

    // Weights grow with index, so growth prefers the higher side.
    check::that(sorted.back() == 9, "growth follows descending stationary weight");
    check::done("neighborhood is connected and follows the weight order");
}

void test_similarity_satisfies_detailed_balance() {
    const auto laplacian = path_laplacian();
    const auto h = graded_weights();

    std::vector<idx> window;
    for (idx i = 2; i < 10; ++i) {
        window.push_back(i);
    }
    const auto subnetwork =
        elsex::laplacian_subnetwork(laplacian, std::span<const real>(h), window);

    check::that(subnetwork.is_reversible(), "the subnetwork carries stationary weights");
    const auto stationary = subnetwork.stationary();
    const num::Matrix generator = num::dense(subnetwork.generator());

    for (idx a = 0; a < subnetwork.size(); ++a) {
        check::close(stationary[a], h[window[a]] * h[window[a]], "stationary weight is h^2");
        for (idx b = 0; b < subnetwork.size(); ++b) {
            if (a == b) {
                continue;
            }
            check::close(stationary[a] * generator(a, b), stationary[b] * generator(b, a),
                         "detailed balance holds");
        }
    }
    check::done("the similarity R = -H^-1 L H is reversible with pi = h^2");
}

void test_escape_only_at_the_window_edges() {
    const auto laplacian = path_laplacian();
    const auto h = uniform_weights();

    std::vector<idx> window;
    for (idx i = 4; i < 8; ++i) {
        window.push_back(i);
    }
    const auto subnetwork =
        elsex::laplacian_subnetwork(laplacian, std::span<const real>(h), window);

    check::that(subnetwork.escape_states().size() == 2, "only the two edges escape");
    check::that(subnetwork.escape_states()[0] == 0, "the lower edge escapes");
    check::that(subnetwork.escape_states()[1] == 3, "the upper edge escapes");

    // The first-exit law must still normalize.
    const std::vector<idx> entrances{0, 1, 2, 3};
    const auto law = elsex::entrance_law(subnetwork, std::span<const idx>(entrances));
    const auto rates = subnetwork.escape_rates();
    for (idx k = 0; k < entrances.size(); ++k) {
        real total = 0.0;
        for (idx j = 0; j < subnetwork.size(); ++j) {
            total += rates[j] * law.occupation(k, j);
        }
        check::close(total, 1.0, "sum_j w_j Z_ij = 1 on a Laplacian restriction", 1e-12);
    }
    check::done("a Laplacian restriction escapes only at its edges");
}

void test_restrictions_chain_outward() {
    const auto laplacian = path_laplacian();
    const auto h = uniform_weights();

    const auto chain =
        elsex::laplacian_restrictions(laplacian, std::span<const real>(h), 6, 3, 4);
    check::that(chain.size() >= 2, "the chain has several links");

    // Each link must hand its boundary to the next.
    for (std::size_t r = 0; r + 1 < chain.size(); ++r) {
        bool handed_over = false;
        for (const auto &transition : chain[r].boundary()) {
            if (chain[r + 1].find(transition.destination) < chain[r + 1].size()) {
                handed_over = true;
            }
        }
        check::that(handed_over, "each link's boundary reaches the next link");
    }
    check::done("successive restrictions chain outward");
}

void test_joint_cut_time_matches_its_definition() {
    const auto laplacian = path_laplacian();
    const auto h = graded_weights();

    std::vector<idx> window;
    for (idx i = 1; i < 11; ++i) {
        window.push_back(i);
    }
    const auto subnetwork =
        elsex::laplacian_subnetwork(laplacian, std::span<const real>(h), window);

    num::Vector mixture(subnetwork.size(), 0.0);
    mixture[1] = 0.7;
    mixture[6] = 0.3;
    const auto state = elsex::shedding_state(subnetwork, mixture);

    real full = 0.0;
    for (idx i = 0; i < subnetwork.size(); ++i) {
        full += mixture[i] * state.exit_time[i];
    }

    const num::Matrix operator_matrix = num::dense(subnetwork.operator_matrix());
    const auto reduced_cut_time = [&](std::span<const idx> removed) {
        std::vector<bool> drop(subnetwork.size(), false);
        for (idx j : removed) {
            drop[j] = true;
        }
        std::vector<idx> kept;
        for (idx j = 0; j < subnetwork.size(); ++j) {
            if (!drop[j]) {
                kept.push_back(j);
            }
        }
        num::Matrix block(kept.size(), kept.size(), 0.0);
        for (idx a = 0; a < kept.size(); ++a) {
            for (idx b = 0; b < kept.size(); ++b) {
                block(a, b) = operator_matrix(kept[a], kept[b]);
            }
        }
        const auto factor = num::lu(num::make_square(block));
        num::Vector ones(kept.size(), 1.0);
        num::Vector exit_time(kept.size(), 0.0);
        num::lu_solve(factor, ones, exit_time);

        real total = 0.0;
        for (idx a = 0; a < kept.size(); ++a) {
            total += mixture[kept[a]] * exit_time[a];
        }
        return total;
    };

    // Sets that avoid the entrances, since the lemma requires rho_J = 0.
    const std::vector<std::vector<idx>> blocks{{3}, {3, 4}, {2, 5, 8}, {0, 3, 4, 7, 9}};
    for (const auto &removed : blocks) {
        const auto joint =
            elsex::joint_cut_time_loss(subnetwork, state, std::span<const idx>(removed));
        const real expected = full - reduced_cut_time(std::span<const idx>(removed));
        check::close(joint.loss, expected, "joint loss equals the time actually lost", 1e-10);
        check::that(joint.backward_residual < 1e-10, "the block solve is well conditioned");
    }

    // One removed state must agree with the single-state rule.
    const auto marginal = elsex::exact_cut_time_losses(subnetwork, state);
    const std::vector<idx> single{3};
    const auto joint = elsex::joint_cut_time_loss(subnetwork, state, std::span<const idx>(single));
    check::close(joint.loss, marginal[3], "joint reduces to the single-state rule", 1e-11);
    check::done("joint cut-time loss equals the time actually lost");
}

int main() {
    std::printf("elsex laplacian\n");
    test_neighborhood_is_connected_and_ordered();
    test_similarity_satisfies_detailed_balance();
    test_escape_only_at_the_window_edges();
    test_restrictions_chain_outward();
    test_joint_cut_time_matches_its_definition();
    return check::report("elsex laplacian");
}
