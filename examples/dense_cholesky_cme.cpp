#include "else/restriction.hpp"
#include "markovkit.hpp"
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

int main() {
    constexpr num::idx n = 32;
    constexpr double birth_rate = 8.0;
    constexpr double death_rate = 1.0;

    // This model has constant immigration and linear death.
    markovkit::ReactionSystem model{
        .changes = {{1}, {-1}},
        .propensities = {
            [](const markovkit::State &, const auto &rates, double) { return rates[0]; },
            [](const markovkit::State &state, const auto &rates, double) {
                return rates[1] * state[0];
            },
        }};
    const std::vector<double> rates{birth_rate, death_rate};

    std::vector<markovkit::State> states;
    // The finite restriction contains molecule counts from zero to n-1.
    for (num::idx state = 0; state < n; ++state) {
        states.push_back(markovkit::State{static_cast<int>(state)});
    }

    // The full immigration-death CME has Poisson stationary weights.
    std::vector<double> h(n, 1.0);
    for (num::idx state = 1; state < n; ++state) {
        h[state] = h[state - 1] * std::sqrt(birth_rate / (death_rate * state));
    }

    // Use reversibility to select the Cholesky solve path.
    auto subnetwork = else_sim::reversible_cme_subnetwork(model, rates, states, h);
    num::Vector source(n, 0.0);
    source[0] = 1.0;
    // Occupation measures expected time spent in each state before exit.
    const auto time_spent = subnetwork.occupation(source);

    auto count = num::linspace(0.0, static_cast<double>(n - 1), n);
    std::vector<double> occupation = time_spent;
    std::vector<double> stationary(n);
    for (num::idx state = 0; state < n; ++state) {
        stationary[state] = h[state] * h[state];
    }
    // Compare normalized occupation and stationary profiles.
    num::clip_and_normalize_nonnegative(occupation);
    num::clip_and_normalize_nonnegative(stationary);

    // The plot checks the occupation profile against equilibrium weights.
    num::plt::plot(count, stationary, "stationary", "lines lw 2");
    num::plt::plot(count, occupation, "occupation", "points pt 7 lc rgb '#c0392b'");
    num::plt::title("Reversible CME: dense Cholesky occupation solve");
    num::plt::xlabel("molecule count");
    num::plt::ylabel("probability");
    num::plt::legend();
    num::plt::savefig("dense_cholesky_cme.png");
}
