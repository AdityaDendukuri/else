#include "io/json.hpp"
#include "io/sparse_json.hpp"
#include "markovkit.hpp"
#include <array>
#include <cmath>
#include <vector>

int main() {
    // Load the graph Laplacian, stationary weights, and committors.
    const auto data = num::io::read_json("laplacians/cs-medium.json");
    const auto laplacian = num::io::sparse_matrix(data.at("L"));
    const auto h = num::io::json_vector<double>(data.at("h"));
    const auto committors = num::io::json_matrix<double>(data.at("C"));
    const cme::ReversibleLaplacian generator(laplacian, h);

    constexpr std::size_t count = 5;
    constexpr std::size_t samples = 20;
    std::array<num::idx, count> starts{};
    // Start each ensemble at its dominant committor state.
    for (std::size_t observable = 0; observable < count; ++observable) {
        starts[observable] = num::argmax(
            laplacian.n_rows(), [&](num::idx state) { return committors[state][observable]; });
    }

    // Observe each path on a logarithmic time grid.
    const auto times = num::logspace(-12.0, -4.0, 41);

    std::array<std::array<std::vector<double>, count>, count> curves;
    for (std::size_t panel = 0; panel < count; ++panel) {
        for (auto &curve : curves[panel]) {
            curve.assign(times.size(), 0.0);
        }

        for (std::size_t sample = 0; sample < samples; ++sample) {
            // Sample one trajectory through local Laplacian restrictions.
            const auto path = else_sim::laplacian_else_trajectory(
                generator, starts[panel], 0.0, times.back(), {.capacity = 60},
                static_cast<int>((1000 * panel) + sample));

            // Accumulate committors at the requested observation times.
            const auto state_indices = markovkit::trajectory_indices_at(path, times);
            for (std::size_t time_index = 0; time_index < times.size(); ++time_index) {
                const auto state = static_cast<num::idx>(path.states[state_indices[time_index]][0]);
                for (std::size_t observable = 0; observable < count; ++observable) {
                    curves[panel][observable][time_index] +=
                        committors[state][observable] / static_cast<double>(samples);
                }
            }
        }
    }

    // Each panel shows ensemble-averaged committors from one basin.
    num::plt::subplot(2, 3);
    for (std::size_t panel = 0; panel < count; ++panel) {
        for (std::size_t observable = 0; observable < count; ++observable) {
            num::plt::plot(times, curves[panel][observable], "C" + std::to_string(observable + 1),
                           "linespoints lw 2 pt 7 ps 0.35");
        }
        num::plt::title("start = argmax(C" + std::to_string(panel + 1) + ")");
        num::plt::xlabel("t");
        num::plt::ylabel("mean C^T p(t)");
        num::plt::semilogx();
        num::plt::ylim(-0.02, 1.02);
        if (panel == 0) {
            num::plt::legend();
        }
        num::plt::next();
    }
    num::plt::savefig("laplacian_trajectories.png");
}
