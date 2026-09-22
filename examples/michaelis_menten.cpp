// Unordered subsweep versus the finite state projection on Michaelis-Menten:
// the marginals of every species at the horizon.
#include "common/models.hpp"
#include "common/rate_matrix.hpp"
#include "fsp/fsp.hpp"
#include "markovkit/statistics.hpp"
#include "plot/plot.hpp"

using namespace subsweep;
using namespace subsweep::examples;

int main() {
    const reaction_model model = michaelis_menten();
    constexpr real time = 10.0;

    const fsp::fsp_problem problem{.model = model.system,
                                   .u0 = model.initial,
                                   .t0 = 0.0,
                                   .tf = time,
                                   .rates = model.rates,
                                   .bc = fsp::rect_boundary({50, 50, 50, 50})};
    const auto [fsp_solution, fsp_diagnostics] =
        fsp::solve_adaptive_fsp(problem, {.eps_dt = 0.1, .flux_tolerance = 1e-10});

    sweep_options options;
    options.capacity = 60;
    const auto chain = unordered_subsweep(reaction_rate_matrix(model.system, model.rates),
                                          model.initial, 30, options);
    const auto density = reconstruct_distribution(chain, time);

    const char *names[] = {"S", "E", "C", "P"};
    num::plt::subplot(2, 2);
    for (idx species = 0; species < 4; ++species) {
        array<real> fsp_x, fsp_y, sweep_x, sweep_y;
        for (const auto &[count, probability] :
             fsp_solution.marginal(fsp_solution.n_snapshots() - 1, species)) {
            num::append(fsp_x, count);
            num::append(fsp_y, probability);
        }
        for (const auto &[count, probability] :
             markovkit::marginal_distribution(density.states, density.probability, species)) {
            num::append(sweep_x, count);
            num::append(sweep_y, probability);
        }
        num::plt::plot(fsp_x, fsp_y, "FSP", "lines lw 2");
        num::plt::plot(sweep_x, sweep_y, "Unordered subsweep", "boxes");
        num::plt::xlabel(names[species]);
        num::plt::ylabel("probability");
        num::plt::legend();
        if (species < 3)
            num::plt::next();
    }
    num::plt::savefig("michaelis_menten.png");
}
