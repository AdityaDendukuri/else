// Subnetwork sweeping (subsweep): first-exit simulation of a Markov chain
// through a sequence of finite subnetworks.
//
//   types.hpp             options and diagnostics
//   linear/               one R_bar: factorization, reversible similarity, probe, Woodbury
//   sweep/                one sweep: solution, shedding, reuse, the subnetwork
//   algorithm/            the paper's algorithms on a system: ordered, unordered, reconstruct
#pragma once

#include "subsweep/algorithm/ordered.hpp"
#include "subsweep/algorithm/reconstruct.hpp"
#include "subsweep/algorithm/row_system.hpp"
#include "subsweep/algorithm/system.hpp"
#include "subsweep/algorithm/unordered.hpp"
#include "subsweep/linear/factorization.hpp"
#include "subsweep/linear/probe.hpp"
#include "subsweep/linear/reversible.hpp"
#include "subsweep/linear/woodbury.hpp"
#include "subsweep/sweep/reuse.hpp"
#include "subsweep/sweep/shedding.hpp"
#include "subsweep/sweep/solution.hpp"
#include "subsweep/sweep/subnetwork.hpp"
#include "subsweep/types.hpp"
