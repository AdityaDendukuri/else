# The numerical experiments of the paper, one executable each; the reference
# problem code and simulation loops they share live in common/.
function(subsweep_example name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE subsweep numerics::plot ${ARGN})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

subsweep_example(oregonator_mean_time markovkit::markovkit OpenMP::OpenMP_CXX)
subsweep_example(toggle_switch markovkit::markovkit)
subsweep_example(michaelis_menten markovkit::markovkit)
subsweep_example(density_laplacian numerics::io OpenMP::OpenMP_CXX)
subsweep_example(laplacian_trajectories numerics::io OpenMP::OpenMP_CXX)
subsweep_example(talbot_vs_fsp_laplacian markovkit::markovkit numerics::io)
subsweep_example(custom_system)
subsweep_example(oregonator_trajectory)
