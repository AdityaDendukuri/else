include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../approxchol/include")

# SSA vs ELSE comparisons
add_executable(else_oregonator else_oregonator.cpp)
target_link_libraries(else_oregonator PRIVATE else markovkit::markovkit numerics::plot OpenMP::OpenMP_CXX)
add_custom_target(ssa_vs_else_oregonator DEPENDS else_oregonator)

add_executable(else_toggle_switch else_toggle_switch.cpp)
target_link_libraries(else_toggle_switch PRIVATE else markovkit::markovkit numerics::plot)
add_custom_target(ssa_vs_else_toggle_switch DEPENDS else_toggle_switch)

add_executable(find_switching find_switching.cpp)
target_link_libraries(find_switching PRIVATE else markovkit::markovkit)

add_executable(michaelis_menten michaelis_menten.cpp)
target_link_libraries(michaelis_menten PRIVATE else markovkit::markovkit numerics::plot)

# ELSE and Shedding examples
add_executable(else_birth_death else_birth_death.cpp)
target_link_libraries(else_birth_death PRIVATE else markovkit::markovkit numerics::plot)

add_executable(else_shedding_comparison else_shedding_comparison.cpp)
target_link_libraries(else_shedding_comparison PRIVATE else markovkit::markovkit numerics::plot)

if(TARGET numerics::io)
    add_executable(density_laplacian density_laplacian.cpp)
    target_link_libraries(density_laplacian PRIVATE else numerics::plot numerics::io OpenMP::OpenMP_CXX)

    add_executable(laplacian_trajectories laplacian_trajectories.cpp)
    target_link_libraries(laplacian_trajectories PRIVATE else markovkit::markovkit numerics::plot numerics::io)

    add_executable(benchmark_laplacians benchmark_laplacians.cpp)
    target_link_libraries(benchmark_laplacians PRIVATE else markovkit::markovkit numerics::solvers numerics::io)

    add_executable(talbot_vs_fsp_laplacian talbot_vs_fsp_laplacian.cpp)
    target_link_libraries(talbot_vs_fsp_laplacian PRIVATE else markovkit::markovkit numerics::plot numerics::io)
endif()

add_executable(dense_cholesky_cme dense_cholesky_cme.cpp)
target_link_libraries(dense_cholesky_cme PRIVATE else markovkit::markovkit numerics::plot)

add_executable(compare_approxchol_cme compare_approxchol_cme.cpp)
target_link_libraries(compare_approxchol_cme PRIVATE else markovkit::markovkit numerics::solvers)
