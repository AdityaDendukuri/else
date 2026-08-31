add_executable(benchmark_else benchmark_else.cpp)
target_link_libraries(benchmark_else PRIVATE else)

add_executable(dense_factor_update dense_factor_update.cpp)
target_link_libraries(dense_factor_update PRIVATE else numerics::plot)

add_executable(factor_reuse_cme factor_reuse_cme.cpp)
target_link_libraries(factor_reuse_cme PRIVATE else markovkit::markovkit)
