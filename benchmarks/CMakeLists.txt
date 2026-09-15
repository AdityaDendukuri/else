add_executable(dense_factor_update dense_factor_update.cpp)
target_link_libraries(dense_factor_update PRIVATE else numerics::plot)

add_executable(factor_reuse_cme factor_reuse_cme.cpp)
target_link_libraries(factor_reuse_cme PRIVATE else markovkit::markovkit)

add_executable(block_reuse_walltime block_reuse_walltime.cpp)
target_link_libraries(block_reuse_walltime PRIVATE else markovkit::markovkit numerics::plot)

add_executable(computational_scaling computational_scaling.cpp)
target_link_libraries(computational_scaling PRIVATE else numerics::plot)
