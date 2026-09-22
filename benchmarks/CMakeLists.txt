# Timing experiments behind the computational-scaling figure.
add_executable(computational_scaling computational_scaling.cpp)
target_link_libraries(computational_scaling PRIVATE subsweep markovkit::markovkit)
target_include_directories(computational_scaling PRIVATE ${CMAKE_SOURCE_DIR}/examples)

add_executable(block_law_reuse block_law_reuse.cpp)
target_link_libraries(block_law_reuse PRIVATE subsweep)
