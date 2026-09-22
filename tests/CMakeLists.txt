# Assertions live in check.hpp rather than <cassert>: this project builds
# Release by default, so `assert` is compiled out.
function(subsweep_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE subsweep)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/examples)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

subsweep_test(test_factorization)
subsweep_test(test_subnetwork)
subsweep_test(test_ordered)
subsweep_test(test_unordered)
