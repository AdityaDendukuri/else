# Test suite for the numerics-based reimplementation in include/elsex/.
#
# Assertions live in check.hpp rather than <cassert>: this project builds Release
# by default, so `assert` is compiled out and an assert-based suite reports
# success without verifying anything.

function(elsex_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE else elsex)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

elsex_test(test_elsex_law)
elsex_test(test_elsex_shedding)
elsex_test(test_elsex_ensemble)
elsex_test(test_elsex_density)
elsex_test(test_elsex_laplacian)
elsex_test(test_elsex_factor_update)
