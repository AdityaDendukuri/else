# Test suite for the Numerics-backed ELSE implementation.
#
# Assertions live in check.hpp rather than <cassert>: this project builds Release
# by default, so `assert` is compiled out and an assert-based suite reports
# success without verifying anything.

function(else_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE else)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

else_test(test_elsex_law)
else_test(test_elsex_shedding)
else_test(test_elsex_ensemble)
else_test(test_elsex_density)
else_test(test_elsex_laplacian)
else_test(test_elsex_factor_update)
