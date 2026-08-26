add_executable(test_else test_else.cpp)
target_link_libraries(test_else PRIVATE else)
add_test(NAME test_else COMMAND test_else)
