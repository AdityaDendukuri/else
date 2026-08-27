cmake_minimum_required(VERSION 3.20)
project(else VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(else INTERFACE)
add_library(else::else ALIAS else)

target_include_directories(else
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(CTest)
option(ELSE_BUILD_BENCHMARKS "Build the dependency-free benchmark" ON)
option(ELSE_BUILD_EXAMPLES "Build examples that use Markovkit and Numerics" OFF)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(ELSE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(ELSE_BUILD_EXAMPLES)
  include(FetchContent)
  FetchContent_Declare(numerics SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../numerics")
  FetchContent_Declare(markovkit SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../markovkit")
  FetchContent_MakeAvailable(numerics markovkit)
  find_package(OpenMP)
  add_subdirectory(examples)
endif()
