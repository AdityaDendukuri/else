cmake_minimum_required(VERSION 3.20)
project(subsweep VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(subsweep INTERFACE)
add_library(subsweep::subsweep ALIAS subsweep)

target_include_directories(subsweep
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Numerics supplies the linear algebra used by the subsweep algorithms.
include(FetchContent)
FetchContent_Declare(numerics SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../numerics")
FetchContent_MakeAvailable(numerics)
target_link_libraries(subsweep INTERFACE numerics::numerics)

include(CTest)
option(SUBSWEEP_BUILD_BENCHMARKS "Build the benchmarks (uses Markovkit)" ON)
option(SUBSWEEP_BUILD_EXAMPLES "Build the examples (use Markovkit and OpenMP)" OFF)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Markovkit supplies the reaction systems the benchmarks and examples restrict.
if(SUBSWEEP_BUILD_BENCHMARKS OR SUBSWEEP_BUILD_EXAMPLES)
  FetchContent_Declare(markovkit SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../markovkit")
  FetchContent_MakeAvailable(markovkit)
endif()

if(SUBSWEEP_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(SUBSWEEP_BUILD_EXAMPLES)
  find_package(OpenMP)
  add_subdirectory(examples)
endif()
