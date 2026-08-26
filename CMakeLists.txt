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

include(FetchContent)
FetchContent_Declare(
    numerics
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../numerics"
)
FetchContent_Declare(
    markovkit
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../markovkit"
)
FetchContent_MakeAvailable(numerics markovkit)
find_package(OpenMP)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(examples)
