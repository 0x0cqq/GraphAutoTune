cmake_minimum_required(VERSION 3.10)

project(GraphAutoTune VERSION 1.0
        DESCRIPTION "Graph AutoTune"
        LANGUAGES CUDA CXX
)

# Build type: Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()


# Set binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# C++ and CUDA standard: C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# CUDA architecture: sm_70
set(CMAKE_CUDA_ARCHITECTURES 70)
# CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native -Wno-sign-compare")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")


# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-march=native -Wno-sign-compare")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

# disable -Wrestrict
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-restrict")

# Set Include Directories
include_directories(include)


# Fetch Catch2 for testing
message(STATUS "Fetching Catch2...")

Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY git@github.com:catchorg/Catch2.git
  GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)
message(STATUS "Catch2 source dir: ${catch2_SOURCE_DIR}")

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# Target: pattern_matching
add_subdirectory(target)
# Tests
add_subdirectory(tests)
# Utils: transform graph e.g.
add_subdirectory(utils)


# message the current parameters
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CUDA_SEPARABLE_COMPILATION: ${CMAKE_CUDA_SEPARABLE_COMPILATION}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
