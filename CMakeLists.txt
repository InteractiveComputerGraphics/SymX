cmake_minimum_required(VERSION 3.15)
project(symx)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build when building standalone (tests + examples)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# Register tests with CTest
enable_testing()

# Allow tests and examples to use SymX's Eigen and fmt
set(SYMX_EIGEN_PUBLIC ON)
set(SYMX_FMT_PUBLIC   ON)

add_subdirectory(symx)
add_subdirectory(tests)
add_subdirectory(examples)
