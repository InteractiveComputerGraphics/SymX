cmake_minimum_required(VERSION 3.15)

# Option to enable compiler diagnostic tests (disabled by default to keep output clean)
option(SYMX_ENABLE_DIAGNOSTIC_TESTS "Enable compiler diagnostic tests (verbose output)" OFF)

set(SOURCE_FILES
    test_main.cpp
    sympy_derivatives.cpp
    utils.h
    test_scalar.cpp
    test_vector.cpp
    test_matrix.cpp
    test_compilation.cpp
    test_diff.cpp
    test_mapped_workspace_make.cpp
    test_loops.cpp
    test_global_potential.cpp
    test_newton_method.cpp
	)

# Conditionally add diagnostic tests
if(SYMX_ENABLE_DIAGNOSTIC_TESTS)
    list(APPEND SOURCE_FILES test_diagnostics.cpp test_user_api.cpp)
    add_compile_definitions(SYMX_DIAGNOSTIC_TESTS_ENABLED)
    message(STATUS "SymX: Compiler diagnostic tests enabled (may produce verbose output)")
endif()

add_executable(tests ${SOURCE_FILES})
target_link_libraries(tests symx)

# Register with CTest: run via `ctest` or `cmake --build build --target test`
add_test(NAME symx_tests COMMAND tests)

