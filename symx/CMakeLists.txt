cmake_minimum_required(VERSION 3.15)

# SymX options
set(SYMX_ENABLE_AVX2 "AUTO" CACHE STRING "Enable AVX2+FMA SIMD: AUTO, ON, or OFF. AUTO enables on x86/x86_64/AMD64.")
set_property(CACHE SYMX_ENABLE_AVX2 PROPERTY STRINGS "AUTO" "ON" "OFF")
string(TOUPPER "${SYMX_ENABLE_AVX2}" _SYMX_AVX2)
if(NOT _SYMX_AVX2 MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "SYMX_ENABLE_AVX2 must be AUTO, ON, or OFF (got: ${SYMX_ENABLE_AVX2})")
endif()
if(_SYMX_AVX2 STREQUAL "AUTO")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|X86_64|amd64|AMD64|i[3-6]86)$")
        set(_SYMX_AVX2 "ON")
    else()
        set(_SYMX_AVX2 "OFF")
    endif()
endif()
message(STATUS "SymX: AVX2 = ${_SYMX_AVX2} (SYMX_ENABLE_AVX2=${SYMX_ENABLE_AVX2}, processor=${CMAKE_SYSTEM_PROCESSOR})")
set(SYMX_COMPILER_PATH "AUTO" CACHE STRING "Path to the compiler to use for runtime compilation. Set to AUTO to auto-detect.")
set(SYMX_HESS_STORAGE_FLOAT "double" CACHE STRING "Storage type for Hessian matrices (float or double)")

# CMake option to make fmt and Eigen PUBLIC to use in parent projects
if(NOT DEFINED SYMX_FMT_PUBLIC)
    set(SYMX_FMT_PUBLIC OFF)
endif()
if(NOT DEFINED SYMX_EIGEN_PUBLIC)
    set(SYMX_EIGEN_PUBLIC OFF)
endif()


# Files
set(SOURCE_FILES
	src/macros.h

	src/symbol/Matrix.cpp
	src/symbol/Matrix.h
	src/symbol/Scalar.cpp
	src/symbol/Scalar.h
	src/symbol/diff.cpp
	src/symbol/diff.h
	src/symbol/substitute.cpp
	src/symbol/substitute.h
	src/symbol/Expr.h
	src/symbol/Expressions.cpp
	src/symbol/Expressions.h
	src/symbol/Workspace.cpp
	src/symbol/Workspace.h
	src/symbol/Vector.cpp
	src/symbol/Vector.h
	src/symbol/utils.cpp
	src/symbol/utils.h
	src/symbol/unordered_array_map.h
	
	src/compile/Compilation.cpp
	src/compile/Compilation.h
	src/compile/static_methods.cpp
	src/compile/static_methods.h
	src/compile/compiler_utils.cpp
	src/compile/compiler_utils.h
	src/compile/Compiled.cpp
	src/compile/Compiled.h
	src/compile/FixedBranchSequence.cpp
	src/compile/FixedBranchSequence.h
	src/compile/Sequence.cpp
	src/compile/Sequence.h
	src/compile/AlignmentAllocator.h
	src/compile/type_utils.h
	src/compile/CompiledInLoop.h
	src/compile/CompiledInLoop_run.h
	src/compile/Element.cpp
	src/compile/Element.h
	src/compile/LabelledConnectivity.h
	src/compile/coloring.cpp
	src/compile/coloring.h
	src/compile/fetch_blocks.h
	src/compile/simd_shuffles.h
	src/compile/data_maps.h
	src/compile/MappedWorkspace.h
	src/compile/View.h
	
	src/solver/DeferredParallelTasks.h
	src/solver/Context.h
	src/solver/Context.cpp
	src/solver/Logger.h
	src/solver/Logger.cpp
	src/solver/Potential.h
	src/solver/Potential.cpp
	src/solver/GlobalPotential.h
	src/solver/GlobalPotential.cpp
	src/solver/SolverCallbacks.h
	src/solver/solver_settings.h
	src/solver/solver_types.h
	src/solver/solver_utils.h
	src/solver/solver_utils.cpp
	src/solver/fem_integrators.h
	src/solver/fem_integrators.cpp
	src/solver/mesh_generation.h
	src/solver/mesh_generation.cpp
	src/solver/mesh_ordering.h
	src/solver/mesh_ordering.cpp
	src/solver/CompiledCondition.h
	src/solver/CompiledCondition.cpp
	src/solver/FirstOrderOptimizer.h
	src/solver/FirstOrderOptimizer.cpp
	src/solver/first_order/FirstOrderAlgorithm.h
	src/solver/first_order/FirstOrderAlgorithm.cpp
	src/solver/first_order/FirstOrderCompiledPotential.h
	src/solver/first_order/FirstOrderCompiledPotential.cpp
	src/solver/first_order/FirstOrderCompiledGlobal.h
	src/solver/first_order/FirstOrderCompiledGlobal.cpp
	
	src/solver/second_order/NewtonsMethod.h
	src/solver/second_order/NewtonsMethod.cpp	
	src/solver/second_order/Assembly.cpp
	src/solver/second_order/Assembly.h
	src/solver/second_order/project_to_PD.cpp
	src/solver/second_order/project_to_PD.h
	src/solver/second_order/chunked_vector.h
	src/solver/second_order/ElementHessians.cpp
	src/solver/second_order/ElementHessians.h
	src/solver/second_order/SecondOrderCompiledPotential.h
	src/solver/second_order/SecondOrderCompiledPotential.cpp
	src/solver/second_order/SecondOrderCompiledGlobal.h
	src/solver/second_order/SecondOrderCompiledGlobal.cpp
	src/solver/second_order/gnuplot.h
	src/solver/second_order/gnuplot.cpp
	
	src/solver/adjoint/AdjointNewton.h
	src/solver/adjoint/AdjointNewton.cpp
	src/solver/adjoint/AdjointCompiledLoss.h
	src/solver/adjoint/AdjointCompiledLoss.cpp
	src/solver/adjoint/AdjointCompiledPotential.h
	src/solver/adjoint/AdjointCompiledPotential.cpp
	src/solver/adjoint/FirstOrderOptimizerAdjoint.h
	src/solver/adjoint/FirstOrderOptimizerAdjoint.cpp
	)

# Add library
add_library(symx ${SOURCE_FILES})

# C++17 is required; propagate to consumers
target_compile_features(symx PUBLIC cxx_std_17)

# OpenMP
if(APPLE)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};/opt/homebrew/opt/libomp)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(symx PUBLIC OpenMP::OpenMP_CXX)
endif()

# SIMD: when AVX2 is enabled its intrinsics appear in public headers
if(_SYMX_AVX2 STREQUAL "ON")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(symx PUBLIC -mavx2 -mfma -Wno-ignored-attributes)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(symx PUBLIC -mavx2 -mfma)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(symx PUBLIC /arch:AVX2)
    endif()
endif()

# MSVC-specific build options (private — only affects symx compilation)
if(MSVC)
    target_compile_options(symx PRIVATE /MP)
    target_compile_definitions(symx PRIVATE _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS)
endif()

# Enable/disable the use of AVX2 instructions
if(_SYMX_AVX2 STREQUAL "ON")
    target_compile_definitions(symx PUBLIC SYMX_ENABLE_AVX2 BSM_ENABLE_AVX2)
endif()

# Hessian storage type (float or double)
target_compile_definitions(symx PUBLIC SYMX_HESS_STORAGE_FLOAT=${SYMX_HESS_STORAGE_FLOAT})
message(STATUS "SymX: Hessian storage type set to: ${SYMX_HESS_STORAGE_FLOAT}")

# Compiler path for runtime JIT compilation
target_compile_definitions(symx PUBLIC SYMX_COMPILER_PATH="${SYMX_COMPILER_PATH}")

# Codegen directory: defaults to <build>/codegen, can be overridden via cache.
set(SYMX_CODEGEN_DIR "" CACHE STRING "Directory for codegen output. Leave empty to use <build>/codegen.")
if(SYMX_CODEGEN_DIR STREQUAL "")
    set(SYMX_CODEGEN_DIR "${CMAKE_BINARY_DIR}/codegen")
endif()
target_compile_definitions(symx PUBLIC SYMX_CODEGEN_DIR="${SYMX_CODEGEN_DIR}")
message(STATUS "SymX: Codegen directory set to: ${SYMX_CODEGEN_DIR}")

# Public include directory
target_include_directories(symx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optional Intel oneMKL/PARDISO support. An existing MKL::MKL target or a
# discoverable oneMKL package enables it automatically.
set(SYMX_ENABLE_MKL "AUTO" CACHE STRING
    "Use Intel oneMKL/PARDISO through Eigen: AUTO, ON, or OFF")
set_property(CACHE SYMX_ENABLE_MKL PROPERTY STRINGS AUTO ON OFF)
set(SYMX_MKL_DIR "" CACHE PATH
    "Optional path to Intel oneMKL's CMake package directory")
string(TOUPPER "${SYMX_ENABLE_MKL}" _symx_mkl_mode)
if(NOT _symx_mkl_mode MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
        "SYMX_ENABLE_MKL must be AUTO, ON, or OFF (got: ${SYMX_ENABLE_MKL})")
endif()
if(NOT _symx_mkl_mode STREQUAL "OFF")
    if(NOT TARGET MKL::MKL)
        if(NOT DEFINED MKL_INTERFACE)
            set(MKL_INTERFACE lp64 CACHE STRING "oneMKL interface width")
        endif()
        if(NOT DEFINED MKL_THREADING AND UNIX
                AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(MKL_THREADING gnu_thread CACHE STRING "oneMKL threading layer")
        endif()

        # Standard CMake package paths (including MKL_DIR and CMAKE_PREFIX_PATH)
        # remain available. These hints also cover oneAPI installations whose
        # environment script has not been sourced.
        set(_symx_mkl_hints "${SYMX_MKL_DIR}")
        if(DEFINED ENV{MKLROOT})
            list(APPEND _symx_mkl_hints "$ENV{MKLROOT}")
        endif()
        if(DEFINED ENV{ONEAPI_ROOT})
            list(APPEND _symx_mkl_hints "$ENV{ONEAPI_ROOT}/mkl/latest")
        endif()
        set(_symx_oneapi_roots "/opt/intel/oneapi" "/usr/local/intel/oneapi")
        if(WIN32)
            list(APPEND _symx_oneapi_roots
                "$ENV{ProgramFiles\(x86\)}/Intel/oneAPI"
                "$ENV{ProgramFiles}/Intel/oneAPI")
        endif()
        foreach(_symx_oneapi_root IN LISTS _symx_oneapi_roots)
            list(APPEND _symx_mkl_hints "${_symx_oneapi_root}/mkl/latest")
            file(GLOB _symx_mkl_versions LIST_DIRECTORIES TRUE
                "${_symx_oneapi_root}/mkl/[0-9]*")
            if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
                list(SORT _symx_mkl_versions COMPARE NATURAL ORDER DESCENDING)
            else()
                list(REVERSE _symx_mkl_versions)
            endif()
            list(APPEND _symx_mkl_hints ${_symx_mkl_versions})
        endforeach()
        list(REMOVE_ITEM _symx_mkl_hints "")
        find_package(MKL CONFIG QUIET HINTS ${_symx_mkl_hints})
    endif()

    if(TARGET MKL::MKL)
        target_compile_definitions(symx PUBLIC SYMX_ENABLE_MKL EIGEN_USE_MKL_ALL)
        target_link_libraries(symx PUBLIC MKL::MKL)
        message(STATUS "SymX: MKL/PARDISO direct solver enabled (${_symx_mkl_mode})")
    elseif(_symx_mkl_mode STREQUAL "ON")
        message(FATAL_ERROR
            "SYMX_ENABLE_MKL=ON requires a usable oneMKL package with MKL::MKL. "
            "Set MKL_DIR to its CMake package directory, or set "
            "CMAKE_PREFIX_PATH or MKLROOT to the oneMKL installation root.")
    else()
        message(STATUS "SymX: oneMKL not found; using Eigen's direct solver")
    endif()
else()
    message(STATUS "SymX: MKL/PARDISO direct solver disabled")
endif()

# Dependencies
## Platform libraries for shared-object loading (empty where none are needed).
target_link_libraries(symx PRIVATE ${CMAKE_DL_LIBS})

## Eigen — header-only, types appear in the public API.
if(SYMX_EIGEN_PUBLIC)
    set(_eigen_vis PUBLIC)
else()
    set(_eigen_vis PRIVATE)
endif()

# Parse the bundled Eigen version once so we can print it in the status message.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/extern/Eigen/Eigen/Version"
    _eigen_ver_str REGEX "^#define EIGEN_VERSION_STRING ")
string(REGEX MATCH "\"([^\"]+)\"" _eigen_ver_match "${_eigen_ver_str}")
set(_eigen_version "${CMAKE_MATCH_1}")

# Source selection: pre-existing target → system installation → bundled copy.
if(TARGET Eigen3::Eigen)
    message(STATUS "SymX: Eigen3 — using pre-existing Eigen3::Eigen target")
    target_link_libraries(symx ${_eigen_vis} Eigen3::Eigen)
else()
    find_package(Eigen3 QUIET NO_CMAKE_PACKAGE_REGISTRY)
    if(Eigen3_FOUND)
        message(STATUS "SymX: Eigen3 — found system installation (v${Eigen3_VERSION})")
        target_link_libraries(symx ${_eigen_vis} Eigen3::Eigen)
    else()
        unset(Eigen3_DIR CACHE)  # prevent NOTFOUND entry from polluting the CMake cache
        message(STATUS "SymX: Eigen3 — using copy bundled with SymX (v${_eigen_version})")
        target_include_directories(symx SYSTEM ${_eigen_vis}
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/Eigen)
    endif()
endif()

## fmt — compiled library; used only inside .cpp files, not in public headers.
# Parse the bundled fmt version once so we can print it in the status message.
# fmt >= 11 defines FMT_VERSION in base.h; older releases define it in core.h.
set(_fmt_ver_str "")
foreach(_fmt_ver_header base.h core.h)
    set(_fmt_ver_path "${CMAKE_CURRENT_SOURCE_DIR}/extern/fmt/include/fmt/${_fmt_ver_header}")
    if(EXISTS "${_fmt_ver_path}")
        file(STRINGS "${_fmt_ver_path}" _fmt_ver_str REGEX "^#define FMT_VERSION ")
    endif()
    if(_fmt_ver_str)
        break()
    endif()
endforeach()
if(_fmt_ver_str MATCHES "FMT_VERSION ([0-9]+)")
    math(EXPR _fmaj "${CMAKE_MATCH_1} / 10000")
    math(EXPR _fmin "(${CMAKE_MATCH_1} % 10000) / 100")
    math(EXPR _fpat "${CMAKE_MATCH_1} % 100")
else()
    set(_fmaj "?")
    set(_fmin "?")
    set(_fpat "?")
endif()

# Prefer the bundled compiled library so an unrelated package installation
# (e.g. Conda) cannot add its runtime libraries to native executables' RPATH.
option(SYMX_USE_SYSTEM_FMT "Search for an installed fmt instead of the bundled copy" OFF)

# Source selection: pre-existing target → optional system installation → bundled copy.
if(TARGET fmt::fmt)
    message(STATUS "SymX: fmt — using pre-existing fmt::fmt target")
else()
    if(SYMX_USE_SYSTEM_FMT)
        find_package(fmt QUIET)
    endif()
    if(TARGET fmt::fmt)
        message(STATUS "SymX: fmt — found system installation (v${fmt_VERSION})")
    else()
        if(SYMX_USE_SYSTEM_FMT)
            unset(fmt_DIR CACHE)  # prevent NOTFOUND entry from polluting the CMake cache
        endif()
        message(STATUS "SymX: fmt — using copy bundled with SymX (v${_fmaj}.${_fmin}.${_fpat})")
        # Mark fmt's headers as system headers for consumers (as before with SYMX_FMT_PUBLIC).
        set(FMT_SYSTEM_HEADERS ON)
        add_subdirectory(extern/fmt EXCLUDE_FROM_ALL)
    endif()
endif()

# Link publicly when consumers use fmt, so they also receive fmt's usage
# requirements (e.g. /utf-8 on MSVC, which fmt >= 11 enforces).
if(SYMX_FMT_PUBLIC)
    target_link_libraries(symx PUBLIC fmt::fmt)
else()
    target_link_libraries(symx PRIVATE fmt::fmt)
endif()

## BlockedSparseMatrix — header-only, types appear in the public API (NewtonsMethod.h).
target_include_directories(symx SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/BlockedSparseMatrix/include)

## picoSHA2 — header-only, type appears in the public API (Scalar.h::_get_checksum).
target_include_directories(symx SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/picoSHA2/include)

# Conditionally increase the stack size (hugely complex expressions might need it)
set(SYMX_STACK_SIZE "10480000" CACHE STRING "Linker stack size in bytes")

target_link_options(symx INTERFACE
  "$<$<CXX_COMPILER_ID:MSVC>:/STACK:${SYMX_STACK_SIZE}>"
  "$<$<CXX_COMPILER_ID:GNU>:-Wl,-z,stack-size=${SYMX_STACK_SIZE}>"
  "$<$<CXX_COMPILER_ID:Clang>:-Wl,-z,--stack_size=${SYMX_STACK_SIZE}>"
  "$<$<CXX_COMPILER_ID:AppleClang>:-Wl,-stack_size,${SYMX_STACK_SIZE}>"
)
