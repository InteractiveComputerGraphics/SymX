# Setup

SymX requires **CMake 3.15+** and a **C++17** compiler.
It bundles all its dependencies (Eigen, fmt) so nothing else needs to be installed.
At runtime, SymX uses your system compiler as a JIT back-end to compile generated code — on Linux and macOS it
finds `g++` automatically.
On Windows, it will check common locations of the MSVC compiler.
In any case, you can always specify SymX which compiler to use; see the
Compiler path section below.


## Project structure

| Folder | Contents |
|---|---|
| `symx/` | The library itself — the only thing you need when embedding SymX in a parent project |
| `tests/` | Catch2 unit tests covering the full API |
| `examples/` | Self-contained example programs |
| `docs/` | Sphinx documentation source |


## Building
Configure project with CMake
```bash
cmake -B build   # Release is default
```

Compile
```bash
cmake --build build --parallel --target examples   # Only examples
cmake --build build --parallel --target tests      # Only tests
cmake --build build --parallel                     # All
```

Build parallelism uses the native build tool's default. To limit it, pass
`--parallel N`, or omit `--parallel` and set `CMAKE_BUILD_PARALLEL_LEVEL`.

Run
```bash
build/tests/tests         # Tests
build/examples/examples   # Examples
```

## CMake options

| Option | Default | Description |
|---|---|---|
| `SYMX_ENABLE_AVX2` | `AUTO` | Enable AVX2 + FMA SIMD paths (`AUTO` / `ON` / `OFF`); `AUTO` enables on x86/x86_64/AMD64 |
| `SYMX_COMPILER_PATH` | `AUTO` | Compiler used for JIT code generation at runtime |
| `SYMX_CODEGEN_DIR` | *(empty)* | Output directory for generated files; defaults to `<build>/codegen` |
| `SYMX_HESS_STORAGE_FLOAT` | `double` | Hessian storage precision (`float` or `double`) |
| `SYMX_ENABLE_MKL` | `AUTO` | Use oneMKL/PARDISO for direct sparse LLT solves when available (`AUTO` / `ON` / `OFF`) |
| `SYMX_MKL_DIR` | *(empty)* | Optional directory containing `MKLConfig.cmake` (compatibility hint; `MKL_DIR` is also supported) |
| `SYMX_USE_SYSTEM_FMT` | `OFF` | Search for an installed fmt; otherwise use the bundled copy. An existing `fmt::fmt` target always takes precedence |
| `SYMX_STACK_SIZE` | `10480000` | Linker stack reservation for generated expressions |

SymX defaults to bundled fmt to avoid accidentally linking a shared library
from an unrelated environment such as Conda. Use `-DSYMX_USE_SYSTEM_FMT=ON`
to allow installed fmt packages, or supply a `fmt::fmt` target from a parent project.


### AVX2 support
SymX uses AVX2 to speedup computations in several locations, such as evaluation and linear system solves.
`SYMX_ENABLE_AVX2` accepts `AUTO` (default), `ON`, or `OFF`.
`AUTO` enables AVX2 on x86/x86_64/AMD64 and disables it everywhere else (e.g. Apple Silicon).
Override explicitly with `-DSYMX_ENABLE_AVX2=ON` or `-DSYMX_ENABLE_AVX2=OFF`.

If you try to compile on a non-AVX2 system with `ON`, you will get something like:

```bash
../immintrin.h:14 error "This header is only meant to be used on x86 and x64 architecture"
```

### `float` Hessian approximation (performance)
Using `SYMX_HESS_STORAGE_FLOAT=float` lets SymX store the global Hessian used in Newton's Method in single point precision.
Importantly, all _operations_ are still performed in `double` (via casting).
This option simply quantizes the global matrix values to `float`, halving memory traffic and significantly increasing performance.
The default remains `double` as very numerically stiff problems might require it, but the user is encouraged to assess whether `float` works for their application.

### MKL
With the default `SYMX_ENABLE_MKL=AUTO`, SymX uses Eigen's PARDISO path when
oneMKL is available and otherwise uses Eigen's built-in solvers. CMake
searches its normal package paths, an existing `MKL::MKL` target supplied by a
parent project, `SYMX_MKL_DIR`, `MKLROOT` and `ONEAPI_ROOT`, and common oneAPI
installation directories (including versioned installations). SymX does not
ship with a copy of MKL and will not download it.

For a standard Intel oneAPI installation on Linux, no environment setup is
needed before configuring:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Use `-DSYMX_ENABLE_MKL=ON` to require MKL and fail configuration if it cannot
be found, or `-DSYMX_ENABLE_MKL=OFF` to use Eigen's solver even when MKL is
installed. For a nonstandard installation, pass a CMake package path or
installation root:

```bash
cmake -S . -B build -DMKL_DIR=/path/to/mkl/lib/cmake/mkl
# Or:
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/mkl
```

An existing build directory keeps its cached `SYMX_ENABLE_MKL` value. Pass
`-DSYMX_ENABLE_MKL=AUTO` to enable automatic discovery in a build previously
configured with MKL disabled.

### Stack size
`SYMX_STACK_SIZE` controls the linker stack reservation. Increase it only for
exceptionally large generated expressions that exhaust the platform default.


## Compiler path

SymX performs JIT compilation at runtime: it writes a `.cpp` file, invokes your compiler, and `dlopen`s the resulting shared object.
You can manually specify the compiler in two ways, via CMake:
```bash
cmake -B build -DSYMX_COMPILER_PATH=/usr/bin/g++-13
```
or at runtime:
```cpp
symx::set_compiler_path("/usr/bin/g++-13");
```

### Linux / macOS
By default, SymX checks if `g++` or `clang++` are available (in that order) and uses them.

### Windows
On Windows, SymX needs the Visual Studio environment batch file.
Most likely, it will need to know where `vcvars64.bat` is.

If no specific compiler path is specified by the user, SymX will try to use `vswhere.exe` to locate the latest Visual Studio installation automatically.
If that fails (or it cannot find it), it will check for `vcvars64.bat` in specific common places.
If that also fails, it will print a message saying so.

For reference, common locations are:
```bash
"C:\\Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat"
"C:\\Program Files (x86)/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build/vcvarsall.bat"
"C:\\Program Files/Microsoft Visual Studio/18/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
```

Make sure you specify the path correctly as string if you use the runtime option:
```cpp
symx::set_compiler_path("\"C:\\Program Files\\Microsoft Visual Studio\\18\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat\"");
```

**Important:** The architecture must match your application.
Use `x64` for 64-bit programs and `x86` for 32-bit.
A mismatch produces errors like *"invalid shared object"* or *"wrong ELF class"* at load time.

**Note:** Compiling in Windows (by using `.bat` scripts) takes _much_ longer than the equivalent compilation with `g++` in Unix systems.
The reason is that loading the `.bat` can take several seconds by itself.


## Compiler diagnostics
If JIT compilation fails or produces wrong results, SymX provides a set of diagnostic utilities to quickly narrow down the problem.

### Suppress / show compiler output
By default SymX hides the compiler's stdout/stderr.
Enable it to see the raw compiler output:
```cpp
symx::suppress_compiler_output(false);
```
Ensure you are not compiling multiple objects in parallel for a clean output.

### Diagnose compilers
You can check which compiler is active by using:
```cpp
std::string path = symx::resolve_compiler_path();  // effective path after AUTO resolution
```

To verify the active (default or otherwise) compiler, the following command runs a small self-contained test: it compiles a trivial function, loads the shared object, evaluates it, and prints a full report:
```cpp
symx::print_compiler_diagnostics();
```

The output looks like this:
```text
Compiler Path: g++
     Available: Yes
     Could Compile Test Program: Yes
     Could Load Test Program: Yes
     Verified Result: Yes
     Supports AVX: Yes
```

If you want to run the above tests for all the default location compilers, you can use:
```cpp
symx::validate_auto_search_compiler_paths();
```

Finally, you can use the following function to generate useful help

```cpp
symx::print_compiler_help();   // prints a numbered checklist of common fixes
```

which outputs
```text
Compiler Help
1. Ensure you have a C++ compiler installed (e.g., g++, clang++, MSVC).
2. Make sure the compiler supports C++17 or higher.
3. If using SIMD features, ensure the compiler supports AVX instructions.
4. You can set the compiler path in symx using symx::set_compiler_path(path).
5. Use symx::suppress_compiler_output(false) to see detailed compilation errors.
6. For specific compiler diagnostics, run symx::print_compiler_diagnostics().
7. To check all auto-detected compilers, run symx::validate_auto_search_compiler_paths().
8. Try again after manually deleting cached compiled shared objects in the codegen folder.
```

This is message is printed when compilation fails.
