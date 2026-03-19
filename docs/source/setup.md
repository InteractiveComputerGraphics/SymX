# Setup

SymX requires **CMake 3.15+** and a **C++17** compiler.
It bundles all its dependencies (Eigen, fmt) so nothing else needs to be installed.
At runtime, SymX uses your system compiler as a JIT back-end to compile generated code — on Linux and macOS it
finds `g++` automatically.
On Windows, it will check common locations of the MSVC compiler.
In any case, you can always specify SymX which compiler to use (see [Compiler path](#compiler-path) below).


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
| `SYMX_HESS_STORAGE_FLOAT` | `float` | Hessian storage precision (`float` or `double`) |


### AVX2 support
SymX uses AVX2 to speedup computations in several locations, such as evaluation and linear system solves.
`SYMX_ENABLE_AVX2` accepts `AUTO` (default), `ON`, or `OFF`.
`AUTO` enables AVX2 on x86/x86_64/AMD64 and disables it everywhere else (e.g. Apple Silicon).
Override explicitly with `-DSYMX_ENABLE_AVX2=ON` or `-DSYMX_ENABLE_AVX2=OFF`.

If you try to compile on a non-AVX2 system with `ON`, you will get something like:

```bash
../immintrin.h:14 error "This header is only meant to be used on x86 and x64 architecture"
```

### `float` Hessian approximation
Using `SYMX_HESS_STORAGE_FLOAT=float` lets SymX store the global Hessian used in Newton's Method in single point precision.
Importantly, all _operations_ are still performed in `double` (via casting). This option simply quantizes the global matrix values to `float`, halving memory traffic and significantly increasing performance.


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
```txt
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
```txt
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
