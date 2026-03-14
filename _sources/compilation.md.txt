# Compilation

Once you have some expressions you want to evaluate efficiently (and probably many times), the next step is to compile them.
This is done very easily by SymX and it typically follows these steps:
1. Optionally, generate a checksum for the expression(s).
2. Check if the compiled object already exists in the codegen folder by matching name, IO and checksum.
3. If it does not exists, SymX will generate a source `.cpp` file with a function `void f(double* in, double* out)`.
4. SymX will use _your_ compiler to generate a `.so` (non-Windows) or `.dll` (Windows) from the source file.
5. SymX will load the compiled function and make it ready to be used.

This process always occurs at _runtime_.
SymX differentiation, code generation and compilation is typically very fast for most expressions (less than one second).
In any case, using the checksum system will skip steps 3 and 4, further reducing setup times.

## Setting up the compiler
See Section [Setup](setup.md) to learn how to point SymX to your compiler.

## `Compiled`
The lowest-level user-facing compilation component in SymX is `Compiled`.
It lets you compile any list of symbolic expressions in any `FLOAT` type.
With `Compiled`, you are responsible to set the numeric values of each symbol and to gather the output.
Here is a simple example:

```cpp
Workspace ws;
Scalar x = ws.make_scalar();
Scalar y = ws.make_scalar();

// Define expressions
Scalar f = x*x + sin(y);
Scalar df_dx = diff(f, x);

// Get default (or CMake specified) codegen dir
std::string codegen_dir = get_codegen_dir();

// Get the checksum of the whole Workspace
std::string checksum = ws.get_checksum();

// Compile both f and its derivative using `double`
Compiled<double> compiled({f, df_dx}, "my_function", codegen_dir, checksum);

// Set values
compiled.set(x, 1.0);
compiled.set(y, 2.0);

// Run compiled function
View<double> result = compiled.run();

// Print result  (output: "f = 1.9093 | df_dx = 2")
std::cout << "f = " << result[0] << " | df_dx = " << result[1] << std::endl;
```

`View` is a thin pointer+length type of wrapper that contains the output of the compiled SymX function.
Under the hood, the function generated in `<codegen_dir>/my_function.cpp` is:
```cpp
EXPORT
void my_function(double* in, double* out)
{
    /*
        Add: 	2
        Mul: 	1
        Sin: 	1
        Total: 	4
    */
    double v2 = in[0] * in[0];
    double v3 = std::sin(in[1]);
    double v4 = v2 + v3;
    out[0] = v4;
    double v6 = in[0] + in[0];
    out[1] = v6;
}
```

### Constructors
You can be more specific on what exactly `Compiled` should do by initializing it empty and then using one of the following methods:
```cpp
Compiled<double> compiled;

// Forces compilation
compiled.compile({f, df_dx}, "my_function", codegen_dir, checksum);

// Try loading
bool success = compiled.load_if_cached("my_function", codegen_dir, checksum);

// Try loading otherwise compile (this is the default constructor)
compiled.try_load_otherwise_compile({f, df_dx}, "my_function", codegen_dir, checksum);
```

A "cached" compiled object that can be loaded, avoiding re-compilation, is one such that:
- An actual `.so` or `.dll` file with the same name exists in the compilation folder.
- The number of function inputs and outputs match.
- The compilation `FLOAT` type matches.
- The checksum matches.

### Status
You can check the status of the compilation process with
```cpp
bool valid = compiled.is_valid();
bool cached = compiled.was_cached() const;
Compilation::Info info compiled.get_compilation_info() const;
```

`info.print()` will produce a report like this one:
```
Number of inputs:             2
Number of outputs:            2
Compiled float type:          double
Was cached:                   No
Runtime code generation (s):  7.531e-05
Runtime compilation (s):      0.14319
Number of bytes for symbols:  144 bytes
```

### Compilation `FLOAT` types
Currently, SymX supports the following SIMD types:
```cpp
Compiled<double> compiled({f, df_dx}, "my_function_double", codegen_dir, checksum);
Compiled<float> compiled({f, df_dx}, "my_function_float", codegen_dir, checksum);
Compiled<__m256d> compiled({f, df_dx}, "my_function_simd4d", codegen_dir, checksum);
Compiled<__m256> compiled({f, df_dx}, "my_function_simd8f", codegen_dir, checksum);
```
Note that you will need to set values of type `FLOAT` and that the result will be returned as `View<FLOAT>`.
If your system does not support these SIMD types, set SymX's the CMake flag `SYMX_ENABLE_AVX2` to `OFF`.

### Parallel evaluation
You can evaluate a `Compiled` expression in parallel in the following way:
```cpp
// Specify the number of concurrent buffers you need
compiled.set_n_threads(4);

// Parallel loop
#pragma omp parallel for schedule(static) num_threads(n_threads)
for (int i = 0; i < n; i++) {
    int tid = omp_get_thread_num();
    
    // Set values thread-safe
    compiled.set(x, 1.0, tid);
    compiled.set(y, 2.0, tid);

    // Evaluate thread-safe
    View<double> result = compiled.run(tid);

    // ... use result ...
}
```

## FEM Example
Let's revisit the Tet4 NeoHookean example.
The energy for a Tet4 was defined like this:
```cpp
// SymX workspace
Workspace ws;

// Create symbols
std::vector<Vector> x = ws.make_vectors(3, 4);
std::vector<Vector> X_rest = ws.make_vectors(3, 4);
Scalar lambda = ws.make_scalar();
Scalar mu = ws.make_scalar();

// FEM potential
Matrix Jx = jac_tet4(ws, x);
Matrix JX = jac_tet4(ws, X_rest);
Matrix F = Jx*JX.inv();
Scalar vol = JX.det()/6.0;
Scalar E = vol*neohookean_energy_density(F, lambda, mu);
```

### Compilation
Let's use the checksum system to avoid differentiation altogether.
If the energy definition and the degrees of freedom are the same, clearly the derivatives must also be the same.
If a compiled object already exist with the same name, IO, compilation type and checksum, then we know that it is safe to load it, even though we skipped differentiation.
SymX actually does this automatically in its high-level infrastructure that we will explore later.

Here is how it can be done:
```cpp
// DoFs
std::vector<Scalar> dofs = collect_scalars(x);

// Checksum (entire WS + dofs)
std::string checksum = ws.get_checksum();
for (const Scalar& s : dofs) {
    checksum += s.get_checksum();
}

// Create empty Compiled
Compiled<double> c_grad;
Compiled<double> c_hess;

// Try loading
c_grad.load_if_cached("NH_grad", codegen_dir, checksum);
c_hess.load_if_cached("NH_hess", codegen_dir, checksum);

// Can't load, must compile
if (!c_grad.was_cached() || !c_hess.was_cached()) {

    // Differentiation
    DiffCache diff_cache;
    Vector grad = gradient(E, dofs, diff_cache);
    Matrix hess = gradient(grad, dofs, /* is_symmetric = */true, diff_cache);

    // Compilation
    c_grad.compile(grad.values(), "NH_grad", codegen_dir, checksum);
    c_hess.compile(hess.values(), "NH_hess", codegen_dir, checksum);
}
```

In a Fedora 43 system equipped with an AMD Ryzen 9 5900X 12 core CPU and using gcc, the following report is obtained when the compiled Hessian is not found:
```
Number of inputs:             26
Number of outputs:            144
Compiled float type:          double
Was cached:                   No
Runtime code generation (s):  0.000605155
Runtime compilation (s):      0.327939
Number of bytes for symbols:  33632 bytes
```
The differentiation runtime for the Hessian was 0.37 ms.
While in this example differentiation does not amount to much, for more complex expressions it can be noticeable.
Therefore, in general avoiding the step when unnecessary is a good practice.
When the compiled function exist, the setup phase is sub-millisecond.

It is worth highlighting that SymX is very fast differentiating and generating code, far faster than compilation which is typically the bottleneck.
While the gradient and Hessian of a Tet4 NeoHookean potential is not extremely complex, it is far from trivial.
Differentiation, code generation and compilation all together stays below one second preparation time.
SymX is designed to be used in far more complex expressions, such as high-order FEM and shell mechanics, and having fast setup times is important.



### Evaluation
The generated code to evaluate the Hessian has a total of 1833 floating point operations.
We can evaluate the function for specific inputs by setting numeric values for the different symbols:
```cpp
struct Data
{
    std::vector<Eigen::Vector3d> x;
    std::vector<Eigen::Vector3d> X_rest;
    double lambda;
    double mu;
} data;

// ... fill data ...

// Set values
for (int v = 0; v < 4; ++v) {
    c_hess.set(x[v], data.x[v]);
    c_hess.set(X_rest[v], data.X_rest[v]);
}
c_hess.set(lambda, data.lambda);
c_hess.set(mu, data.mu);
```

And then we can finally evaluate and print the result:
```cpp
// Evaluate
View<double> result = c_hess.run();

// Print
Eigen::Matrix<double, 12, 12> h;
for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 12; j++) {
        h(i, j) = result[12*i + j];
    }
}
std::cout << "Hessian = \n" << h << std::endl;
```

which for mild irregular compression outputs
```
Hessian = 
  267748  94670.8  94670.8  -152363 -24574.7 -24574.7 -57692.3 -70096.2        0 -57692.3        0 -70096.2
 94670.8   267748  94670.8 -70096.2 -57692.3        0 -24574.7  -152363 -24574.7        0 -57692.3 -70096.2
 94670.8  94670.8   267748 -70096.2        0 -57692.3        0 -70096.2 -57692.3 -24574.7 -24574.7  -152363
 -152363 -70096.2 -70096.2   152363        0        0        0  70096.2        0        0        0  70096.2
-24574.7 -57692.3        0        0  57692.3        0  24574.7        0        0        0        0        0
-24574.7        0 -57692.3        0        0  57692.3        0        0        0  24574.7        0        0
-57692.3 -24574.7        0        0  24574.7        0  57692.3        0        0        0        0        0
-70096.2  -152363 -70096.2  70096.2        0        0        0   152363        0        0        0  70096.2
       0 -24574.7 -57692.3        0        0        0        0        0  57692.3        0  24574.7        0
-57692.3        0 -24574.7        0        0  24574.7        0        0        0  57692.3        0        0
       0 -57692.3 -24574.7        0        0        0        0        0  24574.7        0  57692.3        0
-70096.2 -70096.2  -152363  70096.2        0        0        0  70096.2        0        0        0   152363
```


### Benchmark
We use the Tet4 NeoHookean energy to run a benchmark in order to compare evaluation times for different compilation types and number of threads.
In the aforementioned AMD machine, we obtain

| Compilation type | Runtime (1 thread) | Runtime (12 threads) |
| ---------------- | -----------------: | -------------------: |
| `double`         |            204.886 |               20.142 |
| `float`          |            187.891 |               19.091 |
| `SIMD4d`         |             54.051 |                6.354 |
| `SIMD8f`         |             29.838 |                4.024 |

Runtimes in nanoseconds. 
SIMD times reflect the wall time divided by the number of scalars in the SIMD width.
Note that this benchmark evaluates a single element, thus it is not representative of full-assembly pipelines which are often slowdown by memory traffic.
In any case, it is established that SymX provides great performance and great hardware scalability for almost-zero user involvement.
This benchmark can be found in [`NeoHookean.cpp`](../../examples/NeoHookean.cpp).
