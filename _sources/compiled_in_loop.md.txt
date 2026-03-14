# Loops
Now that we can create SymX symbols bound to our data and connectivity, we can automate loops very easily with `CompiledInLoop`.

## Example
Let's use a simple example to showcase the entire pipeline for symbolic stencil-based loops.
We will use SymX to calculate the area of a triangle mesh.
```cpp
struct Data
{
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::array<int, 3>> triangles;
} data;

// Create MappedWorkspace
auto [mws, element] = MappedWorkspace<double>::create(data.triangles);

// Spawn symbols from data
std::vector<Vector> v = mws->make_vectors(data.vertices, element);

// Expression
Vector cross = (v[1] - v[0]).cross3(v[2] - v[0]);
Scalar area = 0.5*cross.norm();

// Compilation
const std::string codegen_dir = get_codegen_dir();
auto loop = CompiledInLoop<double>::create(mws, {area}, "tri_area", codegen_dir);

// Evaluation
double total_area = 0.0;
const int n_threads = 1;
loop->run(n_threads, 
    [&](View<double> out, int elem_idx, int thread_id, View<int> elem_conn)
    {
        total_area += out[0];
    }
);

// Print
std::cout << "Total triangle mesh area: " << total_area << std::endl;
```

## `CompiledInLoop`
`CompiledInLoop` is the core of most evaluation routines in SymX.
It takes a `MappedWorkspace` and a set of expressions.
It then compiles the expressions, and will evaluate them for all the elements in the connectivity upon calling `run`.

**Important:** `CompiledInLoop` will NOT compile the loop, it will evaluate the compiled function in a loop.

A single `MappedWorkspace` can go into different `CompiledInLoop`.
This is the case in second-order optimization when the same `MappedWorkspace` is liked to different loops (value, gradient and Hessian).

### Float Types
A feature of `CompiledInLoop` is that it can easily handle different floating point types for input, evaluation and output.
This is indicated by its template parameters:
```cpp
CompiledInLoop<INPUT_FLOAT, COMPILED_FLOAT, OUTPUT_FLOAT>;
```
which by default are all equal to `INPUT_FLOAT`.

A common use case is to instruct `CompiledInLoop` to use `double` for IO but SIMD instructions for evaluation, by indicating
```cpp
CompiledInLoop<double, __m256d, double>;
```
In this case, `CompiledInLoop` will compile the expression as `void f(__m256d* in, __m256d* out)` and evaluate it using four elements at a time.
The user will still interface through `double` (their data and `View`) as if they would be using `CompiledInLoop<double>`.

### Constructors
Similarly to `Compiled`, `CompiledInLoop` has three different modes to be initialized:
```cpp
CompiledInLoop<double> loop;

// Forces compilation
loop.compile(mws, {area}, "tri_area", codegen_dir, checksum);

// Try loading
bool success = loop.load_if_cached(mws, "tri_area", codegen_dir, checksum);

// Try loading otherwise compile (this is the default constructor)
loop.try_load_otherwise_compile(mws, {area}, "tri_area", codegen_dir, checksum);
```

### Status
`CompiledInLoop` provides the following functions to check status:
```cpp
bool valid = loop.is_valid();
bool cached = loop.was_cached() const;
CompiledInLoop<>::Info info loop.get_info() const;
```

`info.print()` produces this report for the triangle area example with an empty mesh:
```
Number of inputs:             9
Number of outputs:            1
Number of elements:           0
Connectivity stride:          3
Input float type:             double
Compiled float type:          double
Output float type:            double
Was cached:                   No
Runtime code generation (s):  0.000233242
Runtime compilation (s):      0.165001
Number of bytes for symbols:  544 bytes
```

### Coloring (danger!)
Some applications rely on graph coloring to avoid parallel data races in data.
`CompiledInLoop` supports automatic coloring via these three methods:
```cpp
void enable_coloring(const std::vector<int>& conn_indices);
void disable_coloring();
void update_coloring();
```
To enable coloring, you must indicate which entries of the stencil must not overlap. 
For instance, if your connectivity is numbered or contains group indices, you must not indicate that these are conflicting.
In the previous Tet4 mesh example with `LabelledConnectivity<6> conn = {{ "elem_idx", "group_idx", "v0", "v1", "v2", "v3" }};`, the actual non-overlapping entries for FEM-style assembly must be `conn_indices = {2, 3, 4, 5}`, the actual vertex indices.

**Important:** If your connectivity changes, you must call `update_coloring()` manually as this operation is expensive and not automatically performed by SymX every `run()`.
Failing to do so will result in incorrect element evaluations and possibly in out-of-bounds accesses.


### Run
There are two entry points for `run()`:
```cpp
template<typename CALLBACK_T>
void run(int32_t n_threads, CALLBACK_T callback);

template<typename CALLBACK_T>
void run(int32_t n_threads, CALLBACK_T callback, const std::vector<uint8_t>& is_element_active);
```

The former simply evaluates the compiled expression for every element in the connectivity, while the latter does so selectively based on a boolean mask.
SymX uses OpenMP internally in a fork-join pattern to evaluate the element loop in parallel using `n_threads` in equal partition.

The user can process the output of every element evaluation via `callback`, which should have the following signature:
```cpp
std::function<void(
    View<OUTPUT_FLOAT> solution, 
    int32_t element_idx, 
    int32_t thread_id, 
    View<int32_t> element_conn)>
```
The callback is templated to ensure inlining.


#### Evaluation Loop Overview (if you are curious)
`CompiledInLoop::run` is a critical component of SymX, and as such, it is very carefully optimized.
Internally, `run()` evaluates the element loop in **groups of 8 elements**, in order to amortize buffers and gather/scatter in addition to perfectly fit to SIMD execution.
Each thread owns its own scratch buffers where intermediate computations are carried out.
Every performance sensitive bit of data is aligned, padded and false-sharing-resistant to maximize hardware performance.
If coloring is enabled, the loop is executed one color bin in parallel at a time.
If `is_element_active` is provided, active elements are gathered and processed contiguously.

The stages are:
1. **Gather** element connectivity for the batch.
2. **Gather** inputs into a local AoS buffer from the `MappedWorkspace` data maps.
3. **Cast** inputs if needed (`INPUT_FLOAT → COMPILED_FLOAT` underlying type).
4. **Shuffle** (AoS → SoA) (SIMD only) to prepare the SIMD-friendly layout.
5. **Evaluate** the compiled function accounting for SIMD and a potential summation inner loop.
6. **Shuffle** (SoA → AoS) (SIMD only) so output is always in AoS layout.
7. **Cast** outputs if needed (`COMPILED_FLOAT` underlying type → `OUTPUT_FLOAT`).
8. **Invoke the user callback** for each element in the batch.


## FEM Assembly Example
To exemplify an entire pipeline for `MappedWorkspace` and `CompiledInLoop`, let's take a look again at the Tet4 NeoHookean example, now with evaluation:
```cpp
struct Data
{
    std::vector<std::array<int, 4>> tets;
    std::vector<Eigen::Vector3d> x;
    std::vector<Eigen::Vector3d> X_rest;
    double lambda;
    double mu;
} data;

// Create MappedWorkspace
auto [mws, element] = MappedWorkspace<double>::create(data.tets);

// Create symbols
std::vector<Vector> x = mws->make_vectors(data.x, element);
std::vector<Vector> X_rest = mws->make_vectors(data.X_rest, element);
Scalar lambda = mws->make_scalar(data.lambda);
Scalar mu = mws->make_scalar(data.mu);

// FEM potential
Matrix Jx = jac_tet4(x);
Matrix JX = jac_tet4(X_rest);
Matrix F = Jx*JX.inv();
Scalar vol = JX.det()/6.0;
Scalar E = vol*neohookean_energy_density(F, lambda, mu);

// DoFs
std::vector<Scalar> dofs = collect_scalars(x);

// Differentiation
std::vector<Scalar> values = value_gradient_hessian(E, dofs);  // size: 157

// Compile
auto loop = CompiledInLoop<double>::create(mws, values, "NH_E_grad_hess", codegen_dir);

// Evaluation loop
loop->run(n_threads, 
    [&](View<double> out, int elem_idx, int thread_id, View<int> elem_conn)
    {
        // Energy value at `out[0]`;

        // Gradient
        View<double> out_grad = out.slice(1); // Drop the first value
        // ...

        // Hessian
        View<double> out_hess = out_grad.slice(12); // Drop the first 12 values
        // ...
    }
);
```

Here, we used the `value_gradient_hessian` which conveniently derives the gradient and Hessian, and concatenates all the values in a single output.
The leading value is the energy for the element, the next 12 values correspond to the gradient and the final 144 values are the entries of the 12x12 Hessian (row-major, although symmetric).

### Evaluation
We measure runtime of this entire loop for a mesh with 200k DoFs and 385k tets on the 12 core AMD Ryzen machine:

| Compilation type | Runtime (1 thread) | Runtime (12 threads) |
| ---------------- | -----------------: | -------------------: |
| `double`         |              91.11 |                 9.87 |
| `SIMD4d`         |              37.80 |                 7.16 |
| `SIMD8f`         |              40.10 |                 6.74 |

Runtimes in milliseconds.
Normalizing the runtimes to cost-per-element, the average times range from 236 to 17 nanoseconds.
As expected, this represents a slowdown when compared to evaluating a single element in isolation (no memory traffic) as in the `Compiled` example, which ranged from 204 to 4 ns.


### Assembly (Eigen)
Let's now actually do something with the output.
While SymX provides efficient facilities to assemble this type of sparse systems, let's first use `Eigen::SparseMatrix` to put in perspective performance against a standard approach.
We replace the previous `run()` with the following to process the result:

```cpp
loop->run(n_threads, 
    [&](View<double> out, int elem_idx, int thread_id, View<int> elem_conn)
    {
        // Energy value
        double& thread_energy = energy.get(thread_id);
        thread_energy += out[0];

        // Gradient
        View<double> out_grad = out.slice(1);
        Eigen::VectorXd& thread_grad = grad.get(thread_id);
        for (int v = 0; v < 4; v++) {
            for (int d = 0; d < 3; d++) {
                thread_grad[3*elem_conn[v] + d] += out_grad[3*v + d];
            }
        }
        
        // Hessian
        View<double> out_hess = out_grad.slice(12);
        std::vector<Eigen::Triplet<double>>& thread_triplets = triplets[thread_id];
        for (int vi = 0; vi < 4; vi++) {
            for (int di = 0; di < 3; di++) {
                for (int vj = 0; vj < 4; vj++) {
                    for (int dj = 0; dj < 3; dj++) {
                        thread_triplets.emplace_back(
                            3*elem_conn[vi] + di,
                            3*elem_conn[vj] + dj,
                            out_hess[12*(3*vi + di) + 3*vj + dj]
                        );
                    }
                }
            }
        }
    }
);
```
In this snippet, we use per-thread false-sharing-resistant scratch buffers for evaluation and then aggregate afterwards.
Buffers have enough allocated memory to not need relocation at evaluation time.
These are the updated runtime results (in milliseconds):

| Compilation type | Runtime (1 thread) | Runtime (12 threads) |
| ---------------- | -----------------: | -------------------: |
| `double`         |             463.50 |                95.85 |
| `SIMD4d`         |             415.40 |                93.76 |
| `SIMD8f`         |             412.46 |                89.16 |

This represents a slowdown of 13$\times$ in comparison to evaluation without assembly.
This is due to memory traffic, most notably to writing the Hessian triplets.
Additionally, the assembly from triplets to a sparse matrix must be done after the loop which takes an extra 170 ms.
All things considered, obtaining the global structures using Eigen the fastest way takes 260 ms, while the evaluation itself takes 6.7 ms, a 37x increase.

### Projecting to PD
Finally, let's consider projection to positive definite for all element Hessians.
This can be done with the following line before assembly:
```cpp
project_to_PD_inplace(out_hess.data(), /* size = */ 12, /* eps = */ 1e-10, /* abs = */ false);
```
which uses Eigen internally.
This requires an expensive 12x12 eigendecomposition, which further increases the total evaluation and assembly runtime from 260 ms to 387 ms, a far-cry from the 6.7 ms required to just evaluate every element's energy, gradient and Hessian.

### Assembly (SymX's BCRS)
To alleviate the assembly bottleneck of using naive Eigen Triplets, SymX provides `BlockedSparseMatrix` a Blocked Compressed Row Storage (BCRS) concurrent structure.
It uses in-place insertions of blocks when possible for efficient assembly.
If a block does not exist at the time of insertion, it will temporarily keep it in a block-row bucket.
Both operations are mutex-based to allow for parallelism.
If no new blocks are added, the matrix is built using only concurrent insertions; if there are new blocks, a final rebuild is performed.
`BlockedSparseMatrix` is used internally by SymX when using the `NewtonsMethod` facilities to find the minimum of global potentials.

Here is how we can use `BlockedSparseMatrix` in the example above:
```cpp
bsm::BlockedSparseMatrix<3, 3, double> hess;
hess.start_insertion(ndofs, ndofs);

loop->run(n_threads, 
    [&](View<double> out, int elem_idx, int thread_id, View<int> elem_conn)
    {
        // ...

        // Hessian
        View<double> out_hess = out_grad.slice(12);
        std::array<double, 9> block;
        for (int vi = 0; vi < 4; vi++) {
            for (int vj = 0; vj < 4; vj++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        block[3*i + j] = out_hess[(vi + i)*12 + (vj + j)];
                    }
                }
                hess.add_block_from_ptr(3*elem_conn[vi], 3*elem_conn[vj], block.data());
            }
        }
    }
);

hess.end_insertion(n_threads);
```

In the first run (no previously assembled Hessian), runtimes (excluding projection to PD) are:
| Compilation type | Runtime (1 thread) | Runtime (12 threads) |
| ---------------- | -----------------: | -------------------: |
| `double`         |             197.15 |                62.24 |
| `SIMD4d`         |             147.09 |                38.83 |
| `SIMD8f`         |             144.99 |                38.40 |

Which represent a speedup of up to 2.32$\times$ in comparison to Eigen's approach.
A second assembly run that can reuse the sparsity (e.g. second Newton iteration) results in the following runtimes:
| Compilation type | Runtime (1 thread) | Runtime (12 threads) |
| ---------------- | -----------------: | -------------------: |
| `double`         |             174.32 |                27.78 |
| `SIMD4d`         |             124.40 |                21.66 |
| `SIMD8f`         |             124.52 |                16.62 |

Which is 5.36$\times$ faster than the Eigen assembly.

These benchmarks can be found in [`NeoHookean.cpp`](../../examples/NeoHookean.cpp).

## XPBD Example
Second-order optimization has many particularities surrounding evaluation and assembly.
Delivering on those in a convenient and efficient way is the goal of SymX.

To showcase that SymX can also be used effectively in a non-FEM context, we provide an XPBD cloth simulation example and benchmark in [`xpbd_cloth.cpp`](../../examples/xpbd_cloth.cpp).
There is no projection to PD or assembly in XPBD since it employs a Gauss-Seidel in-place predictor-corrector approach.
SymX can compactly implement such a solver and also provide optimization out-of-the-box such as SIMD and coloring-based parallelism.

