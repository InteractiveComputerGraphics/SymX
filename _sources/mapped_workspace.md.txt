# Symbol-Data Maps

We want SymX to create symbols that represent the actual data we have: the vertices of a mesh, material parametrization, etc.
We can do this directly with `MappedWorkspace`: instead of first creating symbols with `Workspace` and then setting numerical values, `MappedWorkspace` creates symbols that represent data in user's containers.
Then, at evaluation time, SymX knows exactly which numeric value to fetch for each symbol.

## Example
Let's first take a look at an example where we define an expression to evaluate the area of a triangle in a mesh using `MappedWorkspace`:
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

// ... downstream ...
```
As shown, symbols are no longer standalone — they are created from the `vertices` array.
Also, `triangles`, is the **connectivity array** defines the loop and how to index the data arrays in each iteration.


## The `MappedWorkspace`
A `MappedWorkspace` binds a symbolic `Workspace` to actual data for loop-based evaluation.

### Connectivity
The first thing we need to define is the **connectivity** array that defines the length of the loop, as well as each instance and provides indices for data fetching.
Let's say we as users have a triangle mesh with the following connectivity:
```cpp
std::vector<std::array<int, 3>> triangles;
```

Then, we can create the `MappedWorkspace` like this:
```cpp
auto [mws, element] = MappedWorkspace<double>::create(triangles);
```
Where the first return is of type `std::shared_ptr<MappedWorkspace<double>>` and the second one is of type `Element`.
The former is the new workspace itself, that will be used to create symbols and hold the mappings to actual data arrays.
The latter is simply a "symbolic representation" of an element in the connectivity array.
It is simply used to index the actual data arrays in the same way we would do `triangles[tri_i][vertex_j]`.

Let's clarify a few things about connectivities:
- A `MappedWorkspace` cannot exist without connectivity.
- SymX holds a reference to `triangles`. If it is resized (e.g. mesh refinement), it will be correctly accounted for. This is a key feature.
- Connectivity **can** be initialized (and used) empty. You can initialize your SymX-based solver before the user declares entities, as long as the data containers themselves exist.
- SymX always expect contiguous data arrays in row major order.
- SymX only supports fixed-stride connectivities. Dynamic per-entry strides (e.g. SPH) is not allowed in this form.
- If you _move_ or generally invalidate the reference, SymX will produce undefined behavior. This also holds for the data arrays.

#### Connectivity Array Lambdas
Albeit low-level, it is important to know what exactly SymX holds from the user's connectivity:
```cpp
struct ConnectivityMap
{
    std::function<const int32_t*()> data = nullptr; // Lambda to obtain pointer to the beginning
    std::function<int32_t()> n_elements = nullptr;  // Lambda to obtain current number of elements (e.g. triangles)
    int32_t stride = -1;                            // Stride in the array (e.g. 3 for triangles, 4 for quads)
}
```
These lambda functions allow SymX to always get updated sizes and valid pointers to data, even after resizing or clearing the container.

#### Connectivity Array Types
SymX supports other types for connectivity beyond `std::vector<std::array<int, 3>>`.
In fact, the interface is templated to be very flexible since it just needs fill those generic lambdas.
The following are valid initializations for our example:
```cpp
std::vector<int> triangles_flat;
int stride = 3;
auto [mws, element] = MappedWorkspace<double>::create(triangles_flat, stride);
```

```cpp
Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> triangles;
triangles.resize(0, 3);
auto [mws, element] = MappedWorkspace<double>::create(triangles);
```
Note that it's very important to use `RowMajor` and specify the number of columns for Eigen types, even when empty.

Further, you can directly pass the lambdas if your connectivity container does not match any of the available types:
```cpp
std::vector<std::array<int, 3>> triangles;
auto [mws, element] = MappedWorkspace<double>::create(
    [&triangles](){ return (const int*)triangles.data(); },  // ptr to data
    [&triangles](){ return (int)triangles.size(); },         // number of elements
    3                                                        // stride
);
```

Finally, SymX provides a convenient connectivity type (optional to use), `LabelledConnectivity`, that allows `string` labels to each entry in the stencil:
```cpp
LabelledConnectivity<3> triangles = {{ "v0", "v1", "v2" }};
auto [mws, element] = MappedWorkspace<double>::create(triangles);
```
Internally, `LabelledConnectivity<N>` is just `std::vector<std::array<int, N>>` plus `std::array<std::string, N>` to identify each entry.
These labels are now available as indices, e.g. `element["v0"]` instead of the pure numeric default.
While not really necessary for a triangle, they are very useful for more complex stencils such as point-triangle contact between two rigid bodies:
```cpp
LabelledConnectivity<6> rb_point_triangle_contact{ { "rb_a", "rb_b", "p", "v0", "v1", "v2" } };
```

### Symbols
Similarly to connectivity, SymX keeps lambda functions to access updated references to the user's data.
Specifically:
```cpp
struct DataMap
{
    std::function<std::uintptr_t()> id = nullptr;  // Unique identifier: address of the data source container (e.g. &std::vector, &Eigen::Matrix). Used for identity checks.
    std::function<const double*()> data = nullptr; // Lambda to obtain pointer to the beginning of data
    std::function<int32_t()> n_elements = nullptr; // Lambda to obtain current number of elements (e.g. vertices)
    int32_t stride = -1;                           // Stride in the array (e.g. 3 for 3D vectors)
    int32_t connectivity_index = -1;               // Entry idx in the connectivity array the symbol is associated with. -1 for no connectivity (fixed) values
    int32_t first_symbol_idx = -1;                 // Location of the first symbol in the workspace's symbols array.
}
```
While this is not something the user needs to work with, it helps to understand how SymX works internally in order to access the user's data.
The entries are:
- `id`: Pointer to the container. This is used in high level abstraction to match symbols, e.g. for DoF identification.
- `data`: Pointer to the beginning of the data.
- `n_elements`: Number of elements/"rows" in the data array.
- `stride`: Number of entries per element or "columns" in the data array.
- `connectivity_index`: Stencil index associated with this symbol (e.g. `vertices[tri[2]]` -> 2)
- `first_symbol_idx`: Index of the first symbol associated with this array.

This will be populated automatically for supported container types such as `std::vector<Eigen::Vector<>>`, `std::vector<std::array<>>`, `std::vector<>`, etc.


#### Symbols Creation
Just like in `Workspace`, we can create `Scalar`, `Vector` and `Matrix` from `MappedWorkspace`.
The only difference is that now we need to specify an actual data array (and optionally indexing) upon creation:
```cpp
// Creates a constant Scalar
Scalar a = mws->make_scalar(data.a);

// Create a scalar indexed from array `data.v` using `tet[0]` index
Scalar v = mws->make_scalar(data.v, tet[0]);

// Creates multiple Vector bound to `data.vs` indexed with all entries in `tet`
std::vector<Vector> vs = mws->make_vectors(data.vs, tet);

// Creates a Matrix from `data.m` indexed by `tet[1]`
Matrix m = mws->make_matrix(data.m, tet[1]);
```

Further, symbols can be created by explicitly passing the required lambdas for the cases where the user's data container is not recognized by SymX:
```cpp
Vector MappedWorkspace::make_vector(
    std::function<std::uintptr_t()> id, 
    std::function<const FLOAT* ()> data, 
    std::function<int32_t()> n_elements, 
    const int32_t stride, 
    const Index& idx
    );
```

#### Summation
A common operation in many numerical codes is summation, a nested small loop that needs to be evaluated for each element.
A typical use case for summations is FEM numerical integration.
The total evaluation for the loop takes the form:

$$
P = \sum_e^{n_e} \sum_i^{n_i} E(\mathrm x_e, \mathrm p_i)
$$

where $P$ is the total potential, $E$ is the element potential, $n_e$ is the number of elements, $n_i$ is the length of the summation loop, $\mathrm x_e$ is the element-specific input and $\mathrm p_i$ is the summation input.
In the inner loop, $\mathrm x_e$ is constant; the summation occurs for different values of $\mathrm p_i$.

SymX fully supports this logic, including differentiation, CSE, SIMD vectorization, parallelism, etc.
We can indicate summation in SymX using `MappedWorkspace::add_for_each()`, which takes a fixed summation array and a lambda function.
For illustration, this is how the summation of the Lagrange Tet10 element can be written in SymX:

```cpp
std::vector<std::array<double, 4>> integration_rule_tet10 =
    {
        {0.041666667, 0.58541020, 0.13819660, 0.13819660},
        {0.041666667, 0.13819660, 0.58541020, 0.13819660},
        {0.041666667, 0.13819660, 0.13819660, 0.58541020},
        {0.041666667, 0.13819660, 0.13819660, 0.13819660}
    };

Scalar result = mws->add_for_each(integration_rule_tet10,
    [&](Vector& ip)
    {
        Scalar w = ip[0];
        Vector xi = Vector({ ip[1], ip[2], ip[3] });
        return f(xi)*w;
    }
);
```

## FEM Example
Let's use a bit more complex FEM example to showcase several options to create indexed symbols from data arrays.
Consider a single connectivity array representing several concatenated Tet4 meshes where the NeoHookean elastic energy is defined.
Each element has `group_idx` (indicating which mesh it belongs to) and `elem_idx` (indicating the global element index).
The first Lamé parameter (lambda) is defined per group, the second (mu) is a global constant.
Finally, the rest element volume and inverse jacobian is precomputed as they are constant per element.
We can represent the simulation data as such:
```cpp
struct Data
{
    LabelledConnectivity<6> conn = {{ "elem_idx", "group_idx", "v0", "v1", "v2", "v3" }};
    std::vector<Eigen::Vector3d> vertices;
    std::vector<double> group_lambda;
    double mu;
    std::vector<std::array<double, 9>> elem_inv_rest_jacobian;
} data;
```

This is how we conveniently define the potential over the entire mesh on these custom rules and user structures using SymX:
```cpp
// Create MappedWorkspace
auto [mws, element] = MappedWorkspace<double>::create(data.conn);

// Shorthand for Tet4 indices
std::vector<Index> tet = element.slice(2, 6);

// Create symbols from data
std::vector<Vector> x = mws->make_vectors(vertices, tet);
Scalar lambda = mws->make_scalar(data.group_lambda, element["group_idx"]);
Scalar mu = mws->make_scalar(data.mu);
Scalar rest_vol = mws->make_scalar(data.rest_vol, element["elem_idx"]);
Matrix JX_inv = mws->make_matrix(data.elem_inv_rest_jacobian, {3, 3}, element["elem_idx"]);

// FEM potential
Matrix Jx = jac_tet4(x);
Matrix F = Jx*JX_inv;
Scalar E = rest_vol*neohookean_energy_density(F, lambda, mu);

// ... downstream ...
```
Highlights:
- We can slice and index `Element` for more convenient indexing.
- `LabelledConnectivity` makes the mapping readable thanks to indexing by label as opposed to an unlabelled `std::vector<std::array<int, 6>>`.
- We can make multiple scalars, vectors and matrices in a single `make_` call if multiple indices (or the whole `Element`) are passed.
- Global (non-indexed) symbols can be created by simply not passing an index. Applies to `Scalar`, `Vector` and `Matrix`.
- We recommend to **not** use `std::vector<Eigen::Matrix<>>` (e.g. 3x3 matrix) due to potential issues with padding.
    Instead, prefer `std::vector<std::array<double, N*N>>` to guarantee data continuity.

Finally, as we move higher in the abstraction hierarchy, it is important to remark that mappings and definitions are **not** executed at evaluation time.
The SymX snippets shown here are run **once**, usually before the solver starts, in order to define the problem and generate the actual high performance code used at evaluation.

