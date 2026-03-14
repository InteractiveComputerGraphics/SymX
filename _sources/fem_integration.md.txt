# FEM Integration
SymX includes symbolic FEM integration facilities for 3D elastic solids. This section gives a concise overview before the examples that follow.

## Interpolation Functions
In FEM, continuous scalar or vector fields within an element are typically represented by 

$$
\mathbf v(\mathbf \xi) = \sum_i N_i(\mathbf \xi) \, \mathbf v^h_i,
$$

where $\mathbf \xi$ are local coordinates of the element in the reference configuration.
$\mathbf v^h$ are discrete values at fixed locations,
and $N$ are interpolation (or shape) functions that depend on the type of element.

Out of the box, SymX provides symbolic interpolation functions for the Lagrange elements `enum class FEM_Element { Tet4, Tet10, Hex8, Hex27 };`.
The result applying the above summation on a set of symbolic nodal values can be then carried out using
```cpp
template<typename T>
T fem_interpolation(const FEM_Element& element, std::vector<T>& vh, const Vector& xi);
```
where `T` can be `Scalar`, `Vector` or `Matrix`, `vh` is $\mathbf v^h$ and `xi` is $\mathbf \xi$.

For illustration, here is how the operation is defined for `Hex8`:
```cpp
// Hex8 Lagrangian interpolation function
Scalar N0 = 1.0 - xi[0] - xi[1] - xi[2];
Scalar N1 = xi[0];
Scalar N2 = xi[1];
Scalar N3 = xi[2];
std::vector<Scalar> Nh = { 
    N0 * (2.0 * N0 - 1.0),
    N1 * (2.0 * N1 - 1.0),
    N2 * (2.0 * N2 - 1.0),
    N3 * (2.0 * N3 - 1.0),
    4.0 * N0 * N1,
    4.0 * N1 * N2,
    4.0 * N2 * N0,
    4.0 * N0 * N3,
    4.0 * N1 * N3,
    4.0 * N2 * N3 };

// Summation
T x = Nh[0] * vh[0];
for (int i = 1; i < (int)Nh.size(); i++) {
    x += Nh[i] * vh[i];
}

return x;
```

## Jacobian
The Jacobian (or gradient) of the interpolation functions is used to compute important metrics such as deformation gradients.
The Jacobian of an interpolation is simply:

$$
\mathbf J_{\mathbf y}(\mathbf \xi) = \frac{\partial \mathbf v(\mathbf \xi)}{\partial \mathbf y}
$$

SymX's provides a direct calculation for the Jacobian of an element at location $\mathbf \xi$ as

```cpp
Matrix fem_jacobian(const FEM_Element& element, std::vector<Vector>& xh, const Vector& xi);
```

This function will internally call the corresponding `fem_interpolation` and carry out the differentiation.
Then, the deformation gradient $\mathbf F = \mathbf J_{\mathbf x} \mathbf J_{\mathbf X}^{-1}$ for a `Hex8` element can be simply computed as
```cpp
Matrix Jx = fem_jacobian(FEM_Element::Hex8, x, xi);
Matrix JX = fem_jacobian(FEM_Element::Hex8, X, xi);
Matrix F = Jx * JX.inv();
```
where `x` are the current nodal positions and `X` are the rest nodal positions.


## Numerical Integration
In FEM, we typically want to calculate integrals of functions over the element support.
This is done by means of numerical integration rules, a weighted sum of function values at certain locations.
The rule depend on the element type and polynomial integration degree.
SymX's provides the standard rules for the aforementioned supported 3D elements via
```cpp
std::vector<std::array<double, 4>> get_integration_rule(const FEM_Element& element)
```
For illustration, here is the rule for `Hex8`:
```cpp
double p = 1.0 / std::sqrt(3);
return     
    {
        {1.0, -p, -p, -p},
        {1.0,  p, -p, -p},
        {1.0,  p,  p, -p},
        {1.0, -p,  p, -p},
        {1.0, -p, -p,  p},
        {1.0,  p, -p,  p},
        {1.0,  p,  p,  p},
        {1.0, -p,  p,  p}
    };
```

As described in [Symbol-Data Maps](mapped_workspace.md#summation), `MappedWorkspace` provides _summation_ facilities that allows for easy and efficient implementation of numerical integration.
Note that SymX's summation is also _symbolic_ and **not** unrolled, which would explode the number of operations.
Differentiation is efficiently handled as a summation.

## FEM Integrator
Finally, the whole summation logic is wrapped in `fem_integrator` for convenience:
```cpp
Scalar fem_integrator(MappedWorkspace<double>& mws, const FEM_Element& element, std::function<Scalar(Scalar& w, Vector& xi)> summand)
{
    std::vector<std::array<double, 4>> rule = get_integration_rule(element);
    Scalar summation = mws.add_for_each(rule,
        [&](Vector& ip)
        {
            Scalar w = ip[0];
            Vector xi = Vector({ ip[1], ip[2], ip[3] });
            return summand(w, xi);
        }
    );
    return summation;
}
```
With this, the user only needs to indicate the element type and what to do with the integration point.


## Example
The following is a example for the NeoHookean elasticity energy defined for an abstract `FEM_Element` type:
```cpp
struct Data
{
    std::vector<int> connectivity;
    int stride;
    FEM_Element element_type;
    std::vector<Eigen::Vector3d> x;
    std::vector<Eigen::Vector3d> X_rest;
    double lambda;
    double mu;
} data;

// Create MappedWorkspace
auto [mws, element] = MappedWorkspace<double>::create(data.connectivity, data.stride);

// Create symbols from data
std::vector<Vector> x = mws->make_vectors(data.x, element);
std::vector<Vector> X_rest = mws->make_vectors(data.X_rest, element);
Scalar lambda = mws->make_scalar(data.lambda);
Scalar mu = mws->make_scalar(data.mu);

// Define potential
Scalar P = fem_integrator(data.element_type, mws,
    [&](Scalar& w, Vector xi)
    {
        Matrix Jx = fem_jacobian(mesh.element_type, x, xi);
        Matrix JX = fem_jacobian(mesh.element_type, X_rest, xi);
        Matrix F = Jx * JX.inv();
        return neohookean_energy_density(F, lambda, mu) * JX.det() * w;
    }
);

// ... downstream ...
```
Highlights:
- The mesh is declared as a flat `connectivity` and `stride`. This allows for seamless interoperability between different types of elements in the same centralized potential energy definition.
- Both the numerical integration rule and the Jacobians are generically defined for different types of FEM elements.
- As it can be seen by the compactness of the code, SymX delivers great flexibility and expressiveness. Even disregarding the fact that differentiation and assembly will be handled automatically, the expression definition itself can be written in a very readable way. This is especially impressive when considering that it seamlessly supports different types of elements.
- SymX heavily leans on high level abstractions for expression definitions such as lambdas and dynamic containers. However, crucially, we do **not** pay the performance cost for these generic abstractions at evaluation time. One would probably not write a FEM solver using lambdas and `std::vector` at the core of the element evaluations. We can do this in SymX because these definitions will be executed once, and will generate the actual high performance compiled binaries that will be executed at runtime.
