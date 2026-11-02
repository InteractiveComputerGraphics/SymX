# Advanced FEM and Integration

The [FEM integration](fem_integration.md) and [symbol-data maps](symbol_data_maps.md) pages introduced scalar potential definitions and shared, fixed integration rules.
This page covers:

- Factorized FEM potentials for efficient Hessian generation
- Element-specific data at integration points
- Variable summation lengths per element


## Factorized FEM Potentials

For high-order elements, differentiating the complete scalar energy with respect to every nodal coordinate can produce large symbolic expressions and generated kernels.
Declaring the material density separately from the deformation gradient lets SymX differentiate the smaller material expression and apply the remaining matrix products in compact loops.
This reduces symbolic differentiation and code generation work and can significantly improve evaluation speed for high-order elements.

At one integration point, collect the nodal positions in $\mathbf x\in\mathbb R^{n\times3}$.
The deformation gradient is linear in these positions:

$$
\mathbf F = \mathbf x^\mathsf T\mathbf B,
\qquad E_q = \psi(\mathbf F),
\qquad \psi(\mathbf F) = w_q\det(\mathbf J_{\mathbf X})\,\Psi(\mathbf F),
$$

where $\mathbf B\in\mathbb R^{n\times3}$ contains the shape function gradients in rest coordinates and is independent of the current positions.
Writing the coordinates and deformation gradient as vectors $\mathbf q$ and $\mathbf f$, this is a linear map $\mathbf f=\mathbf L\mathbf q$.
The chain rule gives

$$
\nabla_{\mathbf q}E_q = \mathbf L^\mathsf T\nabla_{\mathbf f}\psi,
\qquad
\nabla^2_{\mathbf q}E_q = \mathbf L^\mathsf T\bigl(\nabla^2_{\mathbf f}\psi\bigr)\mathbf L.
$$

For Newton's method, SymX generates the material gradient and Hessian symbolically, then the factorized kernel forms the full nodal gradient and Hessian from them.
The material density $\Psi$ can be nonlinear.

Consider the Stable Neo-Hookean definition from [Newton's method](newtons_method.md).
Here `data.element_type` has type `FEM_Element`, and the flat connectivity has `data.stride` entries per element.
The standard scalar definition is:

```cpp
G->add_potential("stable_neo_hookean_" + get_name(data.element_type),
    data.connectivity, data.stride,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        // Create symbols from data
        std::vector<Vector> x = mws.make_vectors(data.x, elem);
        std::vector<Vector> X = mws.make_vectors(data.X, elem);
        Scalar E = mws.make_scalar(data.youngs_modulus);
        Scalar nu = mws.make_scalar(data.poissons_ratio);

        return fem_integrator(mws, data.element_type,
            [&](Scalar& w, Vector& xi)
            {
                Matrix Jx = fem_jacobian(data.element_type, x, xi);
                Matrix JX = fem_jacobian(data.element_type, X, xi);
                Matrix F = Jx * JX.inv();
                return stable_neohookean_energy_density(F, E, nu) * JX.det() * w;
            }
        );
    }
);
```

With `add_potential_fem()`, return a `FemEnergySplit` containing the nodal position matrix, the coefficients `B`, and the weighted material density:

```cpp
G->add_potential_fem("stable_neo_hookean_" + get_name(data.element_type),
    data.connectivity, data.stride,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        std::vector<Vector> x = mws.make_vectors(data.x, elem);
        std::vector<Vector> X = mws.make_vectors(data.X, elem);
        Scalar E = mws.make_scalar(data.youngs_modulus);
        Scalar nu = mws.make_scalar(data.poissons_ratio);
        Matrix x_m(collect_scalars(x), {int(x.size()), 3});

        auto rule = get_integration_rule(data.element_type);
        return mws.add_for_each(rule, [&](Vector& ip)
        {
            Scalar w = ip[0];
            Vector xi({ip[1], ip[2], ip[3]});
            Matrix JX = fem_jacobian(data.element_type, X, xi);
            Matrix B = fem_deformation_gradient_B(data.element_type, xi, JX);
            auto psi = [=](const Matrix& F) {
                return stable_neohookean_energy_density(F, E, nu) * JX.det() * w;
            };
            return FemEnergySplit{ x_m, B, psi };
        });
    }
);
```

Both versions use the same integration rule and define the same discrete energy.
Capture local symbols by value in `psi`, since SymX calls it after the definition callback returns.
Its dependence on the current positions must be through `F`; `B` must be independent of the state DoFs.
In either version, register the current positions as DoFs:

```cpp
G->add_dof(data.x);
```

The [rubber extrusion showcase](rubber-extrusion-example) includes standard, indexed-quadrature, and factorized definitions in [`rubber_extrusion.cpp`](../../examples/rubber_extrusion.cpp), using the material density from [elasticity_potentials.h](../../examples/elasticity_potentials.h).

Factorized SIMD kernels are not provided. The full symbolic energy remains available to the first-order optimizer and adjoint solver.


### Mapped Node Positions

The previous example uses nodal positions directly as DoFs.
Each node's position can also be a nonlinear expression in that node's three DoFs.
SymX detects this mapping in `FemEnergySplit::x` and handles its derivatives automatically; `B` remains independent of the state DoFs.

Collecting these mapped positions into $\mathbf q=\boldsymbol\chi(\mathbf u)$, let $\mathbf J_\chi=\partial\mathbf q/\partial\mathbf u$.
Writing the position gradient and Hessian as $\mathbf g_q=\nabla_{\mathbf q}E_q$ and $\mathbf H_q=\nabla^2_{\mathbf q}E_q$, the derivatives with respect to the optimization variables are

$$
\nabla_{\mathbf u}E_q = \mathbf J_\chi^\mathsf T\mathbf g_q,
\qquad
\nabla^2_{\mathbf u}E_q = \mathbf J_\chi^\mathsf T\mathbf H_q\mathbf J_\chi
+ \sum_k (\mathbf g_q)_k\,\nabla^2_{\mathbf u}\chi_k.
$$

The final term is the geometric stiffness of the map and is included automatically.


## Indexable Summation Data

In the previous Stable Neo-Hookean example, all elements share the same integration rule.
Some applications need different weights, coordinates, or material data for each integration point of each element.
SymX supports these inputs through `make_quadrature_scalar()`, `make_quadrature_vector()`, and `make_quadrature_matrix()`.
Unlike `make_summation_vector(rule)`, these functions bind element-specific point data rather than a shared rule.

Declare a fixed, positive number of points per element with `make_summation(count)`, or use the equivalent `add_for_each(count, definition)` wrapper for a nested loop style.
Adapting the previous scalar potential definition:

```cpp
// Input stored outside the potential definition; populate before evaluation.
std::vector<std::array<double, 4>> all_element_ip_rules; // {weight, xi0, xi1, xi2}

// Inside the definition, using x, X, E, and nu as before:
Scalar P = mws.add_for_each(n_integration_points_per_element,
    [&]()
    {
        Vector ip = mws.make_quadrature_vector(all_element_ip_rules);
        Scalar w = ip[0];
        Vector xi({ip[1], ip[2], ip[3]});
        Matrix Jx = fem_jacobian(data.element_type, x, xi);
        Matrix JX = fem_jacobian(data.element_type, X, xi);
        Matrix F = Jx * JX.inv();
        return stable_neohookean_energy_density(F, E, nu) * JX.det() * w;
    }
);
return P;
```

The array stores points in `[element][point][component]` order: point `p` of element `e` is entry `e * n_integration_points_per_element + p`.
It must contain exactly `n_elements * n_integration_points_per_element` entries, with four components per entry in this example.
All elements use the same point count, but their data may differ.
Keep the input container alive for all evaluations.
Indexable summation data is compatible with factorized FEM definitions.


## Variable-Length Summation

SymX also supports different summation lengths per element, for example to use integration rules with different numbers of points.
Store the points consecutively as before, and pass an offsets array to `make_summation(offsets)` or `add_for_each(offsets, definition)` instead of a fixed count:

```cpp
// Input stored outside the potential definition, alongside all_element_ip_rules.
std::vector<int64_t> offsets = {0, 2, 2, 5};
// Populate all_element_ip_rules with five {weight, xi0, xi1, xi2} entries.

// Inside the definition, using x, X, E, and nu as before:
Scalar P = mws.add_for_each(offsets,
    [&]()
    {
        Vector ip = mws.make_quadrature_vector(all_element_ip_rules);
        Scalar w = ip[0];
        Vector xi({ip[1], ip[2], ip[3]});
        Matrix Jx = fem_jacobian(data.element_type, x, xi);
        Matrix JX = fem_jacobian(data.element_type, X, xi);
        Matrix F = Jx * JX.inv();
        return stable_neohookean_energy_density(F, E, nu) * JX.det() * w;
    }
);
return P;
```

Element `e` uses point entries in `[offsets[e], offsets[e + 1])`.
Here the three elements have two, zero, and three points, respectively.
An empty range contributes zero energy, gradient, and Hessian without evaluating the point kernel.
Use `n_elements + 1` nondecreasing offsets starting at zero, with `offsets.back()` matching the number of point entries in each input array.

Keep the offsets and point containers alive for all evaluations.
You can update or resize them between evaluations without recompiling, provided the arrays remain consistent; do not modify them during evaluation.
Variable-length summations also work with factorized FEM definitions.
They support scalar `float` and `double` compiled kernels; SIMD compiled types are not supported.
