# Inverse Simulation

This page builds an inverse gravity-loading problem around the density cantilever example in [`cantilever_adjoint.cpp`](../../examples/cantilever_adjoint.cpp). 
We first simulate a beam with a known density to create a target deformation. 
We then reset the density to a wrong value and recover it by minimizing the difference between the simulated and target vertex positions.

It is a small example, but the declaration pattern is the same for many parameters, multiple parameter sets, controls, and shape objectives.

## Build a target state

The physical potential contains elastic energy, gravitational loading, and penalty boundary conditions.
The density is an ordinary mapped scalar in that forward problem; the displacement field is the simulation DoF.

```cpp
G->add_potential("physical", elements,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        std::vector<Vector> x = mws.make_vectors(data.x, elem);
        std::vector<Vector> X = mws.make_vectors(data.X, elem);
        Scalar density = mws.make_scalar(data.density);

        Scalar elastic = stable_neo_hookean(X, x, E, nu) * dV;
        Scalar gravity = -density * gravity_vector.dot(x) * dV;
        return elastic + gravity;
    });
G->add_dof(data.x);

auto newton = NewtonsMethod::create(G, context);
data.density = target_density;
newton->solve();
data.target_vertices = data.x;
```

The target is simply a converged forward simulation.
In an experimental workflow, `target_vertices` would instead come from measurements.
Reset the state to its reference configuration and set `data.density` to its initial guess before beginning the inverse solve.

## Express the mismatch and parameter

The loss measures squared vertex displacement error.
It declares density as a loss DoF, while reading the simulated positions from the forward solve.

```cpp
spGlobalPotential loss = GlobalPotential::create();
loss->add_potential("shape_match", vertices,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        Vector x = mws.make_vector(data.x, elem[0]);
        Vector target = mws.make_vector(data.target_vertices, elem[0]);
        Scalar weight = mws.make_scalar(data.shape_weight, elem[0]);
        Scalar volume = mws.make_scalar(vertex_volume);
        return 0.5 * weight * volume * (x - target).squared_norm();
    });
loss->add_dof(data.density, "density");

LinearSolveSettings linear;
linear.linear_solver = LinearSolver::DirectLLT;
auto adjoint = AdjointNewton::create(newton, loss, linear);
```

The forward potential supplies $P(u,p)$; this loss supplies $L(u,p)$. The adjoint relation presented in [The Adjoint Method](adjoint_method.md) combines them into $dL/dp$.
Registering another loss DoF set appends another block of parameters to that derivative; it does not require a separate adjoint solve.

## Check the derivative before optimizing

The example checks the analytic gradient against finite differences at the initial guess.
This makes a valuable regression test because it exercises the forward solve, loss, mixed derivatives, parameter flattening, and state restoration together.

```cpp
auto value = adjoint->run_forward_and_evaluate_L();
auto analytic = adjoint->evaluate_dL_dp_no_solve();
auto fd = adjoint->evaluate_dL_dp_finite_differences(1e-3);

double relative_error = (analytic.gradient - fd.gradient).norm() /
                        fd.gradient.norm();
assert(value.success && analytic.success && fd.success);
assert(relative_error < 1e-4);
```

Choose the finite-difference step in the scale of the parameter and use a relative comparison.
A finite-difference check is a diagnostic, not the optimizer: it needs a forward solve per perturbed component, whereas the adjoint gradient needs one adjoint solve.

## Optimize the parameters

`FirstOrderOptimizerAdjoint` applies a first-order method to the parameter vector and calls the forward and adjoint operations as needed.
L-BFGS is the default and works well for this smooth, small inverse problem.

```cpp
auto optimizer = FirstOrderOptimizerAdjoint::create(adjoint, context);
optimizer->settings.type = FirstOrderOptType::LBFGS;
optimizer->settings.lbfgs.bootstrap_step_length = 100.0;
optimizer->settings.max_iterations = 200;
optimizer->settings.residual_tolerance_rel = 1e-4;
optimizer->suppress_inner_newton_output = true;

SolverReturn result = optimizer->solve();
```

With inner output suppressed, the outer trace is concise.
This is an actual Hex8 density-identification trace from the example:

```text
Parameters:
density: 1
Total: 1
0. r0: 2.0e-05 | dp_raw: 1.0e+02 | #newton:  4 | #newton_ls:  0 | dp: 1.0e+02 | 
1. r0: 1.7e-05 | dp_raw: 6.5e+02 | #newton:  7 | #newton_ls:  1 | dp: 6.5e+02 | 
2. r0: 1.1e-06 | dp_raw: 4.7e+01 | #newton:  3 | #newton_ls:  0 | dp: 4.7e+01 | 
3. r0: 1.7e-07 | dp_raw: 7.8e+00 | #newton:  3 | #newton_ls:  0 | dp: 7.8e+00 | 
4. r0: 2.9e-09 | dp_raw: 1.4e-01 | #newton:  1 | #newton_ls:  0 | dp: 1.4e-01 | 
5. r0: 7.5e-12 | (converged)
```

`r0` is the outer optimality residual, falling by more than six orders of magnitude.
`dp_raw` is the L-BFGS proposal and `dp` the accepted parameter increment.
`#newton` reports the number of inner equilibrium iterations needed for that accepted outer trial, and `#newton_ls` reports its inner Newton line-search events. 
Overall the problem converges very quickly and with quadratic convergence as expected.
