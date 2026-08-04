# The Adjoint Method

Many simulation tasks ask a question that runs in the opposite direction to a forward solve: which material value, load, actuator, or design parameter makes
the simulated result agree with an observation?
The quantity being optimized is a loss on a state obtained only after solving an equilibrium problem. 
This is sensitivity analysis through a simulation, and it is what the adjoint layer in SymX is for.

The layer deliberately keeps the forward model unchanged.
You declare and solve a `NewtonsMethod` exactly as in the [global-optimization layer](newtons_method.md).
Then you declare a separate loss and give both objects to `AdjointNewton`.
The result is an efficient gradient with respect to all loss parameters, independent of how many of them there are.

SymX is perfect for this type of problems as it has access to the symbolic representation of the entire problem and can handle with the annoying task of calculating cumbersome cross derivatives.

## Equilibrium sensitivities

Let $u$ be the simulation DoFs, $p$ a set of parameters, and $P(u,p)$ the physical potential.
The converged forward state $u^*(p)$ satisfies

$$
P_u(u^*(p), p) = 0.
$$

We want the total derivative of a loss $L(u^*(p),p)$.
Differentiating the equilibrium condition would require a linear solve for every parameter:

$$
P_{uu}\frac{d u^*}{d p} + P_{up} = 0.
$$

Instead, SymX solves one adjoint system,

$$
P_{uu}^{\mathsf T}\lambda = -L_u^{\mathsf T},
$$

and evaluates the gradient as

$$
\frac{dL}{dp} = L_p + \lambda^{\mathsf T} P_{up}.
$$

That one additional solve is why the method is especially useful when the number of parameters is large.
The forward Hessian is reused as the adjoint system matrix; choose the adjoint linear solver just as deliberately as the one used by Newton.

## Declare the forward problem and loss

The forward `GlobalPotential` owns the state DoFs.
The loss is another `GlobalPotential`, whose DoFs are the quantities to optimize. 
Both potentials use the same mapped-stencil declarations, so a loss can naturally read the converged state and any parameter arrays.

```cpp
spGlobalPotential physical = GlobalPotential::create();
// physical->add_potential(...);  // defines P(u, p)
physical->add_dof(state);         // u

auto newton = NewtonsMethod::create(physical, context);

spGlobalPotential loss = GlobalPotential::create();
// loss->add_potential(...);      // defines L(u, p)
loss->add_dof(parameters, "parameters");  // p

LinearSolveSettings adjoint_linear;
adjoint_linear.linear_solver = LinearSolver::DirectLLT;
auto adjoint = AdjointNewton::create(newton, loss, adjoint_linear);
```

The physical potential may depend on `parameters` without registering them as simulation DoFs.
Conversely, the loss does not register the state as a loss parameter: it reads it from the forward model. 
Register every independent parameter set that should receive a gradient with `loss->add_dof`.

## Evaluate a value and gradient

`run_forward_and_evaluate_L()` solves the forward equilibrium and evaluates the loss.
Once that state is current, `evaluate_dL_dp_no_solve()` computes its adjoint gradient without solving the forward problem again.

```cpp
AdjointValueResult value = adjoint->run_forward_and_evaluate_L();
if (!value.success) {
    // value.status explains whether the forward or loss evaluation failed.
}

AdjointGradientResult gradient = adjoint->evaluate_dL_dp_no_solve();
if (!gradient.success) {
    // gradient.status explains the failure.
}

double L = value.value;
const Eigen::VectorXd& dL_dp = gradient.gradient;
```

The `run_forward_and_evaluate_dL_dp()` convenience method performs both operations.
The `*_no_solve` methods are intentional: they let an outer optimizer reuse an equilibrium state, but require the caller to keep that state valid.
A failed value is infinity and a failed gradient is empty, so always inspect `success` before consuming either result.

## Verify an implementation

Finite differences remain the best compact check of a new inverse model.
The adjoint helper evaluates centered finite differences, restoring both the parameters and simulation state even if an evaluation fails.

```cpp
auto analytic = adjoint->run_forward_and_evaluate_dL_dp();
auto finite_difference = adjoint->evaluate_dL_dp_finite_differences(1e-3);

if (analytic.success && finite_difference.success) {
    double relative_error =
        (analytic.gradient - finite_difference.gradient).norm() /
        finite_difference.gradient.norm();
}
```

For temporarily fixed scalar components, use `force_gradient_to_zero(indices)`.
Indices address the flattened vector of all loss DoF sets and are validated by the API.

The complete workflow, including a target deformation, finite-difference check, and L-BFGS solve, is developed in [Inverse Simulation](inverse_simulation.md).
