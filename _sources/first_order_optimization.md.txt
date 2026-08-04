# First-order optimization

`FirstOrderOptimizer` directly minimises a `GlobalPotential` without the need of second-order derivatives.
This is useful in the cases where Hessian calculation is too complex or unstable.
First order runs are also useful for visualization and debugging as they usually produce a smoother progression of the minimization process than Newton's method.

## Basic use

Define the potential and its DoF sets in the usual way, then construct the optimizer with the same context and optional callbacks used by Newton's method:

~~~cpp
spGlobalPotential objective = GlobalPotential::create();
// objective->add_potential(...);
// objective->add_dof(variables, "variables");

spContext context = Context::create();
auto optimizer = FirstOrderOptimizer::create(objective, context);
optimizer->settings.type = FirstOrderOptType::LBFGS;
SolverReturn result = optimizer->solve();
~~~

The optimizer compiles the potential value and direct gradient once.
Each iteration evaluates both, selects a descent direction, and applies the normal step cap, maximum-step callback, intermediate-state check, and optional Armijo backtracking.
A rejected or failed line search restores the exact DoFs that were accepted before that trial.

## Algorithms and settings

`FirstOrderSettings` is shared by the direct and adjoint optimizers.
Its default is `LBFGS`; the available values are:

- `GradientDescent`: `-learning_rate * gradient`.
- `Adam`: component-wise adaptive moments.
- `VectorAdam`: adaptive moments with one second-moment value per fixed-size vector block.
- `LBFGS`: limited-memory quasi-Newton directions with normalized steepest descent for the first step.

`VectorAdam` keeps each DoF set separate when creating blocks.
Therefore every DoF set must be divisible by `settings.vector_adam.vector_block_size`; invalid block configurations throw `std::invalid_argument`.
This preserves vector rotation behavior for three-dimensional fields while allowing scalar parameter sets to use a block size of one.

L-BFGS uses `history_size`, `bootstrap_step_length`, and `curvature_threshold`.
Curvature pairs whose `y.dot(s)` is not larger than the threshold are discarded.
If a produced direction is not a descent direction, `restart_when_non_descent` resets the algorithm state and retries with the default steepest-descent direction.

## Callbacks, results, and diagnostics

The direct optimizer accepts `SolverCallbacks`. It invokes `before_step` and `after_step` as a pair for every attempted iteration, and invokes
`before_energy_evaluation` before each value or gradient evaluation. 
The normal validity, maximum-step, Armijo-failure, convergence, and residual callbacks have the same meanings as in [Newton's method](newtons_method.md).

`solve()` returns `SolverReturn`.
Invalid initial values or gradients return `InvalidInitialState`; failed intermediate-state or Armijo searches return the corresponding line-search status after restoring the prior DoFs.
`get_last_solve_stats()` exposes iteration, line-search, and rejected L-BFGS history counts. `print_summary()` formats the accumulated logger statistics.
