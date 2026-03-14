# Second-Order Optimization
SymX was designed to find the solution to large and complex non-linear optimization problems.
While it is not the only thing it can do, it is the main focus and where most of the advanced infrastructure lies.

Consider the problem

$$
\mathbf x^* = \min_{\mathbf x} G(\mathbf x, \mathbf p),
$$

where the goal is to find the values of the Degrees of Freedom (DoF) $\mathbf x$ so that the *global potential* function $G$ is minimized.
$\mathbf p$ are non-DoF parameters.
Further, we are interested in problems where $G$ is a composition of $n_p$ different potential functions

$$
G = \sum_p^{n_p} P_p.
$$

In turn, each potential $P_p$ is an aggregation of $n_p^e$ element contributions

$$
P_p = \sum_e^{n_p^e} P_p^e,
$$

for which each element contribution depends on (part of) $\mathbf x$.
Therefore, the solution $\mathbf x^*$ is affected by every single element contribution and the solution must be found as a coupled global problem, typically by using Newton's Method.

This is a very common scheme in Finite Element Analysis where a global potential can be defined as the addition of different effects (elasticity, contact, friction, ...).
Each of these are represented in different discretizations (volumetric FEM, contact pairs, ...) that link the global DoFs.

## Newton's Method
Assuming that $G$ is at least, $C^1$ continuous, we can find a local optimum $\mathbf x^*$ by solving $\nabla_{\mathbf x} G(\mathbf x^*) = \mathbf 0$, which generally is a non-linear system of equations.
We can then use Newton's Method to find a solution by iteratively solving the following linearized system of equations for a sequence $\mathbf x^k$

$$
\nabla^2_{\mathbf x} G(\mathbf x^k) \, \Delta \mathbf x = -\nabla_{\mathbf x} G(\mathbf x^k),
$$

for $\mathbf x^{k+1} = \mathbf x^k + \Delta \mathbf x$, where:
- $\nabla^2_{\mathbf x} G$ is the second derivative of the global potential wrt $\mathbf x$ (**Hessian**),
- $\nabla_{\mathbf x} G$ is the first derivative of the global potential wrt $\mathbf x$ (**gradient**),
- $\Delta \mathbf x$ is the iteration increment that takes each $\mathbf x^k$ closer to $\mathbf x^*$.

### Main components:
Without going into details, the following are important stages and considerations needed to implement a robust Newton's Method:

- **Evaluation of Element Derivatives:** First- and second-order derivatives of the elements contributions wrt the DoFs must be computed.
    These will be compact $N \times 1$ and $N \times N$ vectors and matrices.
- **Projection to Positive Definite:** To guarantee global descend towards a minimum, the local Hessians are typically projected to their closest Symmetric Positive Definite (SPD) approximation.
    This guarantees the global Hessian to also be SPD, which in turn guarantees descend step directions.
- **Assembly:** Elemental derivatives must be scattered in a global vector and a (typically sparse) matrix based on the DoFs they influence.
- **Linear System Solve:** Each iteration requires solving a global linear system to obtain a step.
    If numerically feasible, sparse iterative solvers, such as Conjugate Gradient, are preferred due to potentially lower runtimes.
    Otherwise, direct sparse solvers are used.
- **Line Search:** The step resulting from the linear solve is based _solely_ on the derivative information at $\mathbf x^k$.
    This step might overshoot the expected descent and end up at a high energy point, which can in turn diverge or stall the optimization process.
    Line search is a 1D analysis of the global potential profile along the proposed step that ensures that the next iterand lands at a lower energetic configuration.

## `GlobalPotential`

SymX's `GlobalPotential` represents the mathematical definition above for a specific problem, it holds a collection of potentials $P_p$ and DoFs $\mathbf x$.
Once the problem is defined, it is passed to `NewtonsMethod` to obtain a solution.

A `GlobalPotential` is instantiated via

```cpp
spGlobalPotential G = GlobalPotential::create();
```

### Adding Potentials
Each potential $P_p$ is added via `add_potential`, which takes a name, a connectivity array, and a definition callback.
```cpp
G->add_potential("new_potential", connectivity,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        // ... define potential ...
        return P;
    }
);
```
Potential function definitions use `MappedWorkspace` and `Element`, introduced in [Symbol-Data Maps](symbol_data.md), within a lambda.
This time however, these are managed by `GlobalPotential` in order to automate differentiation and evaluation later on.
Multiple potentials can be added freely, and together they form $G = \sum_p P_p$.

### Conditional Potentials
Some potentials are only active under certain conditions — penalties being a typical example.
For these, the user can return a `std::pair<Scalar, Scalar>` in the callback: the energy expression and a condition.
When the condition evaluates to zero or below for a given element, that element contribution is skipped entirely.
This is more efficient than using `branch` since we entirely avoid evaluating and assembling zero contributions.

```cpp
G->add_potential("new_potential_with_condition", connectivity,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        // ... define potential and condition ...
        return std::make_pair(P, cond);
    }
);
```


### Degrees of Freedom
DoFs are registered with `add_dof`, which accepts a wide range of container types: `std::vector` of fixed-size Eigen vectors or arrays, `Eigen::VectorXd`.
For instance:
```cpp
G->add_dof(data.x);  // std::vector<Eigen::Vector3d>
```
You can register multiple independent DoF sets; SymX flattens them into a single global vector internally and tracks their offsets.
At any point the flat DoF vector can be read and written:

```cpp
Eigen::VectorXd u(G->get_total_n_dofs());
G->get_dofs(u.data());             // copy current state out
G->set_dofs(u.data());             // overwrite the full state
G->apply_dof_increment(du.data()); // in-place update: x ← x + Δx
```

These operations are used internally at each Newton iteration, and they are accessible to the user.
