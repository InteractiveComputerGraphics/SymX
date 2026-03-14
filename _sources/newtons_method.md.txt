# Newton's Method
In this section we discuss how to use SymX's Newton's Method infrastructure to efficiently find the solution to non-linear optimization problems.

## Overview
SymX provides facilities that will take care of everything other than the data and energy definition.
These are the new SymX entities required to set up the solve:
- `NewtonsMethod`: Main class the global potential and options are declared in. Hosts the solve.
- `GlobalPotential`: Collection of different potential and DoFs definitions.
- `Context`: Operational context. Gives access to `Logger`, `OutputSink` and options such as `n_threads`.
- `NewtonSettings`: Simple struct with all parameters used in the solve.
- `SolverCallbacks`: Set of customizable user-defined callbacks that will run at certain stages of the solve.

Definitions of potential functions must be declared in `GlobalPotential`, which will keep and manage the `MappedWorkspace`s internally.
Once all potentials and DoFs are declared in `GlobalPotential`, we can instantiate `NewtonsMethod`.
Upon initialization of `NewtonsMethod`, differentiation will occur for all potentials wrt to all DoFs.
Code generation and compilation will happen immediately afterwards.
If the compilation objects already exist in the `codegen` directory, differentiation, code generation and compilation are skipped, and the compiled object is directly loaded instead.
Parameters can then be set in `NewtonSettings` and callbacks can be added to `SolverCallbacks`.
Finally, the execution can occur.

Because SymX has access to the DoF arrays (pointer and size), it can autonomously update their values in-place until the solution is reached.
Multiple sets of degrees of freedom are allowed.

## Example
We will build a 3D static elasticity solver using a FEM discretization for the Stable NeoHookean constitutive model and boundary conditions with the penalty method.
The solver is compatible with the natively supported FEM element types SymX supports: `Tet3, Hex8, Tet10, Hex27`.
Note that we use _Stable_ NeoHookean for the upcoming simulation examples for improved stability.
A demo of solver can be found in [`rubber_extrusion.cpp`](../../examples/rubber_extrusion.cpp).

```cpp
struct Data
{
    // Mesh
    std::vector<int> connectivity;
    int stride;
    std::vector<Eigen::Vector3d> x; // Current positions
    std::vector<Eigen::Vector3d> X; // Rest positions

    // Elasticity
    double youngs_modulus;
    double poissons_ratio;

    // Boundary Conditions
    LabelledConnectivity<2> bc_connectivity{{ "constraint_idx", "vertex_idx" }};
    std::vector<Eigen::Vector3d> bc_target_positions;
    double bc_stiffness;
} data;

// Define Global Potential
spGlobalPotential G = GlobalPotential::create();

//// Stable Neo-Hookean material energy (generic on FEM element type)
G->add_potential("stable_neo_hookean_" + get_name(data.element_type), 
    data.connectivity, data.stride,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        // Create symbols from data
        std::vector<Vector> x = mws.make_vectors(data.x, elem);
        std::vector<Vector> X = mws.make_vectors(data.X, elem);
        Scalar E = mws.make_scalar(data.youngs_modulus);
        Scalar nu = mws.make_scalar(data.poissons_ratio);

        // Define potential
        Scalar P = fem_integrator(mws, data.element_type,
            [&](Scalar& w, Vector xi)
            {
                Matrix Jx = fem_jacobian(data.element_type, x, xi);
                Matrix JX = fem_jacobian(data.element_type, X, xi);
                Matrix F = Jx*JX.inv();
                return stable_neohookean_energy_density(F, E, nu)*JX.det()*w;
            }
        );
        return P;
    }
);

//// Boundary conditions energy
G->add_potential("boundary_conditions", data.bc_connectivity,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        // Create symbols from data
        Vector x = mws.make_vector(data.x, elem["vertex_idx"]);
        Vector target = mws.make_vector(data.bc_target_positions, elem["constraint_idx"]);
        Scalar k = mws.make_scalar(data.bc_stiffness);

        // Define potential
        Scalar P = 0.5*k*(x - target).squared_norm();
        return P;
    }
);

// DoF declaration
G->add_dof(data.x);

// Context
spContext context = Context::create();
context->compilation_directory = symx::get_codegen_dir();

// Initialize Newton solver
NewtonsMethod newton(G, context);  // <- Compilation occurs here

// Newton's Method parameters
newton.settings.step_tolerance = 0.001;  // Converged when corrections are < 1mm

// Solve
SolverReturn ret = newton.solve();
```

Highlights:
- Code generation and compilation will create files with different names depending on the FEM element type used.
- Structures can be declared empty in SymX up until `newton.solve()`.
    If the connectivity is actually of size zero during execution, the loop will be skipped.
    In large complex solvers, many energies might have empty connectivities as those are simply not activated. SymX will work as intended.
- The Newton search is terminated when the step size $\| \Delta \mathbf x \|_{\infty} < 10^{-3}$ m.

We use the solver above to stretch a block of almost incompressible elastic material ($\nu = 0.49$) five times its original size.
This is the outcome for 1.92M Tet10 elements totalling 7.89M DoFs:

<p align="center">
    <img src="_static/rubber_extrusion.jpg" alt="Extruded Rubber Simulation" style="width:95%;">
</p>


## Newton's Method implementation
SymX provides a comprehensive implementation of Newton's Method that is robust and battle-tested, and that is the core of the [STARK](https://github.com/InteractiveComputerGraphics/stark) simulation library. 
While it is naturally possible to use SymX in third-party solvers, the built-in optimizer provides the following advantages:
- The required derivatives are done automatically
- The cache system allows SymX to skip differentiation and compilation if the definitions don't change
- Assembly is taken cared of automatically
- DoF indexing is automatically handled, particularly useful when multiple sets are declared
- Output (log and console) and benchmark facilities are included

The Newton solve procedure can be customized via `NewtonSettings` and `SolverCallbacks`.

### Projection to PD
SymX supports four modes of projecting element Hessians to PD before assembly via `ProjectionToPD`:
- `ProjectionToPD::Newton`: Will not project and exit the solve if non-descend direction encountered.
- `ProjectionToPD::ProjectedNewton`: Will project all elements always (default).
- `ProjectionToPD::ProjectOnDemand`: Will project all elements only if indefiniteness is encountered.
- `ProjectionToPD::Progressive`: Iteratively finds a small and sufficient subset of elements for projection.

Progressive projection is recommended for well regularized problems and dynamic simulations as it can produce large savings from avoiding expensive projection. However, Projected Newton is set as default as it is well-established.


### Line Search
After the step is computed and it is verified that it descends the energy potential, SymX implements four stages to the line search:
- **Step Cap:** The user can define a clamping step length $s_{cap}$ so that $\tilde{\Delta \mathbf x} = \Delta \mathbf x \frac{\min(\| \Delta \mathbf x \|, \, s_{cap})}{\| \Delta \mathbf x \|}$. This could be used to limit huge steps that may arise far from the solution.
- **Max:** The maximum step can be dynamically set via a callback function. This could implement Continuous Collision Detection (CCD).
- **Backtracking Invalid State:** The step is iteratively halved based on a invalid states declared via callback functions. This can be used to avoid element inversions or penetrations.
- **Backtracking Potential Descend:** The step is iteratively halved based on the Armijo condition for sufficient descend.


### Settings
The following are all the settings exposed as `NewtonsMethod.settings`:

```cpp
struct NewtonSettings
{
    // Iteration limits
    int max_iterations = 100;
    int min_iterations = 0;

    // Convergence criteria
    double residual_tolerance_abs = 1e-6;
    double residual_tolerance_rel = 0.0;
    double step_tolerance = 0.0;
    bool max_iterations_as_success = false;

    // Line search
    double step_cap = std::numeric_limits<double>::infinity();
    bool enable_armijo_backtracking = true;
    double line_search_armijo_beta = 1e-4;
    int max_backtracking_armijo_iterations = 20;
    int max_backtracking_invalid_state_iterations = 8;
    bool print_line_search_upon_failure = false;

    // Hessian projection to PD
    ProjectionToPD projection_mode = ProjectionToPD::ProjectedNewton;
    double projection_eps = 1e-10;
    bool project_to_pd_use_mirroring = false;

    // Project-On-Demand
    int project_on_demand_countdown = 4;

    // Progressive Projected Newton (PPN)
    double ppn_tightening_factor = 0.5;
    double ppn_release_factor = 2.0;

    // Linear solver
    LinearSolver linear_solver = LinearSolver::BDPCG;
    int cg_max_iterations = 10000;
    double cg_abs_tolerance = 1e-12;
    double cg_rel_tolerance = 1e-4;
    bool cg_stop_on_indefiniteness = true;
    double bailout_residual = 1e-10;
};
```

Highlights:
- `min_iterations` can be used to enforce at least N corrections are applied, even if already converged.
- There are four main convergence options. Convergence is granted once one of them is true:
    - $\| \nabla_x G \| <$ `residual_tolerance_abs`
    - $\frac{\| \nabla_x G \|}{\| \nabla_x G_0 \|} <$ `residual_tolerance_rel`
    - $\| \Delta \mathbf x \|_{\infty} < $ `step_tolerance` (Newton step)
    - `max_iterations` reached and `max_iterations_as_success = true`
- `print_line_search_upon_failure = true` will evaluate energy potential profiles along the line search when Armijo backtracking fails.
    A file with the line data will be written for further processing and an interactive gnuplot session will start (if available).
- If `project_to_pd_use_mirroring = true` negative element Hessian eigenvalues are mirrored (`abs`), instead of clamped to `projection_eps`.
- While SymX supports `LinearSolver::DirectLLT` (Eigen), the default linear solver is Block Diagonal Preconditioned Conjugate Gradient (BDPCG). Further:
    - Convergence can be granted by absolute or relative residual, whatever happens first.
    - The solver can optionally report failure if an indefinite direction is found (needed for Progressively Projected Newton).
    - If $\| \nabla_x G \| <$ `bailout_residual`, the linear system is not executed and the Newton search is immediately declared successful. This is to avoid tiny residuals in the iterative solver which can cause issues.

### Callbacks
Logic can be inserted in the Newton's Method process by the use of callbacks.
This is the supported way to incorporate things like collision detection aware line search, custom termination criteria and so on.
The following are the callback containers used:
```cpp
class SolverCallbacks
{
    std::vector<std::function<void()>>    before_energy_evaluation;
    std::vector<std::function<bool()>>    is_initial_state_valid;
    std::vector<std::function<bool()>>    is_intermediate_state_valid;
    std::vector<std::function<void()>>    on_intermediate_state_invalid;
    std::vector<std::function<void()>>    on_armijo_fail;
    std::vector<std::function<bool()>>    is_converged;
    std::vector<std::function<bool()>>    is_converged_state_valid;
    std::vector<std::function<double()>>  max_allowed_step;
}
```

Brief descriptions:
- `before_energy_evaluation`: Will be executed before evaluating any value or derivative. Example: it can be used to update contact pairs.
- `is_initial_state_valid`: Exits the Newton process before it starts. Example: Element inversions in the rest pose.
- `is_intermediate_state_valid`: Used to backtrack invalid states in the line search. Example: element inversions as a consequence of a step.
- `on_intermediate_state_invalid`: Used if the invalid state could not be backtracked after the afforded number of iterations. Example: increase penalty stiffness.
- `on_armijo_fail`: Analogous if Armijo's backtracking fails. Example: restart the solve with a shorter time step size.
- `is_converged`: User's custom convergence criteria.
- `is_converged_state_valid`: Let Newton's Method return non-success right before return. Example: penalties are too soft, tighten them before requesting repeating the solve.
- `max_allowed_step`: Allows the user to specify a max step for the current iteration. Example: CCD.

Callbacks can be appended to those lists by using the corresponding `.add_<callback_name>(T f)` method, e.g. `.add_before_energy_evaluation(std::function<void()> f)`.
`SolverCallbacks` can be accessed in `NewtonsMethod.callbacks`.

Further, a custom residual can be specified by `.set_residual(std::function<double(Eigen::VectorXd&)> f)`.



### OutputSink
SymX provides nice console and file output options to monitor and log execution.
By default, the console output of a simulation like the rubber extrusion described above (albeit for a much smaller mesh) would look like this:

```
Second Order Potentials:
    boundary_conditions.............................................loaded
    stable_neo_hookean_Hex8.........................queued for compilation
Compiling... done.
Total time: 1.801947 s

Degrees of freedom:
Set 0: 7623
Total: 7623
     0. r0: 2.50e+09 | ph: 100.0% | CG:    6 | du: 6.6e-01 | 
     1. r0: 2.10e+08 | ph: 100.0% | CG:   21 | du: 1.4e-01 | 
     2. r0: 7.08e+07 | ph: 100.0% | CG:   28 | du: 1.1e-01 | 
     3. r0: 2.16e+07 | ph: 100.0% | CG:   29 | du: 1.7e-01 | 
     4. r0: 7.52e+06 | ph: 100.0% | CG:   29 | du: 1.1e-01 | 
     5. r0: 2.84e+06 | ph: 100.0% | CG:   28 | du: 6.3e-02 | 
     6. r0: 3.13e+06 | ph: 100.0% | CG:   23 | du: 6.0e-02 | 
     7. r0: 2.02e+06 | ph: 100.0% | CG:   24 | du: 7.8e-02 | 
     8. r0: 1.84e+06 | ph: 100.0% | CG:   24 | du: 5.1e-02 | 
     9. r0: 1.65e+06 | ph: 100.0% | CG:   25 | du: 3.5e-02 | 
    10. r0: 1.59e+06 | ph: 100.0% | CG:   27 | du: 3.2e-02 | 
    11. r0: 1.65e+06 | ph: 100.0% | CG:   30 | du: 5.1e-02 | 
    12. r0: 2.60e+06 | ph: 100.0% | CG:   28 | du: 4.0e-02 | 
    13. r0: 1.19e+06 | ph: 100.0% | CG:   39 | du: 2.9e-02 | 
    14. r0: 8.66e+05 | ph: 100.0% | CG:   26 | du: 1.3e-02 | 
    15. r0: 3.15e+05 | ph: 100.0% | CG:   31 | du: 8.8e-03 | 
    16. r0: 1.31e+05 | ph: 100.0% | CG:   28 | du: 7.4e-03 | 
    17. r0: 8.95e+04 | ph: 100.0% | CG:   28 | du: 7.2e-03 | 
    18. r0: 6.55e+04 | ph: 100.0% | CG:   27 | du: 3.6e-03 | 
    19. r0: 5.24e+04 | ph: 100.0% | CG:   27 | du: 1.3e-03 | 
    20. r0: 4.42e+04 | ph: 100.0% | CG:   27 | du: 1.2e-03 | 
    21. r0: 3.77e+04 | ph: 100.0% | CG:   35 | du: 1.0e-03 | 
    22. r0: 3.24e+04 | ph: 100.0% | CG:   35 | du: 9.0e-04 |
```

Highlights:
- Compilation/loading of the system is specified per potential definition along with the total diff, codegen and compilation runtime.
- The number of DoFs at the beginning of the solve is printed
- Succinct information of each Newton step is printed:
    - Iteration number
    - Initial residual
    - Percentage of projected Hessians
    - Number of CG iterations
    - Newton step length

This corresponds to the default verbosity level (`Verbosity::Medium`) although more options are available in `Context.output`.
Most notably, the output can be directed to a file (in addition to the console), and with a different verbosity level by
```cpp
context->output->open_file("path/to/log.log");
context->output->set_file_verbosity(Verbosity::Full);
```

Finally, a summary of the Newton process can be printed by
```cpp
newton.print_summary();
```

which outputs:

```
Solve                         Total      Avg      Min      Max
--------------------------------------------------------------
Newton iterations                22     22.0       22       22
CG iterations                   625     27.2        6       39
Line search cap                   0      0.0        0        0

Line search max                   0      0.0        0        0
Line search inv                   0      0.0        0        0
Line search bt                    0      0.0        0        0
Projected hessians                     100.0%   100.0%   100.0%
--------------------------------------------------------------

Runtime                                    Time (s)       %
------------------------------------------------------------
project_to_PD                              0.142926   48.4%
evaluate_P_grad_hess                       0.086382   29.3%
linear_system_solve                        0.040844   13.8%
assembly                                   0.019503    6.6%
evaluate_P                                 0.004625    1.6%
misc                                       0.001018    0.3%
------------------------------------------------------------
Total                                      0.295323  100.0%
```

Note that `cap`, `max`, `inv` and `bt` refer to the aforementioned line search stages.

### Log
SymX records key information in a log about the solve, such as several types of iterations and residuals as well as runtimes.
This log can be saved anytime as a yaml by
```cpp
context->logger->save_to_disk("path/to/yaml_log.yaml");
```

These logs are meant to be human readable but also are easy to load for further processing.
Here is part of the output yaml file for the previous example:
```yaml
timers:
  "assembly":
    total: 0.144621
    count: 55
    avg: 0.002629
    min: 0.002102
    max: 0.006113
  "linear_system_solve":
    total: 0.445042
    count: 55
    avg: 0.008092
    min: 0.002714
    max: 0.009327

statistics:
  "n_hessians": {total: 2.195600e+05, avg: 3.992000e+03, min: 3.992000e+03, max: 3.992000e+03, count: 55}
  "n_projected_hessians": {total: 2.195600e+05, avg: 3.992000e+03, min: 3.992000e+03, max: 3.992000e+03, count: 55}
  "projected_hessians_ratio": {total: 5.500000e+01, avg: 1.000000e+00, min: 1.000000e+00, max: 1.000000e+00, count: 55}
  "cg_iterations": {total: 6126, avg: 111.381818, min: 32, max: 126, count: 55}
  "ls_bt": {total: 0, avg: 0.000000, min: 0, max: 0, count: 54}
  "ls_cap": {total: 0, avg: 0.000000, min: 0, max: 0, count: 54}
  "ls_inv": {total: 0, avg: 0.000000, min: 0, max: 0, count: 54}
  "ls_max": {total: 0, avg: 0.000000, min: 0, max: 0, count: 54}
  "newton_iterations": {total: 54, avg: 54.000000, min: 54, max: 54, count: 1}

series:
  "n_hessians": [3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03]
  "n_projected_hessians": [3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03, 3.992000e+03]
  "projected_hessians_ratio": [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  "cg_iterations": [32, 47, 59, 86, 58, 113, 108, 119, 112, 121, 111, 125, 111, 111, 110, 111, 112, 113, 113, 112, 114, 113, 114, 112, 112, 112, 112, 112, 112, 113, 112, 112, 113, 113, 116, 115, 116, 118, 119, 119, 120, 120, 122, 123, 123, 123, 125, 125, 126, 126, 126, 126, 126, 126, 126]
  "ls_bt": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  "ls_cap": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  "ls_inv": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  "ls_max": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  "newton_iterations": [54]
```

