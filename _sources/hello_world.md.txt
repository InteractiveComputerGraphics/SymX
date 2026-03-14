# Hello World

Two self-contained SymX "hello world" applications: a low-level and a high-level one.

- **Low-level:** compile a single expression, set values, evaluate, print.
- **High-level:** declare a `GlobalPotential`, hand it to `NewtonsMethod`, solve.

## Low-Level
Compile a single function, evaluate it and print the result:

```cpp
#include <iostream>
#include <symx>
using namespace symx;

// Create a symbol environment
Workspace sws;

// Declare symbols
Scalar x = sws.make_scalar();

// Do math
Scalar dsinx_dx = diff(sin(x), x);

// Compile
Compiled<double> compiled({ dsinx_dx }, "hello_world", "../codegen");

// Set numeric data before evaluation
compiled.set(x, 0.0);

// Evaluate
View<double> result = compiled.run();
	
// Print  (output: "dsinx_dx(0.0) = 1.0")
std::cout << "dsinx_dx(0.0) = " << result[0] << std::endl;
```

Upon execution, symx will create the file `../codegen/hello_world.cpp` containing
```cpp
EXPORT
void hello_world(double* in, double* out)
{
	double v1 = std::cos(in[0]);
	out[0] = v1;
}
```
and its compiled counterpart `../codegen/hello_world.so` or `.dll`, depending on whether you are using a UNIX system or Windows.
The process always holds:

1. Create a workspace
2. Declare symbols
3. Do math
4. Compile the expressions
5. Set values
6. Run the compiled function
7. Process the output

Naturally, steps 1-4 can be done just once, and 5-7 can happen as many times as necessary.

## High-Level
Use Newton's Method to solve a linear spring elasticity problem with penalty boundary conditions:

```cpp
struct Data
{
    // Mesh
    std::vector<std::array<int, 2>> edges;
    std::vector<Eigen::Vector3d> x; // Current positions
    std::vector<Eigen::Vector3d> X; // Rest positions

    // Elasticity
    double elasticity_stiffness;

    // Boundary Conditions
    std::vector<std::array<int, 1>> bc_connectivity;
    double bc_stiffness;
} data;

// Define Global Potential
spGlobalPotential G = GlobalPotential::create();

//// Elasticity
G->add_potential("spring_elasticity_", data.edges,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        // Create symbols from data
        std::vector<Vector> x = mws.make_vectors(data.x, elem);
        std::vector<Vector> X = mws.make_vectors(data.X, elem);
        Scalar k = mws.make_scalar(data.elasticity_stiffness);

        // Define potential
        Scalar l      = (x[0] - x[1]).norm();
        Scalar l_rest = (X[0] - X[1]).norm();
        return 0.5*k*(l - l_rest).powN(2);
    }
);

//// Boundary conditions energy
G->add_potential("boundary_conditions", data.bc_connectivity,
    [&](MappedWorkspace<double>& mws, Element& elem)
    {
        // Create symbols from data
        Vector x = mws.make_vector(data.x, elem[0]);
        Vector X = mws.make_vector(data.X, elem[0]);
        Scalar k = mws.make_scalar(data.bc_stiffness);

        // Define potential
        Scalar P = 0.5*k*(x - X).squared_norm();
        return P;
    }
);

// DoF declaration
G->add_dof(data.x);

// Context
spContext context = Context::create();

// Initialize Newton solver
NewtonsMethod newton(G, context);  // <- Compilation occurs here

// Newton's Method parameters
newton.settings.residual_tolerance_abs = 1e-6;

// Solve
SolverReturn ret = newton.solve();
```

This example covers the entire solver pipeline:
- The problem is declared in a `GlobalPotential`.
- Differentiation occurs automatically based on the defined degrees of freedom (DoF). 
	Gradients and Hessians of each potential are derived.
- Elemental functions to evaluate potentials and derivatives are compiled.
- During execution, `NewtonsMethod` accesses user's data to read the current state, build global derivative data structures and compute the Newton steps via linear system solves.
	Progress is done by writing the solution in-place in the DoF array.




