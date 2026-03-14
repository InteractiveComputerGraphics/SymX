# Examples
SymX comes with a collection of executable examples and benchmarks to demonstrate key features.
These can be found in the `examples/` directory.
Each one can be toggled on or off in [`examples_main.cpp`](../../examples/examples_main.cpp) by flipping the corresponding boolean.
Examples are ordered by level of abstraction, from low-level (`Compiled`) to high-level (`NewtonsMethod`).

SymX includes light mesh generation utilities in `symx/src/solver/mesh_generation.h` — triangle grids, cuboid tetrahedral meshes, and several higher-order variants are available out of the box.
Examples that produce geometry write VTK files to `output/`.
The free software [Paraview](https://www.paraview.org/) can be used to inspect these VTK files.
Further, you can use the [Sequence Loader](https://github.com/InteractiveComputerGraphics/blender-sequence-loader) Blender addon to view and render VTK files and sequences.

---

## Hello World

**File:** [`hello_world.cpp`](../../examples/hello_world.cpp)

The minimal SymX workflow: declare a symbol, compose an expression, differentiate, compile, evaluate, print.

---

## NeoHookean — Print

**File:** [`NeoHookean.cpp`](../../examples/NeoHookean.cpp)

Derives the 12×1 gradient and 12×12 Hessian of the NeoHookean energy for a linear tetrahedral element (Tet4) and prints the result for a single element.
A good reference to understand the low-level symbolic FEM workflow — from expression construction and differentiation to compilation and evaluation.

---

## NeoHookean — Evaluation Benchmark

**File:** [`NeoHookean.cpp`](../../examples/NeoHookean.cpp)

Measures per-element Hessian evaluation throughput across all float configurations supported by SymX:
`double`, `float`, `__m256d` (SIMD4d), and `__m256` (SIMD8f), both single-threaded and multi-threaded.
Compiled objects are cached and reused across runs unless the expression changes.

Establishes SymX evaluation performance and how it scales to available hardware.

---

## Triangle Mesh Area

**File:** [`triangle_mesh_area.cpp`](../../examples/triangle_mesh_area.cpp)

Sums the areas of all triangles in a mesh by running a single `CompiledInLoop` over the triangle connectivity.
This is the minimal `MappedWorkspace` + `CompiledInLoop` example: the expression is trivial, so the focus is entirely on the data-mapping and loop-execution pattern.
Showcase for how to declare loops with compiled expressions that can be evaluated without manually setting values.

---

## XPBD Cloth Simulation

![XPBD cloth simulation](_static/xpbd_cloth.jpg)

**File:** [`xpbd_cloth.cpp`](../../examples/xpbd_cloth.cpp)

A piece of cloth falling onto a sphere, simulated with XPBD.
The simulation models edge-length springs, membrane strain, dihedral bending, boundary conditions, and sphere contact with friction.

The comparison runs four configurations: sequential and multi-threaded, in both `double` and `float` (with their SIMD backends `__m256d`/`__m256`).

This is the main showcase for `CompiledInLoop` in a non-FEM setting.
Each constraint type is its own compiled expression, all defined compactly using `MappedWorkspace`.
It shows that SymX can be effectively used in applications beyond second-order solvers or generally reliant on heavy differentiation.

---

## NeoHookean — Global Assembly Benchmark

**File:** [`NeoHookean.cpp`](../../examples/NeoHookean.cpp)

Assembles the global potential, gradient, and Hessian of the NeoHookean energy over a Tet4 mesh.
The benchmark sweeps across:

- Sequential vs. parallel evaluation
- `double` vs. `__m256d`
- Eigen's sparse assembler vs. SymX's built-in assembler
- With and without projection to positive semi-definiteness (PSD)

Showcases assembly-style pipelines.
Useful for understanding performance and the trade-offs between assembler back-ends.

---

## Rubber Extrusion

![Rubber extrusion](_static/rubber_extrusion.jpg)

**File:** [`rubber_extrusion.cpp`](../../examples/rubber_extrusion.cpp)

Quasistatic simulation of a block of nearly-incompressible rubber (ν = 0.49) being stretched to five times its length.
The solver finds the solution for FEM element types Tet4, Hex8, Tet10, and Hex27.

Key things to observe:
- Usage of `fem_integrator` for abstract element types.
- Newton parametrization for stiff problems: projection to PD, eigval mirroring, convergence.
- Solver callbacks used to write intermediate VTK frames.
- Structured logging.

Showcases how to use the high-level `NewtonsMethod` to solve FEM problems compactly and with minimal involvement from the user.

---

## Dynamic Elasticity with Contact

![Dynamic elasticity with contact](_static/dynamic_with_contact.jpg)

**File:** [`dynamic_elasticity_with_contact.cpp`](../../examples/dynamic_elasticity_with_contact.cpp)

An elastic Stanford bunny is thrown in the air, hits the ground, and rolls.
The simulation uses implicit (backward Euler) time integration, stable Neo-Hookean elasticity and IPC-style frictional contact with the floor.

Each time step is solved with Newton's Method on a `GlobalPotential` that includes inertia, elasticity, contact, and friction terms — all defined symbolically through the same callback-based API.

