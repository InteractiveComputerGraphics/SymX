# Architecture Overview

SymX is organized in distinct layers of abstraction.
SymX can be used to compile a single symbolic expression, and also to solve large and complex non-linear optimization problems.
Most of the provided facilities are used internally between layers; however, depending on your use case, you may only ever need to interact with the top one or two.
Having a mental map of these layers makes the whole system much easier to navigate.

## The Abstraction Layers

The following is a simplified and non-exhaustive view of SymX's structure:

```{mermaid}
flowchart BT

classDef expr  fill:#dcfce7,stroke:#22c55e,stroke-width:1.5px,color:#064e3b
classDef comp  fill:#dbeafe,stroke:#3b82f6,stroke-width:1.5px,color:#1e3a8a
classDef map   fill:#fef9c3,stroke:#eab308,stroke-width:1.5px,color:#713f12
classDef solve fill:#ffe4e6,stroke:#fb7185,stroke-width:1.5px,color:#881337

CO["Compiled&lt;T&gt;"]:::comp

WS["Workspace"]:::expr
SC["Scalar"]:::expr
VE["Vector"]:::expr
MA["Matrix"]:::expr

MW["MappedWorkspace&lt;T&gt;"]:::map
CIL["CompiledInLoop&lt;T&gt;"]:::map
DM["DataMap&lt;T&gt;"]:::map

POT["Potential"]:::solve
GP["GlobalPotential"]:::solve
NM["NewtonsMethod"]:::solve

WS ==>|"creates"| SC
WS ==>|"creates"| VE
WS ==>|"creates"| MA

CO -. "access" .-> WS

MW ==>|"has one"| WS
MW ==>|"has many"| DM
CIL ==>|"has one"| MW

GP ==>|"has many"| POT
NM ==>|"has one"| GP
NM ==>|"has many"| CIL
```

## Layer 1 (Green) · Expression Building

- **Representative Classes:** `Scalar`, `Vector`, `Matrix`
- **Use case:** You will always interact with this layer.

Regardless of which entry point you choose, you will need to use symbols to build expressions.
You declare symbolic variables — `Scalar`, `Vector`, and `Matrix` — and then compose them using ordinary arithmetic, calculus, and linear algebra operators.
These are managed by a `Workspace`.

## Layer 2 (Blue) · Low–level Compilation

- **Representative Classes:** `Compiled<T>`
- **Use case:** You need a single, standalone compiled function.

`Compiled<T>` takes a list of symbolic expressions and compiles them into a fast native shared library (`.so` / `.dll`).
You feed in numeric values via `set(symbol, value)`, call `run()`, and read back results through a lightweight `View<T>` wrapper.
This is the raw mechanism that every higher layer builds on internally, and it is the right entry point when you want to slot a specific SymX-compiled function into an existing application.

## Layer 3 (Yellow) · Symbol–Data Maps

- **Representative Classes:** `MappedWorkspace<T>`, `CompiledInLoop<T>`
- **Use case:** You want to evaluate expressions over a discretized domain.

`MappedWorkspace<T>` lets you create symbols bound to your data containers via `DataMap<T>` (internal).
This allows SymX to evaluate loops completely automatically via `CompiledInLoop<T>`.
You no longer set values manually; SymX holds references (through lambdas) and knows exactly which memory to read for each loop iteration at evaluation time.
Parallelism (OpenMP), SIMD batching, and gather/scatter are handled internally.
You process the output in a (inlined) callback function.

This layer is still generic and powerful, but it is made convenient by handling loop evaluation and hardware optimizations.

## Layer 4 (Red) · Second–Order Solver

- **Representative Classes:** `NewtonsMethod`, `GlobalPotential`
- **Use case:** You want SymX to solve a second-order optimization problem end-to-end.

You declare energy potentials and degrees of freedom in a `GlobalPotential`.
Each potential is defined by a connectivity array and symbolic expression.
`NewtonsMethod` then takes care of everything else: symbolic differentiation, code generation, compilation (with caching), element-level assembly, projection to positive-definite Hessians, global linear system solve, and line search.
