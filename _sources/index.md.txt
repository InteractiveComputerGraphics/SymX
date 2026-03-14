# SymX Documentation

<p align="center">
    <img src="_static/symx1920.png" alt="SymX Logo" style="width:75%;">
</p>

Welcome to the SymX documentation pages. 
SymX is a C++ library for **symbolic differentiation**, **code generation**, and **evaluation**, designed primarily for non-linear optimization in physics simulation and FEM.
Check out the [SymX GitHub repo](https://github.com/InteractiveComputerGraphics/SymX).

In these pages you will find an overview on how SymX works and how to use it through its levels of abstraction.
Every explanation comes with example snippets.
Note that this is more of a tutorial or a walkthrough than an extensive documentation in a traditional _Doxygen sense_.

Another important goal of this guide is to help you assess whether SymX can be useful to you.
There are many different ways to use symbolics, differentiation and just-in-time compilation, and SymX supports a few use cases out-of-the-box.
There are three entry points available in SymX:
1. **Single Expression:** Compose a symbolic expression (possibly differentiated), compile it, set values, run it, read the output.
2. **Stencil-based Loops:** Compose a symbolic expression (possibly differentiated) to be evaluated in a loop over many instances of an stencil. This is the core of FEM assemblers and iterative predictor/corrector solvers such as Jacobi or Gauss-Seidel.
3. **Non-linear Optimization:** Define potentials + discretizations + degrees of freedom and let SymX find the global solution using Newton's Method. It will take care of differentiation, compilation and evaluation automatically.


Depending on your use case, you might choose to skip parts of this tutorial.


## Table of Contents

```{toctree}
:caption: Getting Started
:maxdepth: 1

hello_world
setup
diagram
```

```{toctree}
:caption: Core Concepts (Layers 1, 2)
:maxdepth: 1

symbols
api_scalar_vector_matrix
compilation
```

```{toctree}
:caption: Symbol-Data Maps (Layer 3)
:maxdepth: 1

symbol_data
mapped_workspace
compiled_in_loop
```

```{toctree}
:caption: Second-Order Solver (Layer 4)
:maxdepth: 1

second_order_optimization
fem_integration
newtons_method
```

```{toctree}
:caption: Examples
:maxdepth: 1

examples
```
