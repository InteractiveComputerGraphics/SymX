# SymX Documentation

<p align="center">
    <img src="_static/symx1920.png" alt="SymX Logo" style="width:75%;">
</p>

Welcome to the SymX documentation pages. 
SymX is a C++ library for **symbolic differentiation**, **code generation**, and **evaluation**, designed primarily for non-linear optimization, FEM, and differentiable simulation.
Check out the [SymX GitHub repo](https://github.com/InteractiveComputerGraphics/SymX).

In these pages you will find an overview on how SymX works and how to use it through its levels of abstraction.
Every explanation comes with example snippets.
Note that this is more of a tutorial or a walkthrough than an extensive documentation in a traditional _Doxygen sense_.

Another important goal of this guide is to help you assess whether SymX can be useful to you.
There are many different ways to use symbolics, differentiation and just-in-time compilation, and SymX supports a few use cases out-of-the-box.
There are four entry points available in SymX:
1. **Single Expression:** Compose a symbolic expression (possibly differentiated), compile it, set values, run it, read the output.
2. **Stencil-based Loops:** Compose a symbolic expression (possibly differentiated) to be evaluated in a loop over many instances of an stencil. This is the core of FEM assemblers and iterative predictor/corrector solvers such as Jacobi or Gauss-Seidel.
3. **Global Optimization:** Define potentials, discretizations, and degrees of freedom and let SymX minimize a global objective with Newton's Method or a first-order optimizer. It takes care of differentiation, compilation, and evaluation automatically.
4. **Differentiable Simulation:** Define an equilibrium simulation and a loss on its outcome. SymX differentiates through the converged solve with the adjoint method, making inverse problems with many parameters practical.


Depending on your use case, you might choose to skip parts of this tutorial.


## Table of Contents

```{toctree}
:caption: Getting Started
:maxdepth: 1

hello_world
setup
architecture_overview
```

```{toctree}
:caption: Core Concepts (Layers 1, 2)
:maxdepth: 1

symbols
symbol_reference_api
compilation
```

```{toctree}
:caption: Symbol-Data Maps (Layer 3)
:maxdepth: 1

symbol_data_overview
symbol_data_maps
loops
```

```{toctree}
:caption: Global Optimization (Layer 4)
:maxdepth: 1

global_optimization
fem_integration
newtons_method
first_order_optimization
advanced_integration
```

```{toctree}
:caption: Differentiable Simulation (Layer 5)
:maxdepth: 1

adjoint_method
inverse_simulation
```

```{toctree}
:caption: Examples
:maxdepth: 1

examples
```
