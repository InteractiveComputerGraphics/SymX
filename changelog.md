# Changelog

## 1.1.1

- Added factorized FEM kernels with exact derivatives for node-local position maps.
- Added per-point quadrature data and variable-length summations per element.
- Added symbolic substitution for scalars, vectors, and matrices.
- Added per-potential energy and gradient breakdowns.

## 1.1.0

- Added stencil-based differentiable simulation with adjoint gradients,
  finite-difference validation, and cantilever inverse examples.
- Added adjoint first-order optimizers with GradientDescent, Adam,
  VectorAdam, and L-BFGS.
- Improved solver callbacks, status reporting, and rollback.
- Reused sparse storage and symbolic analysis across direct LLT solves.
- Added optional oneMKL/PARDISO support and the `SYMX_STACK_SIZE` build option.
- Added first-order and adjoint documentation, plus expanded solver and
  inverse-problem tests.
- Updated and expanded docs.

## 1.0.1

- Defaulted `SYMX_HESS_STORAGE_FLOAT` to `double`.
- Clarified branch behavior and added boundary tests.
- Updated bundled Eigen to 5.0.1.
- Added CI coverage.

## 1.0.0

Initial public release.
