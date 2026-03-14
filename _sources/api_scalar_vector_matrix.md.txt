# Symbol Reference API

Reference APIs for `Scalar`, `Vector` and `Matrix` as well as the `diff` facilities.

## `Scalar`

### Arithmetic Operations

| Method | Description |
|--------|-------------|
| `a + b` | Addition of two scalars |
| `a - b` | Subtraction of two scalars |
| `a * b` | Multiplication of two scalars |
| `a / b` | Division of two scalars |
| `a + val` | Add numeric constant |
| `a - val` | Subtract numeric constant |
| `a * val` | Multiply by numeric constant |
| `a / val` | Divide by numeric constant |
| `val + a` | Constant plus scalar |
| `val - a` | Constant minus scalar |
| `val * a` | Constant times scalar |
| `val / a` | Constant divided by scalar |
| `-a` | Unary negation |

### Assignment Operations

| Method | Description |
|--------|-------------|
| `a += b` | Add scalar and assign |
| `a -= b` | Subtract scalar and assign |
| `a *= b` | Multiply by scalar and assign |
| `a /= b` | Divide by scalar and assign |
| `a += val` | Add constant and assign |
| `a -= val` | Subtract constant and assign |
| `a *= val` | Multiply by constant and assign |
| `a /= val` | Divide by constant and assign |

### Mathematical Functions

| Method | Description |
|--------|-------------|
| `a.inv()` | Reciprocal (1/a) |
| `a.powN(n)` | Integer power |
| `a.powF(f)` | Floating point power |
| `a.sqrt()` | Square root |
| `a.exp()` | Exponential (e^x) |
| `a.ln()` | Natural logarithm |
| `a.log()` | Natural logarithm (alias for ln) |
| `a.log10()` | Base-10 logarithm |
| `a.sin()` | Sine |
| `a.cos()` | Cosine |
| `a.tan()` | Tangent |
| `a.asin()` | Arc sine |
| `a.acos()` | Arc cosine |
| `a.atan()` | Arc tangent |

### Free Function Versions

| Method | Description |
|--------|-------------|
| `powN(a, n)` | Integer power function |
| `powF(a, f)` | Floating point power function |
| `sqrt(a)` | Square root function |
| `exp(a)` | Exponential function |
| `ln(a)` | Natural logarithm function |
| `log(a)` | Natural logarithm function |
| `log10(a)` | Base-10 logarithm function |
| `sin(a)` | Sine function |
| `cos(a)` | Cosine function |
| `tan(a)` | Tangent function |
| `asin(a)` | Arc sine function |
| `acos(a)` | Arc cosine function |
| `atan(a)` | Arc tangent function |

### Conditional Operations

| Method | Description |
|--------|-------------|
| `branch(cond, pos, neg)` | Conditional expression with scalar branches |
| `branch(cond, pos, val)` | Conditional with scalar positive, constant negative |
| `branch(cond, val, neg)` | Conditional with constant positive, scalar negative |
| `branch(cond, val1, val2)` | Conditional with constant branches |
| `a > b` | Greater than comparison |
| `a < b` | Less than comparison |
| `a > val` | Greater than constant |
| `a < val` | Less than constant |
| `val > a` | Constant greater than scalar |
| `val < a` | Constant less than scalar |
| `abs(a)` | Absolute value |
| `sign(a)` | Sign function (-1, 0, or 1) |
| `min(a, b)` | Minimum of two scalars |
| `max(a, b)` | Maximum of two scalars |

### Value Operations

| Method | Description |
|--------|-------------|
| `set_value(val)` | Set the numeric value of a symbol |
| `get_value()` | Get the numeric value of a symbol |
| `eval()` | Evaluate the expression to a numeric value |
| `is_zero()` | Check if expression is exactly zero |
| `is_one()` | Check if expression is exactly one |

### Expression Structure

| Method | Description |
|--------|-------------|
| `left()` | Get left operand of binary operation |
| `right()` | Get right operand of binary operation |
| `has_right()` | Check if expression has a right operand |
| `has_not_right()` | Check if expression is unary |
| `has_branch()` | Check if expression is a conditional |
| `get_condition()` | Get condition of a branch expression |

### Symbol Information

| Method | Description |
|--------|-------------|
| `is_symbol()` | Check if expression is a basic symbol |
| `get_symbol_idx()` | Get the index of a symbol |
| `get_expression_graph()` | Get pointer to underlying expression graph |
| `get_checksum()` | Get hash checksum of the expression |

### Factory Methods

| Method | Description |
|--------|-------------|
| `get_zero()` | Create a zero constant |
| `get_one()` | Create a one constant |
| `make_constant(val)` | Create a constant with given value |
| `make_branch(c, p, n)` | Create a conditional expression |

### Utility

| Method | Description |
|--------|-------------|
| `print()` | Print expression for debugging (chainable) |

---

## `Vector`

### Construction

| Method | Description |
|--------|-------------|
| `Vector(values)` | Construct from vector of scalars |
| `Vector::zero(size, seed)` | Create zero vector of given size |

### Properties

| Method | Description |
|--------|-------------|
| `size()` | Get the size of the vector |
| `values()` | Get reference to underlying scalar values |

### Element Access

| Method | Description |
|--------|-------------|
| `operator[](i)` | Access element by index |
| `operator()(i)` | Access element by index (alternative syntax) |
| `segment(begin, n)` | Extract a segment of the vector |
| `set_segment(begin, n, other)` | Set a segment from another vector |

### Vector Operations

| Method | Description |
|--------|-------------|
| `dot(other)` | Dot product with another vector |
| `cross2(other)` | 2D cross product (returns scalar) |
| `cross3(other)` | 3D cross product (returns vector) |
| `squared_norm()` | Squared Euclidean norm |
| `norm()` | Euclidean norm |
| `normalized()` | Return normalized vector |
| `normalize()` | Normalize this vector in-place |

### Arithmetic Operations

| Method | Description |
|--------|-------------|
| `v + other` | Vector addition |
| `v - other` | Vector subtraction |
| `v * val` | Scalar multiplication |
| `v * scalar` | Multiplication by symbolic scalar |
| `v / val` | Scalar division |
| `v / scalar` | Division by symbolic scalar |
| `v += other` | Vector addition assignment |
| `v -= other` | Vector subtraction assignment |
| `v *= val` | Scalar multiplication assignment |
| `v *= scalar` | Symbolic scalar multiplication assignment |
| `v /= val` | Scalar division assignment |
| `v /= scalar` | Symbolic scalar division assignment |

### Free Functions

| Method | Description |
|--------|-------------|
| `val * v` | Scalar times vector |
| `scalar * v` | Symbolic scalar times vector |
| `-v` | Vector negation |
| `dot(a, b)` | Dot product of two vectors |

### Value Operations

| Method | Description |
|--------|-------------|
| `set_values(v)` | Set numeric values from array-like object |

---

## `Matrix`

### Construction

| Method | Description |
|--------|-------------|
| `Matrix(values, shape)` | Construct from values (row-major) and shape |
| `Matrix::zero(shape, seed)` | Create zero matrix of given shape |
| `Matrix::identity(size, seed)` | Create identity matrix |

### Properties

| Method | Description |
|--------|-------------|
| `rows()` | Get number of rows |
| `cols()` | Get number of columns |
| `shape()` | Get shape as array [rows, cols] |
| `values()` | Get reference to underlying scalar values |

### Element Access

| Method | Description |
|--------|-------------|
| `operator()(i, j)` | Access element at row i, column j |
| `block(r, c, nr, nc)` | Extract a block sub-matrix |
| `row(idx)` | Extract a row as a vector |
| `col(idx)` | Extract a column as a vector |
| `set_block(r, c, nr, nc, m)` | Set a block from another matrix |
| `set_row(idx, v)` | Set a row from a vector |
| `set_col(idx, v)` | Set a column from a vector |

### Matrix Operations

| Method | Description |
|--------|-------------|
| `transpose()` | Matrix transpose |
| `det()` | Determinant (for square matrices) |
| `inv()` | Matrix inverse |
| `inv_sym()` | Symmetric matrix inverse (optimized) |
| `trace()` | Matrix trace (sum of diagonal elements) |
| `frobenius_norm_sq()` | Squared Frobenius norm |
| `singular_values_2x2()` | Singular values for 2x2 matrix |
| `dot(other)` | Matrix multiplication with another matrix |
| `dot(vec)` | Matrix-vector multiplication |

### Arithmetic Operations

| Method | Description |
|--------|-------------|
| `m + other` | Matrix addition |
| `m - other` | Matrix subtraction |
| `m * val` | Scalar multiplication |
| `m * scalar` | Multiplication by symbolic scalar |
| `m * vec` | Matrix-vector multiplication |
| `m * other` | Matrix multiplication |
| `m / val` | Scalar division |
| `m / scalar` | Division by symbolic scalar |
| `m += other` | Matrix addition assignment |
| `m -= other` | Matrix subtraction assignment |
| `m *= val` | Scalar multiplication assignment |
| `m *= scalar` | Symbolic scalar multiplication assignment |
| `m /= val` | Scalar division assignment |
| `m /= scalar` | Symbolic scalar division assignment |

### Free Functions

| Method | Description |
|--------|-------------|
| `val * m` | Scalar times matrix |
| `scalar * m` | Symbolic scalar times matrix |
| `-m` | Matrix negation |
| `vec * m` | Vector-matrix multiplication (row vector) |
| `outer(a, b)` | Outer product of two vectors |

### Value Operations

| Method | Description |
|--------|-------------|
| `set_values_row_major(v)` | Set numeric values from row-major array |

---

## Differentiation (`diff`)

### Basic Differentiation

| Method | Description |
|--------|-------------|
| `diff(expr, wrt)` | Compute derivative of scalar expression |
| `diff(expr, wrt, cache)` | Compute derivative with caching |

### Gradient Computation

| Method | Description |
|--------|-------------|
| `gradient(expr, wrt)` | Gradient of scalar w.r.t. vector of variables |
| `gradient(expr, wrt, cache)` | Gradient computation with caching |
| `gradient(vec, wrt, symmetric)` | Jacobian of vector w.r.t. variables |
| `gradient(vec, wrt, symmetric, cache)` | Jacobian computation with caching |

### Hessian Computation

| Method | Description |
|--------|-------------|
| `hessian(expr, wrt)` | Hessian matrix of scalar expression |
| `hessian(expr, wrt, cache)` | Hessian computation with caching |

### Combined Operations

| Method | Description |
|--------|-------------|
| `value_gradient(expr, wrt)` | Compute both value and gradient |
| `value_gradient(expr, wrt, cache)` | Value and gradient with caching |
| `value_gradient_hessian(expr, wrt)` | Compute value, gradient, and Hessian |
| `value_gradient_hessian(expr, wrt, cache)` | All three with caching |



