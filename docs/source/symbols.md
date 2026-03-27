# Symbols

Symbols are the building blocks of SymX expressions.
There are three fundamental symbolic user-facing types: `Scalar`, `Vector`, and `Matrix`.

## The `Workspace`

Symbols are created through a `Workspace`, which manages the underlying expression graph (hidden from the user).
Here is how to create a `Scalar`, `Vector` and `Matrix`

```cpp
#include <symx>
using namespace symx;

Workspace ws;
Scalar x = ws.make_scalar();
Vector v = ws.make_vector(3);        // 3D vector
Matrix M = ws.make_matrix({2, 3});   // 2×3 matrix
```

`Scalar`, `Vector` and `Matrix` are convenient user-friendly wrappers that point to internal representations within the expression graph.
As these types are used to compose expressions, the internal expression graph grows.
SymX uses _Common Subexpression Elimination_ (CSE), a technique to reduce expression complexity, by checking if an expression to be created already exists in the graph.

**Important:**
* Symbols from different `Workspace`s cannot be mixed.
* Working on a expression graph (including differentiation) is **not** thread-safe. However, threads can work on different `Workspace`s concurrently.


## `Scalar`
`Scalar` represent real-valued mathematical expressions.
It implements standard arithmetics via operator overload (`+, -, *, /, +=, -=, *=, /=`) as well as common operations such as `sqrt(), inv(), log10(), sin()` and more.
Scalars can be created in different ways:

```cpp
Workspace ws;
Scalar a = ws.make_scalar();                    // Creates a symbolic scalar
std::vector<Scalar> as = ws.make_scalars(5); // Creates five symbolic scalars
Scalar zero = ws.get_zero();                 // Gets the symbolic zero
Scalar one = ws.get_one();                   // Gets the symbolic one
```
Note that zero and one are **special values** in SymX that, among other things such as simplifications, do not create symbols.

### Branching
A special operation that `Scalar` supports is `branch`.
By using 
```cpp
Scalar a = branch(cond, true_branch, false_branch);
```
we can define functions by parts in SymX. Here is an example for a cubic penalty potential:
```cpp
Scalar cubic_penalty(const Scalar& violation, const Scalar& stiffness)
{
    return branch(violation > 0.0, stiffness*violation.powN(3)/3.0, 0.0);
}
```

These can be differentiated and their generated code will feature actual `if-else` statements.
Operations such as `min(), max()` use `branch` internally.
Note that the use of `branch` prevents emitting SIMD code.

**Important:** The condition passed to `branch` is a `Scalar` whose **sign** determines which branch executes.
When you write `a > b`, SymX internally stores `a - b` as the condition scalar.
Opposite for `a < b`.
The true branch executes when that condition scalar is **strictly positive** (`> 0`), exact zero will then fall to the false branch.
Due to the floating point nature of the comparison, the user might consider writing activation mechanisms with conditions that are `+1` and `-1` (instead of `0`) to more clearly indicate sign.

## `Vector` and `Matrix`
`Vector` and `Matrix` are simply a list of `Scalar`s that provide typical algebraic operator overloads.
Further, `Vector` provides `norm(), dot(), cross3()` and such, while `Matrix` provides `det(), inv(), trace()` etc.
Both are dynamic types meaning that their sizes can be defined at runtime.

Both `Vector` and `Matrix` can be created, as well as being constructed from already existing `Scalar`s or being made of zeros and ones:
```cpp
Workspace ws;
Scalar x = ws.make_scalar();
Scalar y = ws.make_scalar();

// Vector
Vector v = ws.make_vector(3);                            // Creates a symbolic 3D vector
std::vector<Vector> vs = ws.make_vectors(3, 4);          // Creates four symbolic 3D vectors
Vector v_zero = ws.get_zero_vector(2);                   // Gets a vector with two zeros
Vector xy = Vector({x, y});                              // Makes a vector from existing symbols

// Matrix
Matrix m = ws.make_matrix({2, 3});                       // Creates a 2x3 symbolic matrix
std::vector<Matrix> ms = ws.make_matrices({2, 3}, 2);    // Creates two 2x3 symbolic matrices
Matrix I = ws.get_identity_matrix(3);                    // Gets the 3D identity matrix
Matrix m_zero = ws.get_zero_matrix({2, 3});              // Gets a 2x3 matrix of zeros
Matrix xxyy = Matrix({x, x, y, y}, {2, 2});              // Make a matrix from existing symbols (row major)
```
Prefer *getting* zero or identity entities to fill them over *making* them, which creates symbols that will go unused.


Here you can see an example on how SymX can compactly write the NeoHookean energy density function of a 3D object:
```cpp
Scalar neohookean_energy_density(const Matrix& F, const Scalar& l, const Scalar& m)
{
    Matrix C = F.transpose()*F;
    Scalar Ic = C.trace();
    Scalar logdetF = log(F.det());
    return 0.5*m*(Ic - 3) - m*logdetF + 0.5*l*logdetF.powN(2);
}
```


## Differentiation
Differentiation is one of the central features of SymX.
A `Scalar` expression can be differentiated with respect to another `Scalar` with the `diff` operator.
SymX supports multivariate and arbitrarily nested differentiation:

```cpp
Workspace ws;
Scalar x = ws.make_scalar();
Scalar y = ws.make_scalar();

// Simple polynomial
Scalar f = x*x + 3*x + 2;
Scalar df_dx = diff(f, x);  // Result: 2*x + 3

// Multivariate function
Scalar g = x*x + y*y + x*y;
Scalar dg_dx = diff(g, x);  // Result: 2*x + y
Scalar dg_dy = diff(g, y);  // Result: 2*y + x

// High-order differentiation
Scalar d2g_dydx = diff(dg_dy, x);  // Result: 1
```

Some convenient higher-level differentiation routines are:
```cpp
Vector gradient(const Scalar& expr, const Vector& wrt);
Matrix gradient(const Vector& vec, const Vector& wrt, const bool symmetric);
Matrix hessian(const Scalar& expr, const Vector& wrt);
std::vector<Scalar> value_gradient_hessian(const Scalar& expr, const std::vector<Scalar>& wrt);
```

Note that SymX implements _scalar differentiation_ exclusively, and therefore tensor calculus identities will not be applied.

To significantly speed up differentiation, SymX internally uses a `DiffCache` to reuse partial results, which is very impactful in high-order differentiation.
If multiple differentiation calls are applied for symbols in the same `Workspace`, create your own `DiffCache` and pass it to all the differentiation calls to share the pool of intermediate results.

## FEM Example
A common operation in the Finite Element Method (FEM) is to evaluate the gradient or Hessian of an element's energy potential for assembly.
We can calculate these very easily in SymX as shown in the following example:

```cpp
// Lambda function to obtain the jacobian of a Tet4 element given its vertices
auto jac_tet4 = [](Workspace& ws, const std::vector<Vector>& xh)
{
    Matrix J = ws.get_identity_matrix(3);
    J.set_col(0, xh[1] - xh[0]);
    J.set_col(1, xh[2] - xh[0]);
    J.set_col(2, xh[3] - xh[0]);
    return J;
};

// SymX workspace
Workspace ws;

// Create symbols
std::vector<Vector> x = ws.make_vectors(3, 4);
std::vector<Vector> X_rest = ws.make_vectors(3, 4);
Scalar lambda = ws.make_scalar();
Scalar mu = ws.make_scalar();

// FEM potential
Matrix Jx = jac_tet4(ws, x);
Matrix JX = jac_tet4(ws, X_rest);
Matrix F = Jx*JX.inv();
Scalar vol = JX.det()/6.0;
Scalar E = vol*neohookean_energy_density(F, lambda, mu);

// Differentiation
DiffCache diff_cache;
std::vector<Scalar> dofs = collect_scalars(x);

Vector grad = gradient(E, dofs, diff_cache);
Matrix hess = gradient(grad, dofs, /* is_symmetric = */ true, diff_cache);

// ...downstream...
```

**Important:** This is not the most convenient way to write a FEM solver in SymX.
Using the right level of abstraction (which will be shown later), SymX can autonomously compute derivatives, evaluate expressions, assemble sparse systems and use efficient linear solvers.
This example is meant to illustrate the symbolic work in a non-trivial scenario.


