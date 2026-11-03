#include "catch.hpp"

#include <vector>

#include <symx>
#include "utils.h"

using namespace symx;


TEST_CASE("substitute: leaf replacement matches directly-built expressions", "[substitute]")
{
    Workspace ws;
    Scalar x = ws.make_scalar();
    Scalar y = ws.make_scalar();
    Scalar a = ws.make_scalar();  // dof symbols the replacements are built from
    Scalar b = ws.make_scalar();

    a.set_value(0.37);
    b.set_value(-1.21);

    // Replacements: x -> a*a + 1,  y -> sin(b)
    Scalar gx = a * a + 1.0;
    Scalar gy = sin(b);

    SECTION("covers all the unary/binary/pow node types")
    {
        // A single expression exercising +,-,*,/,powN,powF,sqrt,ln,log10,exp,sin,cos,tan,inv
        auto build = [](const Scalar& X, const Scalar& Y) {
            Scalar e = (X * Y - X / (Y + 2.0)) + X.powN(3) + sqrt(X * X + 1.0);
            e = e + ln(X * X + 2.0) + log10(X * X + 3.0) + exp(0.1 * Y);
            e = e + sin(X) * cos(Y) + tan(0.2 * X) + X.powF(1.5) + (1.0 / (Y * Y + 1.0));
            return e;
        };

        Scalar subst = substitute(build(x, y), {x, y}, {gx, gy});
        Scalar direct = build(gx, gy);   // same expression built directly over the replacements

        REQUIRE(approx(subst.eval(), direct.eval(), 1e-12));
    }

    SECTION("partial substitution leaves untouched symbols intact")
    {
        x.set_value(2.0);
        Scalar e = x * y + 3.0;
        Scalar subst = substitute(e, {y}, {gy});  // only y replaced; x stays a leaf
        // expected: x * sin(b) + 3
        REQUIRE(approx(subst.eval(), 2.0 * std::sin(-1.21) + 3.0, 1e-12));
    }
}


TEST_CASE("substitute: chain rule cross-check", "[substitute]")
{
    // d/da [ f(g(a)) ] should equal  substitute( df/dx , x -> g ) * dg/da
    Workspace ws;
    Scalar x = ws.make_scalar();
    Scalar a = ws.make_scalar();
    a.set_value(0.7);

    Scalar f = sin(x) * x.powN(2) + ln(x + 3.0);  // f(x)
    Scalar g = a * a + 1.0;                        // x = g(a)

    Scalar fg = substitute(f, {x}, {g});           // f(g(a))
    Scalar lhs = diff(fg, a);                       // d/da f(g(a))

    Scalar df_dx = diff(f, x);
    Scalar rhs = substitute(df_dx, {x}, {g}) * diff(g, a);  // (df/dx|_{x=g}) * dg/da

    REQUIRE(approx(lhs.eval(), rhs.eval(), 1e-12));
}


TEST_CASE("substitute: symbolic exponent", "[substitute]")
{
    Workspace ws;
    Scalar x = ws.make_scalar();
    Scalar y = ws.make_scalar();
    Scalar a = ws.make_scalar();
    Scalar b = ws.make_scalar();
    a.set_value(1.4);
    b.set_value(0.3);

    Scalar substituted = substitute(x.powF(2.0 * y + 1.0), {x, y}, {a + 1.0, b});
    REQUIRE(approx(substituted.eval(), std::pow(2.4, 1.6), 1e-12));
}


TEST_CASE("substitute: branch nodes", "[substitute]")
{
    Workspace ws;
    Scalar x = ws.make_scalar();
    Scalar a = ws.make_scalar();
    a.set_value(1.5);

    Scalar e = branch(x, x * 2.0, x - 1.0);  // x>0 ? 2x : x-1
    Scalar g = a - 3.0;                       // = -1.5  -> negative branch
    Scalar subst = substitute(e, {x}, {g});

    REQUIRE(approx(subst.eval(), (-1.5) - 1.0, 1e-12));   // negative branch: g - 1

    a.set_value(5.0);                          // g = 2.0 -> positive branch
    REQUIRE(approx(subst.eval(), 2.0 * 2.0, 1e-12));      // positive branch: 2g
}


TEST_CASE("substitute: Vector and Matrix overloads", "[substitute]")
{
    Workspace ws;
    Matrix Fs = ws.make_matrix({2, 2});
    std::vector<Scalar> fs = Fs.values();

    std::vector<Vector> nodes = ws.make_vectors(2, 2);  // 4 dof scalars
    std::vector<Scalar> dofs = collect_scalars(nodes);
    for (int i = 0; i < (int)dofs.size(); i++) {
        dofs[i].set_value(0.1 * (i + 1));
    }

    // F_expr entries as some expressions of the dofs
    std::vector<Scalar> F_expr = {
        dofs[0] * dofs[1],
        dofs[2] + dofs[3],
        dofs[0] - dofs[2],
        dofs[1] * dofs[3] + 1.0,
    };

    // Psi(Fs) and its gradient/Hessian over fresh F-symbols
    Scalar Psi = Fs(0, 0) * Fs(1, 1) - Fs(0, 1) * Fs(1, 0) + sin(Fs(0, 0));
    Vector P = gradient(Psi, fs);     // 4
    Matrix A = hessian(Psi, fs);      // 4x4

    // Substitute Fs -> F_expr to get dof-expressions
    Vector P_dof = substitute(P, fs, F_expr);
    Matrix A_dof = substitute(A, fs, F_expr);

    // Reference: bind Fs to the evaluated F_expr and eval P/A directly
    for (int p = 0; p < (int)fs.size(); p++) {
        fs[p].set_value(F_expr[p].eval());
    }

    for (int i = 0; i < P.size(); i++) {
        REQUIRE(approx(P_dof[i].eval(), P[i].eval(), 1e-12));
    }
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            REQUIRE(approx(A_dof(i, j).eval(), A(i, j).eval(), 1e-12));
        }
    }
}


TEST_CASE("substitute: shared subexpressions handled once (DAG)", "[substitute]")
{
    // Repeated squaring builds an expression whose equivalent *tree* has 2^depth leaves but
    // whose *DAG* is only ~depth nodes (CSE shares e*e). substitute() must be linear in the
    // DAG size via its memo cache.
    Workspace ws;
    Scalar x = ws.make_scalar();
    Scalar a = ws.make_scalar();
    a.set_value(0.85);

    // (1) Check the substituted expression numerically at a depth with a simple reference value.
    {
        Scalar e = x;
        for (int i = 0; i < 12; i++) {
            e = e * e * 0.5;   // contracts toward 0, stays finite
        }
        Scalar subst = substitute(e, {x}, {a * 0.5 + 0.1});

        double ref = a.eval() * 0.5 + 0.1;
        for (int i = 0; i < 12; i++) { ref = ref * ref * 0.5; }

        REQUIRE(std::isfinite(subst.eval()));
        REQUIRE(approx(subst.eval(), ref, 1e-12));
    }

    // (2) Substitution should visit each shared node once, even for a depth-60 DAG.
    // Evaluation is memoized too, but this case focuses on constructing the substituted DAG.
    {
        Scalar e = x;
        for (int i = 0; i < 60; i++) {
            e = e * e * 0.5;
        }
        Scalar subst = substitute(e, {x}, {a});
        REQUIRE(subst.expr_id >= 0);
    }
}
