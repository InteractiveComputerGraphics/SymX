#include "catch.hpp"

#include <symx>
#include "sympy_derivatives.h"
#include "utils.h"

using namespace symx;


TEST_CASE("Derivatives", "[derivatives]")
{
	const double A = 0.2;
	const double X = 0.3;
	const double B = 0.4;
	const double C = 0.5;
	const double D = 0.6;

	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar x = ws.make_scalar();
	Scalar b = ws.make_scalar();
	Scalar c = ws.make_scalar();
	Scalar d = ws.make_scalar();
	a.set_value(A);
	x.set_value(X);
	b.set_value(B);
	c.set_value(C);
	d.set_value(D);

    SECTION("Single Expression")
    {
		REQUIRE(approx(diff(a + b, a).eval(), 1.0));
		REQUIRE(approx(diff(a + b, b).eval(), 1.0));
		REQUIRE(approx(diff(1.0 + b, b).eval(), 1.0));
		REQUIRE(approx(diff(a + 1.0, b).eval(), 0.0));
		REQUIRE(approx(diff(a - b, a).eval(), 1.0));
		REQUIRE(approx(diff(a - b, b).eval(), -1.0));
		REQUIRE(approx(diff(a * b, a).eval(), B));
		REQUIRE(approx(diff(a * b, b).eval(), A));
		REQUIRE(approx(diff(a / b, a).eval(), 1.0 / B));
		REQUIRE(approx(diff(a / b, b).eval(), -A / std::pow(B, 2)));
		REQUIRE(approx(diff(0.0 / b, b).eval(), 0.0));
		REQUIRE(approx(diff(a.powN(2), a).eval(), 2.0 * A));
		REQUIRE(approx(diff(sqrt(a), a).eval(), 0.5 / std::sqrt(A)));
		REQUIRE(approx(diff(ln(a), a).eval(), 1.0 / A));
		REQUIRE(approx(diff(log(a), a).eval(), 1.0 / A));
		REQUIRE(approx(diff(log10(a), a).eval(), 1.0 / (A * std::log(10.0))));
		REQUIRE(approx(diff(exp(a), a).eval(), std::exp(A)));
		REQUIRE(approx(diff(sin(a), a).eval(), std::cos(A)));
		REQUIRE(approx(diff(cos(a), a).eval(), -std::sin(A)));
		REQUIRE(approx(diff(tan(a), a).eval(), 1.0 + std::tan(A) * std::tan(A)));
		REQUIRE(approx(diff(asin(a), a).eval(), 1.0 / std::sqrt(1.0 - A * A)));
		REQUIRE(approx(diff(acos(a), a).eval(), -1.0 / std::sqrt(1.0 - A * A)));
		REQUIRE(approx(diff(atan(a), a).eval(), 1.0 / (1.0 + A * A)));
    }

    SECTION("Double Expressions")
    {
		REQUIRE(approx(diff(a * b + c * d, a).eval(), B));
		REQUIRE(approx(diff(a * b - c * d, c).eval(), -D));
		REQUIRE(approx(diff(a * b * c * d, a).eval(), B * C * D));
		REQUIRE(approx(diff(a * b / (c * d), a).eval(), B / (C * D)));
		REQUIRE(approx(diff(a * b / (c * d), c).eval(), -A * B / D / std::pow(C, 2)));
		REQUIRE(approx(diff(sqrt(a * b), a).eval(), 0.5 / std::sqrt(A * B) * B));
		REQUIRE(approx(diff(ln(a * b), a).eval(), B / (A * B)));
		REQUIRE(approx(diff(log(a * b), a).eval(), B / (A * B)));
		REQUIRE(approx(diff(log10(a * b), a).eval(), B / (A * B * std::log(10.0))));
		REQUIRE(approx(diff(exp(a * b), a).eval(), std::exp(A * B) * B));
		REQUIRE(approx(diff(sin(a * b), a).eval(), std::cos(A * B) * B));
		REQUIRE(approx(diff(cos(a * b), a).eval(), -std::sin(A * B) * B));
		REQUIRE(approx(diff(tan(a * b), a).eval(), (1.0 + std::tan(A * B) * std::tan(A * B)) * B));
    }

    SECTION("Point-triangle distance")
    {
		Workspace ws;

		// Degrees of freedom
		Vector a = ws.make_vector(3);
		Vector b = ws.make_vector(3);
		Vector c = ws.make_vector(3);
		Vector p = ws.make_vector(3);

		// Known variables
		Scalar sign = ws.make_scalar();

		// Expression
		Vector ap = p - a;
		Vector ca = a - c;
		Vector cb = b - c;
		Vector n = ca.cross3(cb).normalized();
		Scalar C_tp = sign * ap.dot(n);

		// Evaluate
		const Eigen::Vector3d A = Eigen::Vector3d::Random();
		const Eigen::Vector3d B = Eigen::Vector3d::Random();
		const Eigen::Vector3d C = Eigen::Vector3d::Random();
		const Eigen::Vector3d P = Eigen::Vector3d::Random();
		const double S = -1.0;
		for (int i = 0; i < 3; i++) {
			a[i].set_value(A[i]);
			b[i].set_value(B[i]);
			c[i].set_value(C[i]);
			p[i].set_value(P[i]);
		}
		sign.set_value(S);

		// Derivatives
		std::vector<Scalar> variables = { a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], p[0], p[1], p[2] };
		Vector grad = gradient(C_tp, variables);
		Matrix hess = hessian(C_tp, variables);

		// Sympy
		std::vector<double> in(13);
		for (int i = 0; i < 3; i++) {
			in[i] = A[i];
			in[3 + i] = B[i];
			in[6 + i] = C[i];
			in[9 + i] = P[i];
		}
		in[12] = S;
		std::vector<double> out(157);
		sympy_point_triangle_distance_derivatives(in.data(), out.data());

		// Test
		REQUIRE(approx(C_tp.eval(), out[0]));

		for (int i = 0; i < 12; i++) {
			REQUIRE(approx(grad[i].eval(), out[1 + i]));
		}

		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				REQUIRE(approx(hess(i, j).eval(), out[13 + i * 12 + j]));
			}
		}
	}
}

TEST_CASE("Gradient operations", "[derivatives]")
{
	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	Scalar c = ws.make_scalar();
	
	const double A = 0.2;
	const double B = 0.4;
	const double C = 0.6;
	
	a.set_value(A);
	b.set_value(B);
	c.set_value(C);
	
	SECTION("Simple gradient")
	{
		Scalar expr = a * b + c.powN(2);
		std::vector<Scalar> vars = {a, b, c};
		Vector grad = gradient(expr, vars);
		
		REQUIRE(grad.size() == 3);
		REQUIRE(approx(grad[0].eval(), B));  // d/da = b
		REQUIRE(approx(grad[1].eval(), A));  // d/db = a
		REQUIRE(approx(grad[2].eval(), 2.0 * C));  // d/dc = 2*c
	}
	
	SECTION("Gradient with trigonometric functions")
	{
		Scalar expr = sin(a) * cos(b) + exp(c);
		std::vector<Scalar> vars = {a, b, c};
		Vector grad = gradient(expr, vars);
		
		REQUIRE(grad.size() == 3);
		REQUIRE(approx(grad[0].eval(), std::cos(A) * std::cos(B)));
		REQUIRE(approx(grad[1].eval(), -std::sin(A) * std::sin(B)));
		REQUIRE(approx(grad[2].eval(), std::exp(C)));
	}
	
	SECTION("Gradient with vector operations")
	{
		Vector v = ws.make_vector(2);
		v[0].set_value(1.0);
		v[1].set_value(2.0);
		
		Scalar expr = v.dot(v);  // v · v = v[0]² + v[1]²
		std::vector<Scalar> vars = {v[0], v[1]};
		Vector grad = gradient(expr, vars);
		
		REQUIRE(grad.size() == 2);
		REQUIRE(approx(grad[0].eval(), 2.0 * 1.0));  // 2*v[0]
		REQUIRE(approx(grad[1].eval(), 2.0 * 2.0));  // 2*v[1]
	}
}

TEST_CASE("Hessian operations", "[derivatives]")
{
	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	
	const double A = 0.3;
	const double B = 0.7;
	
	a.set_value(A);
	b.set_value(B);
	
	SECTION("Simple quadratic hessian")
	{
		Scalar expr = a.powN(2) + 2.0 * a * b + b.powN(2);
		std::vector<Scalar> vars = {a, b};
		Matrix hess = hessian(expr, vars);
		
		REQUIRE(hess.rows() == 2);
		REQUIRE(hess.cols() == 2);
		REQUIRE(approx(hess(0, 0).eval(), 2.0));  // d²/da²
		REQUIRE(approx(hess(0, 1).eval(), 2.0));  // d²/dadb
		REQUIRE(approx(hess(1, 0).eval(), 2.0));  // d²/dbda
		REQUIRE(approx(hess(1, 1).eval(), 2.0));  // d²/db²
	}
	
	SECTION("Hessian symmetry test")
	{
		Scalar expr = sin(a * b) + exp(a) * log(b);
		std::vector<Scalar> vars = {a, b};
		Matrix hess_sym = hessian(expr, vars);
		Matrix hess_full = hessian(expr, vars);
		
		REQUIRE(hess_sym.rows() == 2);
		REQUIRE(hess_sym.cols() == 2);
		REQUIRE(hess_full.rows() == 2);
		REQUIRE(hess_full.cols() == 2);
		
		// Check symmetry
		REQUIRE(approx(hess_sym(0, 1).eval(), hess_sym(1, 0).eval()));
		REQUIRE(approx(hess_full(0, 1).eval(), hess_full(1, 0).eval()));
		REQUIRE(approx(hess_sym(0, 1).eval(), hess_full(0, 1).eval()));
	}
	
	SECTION("Higher-order polynomial hessian")
	{
		Scalar c = ws.make_scalar();
		c.set_value(0.5);
		
		Scalar expr = a.powN(4) + b.powN(3) * c + a.powN(2) * b;
		std::vector<Scalar> vars = {a, b, c};
		Matrix hess = hessian(expr, vars);
		
		REQUIRE(hess.rows() == 3);
		REQUIRE(hess.cols() == 3);
		
		// Check some specific entries
		REQUIRE(approx(hess(0, 0).eval(), 12.0 * A * A + 2.0 * B));  // d²/da²
		REQUIRE(approx(hess(1, 1).eval(), 6.0 * B * 0.5));  // d²/db²
		REQUIRE(approx(hess(2, 2).eval(), 0.0));  // d²/dc²
	}
}

TEST_CASE("Combined value-gradient-hessian", "[derivatives]")
{
	Workspace ws;
	Scalar x = ws.make_scalar();
	Scalar y = ws.make_scalar();
	
	const double X = 1.5;
	const double Y = 2.5;
	
	x.set_value(X);
	y.set_value(Y);
	
	SECTION("Value-gradient computation")
	{
		Scalar expr = x.powN(3) + sin(y) * x;
		std::vector<Scalar> vars = {x, y};
		std::vector<Scalar> vg = value_gradient(expr, vars);
		
		REQUIRE(vg.size() == 3);  // value + 2 gradients
		
		double expected_val = X*X*X + std::sin(Y) * X;
		double expected_dx = 3.0 * X*X + std::sin(Y);
		double expected_dy = std::cos(Y) * X;
		
		REQUIRE(approx(vg[0].eval(), expected_val));
		REQUIRE(approx(vg[1].eval(), expected_dx));
		REQUIRE(approx(vg[2].eval(), expected_dy));
	}
	
	SECTION("Value-gradient-hessian computation")
	{
		Scalar expr = x.powN(2) * y + exp(x) * y.powN(2);
		std::vector<Scalar> vars = {x, y};
		std::vector<Scalar> vgh = value_gradient_hessian(expr, vars);
		
		REQUIRE(vgh.size() == 7);  // value + 2 gradients + 4 hessian entries
		
		// Test value
		double expected_val = X*X * Y + std::exp(X) * Y*Y;
		REQUIRE(approx(vgh[0].eval(), expected_val));
		
		// Test gradients  
		double expected_dx = 2.0 * X * Y + std::exp(X) * Y*Y;
		double expected_dy = X*X + 2.0 * std::exp(X) * Y;
		REQUIRE(approx(vgh[1].eval(), expected_dx));
		REQUIRE(approx(vgh[2].eval(), expected_dy));
		
		// Test some hessian entries
		double expected_d2dx2 = 2.0 * Y + std::exp(X) * Y*Y;
		double expected_d2dxdy = 2.0 * X + 2.0 * std::exp(X) * Y;
		REQUIRE(approx(vgh[3].eval(), expected_d2dx2));  // H[0,0]
		REQUIRE(approx(vgh[4].eval(), expected_d2dxdy)); // H[0,1]
	}
}

TEST_CASE("Branch differentiation", "[derivatives]")
{
	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	Scalar c = ws.make_scalar();
	
	SECTION("Simple branch derivative")
	{
		Scalar condition = a > 0.0;
		Scalar positive_branch = a.powN(2) + b;
		Scalar negative_branch = a * b + c;
		Scalar branch_expr = branch(condition, positive_branch, negative_branch);
		
		Scalar da = diff(branch_expr, a);
		Scalar db = diff(branch_expr, b);
		Scalar dc = diff(branch_expr, c);
		
		// Test with positive a
		a.set_value(1.0);
		b.set_value(2.0);
		c.set_value(3.0);
		REQUIRE(approx(da.eval(), 2.0 * 1.0));  // d/da of a² + b = 2a
		REQUIRE(approx(db.eval(), 1.0));        // d/db of a² + b = 1
		REQUIRE(approx(dc.eval(), 0.0));        // d/dc of a² + b = 0
		
		// Test with negative a
		a.set_value(-1.0);
		REQUIRE(approx(da.eval(), b.eval()));   // d/da of a*b + c = b
		REQUIRE(approx(db.eval(), a.eval()));   // d/db of a*b + c = a
		REQUIRE(approx(dc.eval(), 1.0));        // d/dc of a*b + c = 1
	}
	
	SECTION("Nested branches derivative")
	{
		Scalar inner_condition = b > 0.0;
		Scalar inner_branch = branch(inner_condition, b.powN(2), b.sqrt());
		
		Scalar outer_condition = a > 0.0;
		Scalar outer_branch = branch(outer_condition, a * inner_branch, a + inner_branch);
		
		Scalar da = diff(outer_branch, a);
		Scalar db = diff(outer_branch, b);
		
		// Test various combinations
		a.set_value(1.0);
		b.set_value(4.0);
		// a > 0, b > 0: expr = a * b² = 1 * 16 = 16
		// da = b² = 16, db = a * 2b = 1 * 8 = 8
		REQUIRE(approx(da.eval(), 16.0));
		REQUIRE(approx(db.eval(), 8.0));
		
		a.set_value(-1.0);
		b.set_value(4.0);
		// a < 0, b > 0: expr = a + b² = -1 + 16 = 15
		// da = 1, db = 2b = 8
		REQUIRE(approx(da.eval(), 1.0));
		REQUIRE(approx(db.eval(), 8.0));
	}
}

TEST_CASE("Differentiation error handling", "[derivatives]")
{
	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	
	SECTION("Differentiation with respect to non-existent variable")
	{
		Scalar c = ws.make_scalar();
		Scalar expr = a + b;
		Scalar dc = diff(expr, c);
		
		a.set_value(1.0);
		b.set_value(2.0);
		c.set_value(3.0);
		
		REQUIRE(approx(dc.eval(), 0.0));  // Should be zero
	}
	
	SECTION("Self-differentiation")
	{
		Scalar expr = a.powN(3) + sin(a);
		Scalar da = diff(expr, a);
		
		a.set_value(0.5);
		double expected = 3.0 * 0.5 * 0.5 + std::cos(0.5);
		REQUIRE(approx(da.eval(), expected));
	}
	
	SECTION("Constant differentiation")
	{
		Scalar expr = a.make_constant(5.0);
		Scalar da = diff(expr, a);
		
		a.set_value(10.0);
		REQUIRE(approx(da.eval(), 0.0));
	}
}

TEST_CASE("Large-scale differentiation", "[derivatives]")
{
	Workspace ws;
	const int n = 10;
	std::vector<Scalar> vars;
	
	// Create variables
	for (int i = 0; i < n; i++) {
		vars.push_back(ws.make_scalar());
		vars[i].set_value(0.1 * (i + 1));  // x0=0.1, x1=0.2, ..., x9=1.0
	}
	
	SECTION("Large quadratic form")
	{
		// Create quadratic expression: sum(xi² + xi*x(i+1))
		Scalar expr = vars[0].get_zero();
		for (int i = 0; i < n; i++) {
			expr = expr + vars[i].powN(2);
			if (i < n - 1) {
				expr = expr + vars[i] * vars[i + 1];
			}
		}
		
		Vector grad = gradient(expr, vars);
		Matrix hess = hessian(expr, vars);
		
		REQUIRE(grad.size() == n);
		REQUIRE(hess.rows() == n);
		REQUIRE(hess.cols() == n);
		
		// Check gradient values
		for (int i = 0; i < n; i++) {
			double expected_grad = 2.0 * vars[i].eval();
			if (i > 0) expected_grad += vars[i - 1].eval();
			if (i < n - 1) expected_grad += vars[i + 1].eval();
			
			REQUIRE(approx(grad[i].eval(), expected_grad, 1e-10));
		}
		
		// Check hessian diagonal
		for (int i = 0; i < n; i++) {
			REQUIRE(approx(hess(i, i).eval(), 2.0, 1e-10));
		}
		
		// Check hessian off-diagonal
		for (int i = 0; i < n - 1; i++) {
			REQUIRE(approx(hess(i, i + 1).eval(), 1.0, 1e-10));
			REQUIRE(approx(hess(i + 1, i).eval(), 1.0, 1e-10));
		}
	}
}

TEST_CASE("Logarithm differentiation tests", "[diff]")
{
	const double A = 2.5;
	const double B = 3.7;
	
	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	a.set_value(A);
	b.set_value(B);

	SECTION("Basic logarithm derivatives")
	{
		// d/dx ln(x) = 1/x
		REQUIRE(approx(diff(ln(a), a).eval(), 1.0 / A));
		
		// d/dx log(x) = 1/x (same as ln since they map to same ExprType::Ln internally)
		REQUIRE(approx(diff(log(a), a).eval(), 1.0 / A));
		
		// d/dx log10(x) = 1/(x * ln(10))
		REQUIRE(approx(diff(log10(a), a).eval(), 1.0 / (A * std::log(10.0))));
		
		// Verify log and ln give same derivative (both map to ExprType::Ln)
		REQUIRE(approx(diff(log(a), a).eval(), diff(ln(a), a).eval()));
	}

	SECTION("Chain rule with logarithms")
	{
		// d/dx ln(f(x)) = f'(x)/f(x)
		Scalar expr1 = ln(a * b);
		REQUIRE(approx(diff(expr1, a).eval(), B / (A * B)));
		
		// d/dx log(f(x)) = f'(x)/f(x) (same as ln since both map to ExprType::Ln)
		Scalar expr1_log = log(a * b);
		REQUIRE(approx(diff(expr1_log, a).eval(), B / (A * B)));
		
		// d/dx log10(f(x)) = f'(x)/(f(x) * ln(10))
		Scalar expr2 = log10(a * b);
		REQUIRE(approx(diff(expr2, a).eval(), B / (A * B * std::log(10.0))));
		
		// More complex: d/dx ln(x^2 + 1) = 2x/(x^2 + 1)
		Scalar expr3 = ln(a.powN(2) + 1.0);
		double expected = 2.0 * A / (A * A + 1.0);
		REQUIRE(approx(diff(expr3, a).eval(), expected));
	}

	SECTION("Mixed logarithm expressions")
	{
		// d/dx [ln(x) + log10(x)] = 1/x + 1/(x*ln(10))
		Scalar expr = ln(a) + log10(a);
		double expected = 1.0 / A + 1.0 / (A * std::log(10.0));
		REQUIRE(approx(diff(expr, a).eval(), expected));
		
		// d/dx [ln(x) * log10(x)]
		Scalar expr2 = ln(a) * log10(a);
		double ln_a = std::log(A);
		double log10_a = std::log10(A);
		double expected2 = (1.0/A) * log10_a + ln_a * (1.0/(A * std::log(10.0)));
		REQUIRE(approx(diff(expr2, a).eval(), expected2));
	}
}
