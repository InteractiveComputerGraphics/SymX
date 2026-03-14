#include "catch.hpp"

#include <symx>
#include "utils.h"
#include <cmath>

using namespace symx;

TEST_CASE("vector operations", "[vector]")
{
	Workspace ws;
	
	// Create test vectors with different sizes
	Vector v3 = ws.make_vector(3);
	Vector u3 = ws.make_vector(3);
	Vector v4 = ws.make_vector(4);
	Vector u4 = ws.make_vector(4);
	
	// Set test values
	std::vector<double> vals3 = {1.0, 2.0, 3.0};
	std::vector<double> vals3_2 = {4.0, 5.0, 6.0};
	std::vector<double> vals4 = {1.0, 2.0, 3.0, 4.0};
	std::vector<double> vals4_2 = {5.0, 6.0, 7.0, 8.0};
	
	v3.set_values(vals3);
	u3.set_values(vals3_2);
	v4.set_values(vals4);
	u4.set_values(vals4_2);
	
	const double scalar_val = 2.5;
	Scalar s = ws.make_scalar();
	s.set_value(scalar_val);
	
	SECTION("Basic properties")
	{
		REQUIRE(v3.size() == 3);
		REQUIRE(v4.size() == 4);
		
		// Test accessors
		compare(1.0, v3[0]);
		compare(2.0, v3[1]);
		compare(3.0, v3[2]);
		
		compare(1.0, v3(0));
		compare(2.0, v3(1));
		compare(3.0, v3(2));
		
		// Test values() getter
		const auto& values = v3.values();
		REQUIRE(values.size() == 3);
		compare(1.0, values[0]);
		compare(2.0, values[1]);
		compare(3.0, values[2]);
	}
	
	SECTION("Vector arithmetic with vectors")
	{
		Vector sum = v3 + u3;
		Vector diff = v3 - u3;
		
		compare(5.0, sum[0]);  // 1 + 4
		compare(7.0, sum[1]);  // 2 + 5
		compare(9.0, sum[2]);  // 3 + 6
		
		compare(-3.0, diff[0]); // 1 - 4
		compare(-3.0, diff[1]); // 2 - 5
		compare(-3.0, diff[2]); // 3 - 6
		
		// Assignment operations
		Vector temp = v3;
		temp += u3;
		compare(5.0, temp[0]);
		compare(7.0, temp[1]);
		compare(9.0, temp[2]);
		
		temp = v3;
		temp -= u3;
		compare(-3.0, temp[0]);
		compare(-3.0, temp[1]);
		compare(-3.0, temp[2]);
	}
	
	SECTION("Vector arithmetic with scalars")
	{
		Vector scaled_double = v3 * scalar_val;
		Vector scaled_scalar = v3 * s;
		Vector divided_double = v3 / scalar_val;
		Vector divided_scalar = v3 / s;
		
		compare(2.5, scaled_double[0]);
		compare(5.0, scaled_double[1]);
		compare(7.5, scaled_double[2]);
		
		compare(2.5, scaled_scalar[0]);
		compare(5.0, scaled_scalar[1]);
		compare(7.5, scaled_scalar[2]);
		
		compare(1.0/2.5, divided_double[0]);
		compare(2.0/2.5, divided_double[1]);
		compare(3.0/2.5, divided_double[2]);
		
		compare(1.0/2.5, divided_scalar[0]);
		compare(2.0/2.5, divided_scalar[1]);
		compare(3.0/2.5, divided_scalar[2]);
		
		// Test reverse operations
		Vector rev_scaled_double = scalar_val * v3;
		Vector rev_scaled_scalar = s * v3;
		
		compare(2.5, rev_scaled_double[0]);
		compare(5.0, rev_scaled_double[1]);
		compare(7.5, rev_scaled_double[2]);
		
		compare(2.5, rev_scaled_scalar[0]);
		compare(5.0, rev_scaled_scalar[1]);
		compare(7.5, rev_scaled_scalar[2]);
		
		// Assignment operations with scalars
		Vector temp = v3;
		temp *= scalar_val;
		compare(2.5, temp[0]);
		compare(5.0, temp[1]);
		compare(7.5, temp[2]);
		
		temp = v3;
		temp *= s;
		compare(2.5, temp[0]);
		compare(5.0, temp[1]);
		compare(7.5, temp[2]);
		
		temp = v3;
		temp /= scalar_val;
		compare(1.0/2.5, temp[0]);
		compare(2.0/2.5, temp[1]);
		compare(3.0/2.5, temp[2]);
		
		temp = v3;
		temp /= s;
		compare(1.0/2.5, temp[0]);
		compare(2.0/2.5, temp[1]);
		compare(3.0/2.5, temp[2]);
		
		// Unary minus
		Vector neg = -v3;
		compare(-1.0, neg[0]);
		compare(-2.0, neg[1]);
		compare(-3.0, neg[2]);
	}
	
	SECTION("Vector products")
	{
		// Dot product
		double expected_dot = 1.0*4.0 + 2.0*5.0 + 3.0*6.0;  // 4 + 10 + 18 = 32
		compare(expected_dot, v3.dot(u3));
		compare(expected_dot, dot(v3, u3));
		
		// 2D cross product (returns scalar)
		Vector v2 = ws.make_vector(2);
		Vector u2 = ws.make_vector(2);
		std::vector<double> vals2_1 = {3.0, 4.0};
		std::vector<double> vals2_2 = {1.0, 2.0};
		v2.set_values(vals2_1);
		u2.set_values(vals2_2);
		
		double expected_cross2 = 3.0*2.0 - 4.0*1.0;  // 6 - 4 = 2
		compare(expected_cross2, v2.cross2(u2));
		
		// 3D cross product (returns vector)
		Vector cross_result = v3.cross3(u3);
		// v3 = [1, 2, 3], u3 = [4, 5, 6]
		// cross = [2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4] = [12-15, 12-6, 5-8] = [-3, 6, -3]
		compare(-3.0, cross_result[0]);
		compare(6.0, cross_result[1]);
		compare(-3.0, cross_result[2]);
	}
	
	SECTION("Vector norms and normalization")
	{
		// Squared norm: 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
		compare(14.0, v3.squared_norm());
		
		// Norm: sqrt(14)
		compare(std::sqrt(14.0), v3.norm());
		
		// Normalized vector
		Vector normalized = v3.normalized();
		double norm = std::sqrt(14.0);
		compare(1.0/norm, normalized[0]);
		compare(2.0/norm, normalized[1]);
		compare(3.0/norm, normalized[2]);
		
		// Check that original vector is unchanged
		compare(1.0, v3[0]);
		compare(2.0, v3[1]);
		compare(3.0, v3[2]);
		
		// Test in-place normalization
		Vector temp = v3;
		temp.normalize();
		compare(1.0/norm, temp[0]);
		compare(2.0/norm, temp[1]);
		compare(3.0/norm, temp[2]);
	}
	
	SECTION("Stable norm")
	{
		// Test stable_norm with v3 = [1, 2, 3]
		// squared_norm = 14
		Scalar eps = ws.make_scalar();
		eps.set_value(1e-6);
		
		// stable_norm(eps) = sqrt(squared_norm + eps) - sqrt(eps)
		// = sqrt(14 + 1e-6) - sqrt(1e-6)
		double expected = std::sqrt(14.0 + 1e-6) - std::sqrt(1e-6);
		compare(expected, v3.stable_norm(eps));
		
		// Test with eps = 0 (should give same result as regular norm)
		eps.set_value(0.0);
		expected = std::sqrt(14.0) - 0.0;  // Should be same as norm()
		compare(expected, v3.stable_norm(eps));
		compare(v3.norm().eval(), v3.stable_norm(eps));
	}
	
	SECTION("Segments and slicing")
	{
		// Test segment extraction
		Vector seg = v4.segment(1, 2);  // Extract elements 1 and 2
		REQUIRE(seg.size() == 2);
		compare(2.0, seg[0]);  // v4[1]
		compare(3.0, seg[1]);  // v4[2]
		
		// Test segment setting
		Vector new_vals = ws.make_vector(2);
		std::vector<double> new_data = {10.0, 11.0};
		new_vals.set_values(new_data);
		
		Vector temp = v4;
		temp.set_segment(1, 2, new_vals);
		compare(1.0, temp[0]);   // unchanged
		compare(10.0, temp[1]);  // changed
		compare(11.0, temp[2]);  // changed
		compare(4.0, temp[3]);   // unchanged
	}
	
	SECTION("Static constructors")
	{
		// Test zero vector
		Vector zero_vec = Vector::zero(3, v3[0]);
		REQUIRE(zero_vec.size() == 3);
		for (int i = 0; i < 3; i++) {
			REQUIRE(zero_vec[i].is_zero());
		}
	}
}
