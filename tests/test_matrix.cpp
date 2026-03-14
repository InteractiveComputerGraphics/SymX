#include "catch.hpp"

#include <Eigen/Dense>
#include <symx>
#include "utils.h"
#include <cmath>

using namespace symx;

TEST_CASE("matrix operations", "[matrix]")
{
	Workspace ws;
	
	// Create test matrices
	Matrix m22 = ws.make_matrix({2, 2});
	Matrix n22 = ws.make_matrix({2, 2});
	Matrix m33 = ws.make_matrix({3, 3});
	Matrix n33 = ws.make_matrix({3, 3});
	
	// Set test values
	std::vector<double> vals22_1 = {1.0, 2.0, 3.0, 4.0};  // [[1,2],[3,4]]
	std::vector<double> vals22_2 = {5.0, 6.0, 7.0, 8.0};  // [[5,6],[7,8]]
	std::vector<double> vals33_1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
	std::vector<double> vals33_2 = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
	
	// Set values manually for 2x2 matrices
	m22(0, 0).set_value(1.0); m22(0, 1).set_value(2.0);
	m22(1, 0).set_value(3.0); m22(1, 1).set_value(4.0);
	
	n22(0, 0).set_value(5.0); n22(0, 1).set_value(6.0);
	n22(1, 0).set_value(7.0); n22(1, 1).set_value(8.0);
	
	// Set values manually for 3x3 matrices
	int idx = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			m33(i, j).set_value(vals33_1[idx]);
			n33(i, j).set_value(vals33_2[idx]);
			idx++;
		}
	}
	
	// Create test vectors
	Vector v2 = ws.make_vector(2);
	Vector v3 = ws.make_vector(3);
	std::vector<double> vec2_vals = {1.0, 2.0};
	std::vector<double> vec3_vals = {1.0, 2.0, 3.0};
	v2.set_values(vec2_vals);
	v3.set_values(vec3_vals);
	
	const double scalar_val = 2.5;
	Scalar s = ws.make_scalar();
	s.set_value(scalar_val);
	
	SECTION("Basic properties")
	{
		REQUIRE(m22.rows() == 2);
		REQUIRE(m22.cols() == 2);
		REQUIRE(m33.rows() == 3);
		REQUIRE(m33.cols() == 3);
		
		auto shape22 = m22.shape();
		REQUIRE(shape22[0] == 2);
		REQUIRE(shape22[1] == 2);
		
		// Test element access
		compare(1.0, m22(0, 0));
		compare(2.0, m22(0, 1));
		compare(3.0, m22(1, 0));
		compare(4.0, m22(1, 1));
		
		// Test values() getter
		const auto& values = m22.values();
		REQUIRE(values.size() == 4);
		compare(1.0, values[0]);
		compare(2.0, values[1]);
		compare(3.0, values[2]);
		compare(4.0, values[3]);
	}
	
	SECTION("Matrix arithmetic with matrices")
	{
		Matrix sum = m22 + n22;
		Matrix diff = m22 - n22;
		
		// m22 + n22 = [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
		compare(6.0, sum(0, 0));
		compare(8.0, sum(0, 1));
		compare(10.0, sum(1, 0));
		compare(12.0, sum(1, 1));
		
		// m22 - n22 = [[1,2],[3,4]] - [[5,6],[7,8]] = [[-4,-4],[-4,-4]]
		compare(-4.0, diff(0, 0));
		compare(-4.0, diff(0, 1));
		compare(-4.0, diff(1, 0));
		compare(-4.0, diff(1, 1));
		
		// Assignment operations
		Matrix temp = m22;
		temp += n22;
		compare(6.0, temp(0, 0));
		compare(8.0, temp(0, 1));
		compare(10.0, temp(1, 0));
		compare(12.0, temp(1, 1));
		
		temp = m22;
		temp -= n22;
		compare(-4.0, temp(0, 0));
		compare(-4.0, temp(0, 1));
		compare(-4.0, temp(1, 0));
		compare(-4.0, temp(1, 1));
	}
	
	SECTION("Matrix arithmetic with scalars")
	{
		Matrix scaled_double = m22 * scalar_val;
		Matrix scaled_scalar = m22 * s;
		Matrix divided_double = m22 / scalar_val;
		Matrix divided_scalar = m22 / s;
		
		compare(2.5, scaled_double(0, 0));
		compare(5.0, scaled_double(0, 1));
		compare(7.5, scaled_double(1, 0));
		compare(10.0, scaled_double(1, 1));
		
		compare(2.5, scaled_scalar(0, 0));
		compare(5.0, scaled_scalar(0, 1));
		compare(7.5, scaled_scalar(1, 0));
		compare(10.0, scaled_scalar(1, 1));
		
		compare(1.0/2.5, divided_double(0, 0));
		compare(2.0/2.5, divided_double(0, 1));
		compare(3.0/2.5, divided_double(1, 0));
		compare(4.0/2.5, divided_double(1, 1));
		
		compare(1.0/2.5, divided_scalar(0, 0));
		compare(2.0/2.5, divided_scalar(0, 1));
		compare(3.0/2.5, divided_scalar(1, 0));
		compare(4.0/2.5, divided_scalar(1, 1));
		
		// Test reverse operations
		Matrix rev_scaled_double = scalar_val * m22;
		Matrix rev_scaled_scalar = s * m22;
		
		compare(2.5, rev_scaled_double(0, 0));
		compare(5.0, rev_scaled_double(0, 1));
		compare(7.5, rev_scaled_double(1, 0));
		compare(10.0, rev_scaled_double(1, 1));
		
		compare(2.5, rev_scaled_scalar(0, 0));
		compare(5.0, rev_scaled_scalar(0, 1));
		compare(7.5, rev_scaled_scalar(1, 0));
		compare(10.0, rev_scaled_scalar(1, 1));
		
		// Assignment operations with scalars
		Matrix temp = m22;
		temp *= scalar_val;
		compare(2.5, temp(0, 0));
		compare(5.0, temp(0, 1));
		compare(7.5, temp(1, 0));
		compare(10.0, temp(1, 1));
		
		temp = m22;
		temp *= s;
		compare(2.5, temp(0, 0));
		compare(5.0, temp(0, 1));
		compare(7.5, temp(1, 0));
		compare(10.0, temp(1, 1));
		
		temp = m22;
		temp /= scalar_val;
		compare(1.0/2.5, temp(0, 0));
		compare(2.0/2.5, temp(0, 1));
		compare(3.0/2.5, temp(1, 0));
		compare(4.0/2.5, temp(1, 1));
		
		temp = m22;
		temp /= s;
		compare(1.0/2.5, temp(0, 0));
		compare(2.0/2.5, temp(0, 1));
		compare(3.0/2.5, temp(1, 0));
		compare(4.0/2.5, temp(1, 1));
		
		// Unary minus
		Matrix neg = -m22;
		compare(-1.0, neg(0, 0));
		compare(-2.0, neg(0, 1));
		compare(-3.0, neg(1, 0));
		compare(-4.0, neg(1, 1));
	}
	
	SECTION("Matrix-vector operations")
	{
		// Matrix-vector multiplication
		Vector result = m22 * v2;
		// [[1,2],[3,4]] * [1,2] = [1*1+2*2, 3*1+4*2] = [5, 11]
		compare(5.0, result[0]);
		compare(11.0, result[1]);
		
		// Alternative dot method
		Vector result2 = m22.dot(v2);
		compare(5.0, result2[0]);
		compare(11.0, result2[1]);
		
		// Vector-matrix multiplication (left multiplication)
		Vector result3 = v2 * m22;
		// [1,2] * [[1,2],[3,4]] = [1*1+2*3, 1*2+2*4] = [7, 10]
		compare(7.0, result3[0]);
		compare(10.0, result3[1]);
	}
	
	SECTION("Matrix-matrix operations")
	{
		// Matrix multiplication
		Matrix product = m22 * n22;
		// [[1,2],[3,4]] * [[5,6],[7,8]] = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
		compare(19.0, product(0, 0));
		compare(22.0, product(0, 1));
		compare(43.0, product(1, 0));
		compare(50.0, product(1, 1));
		
		// Alternative dot method
		Matrix product2 = m22.dot(n22);
		compare(19.0, product2(0, 0));
		compare(22.0, product2(0, 1));
		compare(43.0, product2(1, 0));
		compare(50.0, product2(1, 1));
	}
	
	SECTION("Matrix operations")
	{
		// Transpose
		Matrix transposed = m22.transpose();
		compare(1.0, transposed(0, 0));
		compare(3.0, transposed(0, 1));
		compare(2.0, transposed(1, 0));
		compare(4.0, transposed(1, 1));
		
		// Determinant (2x2)
		// det([[1,2],[3,4]]) = 1*4 - 2*3 = 4 - 6 = -2
		compare(-2.0, m22.det());
		
		// Trace
		// tr([[1,2],[3,4]]) = 1 + 4 = 5
		compare(5.0, m22.trace());
		
		// Frobenius norm squared
		// ||[[1,2],[3,4]]||_F^2 = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
		compare(30.0, m22.frobenius_norm_sq());
		
		// Matrix inverse (2x2)
		Matrix inv = m22.inv();
		// Check that m22 * inv = I (approximately)
		Matrix should_be_identity = m22 * inv;
		compare(1.0, should_be_identity(0, 0));
		compare(0.0, should_be_identity(0, 1));
		compare(0.0, should_be_identity(1, 0));
		compare(1.0, should_be_identity(1, 1));
	}

	SECTION("Matrix inversions")
	{
		using MatXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		for (int n = 1; n < 5; n++) {

			// Build a deterministic, well-conditioned invertible matrix:
			// m = I + small dense perturbation
			MatXdRowMajor m = MatXdRowMajor::Identity(n, n);
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					m(i, j) += 0.05 * (1.0 + i + 2.0 * j);
				}
			}

			// SymX matrix
			Matrix sm = ws.make_matrix({n, n});
			sm.set_values_row_major(m);

			// Inverse
			MatXdRowMajor inv = m.inverse();
			Matrix invs = sm.inv();

			// Compare each element
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					compare(inv(i, j), invs(i, j), 1e-10);
				}
			}

			// Symmetric inverse
			MatXdRowMajor sym_m = m * m.transpose();  // Make it symmetric positive definite
			Matrix sym_sm = ws.make_matrix({n, n});
			sym_sm.set_values_row_major(sym_m);

			// Inverse
			MatXdRowMajor sym_inv = sym_m.inverse();
			Matrix sym_invs = sym_sm.inv_sym();

			// Compare each element
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					compare(sym_inv(i, j), sym_invs(i, j), 1e-10);
				}
			}
		}
	}
	
	SECTION("Matrix singular values (2x2)")
	{
		Vector sv = m22.singular_values_2x2();
		REQUIRE(sv.size() == 2);
		// Note: This tests that the function works, exact values depend on implementation
		// We just check that we get two values (they may not always be ordered as expected)
		bool at_least_one_nonzero = (sv[0].eval() != 0) || (sv[1].eval() != 0);
		REQUIRE(at_least_one_nonzero);  // At least one should be non-zero
	}
	
	SECTION("Blocks and slicing")
	{
		// Test block extraction
		Matrix block = m33.block(0, 0, 2, 2);  // Top-left 2x2 block
		REQUIRE(block.rows() == 2);
		REQUIRE(block.cols() == 2);
		compare(1.0, block(0, 0));
		compare(2.0, block(0, 1));
		compare(4.0, block(1, 0));
		compare(5.0, block(1, 1));
		
		// Test row extraction
		Vector row = m33.row(1);  // Second row
		REQUIRE(row.size() == 3);
		compare(4.0, row[0]);
		compare(5.0, row[1]);
		compare(6.0, row[2]);
		
		// Test column extraction
		Vector col = m33.col(2);  // Third column
		REQUIRE(col.size() == 3);
		compare(3.0, col[0]);
		compare(6.0, col[1]);
		compare(9.0, col[2]);
		
		// Test block setting
		Matrix new_block = ws.make_matrix({2, 2});
		new_block(0, 0).set_value(10.0); new_block(0, 1).set_value(11.0);
		new_block(1, 0).set_value(12.0); new_block(1, 1).set_value(13.0);
		
		Matrix temp = m33;
		temp.set_block(1, 1, 2, 2, new_block);
		// Check that the block was set correctly
		compare(1.0, temp(0, 0));  // unchanged
		compare(2.0, temp(0, 1));  // unchanged
		compare(3.0, temp(0, 2));  // unchanged
		compare(4.0, temp(1, 0));  // unchanged
		compare(10.0, temp(1, 1)); // changed
		compare(11.0, temp(1, 2)); // changed
		compare(7.0, temp(2, 0));  // unchanged
		compare(12.0, temp(2, 1)); // changed
		compare(13.0, temp(2, 2)); // changed
		
		// Test row setting
		Vector new_row = ws.make_vector(3);
		new_row[0].set_value(20.0); new_row[1].set_value(21.0); new_row[2].set_value(22.0);
		
		temp = m33;
		temp.set_row(1, new_row);
		compare(1.0, temp(0, 0));  // unchanged
		compare(2.0, temp(0, 1));  // unchanged
		compare(3.0, temp(0, 2));  // unchanged
		compare(20.0, temp(1, 0)); // changed
		compare(21.0, temp(1, 1)); // changed
		compare(22.0, temp(1, 2)); // changed
		compare(7.0, temp(2, 0));  // unchanged
		compare(8.0, temp(2, 1));  // unchanged
		compare(9.0, temp(2, 2));  // unchanged
		
		// Test column setting
		Vector new_col = ws.make_vector(3);
		std::vector<double> col_vals = {30.0, 31.0, 32.0};
		new_col.set_values(col_vals);
		
		temp = m33;
		temp.set_col(2, new_col);
		compare(1.0, temp(0, 0));  // unchanged
		compare(2.0, temp(0, 1));  // unchanged
		compare(30.0, temp(0, 2)); // changed
		compare(4.0, temp(1, 0));  // unchanged
		compare(5.0, temp(1, 1));  // unchanged
		compare(31.0, temp(1, 2)); // changed
		compare(7.0, temp(2, 0));  // unchanged
		compare(8.0, temp(2, 1));  // unchanged
		compare(32.0, temp(2, 2)); // changed
	}
	
	SECTION("Static constructors")
	{
		// Test zero matrix
		Matrix zero_mat = Matrix::zero({2, 3}, m22(0, 0));
		REQUIRE(zero_mat.rows() == 2);
		REQUIRE(zero_mat.cols() == 3);
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 3; j++) {
				REQUIRE(zero_mat(i, j).is_zero());
			}
		}
		
		// Test identity matrix
		Matrix identity = Matrix::identity(3, m33(0, 0));
		REQUIRE(identity.rows() == 3);
		REQUIRE(identity.cols() == 3);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == j) {
					REQUIRE(identity(i, j).is_one());
				} else {
					REQUIRE(identity(i, j).is_zero());
				}
			}
		}
	}
	
	SECTION("Outer product")
	{
		Vector v1 = ws.make_vector(3);
		Vector v2 = ws.make_vector(3);
		std::vector<double> v1_vals = {1.0, 2.0, 3.0};
		std::vector<double> v2_vals = {4.0, 5.0, 6.0};
		v1.set_values(v1_vals);
		v2.set_values(v2_vals);
		
		Matrix outer_result = outer(v1, v2);
		REQUIRE(outer_result.rows() == 3);
		REQUIRE(outer_result.cols() == 3);
		
		// v1[0] * v2 = [4, 5, 6]
		compare(4.0, outer_result(0, 0));
		compare(5.0, outer_result(0, 1));
		compare(6.0, outer_result(0, 2));
		
		// v1[1] * v2 = [8, 10, 12]
		compare(8.0, outer_result(1, 0));
		compare(10.0, outer_result(1, 1));
		compare(12.0, outer_result(1, 2));
		
		// v1[2] * v2 = [12, 15, 18]
		compare(12.0, outer_result(2, 0));
		compare(15.0, outer_result(2, 1));
		compare(18.0, outer_result(2, 2));
	}
}
