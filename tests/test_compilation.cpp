#include "catch.hpp"

#ifdef SYMX_ENABLE_AVX2
#include <immintrin.h>
#endif

#include <ctime>
#include <limits>
#include <cmath>
#include <symx>
#include "utils.h"

using namespace symx;


TEST_CASE("Code generation Compiled<T>", "[compilation]")
{
	Workspace ws; 
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	Scalar c = ws.make_scalar();

	double A = 5.76;
	double B = 1.47;
	double C = 0.79;
	float Af = (float)A;
	float Bf = (float)B;
	float Cf = (float)C;

	Scalar x = a + b;
	x -= c;
	x *= a;
	x /= b;
	x = powN(x, 2);
	x = powF(x, 3.1);
	x = x.inv();
	x = sqrt(x);
	x = log(x);
	x = exp(x);
	x = sin(x);
	x = cos(x);
	x = tan(x);
	x *= asin(c);
	x *= acos(c);
	x = atan(x);

	std::string checksum = x.get_checksum();
	std::vector<Scalar> expr = { x };

	a.set_value(A);
	b.set_value(B);
	c.set_value(C);
	const double sol = expr[0].eval();

	{
		Compiled<double> f(expr, "f_double", symx::get_codegen_dir(), checksum);
		f.set(a, A);
		f.set(b, B);
		f.set(c, C);
		REQUIRE(approx(sol, f.run()[0]));
	}
	{
		Compiled<float> f(expr, "f_float", symx::get_codegen_dir(), checksum);
		f.set(a, Af);
		f.set(b, Bf);
		f.set(c, Cf);
		REQUIRE(std::abs(sol - (double)f.run()[0]) < 1e-2);
	}
#ifdef SYMX_ENABLE_AVX2
	{
		Compiled<__m256d> f(expr, "f_4d", symx::get_codegen_dir(), checksum);
		__m256d A_ = _mm256_set1_pd(A);
		__m256d B_ = _mm256_set1_pd(B);
		__m256d C_ = _mm256_set1_pd(C);

		f.set(a, A_);
		f.set(b, B_);
		f.set(c, C_);
		REQUIRE(approx(sol, reinterpret_cast<double*>(f.run().data())[0]));
	}
	{
		Compiled<__m256> f(expr, "f_8f", symx::get_codegen_dir(), checksum);
		__m256 A_ = _mm256_set1_ps((float)A);
		__m256 B_ = _mm256_set1_ps((float)B);
		__m256 C_ = _mm256_set1_ps((float)C);
        
		f.set(a, A_);
		f.set(b, B_);
		f.set(c, C_);
		REQUIRE(std::abs(sol - (double)(reinterpret_cast<float*>(f.run().data())[0])) < 1e-2);
	}
#endif
}


TEST_CASE("Multi-expression Compiled<T> type combinations", "[compilation]")
{
	Workspace ws;
	Scalar x = ws.make_scalar();
	Scalar y = ws.make_scalar();

	// Multiple expressions of varying complexity
	Scalar expr1 = x + y;
	Scalar expr2 = x * y - 0.5;
	Scalar expr3 = sin(x) + cos(y);
	Scalar expr4 = sqrt(x * x + y * y);
	
	std::vector<Scalar> exprs = { expr1, expr2, expr3, expr4 };

	double X = 1.2, Y = 2.1;
	float Xf = (float)X, Yf = (float)Y;

	// Calculate expected results
	x.set_value(X); y.set_value(Y);
	std::vector<double> expected(4);
	expected[0] = expr1.eval();
	expected[1] = expr2.eval();
	expected[2] = expr3.eval();
	expected[3] = expr4.eval();

	std::string checksum = exprs[0].get_checksum();

	SECTION("Multi-expr Compiled<double>") {
		Compiled<double> compiled(exprs, "multi_type_test_double", symx::get_codegen_dir(), checksum);
		compiled.set(x, X);
		compiled.set(y, Y);
		
		const View<double> results = compiled.run();
		for (int i = 0; i < 4; i++) {
			REQUIRE(approx(expected[i], results[i]));
		}
	}

	SECTION("Multi-expr Compiled<float>") {
		Compiled<float> compiled(exprs, "multi_type_test_float", symx::get_codegen_dir(), checksum);
		compiled.set(x, Xf);
		compiled.set(y, Yf);
		
		const View<float> results = compiled.run();
		for (int i = 0; i < 4; i++) {
			REQUIRE(approx(expected[i], (double)results[i], 1e-5));
		}
	}

#ifdef SYMX_ENABLE_AVX2
	SECTION("Multi-expr Compiled<__m256d> (SIMD4d)") {
		Compiled<__m256d> compiled(exprs, "multi_type_test_simd4d", symx::get_codegen_dir(), checksum);
		__m256d X_vec = _mm256_set1_pd(X);
		__m256d Y_vec = _mm256_set1_pd(Y);
		
		compiled.set(x, X_vec);
		compiled.set(y, Y_vec);
		
		const View<__m256d> results = compiled.run();
		for (int i = 0; i < 4; i++) {
			REQUIRE(approx(expected[i], reinterpret_cast<const double*>(&results[i])[0]));
		}
	}

	SECTION("Multi-expr Compiled<__m256> (SIMD8f)") {
		Compiled<__m256> compiled(exprs, "multi_type_test_simd8f", symx::get_codegen_dir(), checksum);
		__m256 X_vec = _mm256_set1_ps(Xf);
		__m256 Y_vec = _mm256_set1_ps(Yf);
		
		compiled.set(x, X_vec);
		compiled.set(y, Y_vec);
		
		const View<__m256> results = compiled.run();
		for (int i = 0; i < 4; i++) {
			REQUIRE(approx(expected[i], (double)reinterpret_cast<const float*>(&results[i])[0], 1e-5));
		}
	}
#endif
}

TEST_CASE("Branch compilation", "[compilation]")
{
	

	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	Scalar c = ws.make_scalar();

	// Create a branched expression
	Scalar condition = a > b;
	Scalar positive_branch = a * c + sin(b);
	Scalar negative_branch = b * c - cos(a);
	Scalar branch_expr = branch(condition, positive_branch, negative_branch);

	std::vector<Scalar> exprs = {branch_expr};

	SECTION("Branch true case")
	{
		double A = 2.0, B = 1.0, C = 0.5;
		
		Compiled<double> f(exprs, "branch_true", symx::get_codegen_dir(), branch_expr.get_checksum());
		f.set(a, A);
		f.set(b, B);
		f.set(c, C);
		
		View<double>result = f.run();
		double expected = A * C + std::sin(B);  // a > b is true, so positive branch
		
		REQUIRE(approx(result[0], expected));
	}

	SECTION("Branch false case")
	{
		double A = 1.0, B = 2.0, C = 0.5;
		
		Compiled<double> f(exprs, "branch_false", symx::get_codegen_dir(), branch_expr.get_checksum());
		f.set(a, A);
		f.set(b, B);
		f.set(c, C);
		
		View<double>result = f.run();
		double expected = B * C - std::cos(A);  // a > b is false, so negative branch
		
		REQUIRE(approx(result[0], expected));
	}

	SECTION("Nested branches")
	{
		Scalar d = ws.make_scalar();
		Scalar inner_condition = c > 0.0;
		Scalar inner_branch = branch(inner_condition, c.powN(2), c.sqrt());
		
		Scalar outer_condition = a > b;
		Scalar nested_expr = branch(outer_condition, a * inner_branch, b + inner_branch);
		
		std::vector<Scalar> nested_exprs = {nested_expr};
		
		double A = 3.0, B = 1.0, C = 4.0, D = 0.1;
		
		Compiled<double> f(nested_exprs, "nested_branch", symx::get_codegen_dir(), nested_expr.get_checksum());
		f.set(a, A);
		f.set(b, B);
		f.set(c, C);
		f.set(d, D);
		
		View<double>result = f.run();
		
		// a > b (true), c > 0 (true), so: a * c² = 3 * 16 = 48
		double expected = A * (C * C);
		REQUIRE(approx(result[0], expected));
	}
}

TEST_CASE("Compilation caching", "[compilation]")
{
	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	Scalar expr = a * b + sin(a);
	
	std::vector<Scalar> exprs = {expr};
	std::string checksum = expr.get_checksum();

	SECTION("Cache miss then hit")
	{
		// First compilation - should be a cache miss
		Compiled<double> f1;
		f1.compile(exprs, "cache_test", symx::get_codegen_dir(), checksum);
		REQUIRE(!f1.was_cached());
		
		// Second compilation with same checksum - should be a cache hit
		Compiled<double> f2;
		bool loaded = f2.load_if_cached("cache_test", symx::get_codegen_dir(), checksum);
		REQUIRE(loaded);
		REQUIRE(f2.was_cached());
		
		// Verify both produce same results
		double A = 1.5, B = 2.3;
		
		f1.set(a, A);
		f1.set(b, B);
		View<double>result1 = f1.run();
		
		f2.set(a, A);
		f2.set(b, B);
		View<double>result2 = f2.run();
		
		REQUIRE(approx(result1[0], result2[0]));
	}

	SECTION("Different checksums")
	{
		Scalar different_expr = a + b * cos(a);  // Different expression
		std::string different_checksum = different_expr.get_checksum();
		
		REQUIRE(checksum != different_checksum);
		
		Compiled<double> f1(exprs, "checksum_test1", symx::get_codegen_dir(), checksum);
		Compiled<double> f2({different_expr}, "checksum_test2", symx::get_codegen_dir(), different_checksum);
		
		double A = 1.0, B = 2.0;
		
		f1.set(a, A);
		f1.set(b, B);
		View<double>result1 = f1.run();
		
		f2.set(a, A);
		f2.set(b, B);
		View<double>result2 = f2.run();
		
		// Results should be different
		REQUIRE(!approx(result1[0], result2[0], 1e-10));
	}
}

TEST_CASE("Compiled thread safety", "[compilation]")
{
	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar b = ws.make_scalar();
	Scalar expr = a * a + b * b + sin(a) * cos(b);
	
	std::vector<Scalar> exprs = {expr};
	std::string checksum = expr.get_checksum();

	SECTION("Single thread (default behavior)")
	{
		Compiled<double> f(exprs, "single_thread", symx::get_codegen_dir(), checksum);
		REQUIRE(f.get_n_threads() == 1);
		
		double A = 1.5, B = 2.3;
		f.set(a, A);
		f.set(b, B);
		
		View<double>result = f.run();
		double expected = A * A + B * B + std::sin(A) * std::cos(B);
		REQUIRE(approx(result[0], expected));
	}

	SECTION("Multiple threads with independent buffers")
	{
		Compiled<double> f(exprs, "multi_thread", symx::get_codegen_dir(), checksum);
		f.set_n_threads(4);
		REQUIRE(f.get_n_threads() == 4);
		
		// Different input values for each thread
		std::vector<double> A_vals = {1.0, 2.0, 3.0, 4.0};
		std::vector<double> B_vals = {0.5, 1.5, 2.5, 3.5};
		std::vector<double> expected(4);
		
		// Set inputs for each thread and calculate expected results
		for (int thread_id = 0; thread_id < 4; thread_id++) {
			f.set(a, A_vals[thread_id], thread_id);
			f.set(b, B_vals[thread_id], thread_id);
			expected[thread_id] = A_vals[thread_id] * A_vals[thread_id] + 
								  B_vals[thread_id] * B_vals[thread_id] + 
								  std::sin(A_vals[thread_id]) * std::cos(B_vals[thread_id]);
		}
		
		// Run each thread and verify results
		for (int thread_id = 0; thread_id < 4; thread_id++) {
			View<double>result = f.run(thread_id);
			REQUIRE(approx(result[0], expected[thread_id]));
		}
		
		// Verify that thread 0 can still be accessed with the default run()
		View<double>result_default = f.run();
		REQUIRE(approx(result_default[0], expected[0]));
	}
}

TEST_CASE("Compiled matrix support", "[compilation]")
{
	Workspace ws;
	
	// Create a 2x2 matrix and compute its determinant
	Matrix m = Matrix::zero({2, 2}, ws.make_scalar());
	m(0, 0) = ws.make_scalar();
	m(0, 1) = ws.make_scalar();
	m(1, 0) = ws.make_scalar();
	m(1, 1) = ws.make_scalar();
	
	Scalar det_expr = m.det();
	Scalar trace_expr = m.trace();
	
	std::vector<Scalar> exprs = {det_expr, trace_expr};
	std::string checksum = det_expr.get_checksum();

	SECTION("Matrix input handling")
	{
		Compiled<double> f(exprs, "matrix_test", symx::get_codegen_dir(), checksum);
		
		// Input matrix values (row-major order)
		double matrix_data[4] = {1.0, 2.0, 3.0, 4.0};  // [[1, 2], [3, 4]]
		
		f.set(m, matrix_data);
		View<double> results = f.run();
		
		// det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
		// trace([[1, 2], [3, 4]]) = 1 + 4 = 5
		REQUIRE(approx(results[0], -2.0));
		REQUIRE(approx(results[1], 5.0));
	}

	SECTION("Matrix with threads")
	{
		Compiled<double> f(exprs, "matrix_thread_test", symx::get_codegen_dir(), checksum);
		f.set_n_threads(2);
		
		// Different matrices for each thread
		double matrix1[4] = {2.0, 1.0, 1.0, 3.0};  // [[2, 1], [1, 3]]
		Eigen::Matrix<double, 2, 2, Eigen::RowMajor> matrix2(2, 2);
		matrix2 << 1.0, -1.0, 2.0, 0.0; // [[1, -1], [2, 0]]

		f.set(m, &matrix1[0], 0);
		f.set(m, matrix2, 1);

		View<double> results1 = f.run(0);
		View<double> results2 = f.run(1);
		
		// det([[2, 1], [1, 3]]) = 2*3 - 1*1 = 5
		// trace([[2, 1], [1, 3]]) = 2 + 3 = 5
		REQUIRE(approx(results1[0], 5.0));
		REQUIRE(approx(results1[1], 5.0));
		
		// det([[1, -1], [2, 0]]) = 1*0 - (-1)*2 = 2
		// trace([[1, -1], [2, 0]]) = 1 + 0 = 1
		REQUIRE(approx(results2[0], 2.0));
		REQUIRE(approx(results2[1], 1.0));
	}
}

TEST_CASE("Compiled try_load_otherwise_compile", "[compilation]")
{
	Workspace ws;
	Scalar x = ws.make_scalar();
	Scalar y = ws.make_scalar();
	Scalar expr = x * y + sin(x);
	
	std::vector<Scalar> exprs = {expr};
	std::string checksum = expr.get_checksum();

	SECTION("Cache miss - should compile")
	{
		// Use a unique ID and name for this test to avoid conflicts
		std::string test_checksum = checksum + "_cache_miss_test";
		std::string test_name = "try_load_test_" + std::to_string(std::time(nullptr));
		
		// First time should compile
		Compiled<double> f1;
		f1.try_load_otherwise_compile(exprs, test_name, symx::get_codegen_dir(), test_checksum);
		
		REQUIRE(f1.is_valid());
		REQUIRE(!f1.was_cached());  // Should be freshly compiled
		REQUIRE(f1.get_name() == test_name);
		
		// Verify it works
		double X = 1.5, Y = 2.0;
		f1.set(x, X);
		f1.set(y, Y);
		View<double>result1 = f1.run();
		double expected = X * Y + std::sin(X);
		REQUIRE(approx(result1[0], expected));
	}

	SECTION("Cache hit - should load")
	{
		// Use a unique ID and name for this test
		std::string test_checksum = checksum + "_cache_hit_test";
		std::string test_name = "try_load_cached_" + std::to_string(std::time(nullptr) + 1);
		
		// First compilation to populate cache
		Compiled<double> f1;
		f1.try_load_otherwise_compile(exprs, test_name, symx::get_codegen_dir(), test_checksum);
		REQUIRE(!f1.was_cached());  // First time is compiled
		
		// Second time should load from cache
		Compiled<double> f2;
		f2.try_load_otherwise_compile(exprs, test_name, symx::get_codegen_dir(), test_checksum);
		
		REQUIRE(f2.is_valid());
		REQUIRE(f2.was_cached());  // Should be loaded from cache
		REQUIRE(f2.get_name() == test_name);
		
		// Verify both produce same results
		double X = 2.5, Y = 1.5;
		
		f1.set(x, X);
		f1.set(y, Y);
		View<double>result1 = f1.run();
		
		f2.set(x, X);
		f2.set(y, Y);
		View<double>result2 = f2.run();
		
		REQUIRE(approx(result1[0], result2[0]));
	}
}

TEST_CASE("Compiled error handling and edge cases", "[compilation]")
{
	

	Workspace ws;
	Scalar a = ws.make_scalar();
	Scalar expr = a * a;
	std::vector<Scalar> exprs = {expr};

	SECTION("Running invalid compilation")
	{
		Compiled<double> f;  // Not compiled yet
		
		REQUIRE(!f.is_valid());
		REQUIRE_THROWS_AS(f.run(), std::runtime_error);
		REQUIRE_THROWS_AS(f.run(0), std::runtime_error);
	}

	SECTION("Name tracking")
	{
		Compiled<double> f(exprs, "name_test", symx::get_codegen_dir(), expr.get_checksum());
		REQUIRE(f.get_name() == "name_test");
		
		// Try to load the same compilation from cache
		Compiled<double> f2;
		bool loaded = f2.load_if_cached("name_test", symx::get_codegen_dir(), expr.get_checksum());
		if (loaded) {
			REQUIRE(f2.get_name() == "name_test");  // Should preserve name when loading
		} else {
			// If not cached, just verify the load doesn't work but doesn't crash
			REQUIRE(f2.get_name() == "");  // Should remain empty if not loaded
		}
	}

	SECTION("Check for NaN handling")
	{
		Workspace ws;
		Scalar x = ws.make_scalar();
		Scalar sqrt_x = sqrt(x);  // sqrt of negative number will produce NaN
		Compiled<double> f({sqrt_x}, "nan_test", symx::get_codegen_dir(), sqrt_x.get_checksum());

		// Test with NaN checking enabled
		symx::check_mode_ON(true);
		REQUIRE(symx::get_check_mode_ON() == true);

		// Test that works
		double positive_value = 4.0;  // This will not cause NaN
		f.set(x, positive_value);
		REQUIRE_NOTHROW(f.run());

		// Test NaN in input
		double nan = std::numeric_limits<double>::quiet_NaN();
		f.set(x, nan);
		REQUIRE_THROWS_AS(f.run(), std::runtime_error);  // Should throw due
		
		// Test NaN in output
		double negative_value = -1.0;  // This will cause sqrt to return NaN
		f.set(x, negative_value);
		REQUIRE_THROWS_AS(f.run(), std::runtime_error);  // Should throw due
		
		// Reset to default state
		symx::check_mode_ON(false);
	}
}