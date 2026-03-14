#include "catch.hpp"

#include <symx>
#include "utils.h"
#include <cmath>
#include <iostream>

using namespace symx;

TEST_CASE("scalar operations", "[scalar]")
{
    Workspace ws;

    // Create test scalars with different values
    Scalar a = ws.make_scalar();
    Scalar b = ws.make_scalar();
    a.set_value(3.5);
    b.set_value(2.0);

    const double val = 1.5;

    SECTION("Basic arithmetic with doubles")
    {
        compare(3.5 + 1.5, a + val);
        compare(3.5 - 1.5, a - val);
        compare(3.5 * 1.5, a * val);
        compare(3.5 / 1.5, a / val);

        // Reverse operations
        compare(1.5 + 3.5, val + a);
        compare(1.5 - 3.5, val - a);
        compare(1.5 * 3.5, val * a);
        compare(1.5 / 3.5, val / a);

        // Assignment operations with doubles
        Scalar c = a;
        c += val;
        compare(3.5 + 1.5, c);
        c = a;
        c -= val;
        compare(3.5 - 1.5, c);
        c = a;
        c *= val;
        compare(3.5 * 1.5, c);
        c = a;
        c /= val;
        compare(3.5 / 1.5, c);
    }

    SECTION("Basic arithmetic with scalars")
    {
        compare(3.5 + 2.0, a + b);
        compare(3.5 - 2.0, a - b);
        compare(3.5 * 2.0, a * b);
        compare(3.5 / 2.0, a / b);

        // Assignment operations with scalars
        Scalar c = a;
        c += b;
        compare(3.5 + 2.0, c);
        c = a;
        c -= b;
        compare(3.5 - 2.0, c);
        c = a;
        c *= b;
        compare(3.5 * 2.0, c);
        c = a;
        c /= b;
        compare(3.5 / 2.0, c);

        // Unary minus
        compare(-3.5, -a);
    }

    SECTION("Power operations")
    {
        compare(std::pow(3.5, 3), a.powN(3));
        compare(std::pow(3.5, 2.5), a.powF(2.5));
        compare(std::pow(3.5, 3), powN(a, 3));
        compare(std::pow(3.5, 2.5), powF(a, 2.5));
    }

    SECTION("Mathematical functions")
    {
        // Test with positive value for all functions
        Scalar pos = ws.make_scalar();
        pos.set_value(1.5);

        compare(1.0 / 1.5, pos.inv());
        compare(std::sqrt(1.5), pos.sqrt());
        compare(std::sqrt(1.5), sqrt(pos));
        compare(std::log(1.5), pos.log());
        compare(std::log(1.5), log(pos));
        compare(std::log10(1.5), pos.log10());
        compare(std::log10(1.5), log10(pos));
        compare(std::exp(1.5), pos.exp());
        compare(std::exp(1.5), exp(pos));

        // Trigonometric functions
        compare(std::sin(1.5), pos.sin());
        compare(std::sin(1.5), sin(pos));
        compare(std::cos(1.5), pos.cos());
        compare(std::cos(1.5), cos(pos));
        compare(std::tan(1.5), pos.tan());
        compare(std::tan(1.5), tan(pos));

        // Inverse trigonometric functions (test with valid domain values)
        Scalar one_half = ws.make_scalar();
        one_half.set_value(0.5);
        compare(std::asin(0.5), one_half.asin());
        compare(std::asin(0.5), asin(one_half));
        compare(std::acos(0.5), one_half.acos());
        compare(std::acos(0.5), acos(one_half));
        compare(std::atan(0.5), one_half.atan());
        compare(std::atan(0.5), atan(one_half));

        // Test ln(0) returns -infinity
        Scalar zero_val = ws.make_scalar();
        zero_val.set_value(0.0);
        double ln_zero_result = zero_val.ln().eval();
        REQUIRE(std::isinf(ln_zero_result));
        REQUIRE(ln_zero_result < 0);
        
        // Test log10(0) returns -infinity
        double log10_zero_result = zero_val.log10().eval();
        REQUIRE(std::isinf(log10_zero_result));
        REQUIRE(log10_zero_result < 0);

        // Test ln of negative values also returns -infinity
        Scalar neg = ws.make_scalar();
        neg.set_value(-1.5);
        double ln_neg_result = neg.ln().eval();
        REQUIRE(std::isinf(ln_neg_result));
        REQUIRE(ln_neg_result < 0);
        
        // Test log10 of negative values also returns -infinity
        double log10_neg_result = neg.log10().eval();
        REQUIRE(std::isinf(log10_neg_result));
        REQUIRE(log10_neg_result < 0);

        // Test log(1) = 0
        Scalar one_val = ws.make_scalar();
        one_val.set_value(1.0);
        compare(0.0, one_val.log());
        
        // Test ln(1) = 0
        compare(0.0, one_val.ln());
        
        // Test log10(1) = 0
        compare(0.0, one_val.log10());
    }

    SECTION("Chain Expressions")
    {
        const double A = 0.2;
        const double X = 0.3;
        const double B = 0.4;

        Workspace ws;
        Scalar a = ws.make_scalar();
        Scalar x = ws.make_scalar();
        Scalar b = ws.make_scalar();
        a.set_value(A);
        x.set_value(X);
        b.set_value(B);

        REQUIRE(approx((a * x + b).eval(), A * X + B));
        REQUIRE(approx((a * x / b).eval(), A * X / B));
        REQUIRE(approx((sin(a * x) / b).eval(), std::sin(A * X) / B));
        REQUIRE(approx((sin(a * x) / log(b)).eval(), std::sin(A * X) / std::log(B)));
        REQUIRE(approx((a * x * x * x + b).eval(), A * std::pow(X, 3) + B));
        REQUIRE(approx((a * x.powN(3) + b).eval(), A * std::pow(X, 3) + B));
        REQUIRE(approx((a * x.powF(3) + b).eval(), A * std::pow(X, 3) + B));
    }

    SECTION("Utility functions")
    {
        // Test zero and one checks
        Scalar zero = ws.get_zero();
        Scalar one = ws.get_one();

        REQUIRE(zero.is_zero());
        REQUIRE(!one.is_zero());
        REQUIRE(one.is_one());
        REQUIRE(!zero.is_one());

        // Test get_value and eval
        REQUIRE(approx(3.5, a.get_value()));
        REQUIRE(approx(3.5, a.eval()));

        // Test creating constants
        Scalar constant = a.make_constant(7.5);
        compare(7.5, constant);

        // Test zero and one creation
        Scalar new_zero = a.get_zero();
        Scalar new_one = a.get_one();
        REQUIRE(new_zero.is_zero());
        REQUIRE(new_one.is_one());

        SECTION("Checksum consistency")
        {
            std::string checksum_1;
            {
                Workspace ws;
                Vector a = ws.make_vector(3);
                Vector b = ws.make_vector(3);
                checksum_1 = a.dot(b).get_checksum();
            }

            std::string checksum_2;
            {
                Workspace ws;
                Vector a = ws.make_vector(3);
                Vector b = ws.make_vector(3);
                checksum_2 = a.dot(b).get_checksum();
            }

            REQUIRE(checksum_1 == checksum_2);
        }
    }
}

TEST_CASE("Branches", "[scalar]")
{
    auto f = [](double a, double b, double c)
    {
        double x = c - 1.0;
        double f1;
        if (x >= 0.0)
        {
            f1 = a * std::pow(x, 3) - b;
        }
        else
        {
            f1 = std::cos(a) - a * b * x;
        }

        double y = b * a;
        double f2;
        if (y >= 0)
        {
            f2 = f1 * std::pow(y, 2);
        }
        else
        {
            f2 = a;
        }
        return f1 * x + f2;
    };

    Workspace ws;
    Scalar a = ws.make_scalar();
    Scalar b = ws.make_scalar();
    Scalar c = ws.make_scalar();

    Scalar x = c - 1.0;
    Scalar f1_positive = a * x.powN(3) - b;
    Scalar f1_negative = cos(a) - a * b * x;
    Scalar f1 = branch(c > 1.0, f1_positive, f1_negative);

    Scalar y = b * a;
    Scalar f2_positive = f1 * y.powN(2);
    Scalar f2_negative = a;
    Scalar f2 = branch(y > 0.0, f2_positive, f2_negative);

    Scalar E = f1 * x + f2;

    SECTION("Eval")
    {
        a.set_value(1.13);
        b.set_value(-7.37);
        c.set_value(3.3);
        REQUIRE(approx(E.eval(), f(1.13, -7.37, 3.3)));

        b.set_value(7.37);
        REQUIRE(approx(E.eval(), f(1.13, 7.37, 3.3)));
    }

    SECTION("Compiled")
    {
        Compiled<double> compiled({E}, "branch", symx::get_codegen_dir(), E.get_checksum());
        double A = 1.13;
        double B = -7.37;
        double C = 3.3;

        compiled.set(a, A);
        compiled.set(b, B);
        compiled.set(c, C);

        REQUIRE(approx(compiled.run()[0], f(A, B, C)));

        B = -B;
        compiled.set(b, B);
        REQUIRE(approx(compiled.run()[0], f(A, B, C)));
    }

    SECTION("Diff compiled vs evaluated")
    {
        std::vector<Scalar> vals = value_gradient_hessian(E, {a, b, c});
        double A = 1.13;
        double B = -7.37;
        double C = 3.3;

        // Evaluate
        a.set_value(A);
        b.set_value(B);
        c.set_value(C);
        std::vector<double> sol_eval(vals.size());
        for (int i = 0; i < (int)vals.size(); i++)
        {
            sol_eval[i] = vals[i].eval();
        }

        Compiled<double> compiled({vals}, "branch_diff", symx::get_codegen_dir(), E.get_checksum());
        compiled.set(a, A);
        compiled.set(b, B);
        compiled.set(c, C);
        const View<double> sol_compiled = compiled.run();

        for (int i = 0; i < (int)vals.size(); i++)
        {
            REQUIRE(approx(sol_eval[i], sol_compiled[i]));
        }
    }

    SECTION("Diff simple")
    {
        Workspace ws;
        Scalar a = ws.make_scalar();
        Scalar b = ws.make_scalar();
        Scalar c = ws.make_scalar();

        Scalar E = branch(c, a.powN(2), b.sqrt());

        std::vector<Scalar> vals = {diff(E, a), diff(E, b)};
        Compiled<double> compiled({vals}, "branch_diff_simple", symx::get_codegen_dir(), E.get_checksum());

        double A = 1.13;
        double B = 7.37;
        double C = 2.0;
        compiled.set(a, A);
        compiled.set(b, B);
        compiled.set(c, C);
        View<double> sol = compiled.run();

        REQUIRE(approx(sol[0], 2.0 * A));
        REQUIRE(approx(sol[1], 0.0));

        C = -2.0;
        compiled.set(c, C);
        sol = compiled.run();
        REQUIRE(approx(sol[0], 0.0));
        REQUIRE(approx(sol[1], 0.5 / std::sqrt(B)));
    }
}

TEST_CASE("Log edge cases", "[scalar]")
{
    Workspace ws;

    SECTION("Log of zero and negative values - evaluation")
    {
        Scalar x = ws.make_scalar();
        Scalar log_x = log(x);

        // Test log(0) = -infinity
        x.set_value(0.0);
        double result = log_x.eval();
        REQUIRE(std::isinf(result));
        REQUIRE(result < 0);

        // Test log(negative) = -infinity
        x.set_value(-2.5);
        result = log_x.eval();
        REQUIRE(std::isinf(result));
        REQUIRE(result < 0);

        // Test log(1) = 0
        x.set_value(1.0);
        result = log_x.eval();
        REQUIRE(approx(result, 0.0));

        // Test normal positive value
        x.set_value(2.718281828); // approximately e
        result = log_x.eval();
        REQUIRE(approx(result, 1.0, 1e-6));
    }

    SECTION("Log of zero and negative values - compiled")
    {

        Scalar x = ws.make_scalar();
        Scalar log_x = log(x);

        Compiled<double> compiled({log_x}, "log_edge_test", symx::get_codegen_dir());

        double X = -1.0;
        compiled.set(x, X);

        const View<double> result = compiled.run();
        REQUIRE(std::isinf(result[0]));          // it is ±∞
        REQUIRE(std::signbit(result[0]));        // and the sign bit is set → −∞

        #ifdef SYMX_ENABLE_AVX2
        Compiled<__m256d> compiled_simd({log_x}, "log_edge_test_simd", symx::get_codegen_dir());

        __m256d XX = _mm256_set1_pd(-1.0);
        compiled_simd.set(x, XX);

        const double* result_simd = reinterpret_cast<const double*>(compiled_simd.run().data());
        REQUIRE(std::isinf(result_simd[0]));          // it is ±∞
        REQUIRE(std::signbit(result_simd[0]));        // and the sign bit is set → −∞
        #endif
    }
}

TEST_CASE("Logarithm disambiguation tests", "[scalar]")
{
    Workspace ws;
    
    SECTION("Basic logarithm operations")
    {
        Scalar x = ws.make_scalar();
        x.set_value(100.0);
        
        // ln(100) = natural log of 100
        double ln_100 = x.ln().eval();
        REQUIRE(approx(ln_100, std::log(100.0)));
        
        // log10(100) = base-10 log of 100 = 2.0
        double log10_100 = x.log10().eval();
        REQUIRE(approx(log10_100, 2.0));
        
        // log(100) should work as natural logarithm (interchangeable with ln)
        double log_100 = x.log().eval();
        REQUIRE(approx(log_100, std::log(100.0)));
        
        // Verify ln and log give the same result (both map to ExprType::Ln internally)
        REQUIRE(approx(ln_100, log_100));
    }
    
    SECTION("Global function variants")
    {
        Scalar x = ws.make_scalar();
        x.set_value(10.0);
        
        // Test global function versions
        REQUIRE(approx(ln(x).eval(), std::log(10.0)));
        REQUIRE(approx(log10(x).eval(), 1.0));
        REQUIRE(approx(log(x).eval(), std::log(10.0)));
    }
    
    SECTION("Edge cases for all logarithm types")
    {
        Scalar x = ws.make_scalar();
        
        // Test with 1.0 (all should be 0)
        x.set_value(1.0);
        REQUIRE(approx(x.ln().eval(), 0.0));
        REQUIRE(approx(x.log10().eval(), 0.0));
        REQUIRE(approx(x.log().eval(), 0.0));
        
        // Test with e (ln should be 1, others different)
        x.set_value(std::exp(1.0));
        REQUIRE(approx(x.ln().eval(), 1.0));
        REQUIRE(approx(x.log10().eval(), std::log10(std::exp(1.0))));
        
        // Test with 10 (log10 should be 1, others different)
        x.set_value(10.0);
        REQUIRE(approx(x.log10().eval(), 1.0));
        REQUIRE(approx(x.ln().eval(), std::log(10.0)));
        
        // Test zero handling for all types
        x.set_value(0.0);
        REQUIRE((std::isinf(x.ln().eval()) && x.ln().eval() < 0));
        REQUIRE((std::isinf(x.log10().eval()) && x.log10().eval() < 0));
        REQUIRE((std::isinf(x.log().eval()) && x.log().eval() < 0));
        
        // Test negative handling for all types
        x.set_value(-1.0);
        REQUIRE((std::isinf(x.ln().eval()) && x.ln().eval() < 0));
        REQUIRE((std::isinf(x.log10().eval()) && x.log10().eval() < 0));
        REQUIRE((std::isinf(x.log().eval()) && x.log().eval() < 0));
    }
}
