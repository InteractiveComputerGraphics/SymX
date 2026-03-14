#pragma once
#include <symx>
#include <Eigen/Dense>
#include "catch.hpp"

inline bool approx(double a, double b, double tol = 1e-12)
{
	return std::abs(a - b) < tol;
}

template<typename STATIC_ARRAY>
inline bool approx(const STATIC_ARRAY& a, const STATIC_ARRAY& b, double tol = 1e-12)
{
	bool ok = true;
	for (int i = 0; i < (int)a.size(); i++) {
		ok = ok && approx(a[i], b[i], tol);
	}
	return ok;
}

// Helper comparison functions for different types
template<std::size_t N>
inline void compare(const Eigen::Matrix<double, N, N>& A, const symx::Matrix& A_, const double tol = 1e-12)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			REQUIRE(approx(A(i, j), A_(i, j).eval(), tol));
		}
	}
}

template<std::size_t N>
inline void compare(const Eigen::Matrix<double, N, 1>& x, const symx::Vector& x_, const double tol = 1e-12)
{
	for (int i = 0; i < N; i++) {
		REQUIRE(approx(x[i], x_[i].eval(), tol));
	}
}

inline void compare(const double v, const symx::Scalar& v_, const double tol = 1e-12)
{
	REQUIRE(approx(v, v_.eval(), tol));
}

inline void compare(const Eigen::Matrix<double, Eigen::Dynamic, 1>& x, const symx::Vector& x_, const double tol = 1e-12)
{
	for (int i = 0; i < x.size(); i++) {
		REQUIRE(approx(x[i], x_[i].eval(), tol));
	}
}

inline void compare(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& A, const symx::Matrix& A_, const double tol = 1e-12)
{
	for (int i = 0; i < A.rows(); i++) {
		for (int j = 0; j < A.cols(); j++) {
			REQUIRE(approx(A(i, j), A_(i, j).eval(), tol));
		}
	}
}
