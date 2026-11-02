#pragma once
#include <type_traits>
#include "../compile/MappedWorkspace.h"

// Source: https://www.researchgate.net/publication/267082822_Unified_Isoparametric_3D_LagrangeFinite_Elements

namespace symx
{
	enum class FEM_Element { Tet4, Tet10, Hex8, Hex27 };
	std::string get_name(const FEM_Element& element);
	int get_stride(const FEM_Element& element);

	inline std::vector<Scalar> fem_shape_functions(const FEM_Element& element, const Vector& xi);
	Matrix fem_shape_gradients(const FEM_Element& element, const Vector& xi);
	// B = D * JX.inv(), where D[l,a] = dN_l/dxi_a and JX = dX/dxi.
	Matrix fem_deformation_gradient_B(const FEM_Element& element, const Vector& xi, const Matrix& JX);
	template<typename T>
	T fem_interpolation(const FEM_Element& element, std::vector<T>& vh, const Vector& xi);

	// From nodal values and integration point
	Matrix fem_jacobian(const FEM_Element& element, std::vector<Vector>& xh, const Vector& xi);

	// From FEM interpolated point, and integration point
	Matrix fem_jacobian(const FEM_Element& element, const Vector& x, const Vector& xi);
	std::vector<std::array<double, 4>> get_integration_rule(const FEM_Element& element); // Returns {weight, xi0, xi1, xi2} quadrature points for the element type

	// Registers a Summation in the MappedWorkspace so that the compiled kernel is called
	// once per quadrature point. The summand lambda receives the weight w and reference
	// coordinates xi, and returns the integrand scalar. The result is the weighted sum
	// over all integration points (Gauss quadrature).
	Scalar fem_integrator(MappedWorkspace<double>& mws, const FEM_Element& element, std::function<Scalar(Scalar& w, Vector& xi)> summand);
	

	inline std::vector<Scalar> fem_shape_functions(const FEM_Element& element, const Vector& xi)
	{
		if (element == FEM_Element::Tet4) {

			std::vector<Scalar> Nh = { 1.0 - xi[0] - xi[1] - xi[2],
				xi[0],
				xi[1],
				xi[2] };
			return Nh;
		}
		else if (element == FEM_Element::Tet10) {

			Scalar N0 = 1.0 - xi[0] - xi[1] - xi[2];
			Scalar N1 = xi[0];
			Scalar N2 = xi[1];
			Scalar N3 = xi[2];
			std::vector<Scalar> Nh = { 
				N0 * (2.0 * N0 - 1.0),
				N1 * (2.0 * N1 - 1.0),
				N2 * (2.0 * N2 - 1.0),
				N3 * (2.0 * N3 - 1.0),
				4.0 * N0 * N1,
				4.0 * N1 * N2,
				4.0 * N2 * N0,
				4.0 * N0 * N3,
				4.0 * N1 * N3,
				4.0 * N2 * N3 };
			return Nh;
		}
		else if (element == FEM_Element::Hex8) {
			Scalar NXm = 0.5 * (1.0 - xi[0]);
			Scalar NXp = 0.5 * (1.0 + xi[0]);
			Scalar NYm = 0.5 * (1.0 - xi[1]);
			Scalar NYp = 0.5 * (1.0 + xi[1]);
			Scalar NZm = 0.5 * (1.0 - xi[2]);
			Scalar NZp = 0.5 * (1.0 + xi[2]);
			std::vector<Scalar> Nh = { 
				NXm * NYm * NZm,
				NXp * NYm * NZm,
				NXp * NYp * NZm,
				NXm * NYp * NZm,
				NXm * NYm * NZp,
				NXp * NYm * NZp,
				NXp * NYp * NZp,
				NXm * NYp * NZp };
			return Nh;
		}
		else if (element == FEM_Element::Hex27) {

			auto N1m = [](const Scalar& x) { return x * (x - 1.0); };
			auto N1p = [](const Scalar& x) { return x * (x + 1.0); };
			auto N2 = [](const Scalar& x) { return (1.0 - x.powN(2)); };
			const Scalar& x = xi[0];
			const Scalar& y = xi[1];
			const Scalar& z = xi[2];

			std::vector<Scalar> Nh = {
				// Corner nodes
				N1m(x)*N1m(y)*N1m(z)/8.0,
				N1p(x)*N1m(y)*N1m(z)/8.0,
				N1p(x)*N1p(y)*N1m(z)/8.0,
				N1m(x)*N1p(y)*N1m(z)/8.0,
				N1m(x)*N1m(y)*N1p(z)/8.0,
				N1p(x)*N1m(y)*N1p(z)/8.0,
				N1p(x)*N1p(y)*N1p(z)/8.0,
				N1m(x)*N1p(y)*N1p(z)/8.0,

				// Mid-edge nodes
				N2(x)*N1m(y)*N1m(z)/4.0,
				N1p(x)*N2(y)*N1m(z)/4.0,
				N2(x)*N1p(y)*N1m(z)/4.0,
				N1m(x)*N2(y)*N1m(z)/4.0,
				N2(x)*N1m(y)*N1p(z)/4.0,
				N1p(x)*N2(y)*N1p(z)/4.0,
				N2(x)*N1p(y)*N1p(z)/4.0,
				N1m(x)*N2(y)*N1p(z)/4.0,
				N1m(x)*N1m(y)*N2(z)/4.0,
				N1p(x)*N1m(y)*N2(z)/4.0,
				N1p(x)*N1p(y)*N2(z)/4.0,
				N1m(x)*N1p(y)*N2(z)/4.0,

				// Mid-face nodes
				N1m(x)*N2(y)*N2(z)/2.0,
				N1p(x)*N2(y)*N2(z)/2.0,
				N2(x)*N1p(y)*N2(z)/2.0,
				N2(x)*N1m(y)*N2(z)/2.0,
				N2(x)*N2(y)*N1m(z)/2.0,
				N2(x)*N2(y)*N1p(z)/2.0,

				// Mid-volume node
				N2(x)*N2(y)*N2(z)
			};
			return Nh;
		}
		else {
			std::cout << "symx error: fem_shape_functions() got unhandled element type." << std::endl;
			exit(-1);
		}
	}

	template<typename T>
	T fem_interpolation(const FEM_Element& element, std::vector<T>& vh, const Vector& xi)
	{
		if constexpr (!std::is_same_v<T, Scalar> && !std::is_same_v<T, Vector> && !std::is_same_v<T, Matrix>) {
			std::cout << "SymX error: fem_interpolation<T>() got not symx type as input (Scalar, Vector, Matrix)." << std::endl;
			exit(-1);
		}

		std::vector<Scalar> Nh = fem_shape_functions(element, xi);
		if (vh.size() != Nh.size()) {
			std::cout << "symx error: fem_interpolation() got " << vh.size()
			          << " nodal values; expected " << Nh.size() << "." << std::endl;
			exit(-1);
		}

		T x = Nh[0] * vh[0];
		for (size_t i = 1; i < Nh.size(); ++i) {
			x += Nh[i] * vh[i];
		}
		return x;
	}
}
