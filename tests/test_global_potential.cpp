#include "catch.hpp"

#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <symx>

#include "utils.h"

using namespace symx;

// ─────────────────────────────────────────────────────────────────────────────
template<typename STORAGE_FLOAT = double>
inline Eigen::MatrixXd to_eigen(bsm::BlockedSparseMatrix<3, 3, STORAGE_FLOAT>& s, const int ndofs)
{
	Eigen::MatrixXd m(ndofs, ndofs);
	std::vector<Eigen::Triplet<double>> triplets;
	s.to_triplets(triplets);

	m.setZero();
	for (const auto& t : triplets) {
		m(t.row(), t.col()) = t.value();
	}
	return m;
}
// ─────────────────────────────────────────────────────────────────────────────


TEST_CASE("Stable Neo-Hookean", "[GlobalEnergy]")
{
	const int n_elements = 100;

	// Data (grouped for readability)
	struct Data {
		std::vector<Eigen::Vector3d> x1;
		std::vector<Eigen::Vector3d> X;
		std::array<double, 2> lame;
		std::vector<std::array<int, 5>> numbered_tets;
		std::vector<std::array<double, 9>> J;
		std::vector<double> volume;
	} data;

	data.numbered_tets = std::vector<std::array<int, 5>>(n_elements, {0, 0, 1, 2, 3});
	data.lame          = {1e4, 0.3};
	data.X             = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	data.x1            = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}};
	data.volume.push_back(std::abs((1.0 / 6.0) * ((data.X[1] - data.X[0]).cross(data.X[2] - data.X[0])).dot(data.X[3] - data.X[0])));
	data.J.push_back({
		// Row-major 3x3 Jacobian
		-data.X[0][0] + data.X[1][0],  -data.X[0][0] + data.X[2][0],  -data.X[0][0] + data.X[3][0],
		-data.X[0][1] + data.X[1][1],  -data.X[0][1] + data.X[2][1],  -data.X[0][1] + data.X[3][1],
		-data.X[0][2] + data.X[1][2],  -data.X[0][2] + data.X[2][2],  -data.X[0][2] + data.X[3][2]
	});
	const int ndofs = static_cast<int>(data.x1.size()) * 3;

	// Energy
	spGlobalPotential G = GlobalPotential::create();

	G->add_potential("stable_neo_hookean_tet4", data.numbered_tets, 
		[&](MappedWorkspace<double>& mws, Element& tet)
		{
			Vector lame = mws.make_vector(data.lame);
			Scalar volume = mws.make_scalar(data.volume, tet[0]);
			Matrix J = mws.make_matrix(data.J, { 3, 3 }, tet[0]);
			std::vector<Vector> x1 = mws.make_vectors(data.x1, tet.slice(1, 5));

			Scalar E = lame[0];
			Scalar nu = lame[1];
			Scalar mu = E / (2.0 * (1.0 + nu));
			Scalar lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
			Scalar mu_ = 4.0 / 3.0 * mu;
			Scalar lambda_ = lambda + 5.0 / 6.0 * mu;
			Scalar alpha = 1.0 + mu_ / lambda_ - mu_ / (4.0 * lambda_);

			Matrix j = Matrix({
				// 3x3 matrix in row-major form for readability
				-x1[0][0] + x1[1][0],  -x1[0][0] + x1[2][0],  -x1[0][0] + x1[3][0],
				-x1[0][1] + x1[1][1],  -x1[0][1] + x1[2][1],  -x1[0][1] + x1[3][1],
				-x1[0][2] + x1[1][2],  -x1[0][2] + x1[2][2],  -x1[0][2] + x1[3][2]
			}, {3, 3});
			Matrix F = j * J.inv();
			Scalar detF = F.det();
			Scalar Ic = F.frobenius_norm_sq();
			Scalar energy_density = 0.5 * mu_ * (Ic - 3.0) + 0.5 * lambda_ * (detF - alpha).powN(2) - 0.5 * mu_ * log(Ic + 1.0);
			return volume * energy_density;
		}
	);

	// DoF declaration
	G->add_dof(data.x1);

	// Compilation
	spContext context = Context::create();
	context->n_threads = 1;
	context->output->set_enabled(false);
	SecondOrderCompiledGlobal compiled(G, context);

	// Test with FD
	if constexpr (std::is_same_v<ElementHessians::HESS_STORAGE_FLOAT, double>)
	{
		REQUIRE(compiled.test_derivatives_with_FD(/* h = */ 1e-7) < 0.002);
	}
	else {
		REQUIRE(compiled.test_derivatives_with_FD(/* h = */ 1e-5) < 0.5);
	}

	// DoF manipulation
	Eigen::VectorXd u = Eigen::VectorXd::Ones(12);
	Eigen::VectorXd du = Eigen::VectorXd::Ones(12);
	G->get_dofs(u.data());
	u += du;
	REQUIRE(approx(data.x1[0][0], 0.0));
	G->set_dofs(u.data());
	REQUIRE(approx(data.x1[0][0], 1.0));
	du *= -1.0;
	G->apply_dof_increment(du.data());
	REQUIRE(approx(data.x1[0][0], 0.0));

	// Evaluate
    double P = 0.0;
    Eigen::VectorXd grad;
	auto element_hessians = compiled.evaluate_P__dP_du__local_d2P_du2(P, grad);

    // Assemble global Hessian
    auto hess = element_hessians->assemble_global(context->n_threads, ndofs);

	// Build a MappedWorkspace and expressions mirroring the energy
	auto [mws, conn] = MappedWorkspace<double>::create(data.numbered_tets);
	std::vector<Vector> x1 = mws->make_vectors(data.x1, conn.slice(1, 5));
	Vector lame = mws->make_vector(data.lame);
	Scalar E = lame[0];
	Scalar nu = lame[1];
	Scalar mu = E / (2.0 * (1.0 + nu));
	Scalar lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
	Scalar mu_snh = 4.0 / 3.0 * mu;
	Scalar lambda_snh = lambda + 5.0 / 6.0 * mu;
	Scalar alpha = 1.0 + mu_snh / lambda_snh - mu_snh / (4.0 * lambda_snh);
	Matrix J = mws->make_matrix(data.J, {3, 3}, conn[0]);
	Scalar volume = mws->make_scalar(data.volume, conn[0]);
	Matrix j({ -x1[0][0] + x1[1][0], -x1[0][0] + x1[2][0], -x1[0][0] + x1[3][0],
			   -x1[0][1] + x1[1][1], -x1[0][1] + x1[2][1], -x1[0][1] + x1[3] [1],
			   -x1[0][2] + x1[1][2], -x1[0][2] + x1[2][2], -x1[0][2] + x1[3][2] }, {3, 3});
	Matrix F = j * J.inv();
	Scalar detF = F.det();
	Scalar Ic = F.frobenius_norm_sq();
	Scalar energy_density = 0.5 * mu_snh * (Ic - 3.0) + 0.5 * lambda_snh * (detF - alpha).powN(2) - 0.5 * mu_snh * log(Ic + 1.0);
	Scalar energy = volume * energy_density;
	std::vector<Scalar> vals = value_gradient_hessian(energy, collect_scalars(x1));

	CompiledInLoop<double> loop(mws, vals, "snh_loop", symx::get_codegen_dir());
	double sum_E = 0.0;
	Eigen::VectorXd sum_grad = Eigen::VectorXd::Zero(12);
	Eigen::Matrix<double, 12, 12> sum_hess = Eigen::Matrix<double,12,12>::Zero();
	loop.run(1,
		[&](const View<double> solution, const int32_t element_idx, const int32_t thread_id, const View<int32_t> connectivity)
		{
			sum_E += solution[0];
			View<double> grad_v = solution.slice(1);
			for (int i = 0; i < 12; ++i) sum_grad[i] += grad_v[i];
			View<double> hess_v = grad_v.slice(12);
			for (int i = 0; i < 12; ++i) {
				for (int j = 0; j < 12; ++j) {
					sum_hess(i,j) += hess_v[12*i + j];
				}
			}
		}
	);

	// Compare aggregated loop results to assembled values
	REQUIRE(approx(P, sum_E, 1e-8));
	REQUIRE(grad.isApprox(sum_grad, 1e-8));
	Eigen::MatrixXd hessx = to_eigen(*hess, 12);
	
	if constexpr (std::is_same_v<ElementHessians::HESS_STORAGE_FLOAT, double>)
	{
		REQUIRE(hessx.isApprox(sum_hess, 1e-8));
	}
	else {
		REQUIRE(hessx.isApprox(sum_hess, 1e-6));
	}
}

TEST_CASE("Two energies with conditional", "[GlobalEnergy]")
{
	const int n_vertices = 100;

	double k_ = 1.0;
	std::vector<Eigen::Vector3d> x_;
	std::vector<std::array<int, 1>> conn;

	for (int i = 0; i < n_vertices; i++) {
		x_.push_back(Eigen::Vector3d::Random());
		conn.push_back({ i });
	}
	const int ndofs = static_cast<int>(x_.size()) * 3;
	
	// Energy
	spGlobalPotential G = GlobalPotential::create();

	G->add_potential("attract_to_zero", conn, 
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			Index vertex = conn[0];
			Vector x = mws.make_vector(x_, vertex);
			Scalar k = mws.make_scalar(k_);
			Scalar energy = 0.5 * k * x.squared_norm();
			return energy;
		}
	);

	G->add_potential("attract_to_zero_if_z_plus", conn, 
		[&](MappedWorkspace<double>& mws, Element& conn) -> std::pair<Scalar, Scalar>
		{
			Index vertex = conn[0];
			Vector x = mws.make_vector(x_, vertex);
			Scalar k = mws.make_scalar(k_);
			Scalar energy = 0.5 * k * x.squared_norm();
			Scalar condition = x[2] > 0.0;
			return {energy, condition};
		}
	);

	// DoF declaration
	G->add_dof(x_);

	// Compilation
	spContext context = Context::create();
	context->n_threads = 1;
	context->output->set_enabled(false);
	// context->print = [](const std::string& msg) { }; // Suppress output
	SecondOrderCompiledGlobal compiled(G, context);

	// Test with FD
	REQUIRE(compiled.test_derivatives_with_FD(/* h = */ 1e-5) < 1e-5);

	// Evaluate
    double P_symx = 0.0;
    Eigen::VectorXd grad_symx;
	auto element_hessians = compiled.evaluate_P__dP_du__local_d2P_du2(P_symx, grad_symx);

    // Assemble global Hessian
    auto hess_symx = element_hessians->assemble_global(context->n_threads, ndofs);
	Eigen::MatrixXd hess_symx_eigen = to_eigen(*hess_symx, ndofs);

	// Test the result analytically
	double P = 0.0;
	Eigen::VectorXd grad = Eigen::VectorXd::Zero(ndofs);
	Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(ndofs, ndofs);

	// Check the first energy
	for (int i = 0; i < n_vertices; i++) {
		P += 0.5 * k_ * x_[i].squaredNorm();
		grad.segment<3>(3*i) = k_ * x_[i];
		hess.block<3, 3>(3*i, 3*i) = k_ * Eigen::Matrix3d::Identity();
	}

	// Test the conditional energy
	for (int i = 0; i < n_vertices; i++) {
		if (x_[i][2] > 0.0) {
			P += 0.5 * k_ * x_[i].squaredNorm();
			grad.segment<3>(3*i) += k_ * x_[i];
			hess.block<3, 3>(3*i, 3*i) += k_ * Eigen::Matrix3d::Identity();
		}
	}

	// Check the results
	REQUIRE(approx(P_symx, P, 1e-8));
	REQUIRE(grad_symx.isApprox(grad, 1e-8));
	REQUIRE(hess_symx_eigen.isApprox(hess, 1e-8));
}
