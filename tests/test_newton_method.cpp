#include "catch.hpp"

#include <Eigen/Dense>
#include <symx>

#include "utils.h"

using namespace symx;

TEST_CASE("Polynomial Newton's Method", "[NewtonsMethod]")
{
	// Data
	std::vector<std::array<int, 1>> conn_;
	std::vector<Eigen::Vector3d> target_;
	std::vector<Eigen::Vector3d> x_;

    conn_.push_back({0});
    target_.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
    x_.push_back(-Eigen::Vector3d::Ones());

    // Setup
    spGlobalPotential G = GlobalPotential::create();

    // Energy
	G->add_potential("Polynomial", conn_,
		[&](MappedWorkspace<double>& mws, Element& elem)
		{
            Vector x = mws.make_vector(x_, elem[0]);
            Vector target = mws.make_vector(target_, elem[0]);
            Scalar E = (x - target).squared_norm().powN(2);
            return E;
		}
	);

    // DoF
    G->add_dof(x_);

    // Context
    spContext context = Context::create();
    context->output->set_enabled(false);

    // Solver
    NewtonsMethod newton(G, context);
    SolverReturn ret = newton.solve();
    
    // Check
    REQUIRE(ret == SolverReturn::Successful);
    REQUIRE(approx(x_[0], target_[0], 1e-1));
}
