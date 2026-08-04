#include "catch.hpp"

#include <Eigen/Dense>
#include <symx>

#include "utils.h"

using namespace symx;

namespace
{
    spContext make_first_order_context()
    {
        spContext context = Context::create();
        context->output->set_enabled(false);
        context->n_threads = 1;
        return context;
    }

    struct VectorQuadratic
    {
        std::vector<std::array<int, 1>> connectivity = {{0}};
        std::vector<Eigen::Vector3d> x = {Eigen::Vector3d(-1.0, 2.0, -3.0)};
        std::vector<Eigen::Vector3d> target = {Eigen::Vector3d(1.0, -2.0, 0.5)};
        spGlobalPotential potential = GlobalPotential::create();
        spContext context = make_first_order_context();

        VectorQuadratic(const std::string& name = "first_order_quadratic")
        {
            potential->add_potential(name, connectivity,
                [&](MappedWorkspace<double>& mws, Element& element)
                {
                    const Vector xi = mws.make_vector(x, element[0]);
                    const Vector ti = mws.make_vector(target, element[0]);
                    return 0.5 * (xi - ti).squared_norm();
                });
            potential->add_dof(x, "x");
        }
    };

    struct ScalarQuadratic
    {
        std::vector<std::array<int, 1>> connectivity = {{0}};
        std::vector<double> x = {1.0};
        spGlobalPotential potential = GlobalPotential::create();

        ScalarQuadratic(const std::string& name)
        {
            potential->add_potential(name, connectivity,
                [&](MappedWorkspace<double>& mws, Element& element)
                {
                    const Scalar xi = mws.make_scalar(x, element[0]);
                    return 0.5 * xi * xi;
                });
            potential->add_dof(x, "x");
        }
    };

    void configure(FirstOrderOptimizer& optimizer)
    {
        optimizer.settings.max_iterations = 300;
        optimizer.settings.residual_tolerance_abs = 1e-8;
        optimizer.settings.step_tolerance = 1e-12;
        optimizer.settings.max_backtracking_armijo_iterations = 40;
    }
}

TEST_CASE("FirstOrderOptimizer solves a vector quadratic with every method", "[FirstOrderOptimizer]")
{
    const auto solve = [](FirstOrderOptType type, const std::string& name) {
        VectorQuadratic problem(name);
        FirstOrderOptimizer optimizer(problem.potential, problem.context);
        configure(optimizer);
        optimizer.settings.type = type;
        optimizer.settings.gradient_descent.learning_rate = 0.5;
        optimizer.settings.adam.learning_rate = 0.2;
        optimizer.settings.vector_adam.learning_rate = 0.2;
        optimizer.settings.lbfgs.bootstrap_step_length = 1.0;

        INFO("type = " << to_string(type));
        REQUIRE(optimizer.solve() == SolverReturn::Successful);
        REQUIRE(approx(problem.x[0], problem.target[0], 1e-4));
    };

    solve(FirstOrderOptType::GradientDescent, "first_order_quadratic_gd");
    solve(FirstOrderOptType::Adam, "first_order_quadratic_adam");
    solve(FirstOrderOptType::VectorAdam, "first_order_quadratic_vector_adam");
    solve(FirstOrderOptType::LBFGS, "first_order_quadratic_lbfgs");
}

TEST_CASE("FirstOrderOptimizer uses normal solver callbacks", "[FirstOrderOptimizer]")
{
    VectorQuadratic problem("first_order_callbacks");
    spSolverCallbacks callbacks = SolverCallbacks::create(problem.context);
    int before_step = 0;
    int after_step = 0;
    int before_energy = 0;
    int intermediate_valid = 0;
    callbacks->add_before_step([&]() { before_step++; });
    callbacks->add_after_step([&]() { after_step++; });
    callbacks->add_before_energy_evaluation([&]() { before_energy++; });
    callbacks->add_is_intermediate_state_valid([&]() { intermediate_valid++; return true; });

    FirstOrderOptimizer optimizer(problem.potential, problem.context, callbacks);
    configure(optimizer);
    optimizer.settings.type = FirstOrderOptType::GradientDescent;
    optimizer.settings.gradient_descent.learning_rate = 0.5;

    REQUIRE(optimizer.solve() == SolverReturn::Successful);
    REQUIRE(before_step > 0);
    REQUIRE(after_step == before_step);
    REQUIRE(before_energy > 0);
    REQUIRE(intermediate_valid > 0);
}

TEST_CASE("FirstOrderOptimizer VectorAdam is rotation equivariant", "[FirstOrderOptimizer]")
{
    const auto one_step = [](const Eigen::Vector3d& x0, const std::string& name) {
        std::vector<std::array<int, 1>> connectivity = {{0}};
        std::vector<Eigen::Vector3d> x = {x0};
        spGlobalPotential potential = GlobalPotential::create();
        potential->add_potential(name, connectivity,
            [&](MappedWorkspace<double>& mws, Element& element)
            {
                const Vector xi = mws.make_vector(x, element[0]);
                return 0.5 * xi.squared_norm();
            });
        potential->add_dof(x, "x");

        FirstOrderOptimizer optimizer(potential, make_first_order_context());
        optimizer.settings.type = FirstOrderOptType::VectorAdam;
        optimizer.settings.vector_adam.learning_rate = 0.25;
        optimizer.settings.max_iterations = 1;
        optimizer.settings.max_iterations_as_success = true;
        optimizer.settings.residual_tolerance_abs = 0.0;
        optimizer.settings.enable_armijo_backtracking = false;
        REQUIRE(optimizer.solve() == SolverReturn::Successful);
        return x[0];
    };

    const Eigen::AngleAxisd rotation(0.7, Eigen::Vector3d::UnitZ());
    const Eigen::Vector3d x0(2.0, -1.0, 0.5);
    const Eigen::Vector3d result = one_step(x0, "first_order_rotation");
    const Eigen::Vector3d rotated_result = one_step(rotation * x0, "first_order_rotation_rotated");
    REQUIRE(approx(rotated_result, rotation * result, 1e-12));
}

TEST_CASE("FirstOrderOptimizer honors conditional potentials", "[FirstOrderOptimizer]")
{
    std::vector<std::array<int, 1>> connectivity = {{0}, {1}};
    std::vector<Eigen::Vector3d> x = {
        Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(10.0, 0.0, 0.0)};
    std::vector<double> active = {1.0, -1.0};
    spGlobalPotential potential = GlobalPotential::create();
    potential->add_potential("first_order_conditional", connectivity,
        [&](MappedWorkspace<double>& mws, Element& element)
        {
            const Vector xi = mws.make_vector(x, element[0]);
            const Scalar ai = mws.make_scalar(active, element[0]);
            return std::make_pair(0.5 * xi.squared_norm(), ai);
        });
    potential->add_dof(x, "x");

    FirstOrderOptimizer optimizer(potential, make_first_order_context());
    optimizer.settings.type = FirstOrderOptType::GradientDescent;
    optimizer.settings.gradient_descent.learning_rate = 1.0;
    optimizer.settings.max_iterations = 1;
    optimizer.settings.max_iterations_as_success = true;
    optimizer.settings.residual_tolerance_abs = 0.0;
    optimizer.settings.enable_armijo_backtracking = false;

    REQUIRE(optimizer.solve() == SolverReturn::Successful);
    REQUIRE(x[0].norm() < 1e-12);
    REQUIRE(approx(x[1], Eigen::Vector3d(10.0, 0.0, 0.0), 1e-12));
}

TEST_CASE("FirstOrderOptimizer validates VectorAdam blocks", "[FirstOrderOptimizer]")
{
    std::vector<std::array<int, 1>> connectivity = {{0}};
    std::vector<double> x = {1.0};
    spGlobalPotential potential = GlobalPotential::create();
    potential->add_potential("first_order_invalid_vector_adam", connectivity,
        [&](MappedWorkspace<double>& mws, Element& element)
        {
            const Scalar xi = mws.make_scalar(x, element[0]);
            return 0.5 * xi * xi;
        });
    potential->add_dof(x, "x");

    FirstOrderOptimizer optimizer(potential, make_first_order_context());
    optimizer.settings.type = FirstOrderOptType::VectorAdam;
    optimizer.settings.vector_adam.vector_block_size = 3;
    REQUIRE_THROWS_AS(optimizer.solve(), std::invalid_argument);
}

TEST_CASE("FirstOrderOptimizer resets optimizer state for repeated solves", "[FirstOrderOptimizer]")
{
    VectorQuadratic problem("first_order_repeat");
    FirstOrderOptimizer optimizer(problem.potential, problem.context);
    configure(optimizer);
    optimizer.settings.type = FirstOrderOptType::VectorAdam;
    optimizer.settings.vector_adam.learning_rate = 0.2;
    REQUIRE(optimizer.solve() == SolverReturn::Successful);

    problem.x[0] = Eigen::Vector3d(-3.0, 4.0, -2.0);
    REQUIRE(optimizer.solve() == SolverReturn::Successful);
    REQUIRE(approx(problem.x[0], problem.target[0], 1e-4));
}

TEST_CASE("FirstOrderOptimizer restarts a non-descent direction", "[FirstOrderOptimizer]")
{
    VectorQuadratic problem("first_order_restart");
    const double initial_x = problem.x[0][0];
    FirstOrderOptimizer optimizer(problem.potential, problem.context);
    optimizer.settings.type = FirstOrderOptType::GradientDescent;
    optimizer.settings.gradient_descent.learning_rate = -0.1;
    optimizer.settings.max_iterations = 1;
    optimizer.settings.max_iterations_as_success = true;
    optimizer.settings.residual_tolerance_abs = 0.0;
    optimizer.settings.enable_armijo_backtracking = false;

    REQUIRE(optimizer.solve() == SolverReturn::Successful);
    REQUIRE(problem.x[0][0] > initial_x);
}

TEST_CASE("FirstOrderOptimizer records rejected L-BFGS curvature pairs", "[FirstOrderOptimizer]")
{
    std::vector<std::array<int, 1>> connectivity = {{0}};
    std::vector<Eigen::Vector3d> x = {Eigen::Vector3d::Zero()};
    spGlobalPotential potential = GlobalPotential::create();
    potential->add_potential("first_order_linear", connectivity,
        [&](MappedWorkspace<double>& mws, Element& element)
        {
            const Vector xi = mws.make_vector(x, element[0]);
            return xi[0];
        });
    potential->add_dof(x, "x");

    FirstOrderOptimizer optimizer(potential, make_first_order_context());
    optimizer.settings.type = FirstOrderOptType::LBFGS;
    optimizer.settings.max_iterations = 3;
    optimizer.settings.max_iterations_as_success = true;
    optimizer.settings.residual_tolerance_abs = 0.0;
    optimizer.settings.enable_armijo_backtracking = false;
    REQUIRE(optimizer.solve() == SolverReturn::Successful);
    REQUIRE(optimizer.get_last_solve_stats().lbfgs_history_skips > 0);
}

TEST_CASE("FirstOrderOptimizer restores DoFs after a failed line search", "[FirstOrderOptimizer]")
{
    SECTION("invalid intermediate state")
    {
        ScalarQuadratic problem("first_order_invalid_rollback");
        spContext context = make_first_order_context();
        spSolverCallbacks callbacks = SolverCallbacks::create(context);
        int invalid_callback_count = 0;
        callbacks->add_is_intermediate_state_valid([]() { return false; });
        callbacks->add_on_intermediate_state_invalid([&]() { invalid_callback_count++; });

        FirstOrderOptimizer optimizer(problem.potential, context, callbacks);
        optimizer.settings.type = FirstOrderOptType::GradientDescent;
        optimizer.settings.gradient_descent.learning_rate = 0.5;
        optimizer.settings.max_backtracking_invalid_state_iterations = 1;
        REQUIRE(optimizer.solve() == SolverReturn::TooManyInvalidIntermediateIterations);
        REQUIRE(approx(problem.x[0], 1.0));
        REQUIRE(invalid_callback_count == 1);
    }

    SECTION("Armijo failure")
    {
        ScalarQuadratic problem("first_order_armijo_rollback");
        spContext context = make_first_order_context();
        spSolverCallbacks callbacks = SolverCallbacks::create(context);
        int armijo_callback_count = 0;
        callbacks->add_on_armijo_fail([&]() { armijo_callback_count++; });

        FirstOrderOptimizer optimizer(problem.potential, context, callbacks);
        optimizer.settings.type = FirstOrderOptType::GradientDescent;
        optimizer.settings.gradient_descent.learning_rate = 0.1;
        optimizer.settings.line_search_armijo_beta = 2.0;
        optimizer.settings.max_backtracking_armijo_iterations = 2;
        REQUIRE(optimizer.solve() == SolverReturn::TooManyArmijoIterations);
        REQUIRE(approx(problem.x[0], 1.0));
        REQUIRE(armijo_callback_count == 1);
    }
}
