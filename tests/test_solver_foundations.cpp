#include "catch.hpp"

#include <Eigen/Dense>
#include <symx>

#include "utils.h"

using namespace symx;

namespace
{
    struct QuadraticProblem
    {
        std::vector<std::array<int, 1>> conn{{0}};
        std::vector<Eigen::Vector3d> x{Eigen::Vector3d::Ones()};
        std::vector<Eigen::Vector3d> target{Eigen::Vector3d::Zero()};
        spGlobalPotential potential = GlobalPotential::create();
        spContext context = Context::create();

        QuadraticProblem()
        {
            potential->add_potential("quadratic", conn,
                [&](MappedWorkspace<double>& mws, Element& elem)
                {
                    const Vector x_sym = mws.make_vector(x, elem[0]);
                    const Vector target_sym = mws.make_vector(target, elem[0]);
                    return (x_sym - target_sym).squared_norm();
                });
            potential->add_dof(x);
            context->output->set_enabled(false);
        }
    };

    spElementHessians make_element_hessians(const bool coupled)
    {
        auto element_hessians = std::make_shared<ElementHessians>();
        element_hessians->start(1);

        if (coupled) {
            WriteHessian write = element_hessians->new_hessian(0, 2);
            write.block_rows[0] = 0;
            write.block_rows[1] = 1;
            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 6; ++col) {
                    write.values[6 * row + col] = row == col ? 2.0 : 0.1;
                }
            }
        }
        else {
            for (int block = 0; block < 2; ++block) {
                WriteHessian write = element_hessians->new_hessian(0, 1);
                write.block_rows[0] = block;
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        write.values[3 * row + col] = row == col ? 2.0 : 0.0;
                    }
                }
            }
        }
        element_hessians->stop();
        element_hessians->assemble_global(1, 6);
        return element_hessians;
    }
}

TEST_CASE("Solver callbacks and direct solve context", "[Solver]")
{
    SECTION("before and after step callbacks are paired on an early exit")
    {
        QuadraticProblem problem;
        auto callbacks = SolverCallbacks::create(problem.context);
        int before = 0;
        int after = 0;
        callbacks->add_before_step([&] { before++; });
        callbacks->add_after_step([&] { after++; });
        callbacks->set_residual([](const Eigen::VectorXd& residual) {
            return residual.cwiseAbs().maxCoeff();
        });

        auto newton = NewtonsMethod::create(problem.potential, problem.context, callbacks);
        newton->settings.max_iterations = 0;
        newton->settings.iteration_offset = 17;

        REQUIRE(newton->solve() == SolverReturn::TooManyIterations);
        REQUIRE(before == 1);
        REQUIRE(after == 1);
    }

    SECTION("direct LLT reuses its symbolic pattern across compatible solves")
    {
        QuadraticProblem problem;
        auto newton = NewtonsMethod::create(problem.potential, problem.context);
        newton->settings.linear_solver = LinearSolver::DirectLLT;

        REQUIRE(newton->solve() == SolverReturn::Successful);
        problem.x[0] = Eigen::Vector3d::Ones();
        REQUIRE(newton->solve() == SolverReturn::Successful);
        REQUIRE(approx(problem.x[0], Eigen::Vector3d(Eigen::Vector3d::Zero())));
    }

    SECTION("direct LLT reanalyses when the actual pattern changes")
    {
        LinearSolveContext linear_solve_context;
        Eigen::VectorXd rhs = Eigen::VectorXd::Ones(6);
        Eigen::VectorXd x;

        const auto uncoupled = make_element_hessians(false);
        REQUIRE(linear_solve_context.solve_direct_llt(uncoupled->hess, x, rhs, 1));
        const Eigen::VectorXd uncoupled_expected = Eigen::VectorXd::Constant(6, 0.5);
        REQUIRE(approx(x, uncoupled_expected));

        const auto coupled = make_element_hessians(true);
        REQUIRE(linear_solve_context.solve_direct_llt(coupled->hess, x, rhs, 1));
        const Eigen::VectorXd coupled_expected = Eigen::VectorXd::Constant(6, 0.4);
        REQUIRE(approx(x, coupled_expected));

        REQUIRE(linear_solve_context.solve_direct_llt(coupled->hess, x, rhs, 1));
        REQUIRE(approx(x, coupled_expected));
        REQUIRE(linear_solve_context.solve_direct_llt(uncoupled->hess, x, rhs, 1));
        REQUIRE(approx(x, uncoupled_expected));
    }
}

TEST_CASE("Thread-local accumulators require an explicit reduction", "[Solver]")
{
    bsm::ParallelNumber<double> scalar;
    scalar.begin_accumulation(4);
    REQUIRE_THROWS_AS(scalar.result(), std::logic_error);
    for (int thread = 0; thread < 4; ++thread) {
        scalar.local(thread) = double(thread + 1);
    }
    REQUIRE(approx(scalar.sum_locals(), 10.0));
    REQUIRE(approx(scalar.reduce(), 10.0));
    REQUIRE(approx(scalar.result(), 10.0));
    REQUIRE_THROWS_AS(scalar.local(0), std::logic_error);
    REQUIRE_THROWS_AS(scalar.reduce(), std::logic_error);

    bsm::ParallelVector<Eigen::VectorXd> vector;
    vector.begin_accumulation(3, 2);
    REQUIRE_THROWS_AS(vector.result(), std::logic_error);
    vector.local(0) << 1.0, 2.0;
    vector.local(1) << 3.0, 4.0;
    vector.local(2) << 5.0, 6.0;
    const Eigen::VectorXd& reduced = vector.reduce();
    REQUIRE(approx(reduced[0], 9.0));
    REQUIRE(approx(reduced[1], 12.0));
    REQUIRE(approx(vector.result()[0], 9.0));
    REQUIRE(approx(vector.result()[1], 12.0));
    REQUIRE_THROWS_AS(vector.local(0), std::logic_error);
    REQUIRE_THROWS_AS(vector.reduce(), std::logic_error);
}

TEST_CASE("Newton restores rejected line-search states", "[Solver]")
{
    SECTION("invalid intermediate states restore the input DoFs")
    {
        QuadraticProblem problem;
        const Eigen::Vector3d initial = problem.x[0];
        auto callbacks = SolverCallbacks::create(problem.context);
        callbacks->add_is_intermediate_state_valid([] { return false; });

        auto newton = NewtonsMethod::create(problem.potential, problem.context, callbacks);
        newton->settings.max_backtracking_invalid_state_iterations = 1;

        REQUIRE(newton->solve() == SolverReturn::TooManyInvalidIntermediateIterations);
        REQUIRE(approx(problem.x[0], initial));
    }

    SECTION("Armijo failure restores the input DoFs")
    {
        QuadraticProblem problem;
        const Eigen::Vector3d initial = problem.x[0];
        auto newton = NewtonsMethod::create(problem.potential, problem.context);
        newton->settings.line_search_armijo_beta = 2.0;
        newton->settings.max_backtracking_armijo_iterations = 1;

        REQUIRE(newton->solve() == SolverReturn::TooManyArmijoIterations);
        REQUIRE(approx(problem.x[0], initial));
    }
}

TEST_CASE("Newton reports linear-system failures", "[Solver]")
{
    auto solve_indefinite_problem = [](LinearSolver linear_solver) {
        std::vector<std::array<int, 1>> conn{{0}};
        std::vector<Eigen::Vector3d> x{Eigen::Vector3d::Ones()};
        auto potential = GlobalPotential::create();
        potential->add_potential("concave", conn,
            [&](MappedWorkspace<double>& mws, Element& elem)
            {
                const Vector x_sym = mws.make_vector(x, elem[0]);
                return -x_sym.squared_norm();
            });
        potential->add_dof(x);

        auto context = Context::create();
        context->output->set_enabled(false);
        auto newton = NewtonsMethod::create(potential, context);
        newton->settings.projection_mode = ProjectionToPD::Newton;
        newton->settings.linear_solver = linear_solver;
        return newton->solve();
    };

    SECTION("direct LLT")
    {
        REQUIRE(solve_indefinite_problem(LinearSolver::DirectLLT) == SolverReturn::LinearSystemSolveFailure);
    }

    SECTION("BDPCG")
    {
        REQUIRE(solve_indefinite_problem(LinearSolver::BDPCG) == SolverReturn::LinearSystemSolveFailure);
    }
}
