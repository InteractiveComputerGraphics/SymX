#include "catch.hpp"

#include <cmath>
#include <Eigen/Dense>
#include <symx>
#include "../symx/src/solver/adjoint/AdjointNewton.h"
#include "../symx/src/solver/adjoint/FirstOrderOptimizerAdjoint.h"

#include "utils.h"

using namespace symx;

// =============================================================================
//  Shared helpers
// =============================================================================

static LinearSolveSettings make_lss()
{
    LinearSolveSettings lss;
    lss.linear_solver = LinearSolver::DirectLLT;
    return lss;
}

static spContext make_context()
{
    auto ctx = Context::create();
    ctx->output->set_enabled(false);
    ctx->n_threads = 1;
    return ctx;
}

// =============================================================================
//  Spring stiffness identification problem
// =============================================================================
//
//  Forward problem
//  ~~~~~~~~~~~~~~~
//  Two 3-D nodes (only the x-component matters) connected by a zero-rest-length
//  spring with stiffness k.  Soft Dirichlet BCs attach each node to its rest
//  position X[i] with fixed stiffness k_bc.
//
//    P = 0.5 * k  * ||x1 - x0||^2          (spring)
//      + 0.5 * kbc * ||x0 - X0||^2          (BC node 0)
//      + 0.5 * kbc * ||x1 - X1||^2          (BC node 1)
//
//  Analytical equilibrium (x-component only):
//    x0 = k  / (2k + kbc)
//    x1 = (k + kbc) / (2k + kbc)
//    gap = x1 - x0 = kbc / (2k + kbc)
//
//  Optimisation problem
//  ~~~~~~~~~~~~~~~~~~~~
//  Loss  L = (gap - l_target)^2
//  Parameter to optimise: k  (scalar)
//
//  Analytical optimum:
//    gap = l_target  =>  2k + kbc = kbc / l_target
//    k* = kbc * (1 - l_target) / (2 * l_target)
//
//  With kbc = 1000, l_target = 0.4:
//    k* = 1000 * 0.6 / 0.8 = 750
// =============================================================================

namespace {

    struct SpringProblem
    {
        std::vector<Eigen::Vector3d> x;   // simulation DOFs (positions)
        std::vector<Eigen::Vector3d> X;   // rest positions
        double k_spring = 100.0;
        double k_bc     = 1e3;
        double l_target = 0.4;

        std::vector<std::array<int, 2>> spring_conn = {{ 0, 1 }};
        std::vector<std::array<int, 1>> bc_conn     = {{ 0 }, { 1 }};

        spGlobalPotential   G           = nullptr;
        spGlobalPotential   G_loss      = nullptr;
        spNewtonsMethod     newton      = nullptr;
        spAdjointNewton     adjoint     = nullptr;
        spContext           ctx         = nullptr;

        // Analytical equilibrium gap for the current k
        double analytical_gap() const
        {
            return k_bc / (2.0 * k_spring + k_bc);
        }

        // Analytical optimal stiffness
        double analytical_k_star() const
        {
            return k_bc * (1.0 - l_target) / (2.0 * l_target);
        }

        SpringProblem()
        {
            X = { Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0) };
            x = X;

            ctx = make_context();

            // --- Forward potential ---
            G = GlobalPotential::create();
            G->add_potential("spring", spring_conn,
                [&](MappedWorkspace<double>& mws, Element& elem)
                {
                    std::vector<Vector> xe = mws.make_vectors(x, elem.slice(0, 2));
                    Scalar k = mws.make_scalar(k_spring);
                    return 0.5 * k * (xe[1] - xe[0]).squared_norm();
                });
            G->add_potential("soft_bc", bc_conn,
                [&](MappedWorkspace<double>& mws, Element& elem)
                {
                    Vector xi = mws.make_vector(x, elem[0]);
                    Vector Xi = mws.make_vector(X, elem[0]);
                    Scalar kbc = mws.make_scalar(k_bc);
                    return 0.5 * kbc * (xi - Xi).squared_norm();
                });
            G->add_dof(x, "positions");

            newton = NewtonsMethod::create(G, ctx);
            newton->settings.residual_tolerance_abs = 1e-10;

            // --- Loss ---
            G_loss = GlobalPotential::create();
            G_loss->add_potential("target_length", spring_conn,
                [&](MappedWorkspace<double>& mws, Element& elem)
                {
                    std::vector<Vector> xe = mws.make_vectors(x, elem.slice(0, 2));
                    Scalar length  = (xe[1] - xe[0]).norm();
                    Scalar l_t     = mws.make_scalar(l_target);
                    return (length - l_t).powN(2);
                });
            G_loss->add_dof(k_spring, "k");

            adjoint = AdjointNewton::create(newton, G_loss, make_lss());
        }

        // Reset simulation DOFs to rest positions before a fresh optimisation run
        void reset_x() { x = X; }

        // Reset parameter to a fresh start for optimisation
        void set_k(double k) { k_spring = k; }
    };

} // anonymous namespace

// =============================================================================
//  Forward correctness (baseline check)
// =============================================================================

TEST_CASE("SpringOpt - Forward solve correctness", "[FirstOrderOptimizer]")
{
    SpringProblem p;
    p.reset_x();

    REQUIRE(p.newton->solve() == SolverReturn::Successful);

    const double k   = p.k_spring;
    const double kbc = p.k_bc;
    REQUIRE(approx(p.x[0][0], k  / (2*k + kbc), 1e-6));
    REQUIRE(approx(p.x[1][0], (k + kbc) / (2*k + kbc), 1e-6));

    const double gap = p.x[1][0] - p.x[0][0];
    REQUIRE(approx(gap, p.analytical_gap(), 1e-6));
}

// =============================================================================
//  Helper: run optimiser and check it reaches the analytical solution
// =============================================================================

static void run_and_check(
    const std::string& method_name,
    SpringProblem&     prob,
    spFirstOrderOptimizerAdjoint& opt,
    double             k_tol   = 2.0,
    double             L_tol   = 1e-3)
{
    SolverReturn ret = opt->solve();

    INFO("Method: " << method_name);
    INFO("k optimised = " << prob.k_spring);
    INFO("k* analytical = " << prob.analytical_k_star());
    INFO("Return code = " << (int)ret);

    // Either converged or hit iteration limit treated as success
    bool ok = (ret == SolverReturn::Successful || ret == SolverReturn::TooManyIterations);
    REQUIRE(ok);

    // Final loss
    prob.reset_x();
    const auto result = prob.adjoint->run_forward_and_evaluate_L();
    REQUIRE(result.success);
    const double L_final = result.value;
    INFO("L_final = " << L_final);
    REQUIRE(L_final < L_tol);

    // Stiffness close to analytical optimum
    REQUIRE(approx(prob.k_spring, prob.analytical_k_star(), k_tol));
}

// =============================================================================
//  Test 1: Gradient Descent
// =============================================================================

TEST_CASE("SpringOpt - GradientDescent", "[FirstOrderOptimizer]")
{
    SpringProblem prob;
    prob.set_k(100.0);
    prob.reset_x();

    auto opt = FirstOrderOptimizerAdjoint::create(prob.adjoint, prob.ctx);
    opt->settings.type                 = FirstOrderOptType::GradientDescent;
    opt->settings.gradient_descent.learning_rate        = 1e5;        // Large step: gradient is O(1e-3)
    opt->settings.max_iterations       = 300;
    opt->settings.max_iterations_as_success = true;
    opt->settings.residual_tolerance_abs    = 1e-8;
    opt->settings.max_backtracking_armijo_iterations = 30;

    // GD is a first-order method; check convergence primarily via loss value.
    // k_tol is loose because the gradient norm near the optimum is ~1e-8.
    run_and_check("GradientDescent", prob, opt, /*k_tol=*/10.0, /*L_tol=*/1e-3);
}

// =============================================================================
//  Test 2: Adam
// =============================================================================

TEST_CASE("SpringOpt - Adam", "[FirstOrderOptimizer]")
{
    SpringProblem prob;
    prob.set_k(100.0);
    prob.reset_x();

    auto opt = FirstOrderOptimizerAdjoint::create(prob.adjoint, prob.ctx);
    opt->settings.type            = FirstOrderOptType::Adam;
    opt->settings.adam.learning_rate      = 20.0;   // each Adam step ≈ ±lr (normalised by moments)
    opt->settings.adam.beta1      = 0.9;
    opt->settings.adam.beta2      = 0.999;
    opt->settings.max_iterations  = 500;
    opt->settings.max_iterations_as_success = true;
    opt->settings.residual_tolerance_abs    = 1e-8;

    // Adam is a first-order method; check convergence primarily via loss value.
    run_and_check("Adam", prob, opt, /*k_tol=*/10.0, /*L_tol=*/1e-3);
}

TEST_CASE("SpringOpt - VectorAdam", "[FirstOrderOptimizer]")
{
    SpringProblem prob;
    prob.set_k(100.0);
    prob.reset_x();

    auto opt = FirstOrderOptimizerAdjoint::create(prob.adjoint, prob.ctx);
    opt->settings.type = FirstOrderOptType::VectorAdam;
    opt->settings.vector_adam.learning_rate = 20.0;
    opt->settings.vector_adam.vector_block_size = 1;
    opt->settings.max_iterations = 500;
    opt->settings.max_iterations_as_success = true;
    opt->settings.residual_tolerance_abs = 1e-8;

    run_and_check("VectorAdam", prob, opt, /*k_tol=*/10.0, /*L_tol=*/1e-3);
}

TEST_CASE("SpringOpt - Failed line search restores accepted state", "[FirstOrderOptimizer]")
{
    SpringProblem prob;
    prob.set_k(100.0);
    prob.reset_x();

    auto callbacks = SolverCallbacks::create(prob.ctx);
    callbacks->add_is_intermediate_state_valid([]() { return false; });
    auto opt = FirstOrderOptimizerAdjoint::create(prob.adjoint, prob.ctx, callbacks);
    opt->settings.type = FirstOrderOptType::GradientDescent;
    opt->settings.gradient_descent.learning_rate = 1e5;
    opt->settings.max_backtracking_invalid_state_iterations = 1;

    REQUIRE(opt->solve() == SolverReturn::TooManyInvalidIntermediateIterations);
    REQUIRE(approx(prob.k_spring, 100.0, 1e-12));
    REQUIRE(approx(prob.x[0][0], 100.0 / 1200.0, 1e-10));
    REQUIRE(approx(prob.x[1][0], 1100.0 / 1200.0, 1e-10));
}

// =============================================================================
//  Test 3: L-BFGS
// =============================================================================

TEST_CASE("SpringOpt - LBFGS", "[FirstOrderOptimizer]")
{
    SpringProblem prob;
    prob.set_k(100.0);
    prob.reset_x();

    auto opt = FirstOrderOptimizerAdjoint::create(prob.adjoint, prob.ctx);
    opt->settings.type             = FirstOrderOptType::LBFGS;
    opt->settings.lbfgs.history_size = 10;
    opt->settings.max_iterations   = 100;
    opt->settings.max_iterations_as_success = true;
    opt->settings.residual_tolerance_abs    = 1e-8;
    opt->settings.max_backtracking_armijo_iterations = 30;

    run_and_check("LBFGS", prob, opt, /*k_tol=*/1.0, /*L_tol=*/1e-6);
}

// =============================================================================
//  Test 4: All three solvers reach the same analytical optimum
// =============================================================================

TEST_CASE("SpringOpt - Analytical solution comparison", "[FirstOrderOptimizer]")
{
    // k_bc = 1000, l_target = 0.4  =>  k* = 750
    const double k_bc     = 1e3;
    const double l_target = 0.4;
    const double k_star   = k_bc * (1.0 - l_target) / (2.0 * l_target);

    INFO("Analytical k* = " << k_star);
    REQUIRE(approx(k_star, 750.0, 1e-10));  // sanity check

    // --- Run each solver and compare against analytical solution ---
    auto test_solver = [&](FirstOrderOptType type, double lr_or_alpha, int max_iter, double k_tol) {

        SpringProblem prob;
        prob.set_k(100.0);
        prob.reset_x();

        auto opt = FirstOrderOptimizerAdjoint::create(prob.adjoint, prob.ctx);
        opt->settings.type                      = type;
        opt->settings.max_iterations            = max_iter;
        opt->settings.max_iterations_as_success = true;
        opt->settings.residual_tolerance_abs    = 1e-8;
        opt->settings.max_backtracking_armijo_iterations = 30;

        if (type == FirstOrderOptType::GradientDescent) {
            opt->settings.gradient_descent.learning_rate = lr_or_alpha;
        } else if (type == FirstOrderOptType::Adam) {
            opt->settings.adam.learning_rate = lr_or_alpha;
        }

        opt->solve();

        prob.reset_x();
        const auto result = prob.adjoint->run_forward_and_evaluate_L();
        REQUIRE(result.success);
        const double L_final = result.value;

        INFO("Type: " << to_string(type));
        INFO("k found: " << prob.k_spring << "  k*: " << k_star);
        INFO("L_final: " << L_final);

        REQUIRE(approx(prob.k_spring, k_star, k_tol));
        REQUIRE(L_final < 1e-3);
    };

    test_solver(FirstOrderOptType::GradientDescent, 1e5, 300, 2.0);
    test_solver(FirstOrderOptType::Adam,            20.0, 500, 2.0);
    test_solver(FirstOrderOptType::LBFGS,            1.0, 100, 1.0);
}

TEST_CASE("SpringOpt - Outer callbacks match first-order optimizer semantics", "[FirstOrderOptimizer]")
{
    SpringProblem prob;
    auto callbacks = SolverCallbacks::create(prob.ctx);
    int before_steps = 0;
    int after_steps = 0;
    callbacks->add_before_step([&] { ++before_steps; });
    callbacks->add_after_step([&] { ++after_steps; });

    auto opt = FirstOrderOptimizerAdjoint::create(prob.adjoint, prob.ctx, callbacks);
    opt->settings.max_iterations = 0;
    opt->settings.iteration_offset = 23;

    REQUIRE(opt->solve() == SolverReturn::TooManyIterations);
    REQUIRE(before_steps == 1);
    REQUIRE(after_steps == 1);
}
