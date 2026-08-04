#include "catch.hpp"

#include <cmath>
#include <Eigen/Dense>
#include <symx>
#include "../symx/src/solver/adjoint/AdjointNewton.h"

#include "utils.h"

using namespace symx;

// ─────────────────────────────────────────────────────────────────────────────
//  Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

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

static void check_adjoint_vs_fd(
    AdjointNewton& adjoint,
    std::vector<Eigen::Vector3d>& x,
    const std::vector<Eigen::Vector3d>& X_rest,
    double fd_eps   = 1e-3,
    double rel_tol  = 1e-2,
    double eps_zero = 1e-8)
{
    const auto fd_result = adjoint.evaluate_dL_dp_finite_differences(fd_eps);
    const auto gradient_result = adjoint.run_forward_and_evaluate_dL_dp();
    REQUIRE(fd_result.success);
    REQUIRE(gradient_result.success);
    const Eigen::VectorXd& dL_dp_fd = fd_result.gradient;
    const Eigen::VectorXd& dL_dp = gradient_result.gradient;

    REQUIRE(dL_dp.size() == dL_dp_fd.size());
    for (int i = 0; i < (int)dL_dp.size(); i++) {
        const double fd  = dL_dp_fd[i];
        const double adj = dL_dp[i];
        if (std::abs(fd) < eps_zero && std::abs(adj) < eps_zero) continue;
        const double rel_err = std::abs(adj - fd) / (std::abs(fd) + eps_zero);
        INFO("dL/dp[" << i << "]:  adj = " << adj
                      << "  fd = " << fd
                      << "  rel_err = " << rel_err);
        REQUIRE(rel_err < rel_tol);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 0: Single spring – baseline
//
//  Two 3D nodes connected by a zero-rest-length spring (stiffness k).
//  Soft Dirichlet BCs pull each node toward its rest position X[i] (stiffness
//  k_bc, fixed).  Parameter: k_spring (scalar).  Loss: (|gap| − l_target)².
//
//  Analytical equilibrium:
//    x[0][0] = k / (2k + k_bc),  x[1][0] = (k + k_bc) / (2k + k_bc)
//    gap = k_bc / (2k + k_bc)
//
//  Also checks FD gradient vs analytic and adjoint vs FD.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Adjoint - Single Spring", "[Adjoint]")
{
    // Data
    struct Data {
        std::vector<Eigen::Vector3d> x;
        std::vector<Eigen::Vector3d> X;
        double k_spring = 100.0;
        double k_bc     = 1e3;
        double l_target = 0.5;
    } data;

    data.X = { Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0) };
    data.x = data.X;

    std::vector<std::array<int, 2>> spring_conn = {{ 0, 1 }};
    std::vector<std::array<int, 1>> bc_conn     = {{ 0 }, { 1 }};

    // Define Global potential
    auto G = GlobalPotential::create();
    G->add_potential("spring", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar k = mws.make_scalar(data.k_spring);
            return 0.5 * k * (xe[1] - xe[0]).squared_norm();
        }
    );
    G->add_potential("soft_bc", bc_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector xi = mws.make_vector(data.x, elem[0]);
            Vector Xi = mws.make_vector(data.X, elem[0]);
            Scalar k = mws.make_scalar(data.k_bc);
            return 0.5 * k * (xi - Xi).squared_norm();
        }
    );
    G->add_dof(data.x, "positions");
    
    // Define Newton solver
    auto ctx    = make_context();
    auto newton = NewtonsMethod::create(G, ctx);
    newton->settings.residual_tolerance_abs = 1e-10;
    
    // Define Global Loss
    auto L_global = GlobalPotential::create();
    L_global->add_potential("target_length", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar length  = (xe[1] - xe[0]).norm();
            Scalar l_target = mws.make_scalar(data.l_target);
            return (length - l_target).powN(2);
        }
    );
    L_global->add_dof(data.k_spring, "k_spring");

    // Define Adjoint
    auto adjoint = AdjointNewton::create(newton, L_global, make_lss());

    // Section 1: Forward solve correctness
    {
        data.x = data.X;
        REQUIRE(newton->solve() == SolverReturn::Successful);
        const double k = data.k_spring, kbc = data.k_bc;
        REQUIRE(approx(data.x[0][0], k / (2*k + kbc), 1e-6));
        REQUIRE(approx(data.x[1][0], (k + kbc) / (2*k + kbc), 1e-6));
        REQUIRE(approx(data.x[0][1], 0.0, 1e-10));
        REQUIRE(approx(data.x[1][1], 0.0, 1e-10));
    }

    // Section 2: Loss evaluation
    {
        data.x = data.X;
        const auto result = adjoint->run_forward_and_evaluate_L();
        REQUIRE(result.success);
        const double L = result.value;
        const double k = data.k_spring, kbc = data.k_bc;
        const double l = kbc / (2*k + kbc);
        REQUIRE(approx(L, (l - data.l_target) * (l - data.l_target), 1e-6));
    }

    // Section 3: FD gradient vs analytic
    {
        const double fd_eps = 1e-3;
        data.x = data.X;
        const auto result = adjoint->evaluate_dL_dp_finite_differences(fd_eps);
        REQUIRE(result.success);
        const Eigen::VectorXd& dL_dp_fd = result.gradient;
        REQUIRE(dL_dp_fd.size() == 1);
        const double k = data.k_spring, kbc = data.k_bc;
        const double l    = kbc / (2*k + kbc);
        const double dl_dk = -2.0*kbc / ((2*k + kbc)*(2*k + kbc));
        REQUIRE(approx(dL_dp_fd[0], 2.0*(l - data.l_target)*dl_dk, 1e-4));
        REQUIRE(approx(data.k_spring, k, 1e-12));
        REQUIRE((data.x[0] - data.X[0]).norm() < 1e-12);
        REQUIRE((data.x[1] - data.X[1]).norm() < 1e-12);
    }

    // Section 4: Adjoint gradient vs FD
    {
        check_adjoint_vs_fd(*adjoint, data.x, data.X, /*fd_eps=*/1e-3, /*rel_tol=*/1e-3);
    }

    // Fixed-gradient validation is checked before it can modify solver state.
    REQUIRE_THROWS_AS(adjoint->force_gradient_to_zero({1}), std::out_of_range);
    adjoint->force_gradient_to_zero({0});
    const auto fixed_gradient = adjoint->run_forward_and_evaluate_dL_dp();
    REQUIRE(fixed_gradient.success);
    REQUIRE(approx(fixed_gradient.gradient[0], 0.0, 1e-14));
    adjoint->force_gradient_to_zero({});

    const auto invalid_fd = adjoint->evaluate_dL_dp_finite_differences(0.0);
    REQUIRE_FALSE(invalid_fd.success);
    REQUIRE(invalid_fd.status == SolverReturn::InvalidEvaluation);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 1: Conditional forward potential
//
//  4 nodes, 2 independent springs sharing a global stiffness k.
//  The forward spring potential is conditional: active only when x-gap > 0.5.
//    Spring A (nodes 0–1): rest gap = 0.2 → INACTIVE
//    Spring B (nodes 2–3): rest gap = 1.0 → ACTIVE  (equilibrium gap ≈ 5/6)
//  Loss: (gap − l_target)² summed over both springs (unconditional).
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Adjoint - Conditional forward potential", "[Adjoint]")
{
    // Data
    struct Data {
        std::vector<Eigen::Vector3d> x;
        std::vector<Eigen::Vector3d> X;
        double k_spring = 100.0;
        double k_bc     = 1e3;
        double l_target = 0.5;
    } data;

    data.X = {
        Eigen::Vector3d(0.0, 0, 0), Eigen::Vector3d(0.2, 0, 0),
        Eigen::Vector3d(0.0, 0, 0), Eigen::Vector3d(1.0, 0, 0),
    };
    data.x = data.X;

    std::vector<std::array<int, 2>> spring_conn = {{0, 1}, {2, 3}};
    std::vector<std::array<int, 1>> bc_conn     = {{0}, {1}, {2}, {3}};

    // Define Global Potential
    auto G = GlobalPotential::create();
    G->add_potential("cond_spring_t1", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar k   = mws.make_scalar(data.k_spring);
            Scalar gap = xe[1][0] - xe[0][0];
            return std::make_pair( 0.5 * k * (xe[1] - xe[0]).squared_norm(), gap - 0.5 );
        }
    );
    G->add_potential("bc_t1", bc_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector xi = mws.make_vector(data.x, elem[0]);
            Vector Xi = mws.make_vector(data.X, elem[0]);
            Scalar k = mws.make_scalar(data.k_bc);
            return 0.5 * k * (xi - Xi).squared_norm();
        }
    );
    G->add_dof(data.x, "positions");

    // Define Newton Solver
    auto ctx    = make_context();
    auto newton = NewtonsMethod::create(G, ctx);
    newton->settings.residual_tolerance_abs = 1e-10;

    // Test spring forward
    // Spring A stays at rest (inactive), Spring B reaches standard equilibrium
    {
        data.x = data.X;
        REQUIRE(newton->solve() == SolverReturn::Successful);
        REQUIRE(approx(data.x[0][0], 0.0, 1e-6));
        REQUIRE(approx(data.x[1][0], 0.2, 1e-6));
        const double k = data.k_spring, kbc = data.k_bc;
        REQUIRE(approx(data.x[2][0], k / (2*k + kbc), 1e-6));
        REQUIRE(approx(data.x[3][0], (k + kbc) / (2*k + kbc), 1e-6));
    }

    // Define Global Loss
    auto L_global = GlobalPotential::create();
    L_global->add_potential("loss_t1", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar gap = xe[1][0] - xe[0][0];
            Scalar l_t = mws.make_scalar(data.l_target);
            return (gap - l_t) * (gap - l_t);
        }
    );
    L_global->add_dof(data.k_spring, "k");

    // Define Adjoint
    auto adjoint = AdjointNewton::create(newton, L_global, make_lss());

    // Test adjoint
    check_adjoint_vs_fd(*adjoint, data.x, data.X);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 2: Conditional loss
//
//  Same 4-node, 2-spring setup. Forward is unconditional (both springs active).
//  The LOSS is conditional: active only when gap > 0.5.
//    Spring A (gap ≈ 0.2): loss inactive → L_A = 0
//    Spring B (gap ≈ 5/6): loss active   → L_B = (gap − l_target)²
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Adjoint - Conditional loss", "[Adjoint]")
{
    // Define Data
    struct Data {
        std::vector<Eigen::Vector3d> x;
        std::vector<Eigen::Vector3d> X;
        double k_spring = 100.0;
        double k_bc     = 1e3;
        double l_target = 0.5;
    } data;

    data.X = {
        Eigen::Vector3d(0.0, 0, 0), Eigen::Vector3d(0.2, 0, 0),
        Eigen::Vector3d(0.0, 0, 0), Eigen::Vector3d(1.0, 0, 0),
    };
    data.x = data.X;

    std::vector<std::array<int, 2>> spring_conn = {{0, 1}, {2, 3}};
    std::vector<std::array<int, 1>> bc_conn     = {{0}, {1}, {2}, {3}};

    // Define Global Potential
    auto G = GlobalPotential::create();
    G->add_potential("uncond_spring_t2", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar k = mws.make_scalar(data.k_spring);
            return 0.5 * k * (xe[1] - xe[0]).squared_norm();
        }
    );
    G->add_potential("bc_t2", bc_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector xi = mws.make_vector(data.x, elem[0]);
            Vector Xi = mws.make_vector(data.X, elem[0]);
            Scalar k = mws.make_scalar(data.k_bc);
            return 0.5 * k * (xi - Xi).squared_norm();
        }
    );
    G->add_dof(data.x, "positions");

    // Define Newtons Solver
    auto ctx    = make_context();
    auto newton = NewtonsMethod::create(G, ctx);
    newton->settings.residual_tolerance_abs = 1e-10;

    // Define Global Loss
    auto L_global = GlobalPotential::create();
    L_global->add_potential("cond_loss_t2", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem) 
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar gap = xe[1][0] - xe[0][0];
            Scalar l_t = mws.make_scalar(data.l_target);
            return std::make_pair( (gap - l_t) * (gap - l_t), gap - 0.5 );
        }
    );
    L_global->add_dof(data.k_spring, "k");

    auto adjoint = AdjointNewton::create(newton, L_global, make_lss());
    check_adjoint_vs_fd(*adjoint, data.x, data.X);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 3: Multiple loss terms
//
//  Single spring, stiffness k as parameter.  Loss = L_A + L_B where:
//    L_A = (gap − l_target)²   (target spring length)
//    L_B = x[0].x² + x[1].x²  (penalise node x-positions)
//  Tests the accumulation loop over compiled_loss in AdjointNewton.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Adjoint - Multiple loss terms", "[Adjoint]")
{
    // Define Data
    struct Data {
        std::vector<Eigen::Vector3d> x;
        std::vector<Eigen::Vector3d> X;
        double k_spring = 100.0;
        double k_bc     = 1e3;
        double l_target = 0.5;
    } data;

    data.X = { Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0) };
    data.x = data.X;

    std::vector<std::array<int, 2>> spring_conn = {{0, 1}};
    std::vector<std::array<int, 1>> bc_conn     = {{0}, {1}};
    std::vector<std::array<int, 1>> node_conn   = {{0}, {1}};

    // Define Global Potential
    auto G = GlobalPotential::create();
    G->add_potential("spring_t3", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar k = mws.make_scalar(data.k_spring);
            return 0.5 * k * (xe[1] - xe[0]).squared_norm();
        }
    );
    G->add_potential("bc_t3", bc_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector xi = mws.make_vector(data.x, elem[0]);
            Vector Xi = mws.make_vector(data.X, elem[0]);
            Scalar k = mws.make_scalar(data.k_bc);
            return 0.5 * k * (xi - Xi).squared_norm();
        });
    G->add_dof(data.x, "positions");

    // Define Newton Solver
    auto ctx    = make_context();
    auto newton = NewtonsMethod::create(G, ctx);
    newton->settings.residual_tolerance_abs = 1e-10;

    // Define Global Loss
    auto L_global = GlobalPotential::create();
    L_global->add_potential("loss_a_t3", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar gap = xe[1][0] - xe[0][0];
            Scalar l_t = mws.make_scalar(data.l_target);
            return (gap - l_t) * (gap - l_t);
        }
    );
    L_global->add_potential("loss_b_t3", node_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector xi = mws.make_vector(data.x, elem[0]);
            return xi[0] * xi[0];
        }
    );
    L_global->add_dof(data.k_spring, "k");

    // Define Adjoint
    auto adjoint = AdjointNewton::create(newton, L_global, make_lss());

    // Test Adjoint
    check_adjoint_vs_fd(*adjoint, data.x, data.X);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 4: Per-element parameter
//
//  4 nodes, 2 independent springs, each with its own stiffness from a vector:
//    k_vec[0] = 100  (spring 0–1)
//    k_vec[1] = 200  (spring 2–3)
//  A third connectivity column carries the spring index so that
//  make_scalar(k_vec, elem[2]) indexes the right stiffness per element.
//  This exercises the non-global (conn_idx ≥ 0) parameter path.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Adjoint - Per-element parameter", "[Adjoint]")
{
    // Define Data
    struct Data {
        std::vector<Eigen::Vector3d> x;
        std::vector<Eigen::Vector3d> X;
        std::vector<double> k_vec;
        double k_bc     = 1e3;
        double l_target = 0.5;
    } data;
    data.k_vec = { 100.0, 200.0 };

    data.X = {
        Eigen::Vector3d(0.0, 0, 0), Eigen::Vector3d(1.0, 0, 0),
        Eigen::Vector3d(0.0, 0, 0), Eigen::Vector3d(1.0, 0, 0),
    };
    data.x = data.X;

    std::vector<std::array<int, 3>> spring_conn_k = {{0, 1, 0}, {2, 3, 1}}; // Enumerated at the back
    std::vector<std::array<int, 1>> bc_conn       = {{0}, {1}, {2}, {3}};

    // Define Global Potential
    auto G = GlobalPotential::create();
    G->add_potential("spring_per_k_t4", spring_conn_k,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar k = mws.make_scalar(data.k_vec, elem[2]);
            return 0.5 * k * (xe[1] - xe[0]).squared_norm();
        }
    );
    G->add_potential("bc_t4", bc_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector xi = mws.make_vector(data.x, elem[0]);
            Vector Xi = mws.make_vector(data.X, elem[0]);
            Scalar k = mws.make_scalar(data.k_bc);
            return 0.5 * k * (xi - Xi).squared_norm();
        }
    );
    G->add_dof(data.x, "positions");

    // Define Newtons Solver
    auto ctx    = make_context();
    auto newton = NewtonsMethod::create(G, ctx);
    newton->settings.residual_tolerance_abs = 1e-10;

    // Test forward solve
    {
        data.x = data.X;
        REQUIRE(newton->solve() == SolverReturn::Successful);
        const double kbc = data.k_bc;
        for (int s = 0; s < 2; s++) {
            const double k = data.k_vec[s];
            REQUIRE(approx(data.x[2*s    ][0], k / (2*k + kbc), 1e-6));
            REQUIRE(approx(data.x[2*s + 1][0], (k + kbc) / (2*k + kbc), 1e-6));
        }
    }

    // Define Global Loss
    auto L_global = GlobalPotential::create();
    L_global->add_potential("loss_per_k_t4", spring_conn_k,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar gap = xe[1][0] - xe[0][0];
            Scalar l_t = mws.make_scalar(data.l_target);
            return (gap - l_t) * (gap - l_t);
        }
    );
    L_global->add_dof(data.k_vec, "k_vec");

    // Define Adjoint
    auto adjoint = AdjointNewton::create(newton, L_global, make_lss());

    // Test adjoint
    check_adjoint_vs_fd(*adjoint, data.x, data.X);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 5: Multiple parameter sets (stiffness + rest positions)
//
//  Single spring, two parameter sets:
//    Set 0: k_spring (1 DOF)   – spring stiffness
//    Set 1: X_param  (6 DOFs)  – rest positions of both nodes (2 × Vector3d)
//  BC pulls nodes toward X_param, so equilibrium depends on both.
//  Loss: (x-gap − l_target)².  Adjoint gradient has 7 components total.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Adjoint - Multiple parameter sets", "[Adjoint]")
{
    // Define Data
    struct Data {
        std::vector<Eigen::Vector3d> x;
        std::vector<Eigen::Vector3d> X_param;
        double k_spring = 100.0;
        double k_bc     = 1e3;
        double l_target = 0.5;
    } data;

    data.X_param = { Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0) };
    data.x       = data.X_param;

    std::vector<std::array<int, 2>> spring_conn = {{0, 1}};
    std::vector<std::array<int, 1>> bc_conn     = {{0}, {1}};

    // Define Global Potential
    auto G = GlobalPotential::create();
    G->add_potential("spring_t5", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar k = mws.make_scalar(data.k_spring);
            return 0.5 * k * (xe[1] - xe[0]).squared_norm();
        });
    G->add_potential("bc_t5", bc_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector xi = mws.make_vector(data.x,       elem[0]);
            Vector Xi = mws.make_vector(data.X_param, elem[0]);  // parameter
            Scalar k = mws.make_scalar(data.k_bc);
            return 0.5 * k * (xi - Xi).squared_norm();
        });
    G->add_dof(data.x, "positions");

    // Define Newtons Solver
    auto ctx    = make_context();
    auto newton = NewtonsMethod::create(G, ctx);
    newton->settings.residual_tolerance_abs = 1e-10;

    // Define Global Potential
    auto L_global = GlobalPotential::create();
    L_global->add_potential("loss_t5", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xe = mws.make_vectors(data.x, elem.slice(0, 2));
            Scalar gap = xe[1][0] - xe[0][0];
            Scalar l_t = mws.make_scalar(data.l_target);
            return (gap - l_t) * (gap - l_t);
        }
    );
    L_global->add_dof(data.k_spring, "k");
    L_global->add_dof(data.X_param,  "X_param");

    // Define Adjoint
    auto adjoint = AdjointNewton::create(newton, L_global, make_lss());

    // Test Adjoint
    check_adjoint_vs_fd(*adjoint, data.x, data.X_param);
}

TEST_CASE("Adjoint - Parallel large per-element parameter regression", "[Adjoint]")
{
    struct ChainResult {
        double loss;
        Eigen::VectorXd gradient;
    };

    auto run_chain = [](const int n_threads) {
        constexpr int n_edges = 257;
        std::vector<Eigen::Vector3d> x(n_edges + 1);
        std::vector<Eigen::Vector3d> X(n_edges + 1);
        std::vector<Eigen::Vector3d> target(n_edges + 1);
        std::vector<double> stiffness(n_edges);
        std::vector<std::array<int, 3>> edge_conn(n_edges);
        std::vector<std::array<int, 1>> node_conn(n_edges + 1);
        for (int i = 0; i <= n_edges; ++i) {
            X[i] = Eigen::Vector3d(0.02 * i, 0.0, 0.0);
            x[i] = X[i];
            target[i] = X[i] + Eigen::Vector3d(0.03 * std::sin(0.17 * i), 0.0, 0.0);
            node_conn[i] = {i};
            if (i < n_edges) {
                stiffness[i] = 1.0 + 0.01 * (i % 11);
                edge_conn[i] = {i, i + 1, i};
            }
        }

        auto G = GlobalPotential::create();
        G->add_potential("adjoint_parallel_chain_springs", edge_conn,
            [&](MappedWorkspace<double>& mws, Element& elem) {
                const std::vector<Vector> xe = mws.make_vectors(x, elem.slice(0, 2));
                const Scalar k = mws.make_scalar(stiffness, elem[2]);
                return 0.5 * k * (xe[1] - xe[0]).squared_norm();
            });
        G->add_potential("adjoint_parallel_chain_pins", node_conn,
            [&](MappedWorkspace<double>& mws, Element& elem) {
                const Vector xi = mws.make_vector(x, elem[0]);
                const Vector Xi = mws.make_vector(X, elem[0]);
                return 2.5 * (xi - Xi).squared_norm();
            });
        G->add_dof(x, "state");

        auto L = GlobalPotential::create();
        L->add_potential("adjoint_parallel_chain_loss", node_conn,
            [&](MappedWorkspace<double>& mws, Element& elem) {
                const Vector xi = mws.make_vector(x, elem[0]);
                const Vector ti = mws.make_vector(target, elem[0]);
                return 0.5 * (xi - ti).squared_norm();
            });
        L->add_dof(stiffness, "element_stiffness");

        auto context = make_context();
        context->n_threads = n_threads;
        auto newton = NewtonsMethod::create(G, context);
        newton->settings.linear_solver = LinearSolver::DirectLLT;
        newton->settings.projection_mode = ProjectionToPD::Newton;
        newton->settings.residual_tolerance_abs = 1e-11;
        auto adjoint = AdjointNewton::create(newton, L, make_lss());

        const auto value = adjoint->run_forward_and_evaluate_L();
        REQUIRE(value.success);
        double manual_loss = 0.0;
        for (int i = 0; i <= n_edges; ++i) {
            manual_loss += 0.5 * (x[i] - target[i]).squaredNorm();
        }
        REQUIRE(approx(value.value, manual_loss, 1e-11));

        const std::vector<Eigen::Vector3d> x_before_fd = x;
        const std::vector<double> stiffness_before_fd = stiffness;
        const auto fd = adjoint->evaluate_dL_dp_finite_differences(1e-4);
        const auto gradient = adjoint->evaluate_dL_dp_no_solve();
        REQUIRE(fd.success);
        REQUIRE(gradient.success);
        REQUIRE(gradient.gradient.size() == n_edges);
        for (int i = 0; i < n_edges; ++i) {
            INFO("thread count = " << n_threads << ", parameter = " << i);
            REQUIRE(std::abs(gradient.gradient[i] - fd.gradient[i])
                < 5e-7 + 2e-3 * std::abs(fd.gradient[i]));
            REQUIRE(approx(stiffness[i], stiffness_before_fd[i], 1e-14));
        }
        for (int i = 0; i <= n_edges; ++i) {
            REQUIRE(approx(x[i], x_before_fd[i], 1e-13));
        }
        return ChainResult{value.value, gradient.gradient};
    };

    const ChainResult sequential = run_chain(1);
    const ChainResult parallel = run_chain(4);
    REQUIRE(approx(sequential.loss, parallel.loss, 1e-11));
    REQUIRE(sequential.gradient.size() == parallel.gradient.size());
    for (int i = 0; i < sequential.gradient.size(); ++i) {
        REQUIRE(std::abs(sequential.gradient[i] - parallel.gradient[i]) < 1e-10);
    }
}

TEST_CASE("Adjoint - Multiple state and parameter sets", "[Adjoint]")
{
    std::vector<Eigen::Vector3d> x{Eigen::Vector3d(0.2, 0, 0)};
    std::vector<Eigen::Vector3d> y{Eigen::Vector3d(-0.3, 0, 0)};
    std::vector<Eigen::Vector3d> tx{Eigen::Vector3d(0.6, 0, 0)};
    std::vector<Eigen::Vector3d> ty{Eigen::Vector3d(-0.5, 0, 0)};
    double kx = 2.0;
    double ky = 3.0;
    const std::vector<std::array<int, 1>> conn{{0}};

    auto G = GlobalPotential::create();
    G->add_potential("adjoint_two_state_sets", conn,
        [&](MappedWorkspace<double>& mws, Element& elem) {
            const Vector xs = mws.make_vector(x, elem[0]);
            const Vector ys = mws.make_vector(y, elem[0]);
            const Scalar kxs = mws.make_scalar(kx);
            const Scalar kys = mws.make_scalar(ky);
            return 0.5 * kxs * xs.squared_norm() + 0.5 * kys * ys.squared_norm()
                + 0.25 * (xs - ys).squared_norm();
        });
    G->add_dof(x, "x");
    G->add_dof(y, "y");

    auto L = GlobalPotential::create();
    L->add_potential("adjoint_two_state_sets_loss", conn,
        [&](MappedWorkspace<double>& mws, Element& elem) {
            const Vector xs = mws.make_vector(x, elem[0]);
            const Vector ys = mws.make_vector(y, elem[0]);
            const Vector x_target = mws.make_vector(tx, elem[0]);
            const Vector y_target = mws.make_vector(ty, elem[0]);
            return 0.5 * (xs - x_target).squared_norm() + 0.5 * (ys - y_target).squared_norm();
        });
    L->add_dof(kx, "kx");
    L->add_dof(ky, "ky");

    auto context = make_context();
    auto newton = NewtonsMethod::create(G, context);
    newton->settings.linear_solver = LinearSolver::DirectLLT;
    newton->settings.projection_mode = ProjectionToPD::Newton;
    auto adjoint = AdjointNewton::create(newton, L, make_lss());
    check_adjoint_vs_fd(*adjoint, x, tx, 1e-5, 1e-4);
}

TEST_CASE("Adjoint - Physics and loss conditions together", "[Adjoint]")
{
    std::vector<Eigen::Vector3d> x{
        Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
        Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0)};
    const std::vector<Eigen::Vector3d> X = x;
    double k = 1.0;
    const double k_bc = 4.0;
    const std::vector<std::array<int, 2>> spring_conn{{0, 1}, {2, 3}};
    const std::vector<std::array<int, 1>> node_conn{{0}, {1}, {2}, {3}};

    auto G = GlobalPotential::create();
    G->add_potential("adjoint_both_conditions_spring", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem) {
            const std::vector<Vector> xe = mws.make_vectors(x, elem.slice(0, 2));
            const Scalar gap = (xe[1] - xe[0]).norm();
            const Scalar ks = mws.make_scalar(k);
            return std::make_pair(0.5 * ks * (xe[1] - xe[0]).squared_norm(), gap - 0.5);
        });
    G->add_potential("adjoint_both_conditions_pin", node_conn,
        [&](MappedWorkspace<double>& mws, Element& elem) {
            const Vector xi = mws.make_vector(x, elem[0]);
            const Vector Xi = mws.make_vector(X, elem[0]);
            return 0.5 * k_bc * (xi - Xi).squared_norm();
        });
    G->add_dof(x, "state");

    auto L = GlobalPotential::create();
    L->add_potential("adjoint_both_conditions_loss", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem) {
            const std::vector<Vector> xe = mws.make_vectors(x, elem.slice(0, 2));
            const Scalar gap = (xe[1] - xe[0]).norm();
            return std::make_pair((gap - 0.8).powN(2), gap - 0.5);
        });
    L->add_dof(k, "k");

    auto context = make_context();
    auto newton = NewtonsMethod::create(G, context);
    newton->settings.linear_solver = LinearSolver::DirectLLT;
    newton->settings.residual_tolerance_abs = 1e-11;
    auto adjoint = AdjointNewton::create(newton, L, make_lss());
    check_adjoint_vs_fd(*adjoint, x, X, 1e-5, 1e-4);
}

TEST_CASE("Adjoint - BDPCG gradient agrees with finite differences", "[Adjoint]")
{
    std::vector<Eigen::Vector3d> x{Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX()};
    const std::vector<Eigen::Vector3d> X = x;
    double k = 2.0;
    const std::vector<std::array<int, 2>> spring_conn{{0, 1}};
    const std::vector<std::array<int, 1>> node_conn{{0}, {1}};

    auto G = GlobalPotential::create();
    G->add_potential("adjoint_pcg_spring", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem) {
            const std::vector<Vector> xe = mws.make_vectors(x, elem.slice(0, 2));
            return 0.5 * mws.make_scalar(k) * (xe[1] - xe[0]).squared_norm();
        });
    G->add_potential("adjoint_pcg_pin", node_conn,
        [&](MappedWorkspace<double>& mws, Element& elem) {
            const Vector xi = mws.make_vector(x, elem[0]);
            const Vector Xi = mws.make_vector(X, elem[0]);
            return 5.0 * (xi - Xi).squared_norm();
        });
    G->add_dof(x, "state");

    auto L = GlobalPotential::create();
    L->add_potential("adjoint_pcg_loss", spring_conn,
        [&](MappedWorkspace<double>& mws, Element& elem) {
            const std::vector<Vector> xe = mws.make_vectors(x, elem.slice(0, 2));
            return ((xe[1] - xe[0]).norm() - 0.7).powN(2);
        });
    L->add_dof(k, "k");

    auto context = make_context();
    auto newton = NewtonsMethod::create(G, context);
    newton->settings.linear_solver = LinearSolver::BDPCG;
    newton->settings.residual_tolerance_abs = 1e-11;
    LinearSolveSettings pcg;
    pcg.linear_solver = LinearSolver::BDPCG;
    pcg.cg_abs_tolerance = 1e-13;
    pcg.cg_rel_tolerance = 1e-10;
    auto adjoint = AdjointNewton::create(newton, L, pcg);
    check_adjoint_vs_fd(*adjoint, x, X, 1e-5, 2e-4);
}
