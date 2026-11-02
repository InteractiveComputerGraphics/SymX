#include "catch.hpp"

#include <symx>
#include "../symx/src/solver/adjoint/AdjointNewton.h"
#include "utils.h"

using namespace symx;

template<typename COMPILED_FLOAT>
static void check_quadrature_refresh(const std::string& name, const double tolerance = 1e-9)
{
    std::vector<std::array<int32_t, 2>> connectivity = {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {1, 3}};
    connectivity.reserve(8);
    std::vector<double> x = {0.1, 0.4, -0.2, 0.8};
    constexpr double cross_coefficient = 0.35;
    std::vector<double> weights;
    std::vector<std::array<double, 2>> targets;
    weights.reserve(8 * 3);
    targets.reserve(8 * 3);
    auto fill_data = [&](double shift) {
        weights.resize(connectivity.size() * 3);
        targets.resize(connectivity.size() * 3);
        for (size_t e = 0; e < connectivity.size(); e++) {
            for (int q = 0; q < 3; q++) {
                const size_t i = e * 3 + q;
                weights[i] = 0.2 + 0.07 * e + 0.03 * q + shift;
                targets[i] = {0.1 * e + 0.02 * q, -0.05 * e + 0.04 * q};
            }
        }
    };
    fill_data(0.0);

    auto [mws, elem] = MappedWorkspace<double>::create(connectivity);
    Scalar a = mws->make_scalar(x, elem[0]);
    Scalar b = mws->make_scalar(x, elem[1]);
    Scalar energy = mws->add_for_each(3, [&, mws = mws]() {
        Scalar w = mws->make_quadrature_scalar(weights);
        Vector t = mws->make_quadrature_vector(targets);
        // The cross term exercises the two off-diagonal Hessian entries.
        return w * (0.5 * ((a - t[0]).powN(2) + (b - t[1]).powN(2)) + cross_coefficient * a * b);
    });
    CompiledInLoop<double, COMPILED_FLOAT> loop(mws, value_gradient_hessian(energy, {a, b}), name, get_codegen_dir());
    loop.enable_coloring({0, 1});
    std::vector<uint8_t> active = {1, 0, 1, 1, 1};

    auto check = [&]() {
        std::vector<std::array<double, 7>> output(connectivity.size());
        std::vector<uint8_t> seen(connectivity.size(), 0);
        loop.run(2, [&](const View<double> sol, int32_t e, int32_t, const View<int32_t>) {
            for (int i = 0; i < 7; i++) { output[e][i] = sol[i]; }
            seen[e] = 1;
        }, active);
        for (size_t e = 0; e < connectivity.size(); e++) {
            REQUIRE(seen[e] == active[e]);
            if (!active[e]) { continue; }
            double expected[7] = {};
            const double av = x[connectivity[e][0]];
            const double bv = x[connectivity[e][1]];
            for (int q = 0; q < 3; q++) {
                const size_t i = e * 3 + q;
                const double w = weights[i];
                expected[0] += w * (0.5 * ((av - targets[i][0]) * (av - targets[i][0])
                                           + (bv - targets[i][1]) * (bv - targets[i][1])) + cross_coefficient * av * bv);
                expected[1] += w * (av - targets[i][0] + cross_coefficient * bv);
                expected[2] += w * (bv - targets[i][1] + cross_coefficient * av);
                expected[3] += w;
                expected[4] += cross_coefficient * w;
                expected[5] += cross_coefficient * w;
                expected[6] += w;
            }
            for (int i = 0; i < 7; i++) {
                INFO(name << " element " << e << " output " << i);
                REQUIRE(approx(output[e][i], expected[i], tolerance));
            }
        }
    };
    check();
    const bool previous_check_mode = get_check_mode_ON();
    check_mode_ON(true);
    check();
    check_mode_ON(previous_check_mode);

    // Grow then shrink without recompiling; data, filtering, and coloring follow the current rows.
    connectivity.resize(8);
    connectivity[5] = {3, 0};
    connectivity[6] = {0, 2};
    connectivity[7] = {3, 1};
    active = {1, 0, 1, 1, 1, 0, 1, 1};
    fill_data(0.11);
    loop.update_coloring();
    check();

    connectivity.resize(3);
    active = {1, 0, 1};
    fill_data(-0.04);
    loop.update_coloring();
    check();
}

TEST_CASE("Quadrature data refresh follows element rows and coloring", "[quadrature]")
{
    check_quadrature_refresh<double>("quadrature_refresh_scalar");
#ifdef SYMX_ENABLE_AVX2
    check_quadrature_refresh<__m256d>("quadrature_refresh_simd");
    check_quadrature_refresh<__m256>("quadrature_refresh_simd_float", 1e-5);
#endif
}

TEST_CASE("Quadrature data and fixed summation have the same derivatives", "[quadrature]")
{
    std::vector<std::array<int32_t, 1>> connectivity = {{0}, {1}};
    std::vector<double> x = {0.3, -0.4};
    std::vector<std::array<double, 2>> rule = {{0.2, -0.1}, {0.5, 0.4}, {0.3, 0.7}};
    std::vector<std::array<double, 2>> per_element_rule;
    for (size_t e = 0; e < connectivity.size(); e++) {
        per_element_rule.insert(per_element_rule.end(), rule.begin(), rule.end());
    }
    auto [fixed, fixed_elem] = MappedWorkspace<double>::create(connectivity);
    Scalar xf = fixed->make_scalar(x, fixed_elem[0]);
    Scalar ef = fixed->add_for_each(rule, [&](Vector& qp) { return qp[0] * (xf - qp[1]).powN(2); });
    CompiledInLoop<double> fixed_loop(fixed, value_gradient_hessian(ef, {xf}), "quadrature_fixed_parity", get_codegen_dir());

    auto [mapped, mapped_elem] = MappedWorkspace<double>::create(connectivity);
    Scalar xm = mapped->make_scalar(x, mapped_elem[0]);
    mapped->make_summation(static_cast<int32_t>(rule.size()));
    Vector qp = mapped->make_quadrature_vector(per_element_rule);
    Scalar em = qp[0] * (xm - qp[1]).powN(2);
    CompiledInLoop<double> mapped_loop(mapped, value_gradient_hessian(em, {xm}), "quadrature_mapped_parity", get_codegen_dir());

    std::vector<int64_t> offsets = {0, 3, 6};
    auto [dynamic, dynamic_elem] = MappedWorkspace<double>::create(connectivity);
    Scalar xd = dynamic->make_scalar(x, dynamic_elem[0]);
    Scalar ed = dynamic->add_for_each(offsets, [&, dynamic = dynamic]() {
        Vector point = dynamic->make_quadrature_vector(per_element_rule);
        return point[0] * (xd - point[1]).powN(2);
    });
    CompiledInLoop<double> dynamic_loop(dynamic, value_gradient_hessian(ed, {xd}), "quadrature_dynamic_parity", get_codegen_dir());

    // A fixed rule can share its summation with a nontrivial per-element matrix.
    std::vector<double> scale_data;
    scale_data.reserve(per_element_rule.size() * 4);
    for (size_t i = 0; i < per_element_rule.size(); i++) {
        scale_data.insert(scale_data.end(), {2.0, 3.0, 4.0, 7.0});
    }
    auto [mixed, mixed_elem] = MappedWorkspace<double>::create(connectivity);
    Scalar xx = mixed->make_scalar(x, mixed_elem[0]);
    Matrix scale = mixed->make_quadrature_matrix(scale_data, {2, 2});
    Scalar ex = mixed->add_for_each(rule, [&](Vector& point) {
        Scalar unit_scale = 0.5 * (scale(0, 0) * scale(1, 1) - scale(0, 1) * scale(1, 0));
        return unit_scale * point[0] * (xx - point[1]).powN(2);
    });
    CompiledInLoop<double> mixed_loop(mixed, value_gradient_hessian(ex, {xx}), "quadrature_mixed_parity", get_codegen_dir());

    std::array<std::array<double, 3>, 2> reference{};
    std::array<uint8_t, 2> fixed_seen{};
    fixed_loop.run(1, [&](const View<double> sol, int32_t e, int32_t, const View<int32_t>) {
        for (int i = 0; i < 3; i++) { reference[e][i] = sol[i]; }
        fixed_seen[e] = 1;
    });
    for (int e = 0; e < 2; e++) {
        double expected[3] = {};
        for (const auto& point : rule) {
            expected[0] += point[0] * (x[e] - point[1]) * (x[e] - point[1]);
            expected[1] += 2.0 * point[0] * (x[e] - point[1]);
            expected[2] += 2.0 * point[0];
        }
        REQUIRE(fixed_seen[e]);
        for (int i = 0; i < 3; i++) { REQUIRE(approx(reference[e][i], expected[i], 1e-10)); }
    }
    std::array<uint8_t, 2> mapped_seen{};
    mapped_loop.run(1, [&](const View<double> sol, int32_t e, int32_t, const View<int32_t>) {
        for (int i = 0; i < 3; i++) { REQUIRE(approx(sol[i], reference[e][i], 1e-10)); }
        mapped_seen[e] = 1;
    });
    std::array<uint8_t, 2> mixed_seen{};
    mixed_loop.run(1, [&](const View<double> sol, int32_t e, int32_t, const View<int32_t>) {
        for (int i = 0; i < 3; i++) { REQUIRE(approx(sol[i], reference[e][i], 1e-10)); }
        mixed_seen[e] = 1;
    });
    std::array<uint8_t, 2> dynamic_seen{};
    dynamic_loop.run(1, [&](const View<double> sol, int32_t e, int32_t, const View<int32_t>) {
        for (int i = 0; i < 3; i++) { REQUIRE(approx(sol[i], reference[e][i], 1e-10)); }
        dynamic_seen[e] = 1;
    });
    for (int e = 0; e < 2; e++) {
        REQUIRE(mapped_seen[e]);
        REQUIRE(mixed_seen[e]);
        REQUIRE(dynamic_seen[e]);
    }
}

#if !defined(_WIN32)
TEST_CASE("Quadrature data validates its summation and extent", "[quadrature]")
{
    const auto second_summation = run_expected_failure([] {
        std::vector<std::array<int32_t, 1>> connectivity = {{0}};
        auto [mws, elem] = MappedWorkspace<double>::create(connectivity);
        mws->make_summation(2);
        mws->make_summation_vector(std::vector<std::array<double, 1>>{{1.0}});
    });
    INFO(second_summation.diagnostic);
    REQUIRE(second_summation.failed);
    REQUIRE(second_summation.diagnostic.find("only one summation") != std::string::npos);

    const auto short_data = run_expected_failure([] {
        std::vector<std::array<int32_t, 1>> connectivity = {{0}, {1}};
        std::vector<double> short_data = {1.0, 2.0, 3.0};
        auto [mws, elem] = MappedWorkspace<double>::create(connectivity);
        mws->make_summation(2);
        mws->make_quadrature_scalar(short_data);
        mws->verify_iteration_maps();
    });
    INFO(short_data.diagnostic);
    REQUIRE(short_data.failed);
    REQUIRE(short_data.diagnostic.find("expected 4") != std::string::npos);
}
#endif

TEST_CASE("Adjoint reads per-element quadrature data", "[quadrature][Adjoint]")
{
    bool dynamic = false;
    SECTION("Fixed summation") {}
    SECTION("Dynamic summation") { dynamic = true; }
    struct Data {
        std::vector<Eigen::Vector3d> x = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0)};
        std::vector<Eigen::Vector3d> rest = x;
        std::vector<double> spring_weights = {0.3, 0.7};
        std::vector<std::array<double, 2>> loss_points = {{0.4, 0.6}, {0.6, 0.8}};
        double stiffness = 100.0;
        double bc_stiffness = 1000.0;
    } data;
    std::vector<std::array<int32_t, 2>> spring = {{0, 1}};
    std::vector<std::array<int32_t, 1>> bc = {{0}, {1}};
    std::vector<int64_t> offsets = {0, 2};

    auto G = GlobalPotential::create();
    G->add_potential("qp_adjoint_spring", spring, [&](MappedWorkspace<double>& mws, Element& e) {
        auto x = mws.make_vectors(data.x, e);
        Scalar k = mws.make_scalar(data.stiffness);
        if (dynamic) { mws.make_summation(offsets); }
        else { mws.make_summation(2); }
        Scalar w = mws.make_quadrature_scalar(data.spring_weights);
        return 0.5 * k * w * (x[1] - x[0]).squared_norm();
    });
    G->add_potential("qp_adjoint_bc", bc, [&](MappedWorkspace<double>& mws, Element& e) {
        Vector x = mws.make_vector(data.x, e[0]);
        Vector rest = mws.make_vector(data.rest, e[0]);
        Scalar k = mws.make_scalar(data.bc_stiffness);
        return 0.5 * k * (x - rest).squared_norm();
    });
    G->add_dof(data.x);
    auto context = Context::create();
    context->output->set_enabled(false);
    context->n_threads = 1;
    auto newton = NewtonsMethod::create(G, context);
    newton->settings.residual_tolerance_abs = 1e-10;

    auto L = GlobalPotential::create();
    L->add_potential("qp_adjoint_loss", spring, [&](MappedWorkspace<double>& mws, Element& e) {
        auto x = mws.make_vectors(data.x, e);
        if (dynamic) { mws.make_summation(offsets); }
        else { mws.make_summation(2); }
        Vector qp = mws.make_quadrature_vector(data.loss_points);
        return qp[0] * (x[1][0] - x[0][0] - qp[1]).powN(2);
    });
    L->add_dof(data.stiffness);
    LinearSolveSettings linear;
    linear.linear_solver = LinearSolver::DirectLLT;
    auto adjoint = AdjointNewton::create(newton, L, linear);
    const auto finite_difference = adjoint->evaluate_dL_dp_finite_differences(1e-3);
    const auto analytic = adjoint->run_forward_and_evaluate_dL_dp();
    REQUIRE(finite_difference.success);
    REQUIRE(analytic.success);
    REQUIRE(analytic.gradient.size() == 1);
    REQUIRE(approx(analytic.gradient[0], finite_difference.gradient[0], 1e-3));
    REQUIRE(approx(analytic.gradient[0], -0.0003148148148148148, 1e-10));
}
