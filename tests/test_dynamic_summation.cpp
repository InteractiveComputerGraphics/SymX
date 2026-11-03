#include "catch.hpp"
#include "utils.h"

using namespace symx;

template<typename FLOAT>
static void check_dynamic_refresh(const std::string& name, double tolerance)
{
    std::vector<std::array<int32_t, 1>> connectivity;
    std::vector<int64_t> offsets;
    std::vector<std::array<FLOAT, 2>> points;
    std::vector<FLOAT> x(41);
    for (size_t i = 0; i < x.size(); i++) { x[i] = static_cast<FLOAT>(0.1 * i); }
    auto fill = [&](size_t n, int phase) {
        connectivity.resize(n);
        // Replace allocations to check that bindings retain containers, not pointers.
        std::vector<int64_t> next_offsets(n + 1, 0);
        std::vector<std::array<FLOAT, 2>> next_points;
        for (size_t e = 0; e < n; e++) {
            connectivity[e] = {static_cast<int32_t>((17 * e) % x.size())};
            for (size_t q = 0; q < (e + phase) % 5; q++) {
                next_points.push_back({static_cast<FLOAT>(0.2 + 0.01 * q),
                                       static_cast<FLOAT>(0.03 * e - 0.04 * q)});
            }
            next_offsets[e + 1] = static_cast<int64_t>(next_points.size());
        }
        offsets.swap(next_offsets);
        points.swap(next_points);
    };
    // Each large color has enough groups to enter the OpenMP parallel loop,
    // even after filtering. The last group is partially filled.
    fill(83, 0);
    auto [mws, elem] = MappedWorkspace<FLOAT>::create(connectivity);
    Scalar a = mws->make_scalar(x, elem[0]);
    int definitions = 0;
    Scalar energy = mws->add_for_each(
        [&]() { return offsets.data(); },
        [&]() { return static_cast<int64_t>(offsets.size()); },
        [&, mws = mws]() {
            ++definitions;
            Vector ip = mws->make_quadrature_vector(points);
            return 1.0 + ip[0] * (a - ip[1]).powN(2);
        });
    REQUIRE(definitions == 1);
    CompiledInLoop<FLOAT> loop(mws, value_gradient_hessian(energy, {a}), name, get_codegen_dir());

    auto check = [&]() {
        std::vector<std::array<FLOAT, 3>> result(connectivity.size());
        std::vector<uint8_t> active(connectivity.size(), 1), seen(connectivity.size(), 0);
        for (size_t e = 0; e < active.size(); e++) { active[e] = e % 11 != 1; }
        loop.run(2, [&](View<FLOAT> sol, int32_t e, int32_t, View<int32_t>) {
            for (int i = 0; i < 3; i++) { result[e][i] = sol[i]; }
            seen[e] = 1;
        }, active);
        for (size_t e = 0; e < connectivity.size(); e++) {
            REQUIRE(seen[e] == active[e]);
            if (!active[e]) { continue; }
            double expected[3] = {};
            const double av = x[connectivity[e][0]];
            for (int64_t q = offsets[e]; q < offsets[e + 1]; q++) {
                const double w = points[q][0], delta = av - points[q][1];
                expected[0] += 1.0 + w * delta * delta;
                expected[1] += 2.0 * w * delta;
                expected[2] += 2.0 * w;
            }
            for (int i = 0; i < 3; i++) {
                INFO(name << " element " << e << " output " << i);
                REQUIRE(approx(result[e][i], expected[i], tolerance));
            }
        }
    };
    check();
    loop.enable_coloring({0});
    const bool previous_check_mode = get_check_mode_ON();
    check_mode_ON(true);
    check();
    fill(91, 2);
    loop.update_coloring();
    check();
    fill(5, 1);
    loop.update_coloring();
    check();
    offsets.assign(connectivity.size() + 1, 0);
    points.clear();
    check();
    connectivity.clear();
    offsets = {0};
    loop.update_coloring();
    check();
    // Reuse the compiled loop after transitioning through an empty mesh.
    fill(3, 3);
    loop.update_coloring();
    check();
    check_mode_ON(previous_check_mode);
    REQUIRE(definitions == 1);
}

TEST_CASE("Dynamic summation refreshes ranges and handles empty sums", "[quadrature][dynamic_summation]")
{
    SECTION("Double") { check_dynamic_refresh<double>("dynamic_refresh_double", 1e-10); }
    SECTION("Float") { check_dynamic_refresh<float>("dynamic_refresh_float", 1e-4); }
    SECTION("Empty sums do not evaluate the kernel, even without quadrature bindings") {
        std::vector<std::array<int32_t, 1>> conn = {{0}};
        std::vector<int64_t> offsets = {0, 0};
        double x = -1.0;
        auto [mws, elem] = MappedWorkspace<double>::create(conn);
        Scalar a = mws->make_scalar(x);
        mws->make_summation(offsets);
        Scalar energy = log(a);
        CompiledInLoop<double> loop(mws, value_gradient_hessian(energy, {a}), "dynamic_empty_no_points", get_codegen_dir());
        int calls = 0;
        loop.run(1, [&](View<double> sol, int32_t, int32_t, View<int32_t>) {
            ++calls;
            for (int i = 0; i < 3; i++) { REQUIRE(sol[i] == 0.0); }
        });
        REQUIRE(calls == 1);
    }
}

#if !defined(_WIN32)
TEST_CASE("Dynamic summation rejects invalid ranges and data", "[quadrature][dynamic_summation]")
{
    std::vector<int64_t> offsets = {0, 2};
    std::vector<double> values = {1.0, 2.0};
    int stride = 1;
    bool duplicate = false;
    std::string diagnostic;
    SECTION("Missing sentinel") { offsets = {0}; diagnostic = "requires 2 offsets"; }
    SECTION("Nonzero start") { offsets = {1, 2}; diagnostic = "start at zero"; }
    SECTION("Negative or decreasing range") { offsets = {0, -1}; diagnostic = "nondecreasing"; }
    SECTION("Wrong point extent") { values.resize(1); diagnostic = "expected 2"; }
    SECTION("Extent overflow") {
        offsets = {0, std::numeric_limits<int64_t>::max()};
        stride = 2;
        diagnostic = "invalid quadrature map metadata";
    }
    SECTION("Empty dynamic summation still prevents a second summation") {
        offsets = {0, 0}; duplicate = true; diagnostic = "only one summation";
    }
    const auto failure = run_expected_failure([&]() {
        std::vector<std::array<int32_t, 1>> conn = {{0}};
        auto [mws, elem] = MappedWorkspace<double>::create(conn);
        mws->make_summation(offsets);
        if (duplicate) { mws->make_summation(1); }
        mws->make_quadrature_vector(values, stride);
        mws->verify_iteration_maps();
    });
    INFO(failure.diagnostic);
    REQUIRE(failure.failed);
    REQUIRE(failure.diagnostic.find(diagnostic) != std::string::npos);
}

#ifdef SYMX_ENABLE_AVX2
TEST_CASE("Dynamic summation rejects SIMD before compilation, loading, and empty evaluation", "[quadrature][dynamic_summation]")
{
    enum class Operation { Compile, Load, Run };
    Operation operation = Operation::Compile;
    SECTION("Compilation") {}
    SECTION("Cache loading") { operation = Operation::Load; }
    SECTION("Empty evaluation after declaring dynamic summation") { operation = Operation::Run; }
    const auto failure = run_expected_failure([&]() {
        std::vector<std::array<int32_t, 1>> conn;
        std::vector<int64_t> offsets = {0};
        auto [mws, elem] = MappedWorkspace<double>::create(conn);
        double value = 1.0;
        Scalar a = mws->make_scalar(value);
        CompiledInLoop<double, __m256d> loop;
        if (operation == Operation::Run) {
            loop.compile(mws, {a}, "dynamic_simd_rejection", get_codegen_dir());
        }
        mws->make_summation(offsets);
        if (operation == Operation::Compile) {
            loop.compile(mws, {a}, "dynamic_simd_rejection", get_codegen_dir());
        }
        else if (operation == Operation::Load) {
            loop.load_if_cached(mws, "dynamic_simd_rejection", get_codegen_dir(), "unused");
        }
        else {
            loop.run(1, [](View<double>, int32_t, int32_t, View<int32_t>) {});
        }
    });
    INFO(failure.diagnostic);
    REQUIRE(failure.failed);
    REQUIRE(failure.diagnostic.find("dynamic summation does not support SIMD") != std::string::npos);
}
#endif
#endif
