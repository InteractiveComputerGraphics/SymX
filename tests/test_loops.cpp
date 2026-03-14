#include "catch.hpp"

#ifdef SYMX_ENABLE_AVX2
#include <immintrin.h>
#endif

#include <omp.h>
#include <symx>
#include "sympy_derivatives.h"
#include "utils.h"

using namespace symx;

TEST_CASE("triangle_point_new", "[loops_new]")
{
	// Data
    EigenMatrixRM<double> points = EigenMatrixRM<double>::Random(4, 3);
	const double S = -1.0;
	std::vector<std::array<int, 4>> triangle_point_pairs = { {0, 1, 2, 3}, {0, 1, 2, 3} };

	// SymPy reference
	std::vector<double> in(13), ref(157);
    std::copy(points.data(), points.data() + points.size(), in.begin());
	in[12] = S;
	sympy_point_triangle_distance_derivatives(in.data(), ref.data());

    // Workspace
    auto [mws, conn] = MappedWorkspace<double>::create(triangle_point_pairs);

    // Make Symbols
    std::vector<Vector> x = mws->make_vectors(points, conn);
    Scalar sign = mws->make_scalar(S);

    // Distance calculation
    Vector ap = x[3] - x[0];
    Vector ca = x[0] - x[2];
    Vector cb = x[1] - x[2];
    Vector n = ca.cross3(cb).normalized();
    Scalar dist = sign * ap.dot(n);

    // Derivatives
    std::vector<Scalar> dofs = collect_scalars(x);
    std::vector<Scalar> vals = value_gradient_hessian(dist, dofs);

    // Compiled in loop
    CompiledInLoop<double> loop(mws, vals, "triangle_point", symx::get_codegen_dir());

    loop.run(/*n_threads = */ 1,
        [&ref](const View<double> solution, const int32_t iteration, const int32_t thread_id, const View<int32_t> connectivity)
        {
            for (int i = 0; i < (int)ref.size(); i++) {
                REQUIRE(approx(ref[i], solution[i]));
            }
        }
    );
}

template <typename COMPILED_FLOAT>
void test_triangle_area(int n_threads, bool use_coloring, bool use_conditional, int n_summations = 1, int num_elements_override = -1)
{
    // Generate mesh
    Eigen::Vector2d size(1.0, 1.0);
    std::array<int32_t, 2> quads = {5, 5}; // 25 quads -> 50 triangles
    symx::TriangleMesh<double> mesh = symx::generate_triangle_grid(size, quads);

    if (num_elements_override > 0) {
        if (num_elements_override < (int)mesh.triangles.size()) {
            mesh.triangles.resize(num_elements_override);
        }
    }

    // Create workspace
    auto [mws, conn] = MappedWorkspace<double>::create(mesh.triangles);

    // Symbols
    std::vector<Vector> x = mws->make_vectors(mesh.vertices, conn);

    // Area calculation: 0.5 * ||(B-A) x (C-A)||
    Vector A = x[0];
    Vector B = x[1];
    Vector C = x[2];
    Vector AB = B - A;
    Vector AC = C - A;
    Vector cross = AB.cross3(AC);
    Scalar area = 0.5 * cross.norm();

    // Expression
    std::vector<Scalar> exprs;

    // Summation
    if (n_summations > 1) {
        std::vector<std::array<double, 1>> iteration_vectors;
        for (int i = 0; i < n_summations; i++) {
            iteration_vectors.push_back({1.0});
        }
        Scalar summation_area = mws->add_for_each(iteration_vectors,
            [&](Vector& vec) {
                return area * vec[0];
            }
        );
        exprs.push_back(summation_area);
    } 
    else {
        exprs.push_back(area);
    }

    // Compile
    CompiledInLoop<double, COMPILED_FLOAT> loop(mws, exprs, "triangle_area", symx::get_codegen_dir());

    if (use_coloring) {
        std::vector<int> indices = {0, 1, 2};
        loop.enable_coloring(indices);
    }

    // Reference calculation
    double ref_area = 0.0;
    std::vector<double> ref_areas(mesh.triangles.size());
    for (size_t i = 0; i < mesh.triangles.size(); ++i) {
        Eigen::Vector3d v0 = mesh.vertices[mesh.triangles[i][0]];
        Eigen::Vector3d v1 = mesh.vertices[mesh.triangles[i][1]];
        Eigen::Vector3d v2 = mesh.vertices[mesh.triangles[i][2]];
        ref_areas[i] = 0.5 * (v1 - v0).cross(v2 - v0).norm();
        ref_areas[i] *= (double)n_summations;
        double cx = (v0.x() + v1.x() + v2.x()) / 3.0;
        if (use_conditional) {
            if (cx > 0.0) {
                ref_area += ref_areas[i];
            }
        } 
        else {
            ref_area += ref_areas[i];
        }
    }
    
    // Conditional flags (centroid x > 0.0)
    std::vector<uint8_t> active(mesh.triangles.size(), 1);
    if (use_conditional) {
        for (size_t i = 0; i < mesh.triangles.size(); ++i) {
            const auto& tri = mesh.triangles[i];
            Eigen::Vector3d v0 = mesh.vertices[tri[0]];
            Eigen::Vector3d v1 = mesh.vertices[tri[1]];
            Eigen::Vector3d v2 = mesh.vertices[tri[2]];
            double cx = (v0.x() + v1.x() + v2.x()) / 3.0;
            active[i] = static_cast<uint8_t>(cx > 0.0);
        }
    }
    
    // Execution check
    double sum_area = 0.0;
    auto execute_callback = [&](const View<double> solution, const int32_t element_idx, const int32_t thread_id, const View<int32_t> connectivity)
    {
        // Check conditional if active
        if (use_conditional) {
            Eigen::Vector3d v0 = mesh.vertices[connectivity[0]];
            Eigen::Vector3d v1 = mesh.vertices[connectivity[1]];
            Eigen::Vector3d v2 = mesh.vertices[connectivity[2]];
            double cx = (v0.x() + v1.x() + v2.x()) / 3.0;
            if (cx < 0.0) {
                REQUIRE(false);
            }
        }

        double ref = ref_areas[element_idx];
        double sol = solution[0];
        if constexpr (std::is_same_v<COMPILED_FLOAT, float>
#ifdef SYMX_ENABLE_AVX2
            || std::is_same_v<COMPILED_FLOAT, __m256>
#endif
            ) {
            REQUIRE(std::abs(sol - ref) < 1e-4);
        } else {
            REQUIRE(approx(ref, sol));
        }

        #pragma omp atomic
        sum_area += (double)sol;
    };
    
    // Run
    loop.run(n_threads, execute_callback, active);
    REQUIRE(sum_area == Approx(ref_area));
}

TEST_CASE("triangle_area_combinations", "[loops_new]")
{
    // void test_triangle_area(int n_threads, bool use_coloring, bool use_conditional, int n_summations = 1, int num_elements_override = -1)
    
    SECTION("Double") {
        test_triangle_area<double>(1, false, false);
        test_triangle_area<double>(4, true, false);
        test_triangle_area<double>(1, false, true);
        test_triangle_area<double>(4, false, false);

        test_triangle_area<double>(1, false, false, 3);
        test_triangle_area<double>(4, true, false, 3);
        test_triangle_area<double>(1, false, true, 3);
        test_triangle_area<double>(4, false, false, 3);
    }
    
    SECTION("Float") {
        test_triangle_area<float>(1, false, false);
    }
    
#ifdef SYMX_ENABLE_AVX2
    SECTION("AVX2 Double") {
        // Mid mesh
        test_triangle_area<__m256d>(1, false, false);
        test_triangle_area<__m256d>(4, true, true);
        test_triangle_area<__m256d>(4, false, true);
        test_triangle_area<__m256d>(4, false, false);

        // Precise number of elements
        for (int n_threads : {1, 4}) {
            for (bool use_coloring : {false, true}) {
                for (bool use_conditional : {false, true}) {
                    for (int n_summations : {1, 3}) {
                        for (int num_elements : {1, 4, 8, 9}) {
                            test_triangle_area<__m256d>(n_threads, use_coloring, use_conditional, n_summations, num_elements);
                        }
                    }
                }
            }
        }
    }

    SECTION("AVX2 Float") {
        test_triangle_area<__m256>(1, false, false);
        test_triangle_area<__m256>(1, false, false, 1);
        test_triangle_area<__m256>(1, false, false, 4);
        test_triangle_area<__m256>(1, false, false, 8);
    }
#endif
}
