#include "examples_main.h"

#include <Eigen/Sparse>
#include <fmt/format.h>
#include <symx>

#include "elasticity_potentials.h"

using namespace symx;


void NeoHookean_print()
{
    std::cout << "\n=================== NeoHookean_print() ===================" << std::endl;

    // Lambda function to obtain the jacobian of a Tet4 element given its vertices
    auto jac_tet4 = [](Workspace& ws, const std::vector<Vector>& xh)
    {
        Matrix J = ws.get_identity_matrix(3);
        J.set_col(0, xh[1] - xh[0]);
        J.set_col(1, xh[2] - xh[0]);
        J.set_col(2, xh[3] - xh[0]);
        return J;
    };

    // SymX workspace
    Workspace ws;

    // Create symbols
    std::vector<Vector> x = ws.make_vectors(3, 4);
    std::vector<Vector> X_rest = ws.make_vectors(3, 4);
    Scalar lambda = ws.make_scalar();
    Scalar mu = ws.make_scalar();

    // FEM potential
    Matrix Jx = jac_tet4(ws, x);
    Matrix JX = jac_tet4(ws, X_rest);
    Matrix F = Jx*JX.inv();
    Scalar vol = JX.det()/6.0;
    Scalar E = vol*neohookean_energy_density(F, lambda, mu);

    // Differentiation
    DiffCache diff_cache;
    std::vector<Scalar> dofs = collect_scalars(x);
    Vector grad = gradient(E, dofs, diff_cache);
    Matrix hess = gradient(grad, dofs, /* is_symmetric = */true, diff_cache);

    // Compilation
    std::string codegen_dir = symx::get_codegen_dir();
    Compiled<double> c_grad(grad.values(), "NH_grad", codegen_dir);
    Compiled<double> c_hess(hess.values(), "NH_hess", codegen_dir);

    // Data
    struct Data
    {
        std::vector<Eigen::Vector3d> x;
        std::vector<Eigen::Vector3d> X_rest;
        double lambda;
        double mu;
    } data;
    data.x.push_back(Eigen::Vector3d::Zero());
    data.x.push_back(Eigen::Vector3d::UnitX());
    data.x.push_back(Eigen::Vector3d::UnitY());
    data.x.push_back(Eigen::Vector3d::UnitZ());
    data.X_rest.push_back(Eigen::Vector3d::Zero());
    data.X_rest.push_back(0.9*Eigen::Vector3d::UnitX());
    data.X_rest.push_back(0.8*Eigen::Vector3d::UnitY());
    data.X_rest.push_back(0.7*Eigen::Vector3d::UnitZ());
    double E_ = 1e6;
    double nu_ = 0.3;
    data.lambda = (E_ * nu_) / ((1.0 + nu_) * (1.0 - 2.0 * nu_));
    data.mu = E_ / (2.0 * (1.0 + nu_));
    
    // Gradient
    {
        // Set values
        for (int v = 0; v < 4; ++v) {
            c_grad.set(x[v], data.x[v]);
            c_grad.set(X_rest[v], data.X_rest[v]);
        }
        c_grad.set(lambda, data.lambda);
        c_grad.set(mu, data.mu);
    
        // Evaluate
        View<double> result = c_grad.run();
    
        // Print
        Eigen::Matrix<double, 12, 1> g;
        for (int i = 0; i < 12; i++) {
            g[i] = result[i];
        }
        std::cout << "Gradient = \n" << g.transpose() << std::endl;
    }
    
    // Hessian
    {
        // Set values
        for (int v = 0; v < 4; ++v) {
            c_hess.set(x[v], data.x[v]);
            c_hess.set(X_rest[v], data.X_rest[v]);
        }
        c_hess.set(lambda, data.lambda);
        c_hess.set(mu, data.mu);
    
        // Evaluate
        View<double> result = c_hess.run();
    
        // Print
        Eigen::Matrix<double, 12, 12> h;
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                h(i, j) = result[12*i + j];
            }
        }
        std::cout << "\nHessian = \n" << h << std::endl;
    }
}

template<typename FLOAT, typename COMPILATION_FLOAT>
void NeoHookean_eval_microbenchmark(int n_threads)
{
    if (n_threads == -1) {
        n_threads = omp_get_max_threads()/2;
    }

    // Lambda function to obtain the jacobian of a Tet4 element given its vertices
    auto jac_tet4 = [](Workspace& ws, const std::vector<Vector>& xh)
    {
        Matrix J = ws.get_identity_matrix(3);
        J.set_col(0, xh[1] - xh[0]);
        J.set_col(1, xh[2] - xh[0]);
        J.set_col(2, xh[3] - xh[0]);
        return J;
    };

    // SymX workspace
    Workspace ws;

    // Create symbols
    std::vector<Vector> x = ws.make_vectors(3, 4);
    std::vector<Vector> X_rest = ws.make_vectors(3, 4);
    Scalar lambda = ws.make_scalar();
    Scalar mu = ws.make_scalar();

    // FEM potential
    Matrix Jx = jac_tet4(ws, x);
    Matrix JX = jac_tet4(ws, X_rest);
    Matrix F = Jx*JX.inv();
    Scalar vol = JX.det()/6.0;
    Scalar E = vol*neohookean_energy_density(F, lambda, mu);

    // DoFs
    std::vector<Scalar> dofs = collect_scalars(x);
    
    // Checksum
    std::string checksum = ws.get_checksum();
    for (const Scalar& s : dofs) {
        checksum += s.get_checksum();
    }

    // Try load compiled functions
    std::string codegen_dir = symx::get_codegen_dir();
    Compiled<COMPILATION_FLOAT> c_grad;
    Compiled<COMPILATION_FLOAT> c_hess;
    std::string grad_name = "NH_grad_" + get_float_type_as_string<COMPILATION_FLOAT>();
    std::string hess_name = "NH_hess_" + get_float_type_as_string<COMPILATION_FLOAT>();

    c_grad.load_if_cached(grad_name, codegen_dir, checksum);
    c_hess.load_if_cached(hess_name, codegen_dir, checksum);
    
    // Can't load, must compile
    if (!c_grad.was_cached() || !c_hess.was_cached()) {

        // Differentiation
        DiffCache diff_cache;
        Vector grad = gradient(E, dofs, diff_cache);

        double t0 = omp_get_wtime();
        Matrix hess = gradient(grad, dofs, /* is_symmetric = */true, diff_cache);
        double t1 = omp_get_wtime();
        std::cout << "Diff hess " << t1 - t0 << std::endl;

        // Compilation
        c_grad.compile(grad.values(), grad_name, codegen_dir, checksum);
        c_hess.compile(hess.values(), hess_name, codegen_dir, checksum);
    }
    // c_hess.get_compilation_info().print();


    // Data
    using Vec3 = Eigen::Matrix<FLOAT, 3, 1>;
    struct Data
    {
        std::vector<Vec3> x;
        std::vector<Vec3> X_rest;
        FLOAT lambda;
        FLOAT mu;
    } data;
    data.x.push_back(Vec3::Zero());
    data.x.push_back(Vec3::UnitX());
    data.x.push_back(Vec3::UnitY());
    data.x.push_back(Vec3::UnitZ());
    data.X_rest.push_back(Vec3::Zero());
    data.X_rest.push_back(0.9*Vec3::UnitX());
    data.X_rest.push_back(0.8*Vec3::UnitY());
    data.X_rest.push_back(0.7*Vec3::UnitZ());
    FLOAT E_ = 1e6;
    FLOAT nu_ = 0.3;
    data.lambda = (E_ * nu_) / ((1.0 + nu_) * (1.0 - 2.0 * nu_));
    data.mu = E_ / (2.0 * (1.0 + nu_));

    // Run benchmark
    int n_runs = 20'000'000;
    const int thread_id = 0; // Each thread uses its own buffer. No need to specify for single-threaded, done here for illustration.
    c_hess.set_n_threads(n_threads); // No need in the case of single-threaded, done here for illustration
    bsm::ParallelNumber<double> dummy;
    dummy.reset(n_threads);
    
    // Set values
    auto set = [](const FLOAT& v)
    {
#ifdef SYMX_ENABLE_AVX2
        if constexpr (std::is_same_v<COMPILATION_FLOAT, __m256d>) {
            return _mm256_set1_pd(v);
        }
        else if constexpr (std::is_same_v<COMPILATION_FLOAT, __m256>) {
            return _mm256_set1_ps(v);
        }
        else
#endif
        {
            return static_cast<COMPILATION_FLOAT>(v);
        }
    };
    for (int tid = 0; tid < n_threads; tid++) {
        for (int v = 0; v < 4; ++v) {
            for (int i = 0; i < 3; i++) {
                c_hess.set(x[v][i], set(data.x[v][i]), tid);
                c_hess.set(X_rest[v][i], set(data.X_rest[v][i]), tid);
            }
        }
        c_hess.set(lambda, set(data.lambda), tid);
        c_hess.set(mu, set(data.mu), tid);
    }
    
    // Run
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int i = 0; i < n_runs; i++) {
        int tid = omp_get_thread_num();

        // Run
        View<COMPILATION_FLOAT> result = c_hess.run(tid);
        
        // Prevent optimizing away
        FLOAT* result_f = reinterpret_cast<FLOAT*>(&result[0]);
        dummy.get(tid) += (double)result_f[0];
    }
    double t1 = omp_get_wtime();
    int per_run = get_n_items_in_simd<COMPILATION_FLOAT>();
    std::cout << fmt::format("[{:2d} threads] runtime: {:6f} microseconds | checksum: {:6e}", n_threads, 1e6 * (t1 - t0)/(n_runs*per_run), dummy.end()) << std::endl;
}
void NeoHookean_eval_comparison()
{
    std::cout << "\n=================== NeoHookean_eval_comparison() ===================" << std::endl;

    const int n_threads = omp_get_max_threads()/2;

    std::cout << "double" << std::endl;
    NeoHookean_eval_microbenchmark<double, double>(1);
    NeoHookean_eval_microbenchmark<double, double>(n_threads);
    std::cout << std::endl;
    
    std::cout << "float" << std::endl;
    NeoHookean_eval_microbenchmark<float, float>(1);
    NeoHookean_eval_microbenchmark<float, float>(n_threads);
    std::cout << std::endl;

#ifdef SYMX_ENABLE_AVX2
    std::cout << "SIMD4d" << std::endl;
    NeoHookean_eval_microbenchmark<double, __m256d>(1);
    NeoHookean_eval_microbenchmark<double, __m256d>(n_threads);
    std::cout << std::endl;
    
    std::cout << "SIMD8f" << std::endl;
    NeoHookean_eval_microbenchmark<float, __m256>(1);
    NeoHookean_eval_microbenchmark<float, __m256>(n_threads);
    std::cout << std::endl;
#endif
}

template<typename COMPILATION_FLOAT, bool PROJECT_TO_PD, bool USE_EIGEN>
void NeoHookean_assembly_microbenchmark(int size, int n_threads)
{
    auto jac_tet4 = [](const std::vector<Vector>& xh)
    {
        return Matrix(collect_scalars({ xh[1] - xh[0], xh[2] - xh[0] , xh[3] - xh[0] }), { 3, 3 }).transpose();
    };

    struct Data
    {
        std::vector<std::array<int, 4>> tets;
        std::vector<Eigen::Vector3d> x;
        std::vector<Eigen::Vector3d> X_rest;
        double lambda;
        double mu;
    } data;

    // Initialization
    auto mesh = generate_cuboid_Tet4_mesh<double>({1.0, 1.0, 1.0}, {size, size, size});
    data.x = mesh.vertices;
    data.X_rest = mesh.vertices;
    data.tets = as_array_vec<4>(mesh.connectivity);
    data.lambda = 2.66e6;
    data.mu = 1.7e5;

    const int ndofs = (int)data.x.size()*3;
    const int ntets = (int)data.tets.size();
    for (int i = 0; i < (int)data.x.size(); i++) {
        data.x[i][0] *= 0.99;
        data.x[i][1] *= 0.83;
        data.x[i][2] *= 0.77;
    }

    // Create MappedWorkspace
    auto [mws, element] = MappedWorkspace<double>::create(data.tets);

    // Create symbols
    std::vector<Vector> x = mws->make_vectors(data.x, element);
    std::vector<Vector> X_rest = mws->make_vectors(data.X_rest, element);
    Scalar lambda = mws->make_scalar(data.lambda);
    Scalar mu = mws->make_scalar(data.mu);

    // FEM potential
    Matrix Jx = jac_tet4(x);
    Matrix JX = jac_tet4(X_rest);
    Matrix F = Jx*JX.inv();
    Scalar vol = JX.det()/6.0;
    Scalar E = vol*neohookean_energy_density(F, lambda, mu);

    // DoFs
    std::vector<Scalar> dofs = collect_scalars(x);

    // Differentiation
    std::vector<Scalar> values = value_gradient_hessian(E, dofs);

    // Compile
    const std::string codegen_dir = get_codegen_dir();
    const std::string checksum = mws->get_workspace().get_checksum();
    auto loop = CompiledInLoop<double, COMPILATION_FLOAT>::create(mws, values, "NH_E_grad_hess", codegen_dir, checksum);

    // Prepare evaluation
    bsm::ParallelNumber<double> energy;
    bsm::ParallelVector<Eigen::VectorXd> grad;
    energy.reset(n_threads);
    grad.reset(n_threads, ndofs);

    // Eigen evaluation
    if constexpr (USE_EIGEN) {
        // Prepare parallel structs
        using Triplet = Eigen::Triplet<double>;
        std::vector<std::vector<Triplet>> triplets;
        
        triplets.resize(n_threads);
        for (int tid = 0; tid < n_threads; tid++) {
            triplets[tid].reserve(144*ntets);
        }
        
        // Evaluate
        double t0 = omp_get_wtime();
        loop->run(n_threads, 
        [&](View<double> out, int32_t elem_idx, 
            int32_t thread_id, View<int32_t> elem_conn)
            {
                // Energy value
                energy.get(thread_id) += out[0];

                // Gradient
                View<double> out_grad = out.slice(1); // Drop the first value
                Eigen::VectorXd& thread_grad = grad.get(thread_id);
                for (int v = 0; v < 4; v++) {
                    for (int d = 0; d < 3; d++) {
                        thread_grad[3*elem_conn[v] + d] += out_grad[3*v + d];
                    }
                }
                
                // Hessian
                View<double> out_hess = out_grad.slice(12); // Drop the first 12 values
                
                if constexpr (PROJECT_TO_PD) {
                    project_to_PD_inplace(out_hess.data(), /* size = */ 12, /* eps = */ 1e-10, /* abs = */ false);
                }

                auto& thread_triplets = triplets[thread_id];
                for (int vi = 0; vi < 4; vi++) {
                    for (int di = 0; di < 3; di++) {
                        for (int vj = 0; vj < 4; vj++) {
                            for (int dj = 0; dj < 3; dj++) {
                                thread_triplets.emplace_back(
                                    3*elem_conn[vi] + di,
                                    3*elem_conn[vj] + dj,
                                    out_hess[12*(3*vi + di) + 3*vj + dj]
                                );
                            }
                        }
                    }
                }
            }
        );
        double t1 = omp_get_wtime();
        const double global_energy = energy.end();
        std::cout << fmt::format("Eigen - {}: [{:2d}][{:6}]({:.2e} dofs, {:.2e} tets) Eval time: {:6.2f} ms | Check: {:10.4f}", 
            (PROJECT_TO_PD) ? "SPD" : "Unp",
            n_threads, get_float_type_as_string<COMPILATION_FLOAT>(), (double)ndofs, (double)ntets, 1000.0*(t1 - t0), global_energy);

        // Reduce parallel structures
        t0 = omp_get_wtime();
        const Eigen::VectorXd& global_gradient = grad.end();

        std::vector<int> triplet_offsets;
        triplet_offsets.push_back(0);
        for (int tid = 0; tid < n_threads; tid++) {
            triplet_offsets.push_back(triplet_offsets.back() + (int)triplets[tid].size());
        }
        const int n_triplets = triplet_offsets.back();

        #pragma omp parallel num_threads(n_threads)
        {
            const int tid = omp_get_thread_num();
            const int begin = triplet_offsets[tid];
            const int end = triplet_offsets[tid + 1];
            const int n = end - begin;

            memcpy(triplets[0].data() + begin, triplets[tid].data(), sizeof(Triplet)*n);
        }
        
        Eigen::SparseMatrix<double, Eigen::RowMajor> s;
        s.resize(ndofs, ndofs);
        s.setFromTriplets(triplets[0].begin(), triplets[0].end());
        s.makeCompressed();

        t1 = omp_get_wtime();
        std::cout << " | Reduction time (ms): " << 1000.0*(t1 - t0) << std::endl;
    }

    // Use BSM
    else {
        bsm::BlockedSparseMatrix<3, 3, double> hess;

        std::array<double, 2> times;
        for (int i = 0; i < 2; i++) {
            hess.start_insertion(ndofs, ndofs);
            energy.reset(n_threads);
            grad.reset(n_threads, ndofs);

            // Evaluate
            double t0 = omp_get_wtime();
            loop->run(n_threads, 
            [&](View<double> out, int32_t elem_idx, 
                int32_t thread_id, View<int32_t> elem_conn)
                {
                    // Energy value
                    energy.get(thread_id) += out[0];

                    // Gradient
                    View<double> out_grad = out.slice(1); // Drop the first value
                    Eigen::VectorXd& thread_grad = grad.get(thread_id);
                    for (int v = 0; v < 4; v++) {
                        for (int d = 0; d < 3; d++) {
                            thread_grad[3*elem_conn[v] + d] += out_grad[3*v + d];
                        }
                    }
                    
                    // Hessian
                    View<double> out_hess = out_grad.slice(12); // Drop the first 12 values
                    
                    if constexpr (PROJECT_TO_PD) {
                        project_to_PD_inplace(out_hess.data(), /* size = */ 12, /* eps = */ 1e-10, /* abs = */ false);
                    }

                    std::array<double, 9> block;
                    for (int vi = 0; vi < 4; vi++) {
                        for (int vj = 0; vj < 4; vj++) {
                            for (int i = 0; i < 3; i++) {
                                for (int j = 0; j < 3; j++) {
                                    block[3*i + j] = out_hess[(vi + i)*12 + (vj + j)];
                                }
                            }
                            hess.add_block_from_ptr<double, bsm::Ordering::RowMajor>(3*elem_conn[vi], 3*elem_conn[vj], block.data());
                        }
                    }
                }
            );
            hess.end_insertion(n_threads);
            double t1 = omp_get_wtime();
            times[i] = 1000.0*(t1 - t0);
        }
        
        const double global_energy = energy.end();
        std::cout << fmt::format("SymX - {}: [{:2d}][{:6}]({:.2e} dofs, {:.2e} tets) Eval time 0: {:6.2f} ms | Eval time 1: {:6.2f} ms | Check: {:10.4f}", 
            (PROJECT_TO_PD) ? "SPD" : "Unp",
            n_threads, get_float_type_as_string<COMPILATION_FLOAT>(), (double)ndofs, (double)ntets, times[0], times[1], global_energy);

        std::cout << std::endl;
    }
}
void NeoHookean_assembly_comparison()
{
    std::cout << "\n=================== NeoHookean_assembly_comparison() ===================" << std::endl;

    constexpr bool PROJECT_TO_PD = false;
    const int n_threads = omp_get_max_threads()/2;
    const int n = 40;

    // Eigen
    if (true) {
        constexpr bool USE_EIGEN = true;
        NeoHookean_assembly_microbenchmark<double, PROJECT_TO_PD, USE_EIGEN>(n, 1);
#ifdef SYMX_ENABLE_AVX2
        NeoHookean_assembly_microbenchmark<__m256d, PROJECT_TO_PD, USE_EIGEN>(n, 1);
        NeoHookean_assembly_microbenchmark<__m256, PROJECT_TO_PD, USE_EIGEN>(n, 1);
#endif
        NeoHookean_assembly_microbenchmark<double, PROJECT_TO_PD, USE_EIGEN>(n, n_threads);
#ifdef SYMX_ENABLE_AVX2
        NeoHookean_assembly_microbenchmark<__m256d, PROJECT_TO_PD, USE_EIGEN>(n, n_threads);
        NeoHookean_assembly_microbenchmark<__m256, PROJECT_TO_PD, USE_EIGEN>(n, n_threads);
#endif
    }

    // BSM
    if (true) {
        constexpr bool USE_EIGEN = false;
        NeoHookean_assembly_microbenchmark<double, PROJECT_TO_PD, USE_EIGEN>(n, 1);
#ifdef SYMX_ENABLE_AVX2
        NeoHookean_assembly_microbenchmark<__m256d, PROJECT_TO_PD, USE_EIGEN>(n, 1);
        NeoHookean_assembly_microbenchmark<__m256, PROJECT_TO_PD, USE_EIGEN>(n, 1);
#endif
        NeoHookean_assembly_microbenchmark<double, PROJECT_TO_PD, USE_EIGEN>(n, n_threads);
#ifdef SYMX_ENABLE_AVX2
        NeoHookean_assembly_microbenchmark<__m256d, PROJECT_TO_PD, USE_EIGEN>(n, n_threads);
        NeoHookean_assembly_microbenchmark<__m256, PROJECT_TO_PD, USE_EIGEN>(n, n_threads);
#endif
    }
}
