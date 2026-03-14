#include "examples_main.h"

#include <fmt/format.h>
#include <symx>
#include <iostream>
#include <filesystem>
#include <fmt/format.h>
#include <vtkio>
#include <set>
#include <algorithm>
#include <random>

#include "examples_utils.h"

using namespace symx;

// XPBD Correction functions =====================================================================
std::pair<Vector, Scalar> xpbd_correction_scalar(const Scalar& constraint, const Vector& dofs, std::vector<Scalar>& inv_mass, const Scalar& k, const Scalar& dt, const Scalar& lambda)
{
    if (dofs.size() != inv_mass.size()*3) {
        throw std::runtime_error("xpbd_correction_scalar: dofs size must be 3 times inv_mass size.");
    }
    
    // Inverse mass matrix
    const int n_particles = (int)inv_mass.size();
    Matrix M_inv = Matrix::identity(3*n_particles, constraint);
    for (int i = 0; i < n_particles; i++) {
        for (int j = 0; j < 3; j++) {
            M_inv(3*i + j, 3*i + j) = inv_mass[i];
        }
    }
    
    // Constraint gradient
    Vector grad = gradient(constraint, dofs);
    
    // XPBD solve for lagrange multiplier lambda
    Scalar alpha = 1.0/k; // compliance
    Scalar alpha_tilde = alpha/(dt*dt);
    Scalar denominator = grad.dot(M_inv.dot(grad)) + alpha_tilde;
    Scalar delta_lambda = (-constraint - alpha_tilde*lambda)/(denominator + 1e-12);
    Vector correction = M_inv*grad*delta_lambda;
    
    return std::make_pair(correction, delta_lambda);
}
std::pair<Vector, Vector> xpbd_correction_tensor(const Vector& constraint, const Vector& dofs, std::vector<Scalar>& inv_mass, const Matrix& alpha_tilde, const Scalar& dt, const Vector& lambda)
{
    if (dofs.size() != inv_mass.size()*3) {
        throw std::runtime_error("xpbd_correction_tensor: dofs size must be 3 times inv_mass size.");
    }
    
    // Inverse mass matrix
    const int n_particles = (int)inv_mass.size();
    Matrix M_inv = Matrix::identity(3*n_particles, constraint[0]);
    for (int i = 0; i < n_particles; i++) {
        for (int j = 0; j < 3; j++) {
            M_inv(3*i + j, 3*i + j) = inv_mass[i];
        }
    }
    
    // Constraint gradient
    Matrix grad = gradient(constraint, dofs, /* symmetric = */ false);
    
    // XPBD solve for lagrange multiplier lambda
    Matrix lhs = grad.dot(M_inv.dot(grad.transpose())) + alpha_tilde;
    Vector rhs = -constraint - alpha_tilde*lambda;
    Vector delta_lambda = lhs.inv().dot(rhs);
    Vector correction = M_inv*grad.transpose()*delta_lambda;
    
    return std::make_pair(correction, delta_lambda);
}
// ===============================================================================================


template<typename INPUT_FLOAT, typename COMPILED_FLOAT>
void xpbd_cloth_sim(int n_threads, int res, bool enable_output)
{
    // Type definitions
    using CIL = CompiledInLoop<INPUT_FLOAT, COMPILED_FLOAT>;
    using Vec3 = Eigen::Matrix<INPUT_FLOAT, 3, 1>;
    using Vec2 = Eigen::Matrix<INPUT_FLOAT, 2, 1>;
    constexpr INPUT_FLOAT zero = INPUT_FLOAT(0.0);
    constexpr INPUT_FLOAT one = INPUT_FLOAT(1.0);
    constexpr INPUT_FLOAT two = INPUT_FLOAT(2.0);

    if (n_threads <= 0) {
        n_threads = omp_get_num_procs()/2;
    }

    std::cout << "\nStarting XPBD Cloth Simulation..." << std::endl;
    std::cout << "Types: INPUT_FLOAT=" << get_float_type_as_string<INPUT_FLOAT>() << ", COMPILED_FLOAT=" << get_float_type_as_string<COMPILED_FLOAT>() << std::endl;
    std::cout << "n_threads = " << n_threads << " | resolution = " << res << "x" << res << std::endl;

    struct Data
    {
        // Parameters
        INPUT_FLOAT duration = INPUT_FLOAT(1.0);
        int fps = 30;
        INPUT_FLOAT dt = INPUT_FLOAT(0.0025);
        int n_substeps = 5;
        Vec2 size = Vec2(INPUT_FLOAT(1.0), INPUT_FLOAT(1.0));
        std::array<int32_t, 2> resolution = {0, 0};

        INPUT_FLOAT density = INPUT_FLOAT(1.0);
        INPUT_FLOAT damping = INPUT_FLOAT(0.99);
        INPUT_FLOAT rest_angle = INPUT_FLOAT(0.0);
        INPUT_FLOAT strain_stiffness = INPUT_FLOAT(1.0);
        INPUT_FLOAT bending_stiffness = INPUT_FLOAT(2e-4);
        INPUT_FLOAT youngs_modulus = INPUT_FLOAT(5e3);
        INPUT_FLOAT poissons_ratio = INPUT_FLOAT(0.3);
        INPUT_FLOAT thickness = INPUT_FLOAT(0.005);
        Vec3 gravity = Vec3(0.0, 0.0, INPUT_FLOAT(-9.81));

        Vec3 sphere_center = Vec3(0.0, 0.0, INPUT_FLOAT(-0.5));
        INPUT_FLOAT sphere_radius = INPUT_FLOAT(0.25);
        INPUT_FLOAT friction = INPUT_FLOAT(0.9);

        // Simulation state
        std::vector<Vec3> X;
        std::vector<Vec3> x0;
        std::vector<Vec3> x1;
        std::vector<Vec3> v;
        std::vector<INPUT_FLOAT> spring_lambdas;
        std::vector<Vec3> strain_lambdas;
        std::vector<INPUT_FLOAT> bending_lambdas;
        std::vector<INPUT_FLOAT> inv_mass;

        // Precomputed data
        std::vector<std::array<INPUT_FLOAT, 4>> DX_inv;
        Eigen::Matrix<INPUT_FLOAT, 3, 3, Eigen::RowMajor> alpha_tilde;

    } data;
    
    // Initialize resolution in data
    data.resolution = {res, res};
    
    // ============================ MESH GENERATION ============================
    TriangleMesh<INPUT_FLOAT> mesh = generate_triangle_grid(data.size, data.resolution);
    LabelledConnectivity<3> edge_conn = get_unique_edges_conn(mesh.triangles);
    LabelledConnectivity<5> dihedral_conn = build_dihedral_connectivity(mesh.triangles);
    LabelledConnectivity<4> triangle_conn = get_triangle_conn(mesh.triangles);
    const int n_vertices = (int)mesh.vertices.size();
    const INPUT_FLOAT area_per_vertex = data.size[0]*data.size[1]/(INPUT_FLOAT)n_vertices;

    std::cout << "Generated mesh with " << n_vertices << " vertices and " 
              << mesh.triangles.size() << " triangles" << std::endl;
              
    // ============================ SIMULATION DATA ============================
    data.X = mesh.vertices;
    data.x0 = mesh.vertices;
    data.x1 = mesh.vertices;
    data.v.assign(n_vertices, Vec3::Zero());
    data.spring_lambdas.assign(edge_conn.size(), INPUT_FLOAT(0.0));
    data.strain_lambdas.assign(triangle_conn.size(), Vec3::Zero());
    data.bending_lambdas.assign(dihedral_conn.size(), INPUT_FLOAT(0.0));
    INPUT_FLOAT mass = data.density*area_per_vertex; // Mass per vertex
    data.inv_mass.assign(n_vertices, one/mass);

    // ============================ PRECOMPUTATION ============================
    // Strain
    const INPUT_FLOAT m = data.youngs_modulus/(two*(one + data.poissons_ratio));
    const INPUT_FLOAT l = (data.youngs_modulus*data.poissons_ratio)/((one + data.poissons_ratio)*(one - two*data.poissons_ratio));
    Eigen::Matrix<INPUT_FLOAT, 3, 3> K = Eigen::Matrix<INPUT_FLOAT, 3, 3>::Zero();
    K(0, 0) = l + two*m;
    K(1, 1) = l + two*m;
    K(2, 2) = two*m;
    K(0, 1) = l;
    K(1, 0) = l;
    data.alpha_tilde = K.inverse()/(data.thickness*area_per_vertex*data.dt*data.dt);

    // Rest shape inverses
    data.DX_inv.resize(triangle_conn.size());
    for (int i = 0; i < triangle_conn.size(); i++) {
        const Vec3& x0 = data.X[mesh.triangles[i][0]];
        const Vec3& x1 = data.X[mesh.triangles[i][1]];
        const Vec3& x2 = data.X[mesh.triangles[i][2]];

        const Vec3 u = (x1 - x0).normalized();
        const Vec3 n = (u.cross(x2 - x0)).normalized();
        const Vec3 v = n.cross(u);
        Eigen::Matrix<INPUT_FLOAT, 2, 3> P;
        P.row(0) = u.transpose();
        P.row(1) = v.transpose();

        const Vec2 x0p = P*x0;
        const Vec2 x1p = P*x1;
        const Vec2 x2p = P*x2;

        Eigen::Matrix<INPUT_FLOAT, 2, 2> DX;
        DX.col(0) = x1p - x0p;
        DX.col(1) = x2p - x0p;
        Eigen::Matrix<INPUT_FLOAT, 2, 2> DXinv = DX.inverse();

        data.DX_inv[i] = {DXinv(0, 0), DXinv(0, 1), DXinv(1, 0), DXinv(1, 1)};
    }

    // Pinned vertices
    const INPUT_FLOAT eps = INPUT_FLOAT(1e-9);
    for (int i = 0; i < n_vertices; ++i) {
        bool top = data.x1[i].y() > 0.5*data.size.y() - eps;
        bool left_corner = data.x1[i].x() < -0.5*data.size.x() + eps;
        bool right_corner = data.x1[i].x() > 0.5*data.size.x() - eps;
        if (top && (left_corner || right_corner)) {
            data.inv_mass[i] = INPUT_FLOAT(0.0);
        }
    }

    // ============================ SIMULATION XPBD CORRECTION DEFINITIONS ============================
    // Spring
    auto make_spring_loop = [&]() 
    {
        // Create workspace for symbols mapped to the simulation data
        auto [mws, element] = MappedWorkspace<INPUT_FLOAT>::create(edge_conn);

        // Unpack connectivity indices
        Index edge_idx = element[0];
        std::vector<Index> edge = element.slice(1, 3);

        // Create symbols from data
        std::vector<Vector> x1 = mws->make_vectors(data.x1, edge);
        std::vector<Vector> X = mws->make_vectors(data.X, edge);
        std::vector<Scalar> inv_mass = mws->make_scalars(data.inv_mass, edge);
        Scalar k = mws->make_scalar(data.strain_stiffness);
        Scalar dt = mws->make_scalar(data.dt);
        Scalar lambda = mws->make_scalar(data.spring_lambdas, edge_idx);

        // Define constraint
        Scalar curr_length = (x1[1] - x1[0]).stable_norm(1e-9);
        Scalar rest_length = (X[1] - X[0]).norm();
        Scalar constraint = curr_length - rest_length;

        // Define what we want to compute: correction and delta_lambda
        Vector dofs = collect_scalars(x1);
        auto [correction, delta_lambda] = xpbd_correction_scalar(constraint, dofs, inv_mass, k, dt, lambda);
        Vector output = collect_scalars({correction.values(), {delta_lambda}});

        // Create compiled in-loop object
        auto loop = CIL::create(mws, output.values(), "spring_xpbd_corr", symx::get_codegen_dir());
        loop->enable_coloring({1, 2});  // Color by vertex indices

        return loop;
    };
    auto spring_loop = make_spring_loop();

    // Triangle strain
    auto make_triangle_loop = [&]() 
    {
        // Create workspace for symbols mapped to the simulation data
        auto [mws, element] = MappedWorkspace<INPUT_FLOAT>::create(triangle_conn);

        // Unpack connectivity indices
        Index idx = element[0];
        std::vector<Index> triangle = element.slice(1, 4);

        // Create symbols from data
        std::vector<Vector> x1 = mws->make_vectors(data.x1, triangle);
        Matrix DX_inv = mws->make_matrix(data.DX_inv, {2, 2}, idx);
        Matrix alpha_tilde = mws->make_matrix(data.alpha_tilde, {3, 3});
        std::vector<Scalar> inv_mass = mws->make_scalars(data.inv_mass, triangle);
        Scalar dt = mws->make_scalar(data.dt);
        Vector lambda = mws->make_vector(data.strain_lambdas, idx);

        // Define constraint
        Matrix Dx_32 = symx::Matrix(symx::collect_scalars({x1[1] - x1[0], x1[2] - x1[0]}), {2, 3}).transpose();
        Matrix F = Dx_32*DX_inv;
        Matrix C = F.transpose()*F;
        Matrix E = 0.5*(C - mws->make_identity_matrix(2));
        Vector constraint = Vector({E(0, 0), E(1, 1), 2.0*E(0, 1)});

        // Define what we want to compute: correction and delta_lambda
        Vector dofs = collect_scalars(x1);
        auto [correction, delta_lambda] = xpbd_correction_tensor(constraint, dofs, inv_mass, alpha_tilde, dt, lambda);
        Vector output = collect_scalars({correction.values(), {delta_lambda}});

        // Create compiled in-loop object
        auto loop = CIL::create(mws, output.values(), "triangle_strain_xpbd_corr", symx::get_codegen_dir());
        loop->enable_coloring({1, 2, 3});  // Color by vertex indices

        return loop;
    };
    auto triangle_loop = make_triangle_loop();

    // Bending
    auto make_bending_loop = [&]() 
    {
        // Create workspace for symbols mapped to the simulation data
        auto [mws, element] = MappedWorkspace<INPUT_FLOAT>::create(dihedral_conn);

        // Unpack connectivity indices
        Index idx = element[0];
        std::vector<Index> dihedral = element.slice(1, 5);

        // Create symbols from data
        std::vector<Vector> x1 = mws->make_vectors(data.x1, dihedral);
        std::vector<Scalar> inv_mass = mws->make_scalars(data.inv_mass, dihedral);
        Scalar rest_angle = mws->make_scalar(data.rest_angle);
        Scalar k = mws->make_scalar(data.bending_stiffness);
        Scalar dt = mws->make_scalar(data.dt);
        Scalar lambda = mws->make_scalar(data.bending_lambdas, idx);

        // Define constraint
        Vector e0 = x1[1] - x1[0];
        Vector e1 = x1[2] - x1[0];
        Vector e2 = x1[3] - x1[0];
        Vector n0 = e0.cross3(e1);
        Vector n1 = -e0.cross3(e2);
        Scalar dihedral_angle_rad = ((1.0 - 1e-5)*n0.normalized().dot(n1.normalized())).acos();
        Scalar constraint = dihedral_angle_rad - rest_angle;

        // Define what we want to compute: correction and delta_lambda
        Vector dofs = collect_scalars(x1);
        auto [correction, delta_lambda] = xpbd_correction_scalar(constraint, dofs, inv_mass, k, dt, lambda);
        Vector output = collect_scalars({correction.values(), {delta_lambda}});

        // Create compiled in-loop object
        auto loop = CIL::create(mws, output.values(), "bending_xpbd_corr", symx::get_codegen_dir());
        loop->enable_coloring({1, 2, 3, 4});  // Color by vertex indices

        return loop;
    };
    auto bending_loop = make_bending_loop();
    
    // ============================ SIMULATION LOOP ============================
    std::filesystem::create_directories(std::string(SYMX_EXAMPLES_OUTPUT_DIR) + "/xpbd_cloth");

    int frame_idx = 0;
    const double t0 = omp_get_wtime();
    for (double time = 0.0; time < data.duration; time += data.dt) {
        // Store initial positions
        data.x0 = data.x1;
        
        // Semi-implicit Euler integration
        for (int i = 0; i < n_vertices; i++) {
            if (data.inv_mass[i] > 1e-6) {
                data.v[i] += data.gravity*data.dt;
                data.x1[i] += data.v[i]*data.dt;
            }
        }
        
        // XPBD constraint loops
        std::fill(data.spring_lambdas.begin(), data.spring_lambdas.end(), zero);
        std::fill(data.strain_lambdas.begin(), data.strain_lambdas.end(), Vec3::Zero());
        std::fill(data.bending_lambdas.begin(), data.bending_lambdas.end(), zero);
        for (int substep = 0; substep < data.n_substeps; substep++) {

            // Springs
            spring_loop->run(n_threads,
                [&](const View<INPUT_FLOAT> out, const int32_t spring_idx,
                    const int32_t thread_id, const View<int32_t> conn) 
                {
                    const Vec3 corr_i(out[0], out[1], out[2]);
                    const Vec3 corr_j(out[3], out[4], out[5]);
                    INPUT_FLOAT delta_lambda = out[6];

                    data.spring_lambdas[conn[0]] += delta_lambda;
                    data.x1[conn[1]] += corr_i;
                    data.x1[conn[2]] += corr_j;
                }
            );

            // Triangle strain
            triangle_loop->run(n_threads,
                [&](const View<INPUT_FLOAT> out, const int32_t hinge_idx,
                    const int32_t thread_id, const View<int32_t> conn) 
                {
                    const Vec3 corr_a(out[0], out[1], out[2]);
                    const Vec3 corr_b(out[3], out[4], out[5]);
                    const Vec3 corr_c(out[6], out[7], out[8]);
                    const Vec3 delta_lambda(out[9], out[10], out[11]);

                    data.strain_lambdas[conn[0]] += delta_lambda;
                    data.x1[conn[1]] += corr_a;
                    data.x1[conn[2]] += corr_b;
                    data.x1[conn[3]] += corr_c;
                }
            );

            // Bending
            bending_loop->run(n_threads,
                [&](const View<INPUT_FLOAT> out, const int32_t hinge_idx,
                    const int32_t thread_id, const View<int32_t> conn) 
                {
                    const Vec3 corr_a(out[0], out[1], out[2]);
                    const Vec3 corr_b(out[3], out[4], out[5]);
                    const Vec3 corr_c(out[6], out[7], out[8]);
                    const Vec3 corr_d(out[9], out[10], out[11]);
                    INPUT_FLOAT delta_lambda = out[12];

                    data.bending_lambdas[conn[0]] += delta_lambda;
                    data.x1[conn[1]] += corr_a;
                    data.x1[conn[2]] += corr_b;
                    data.x1[conn[3]] += corr_c;
                    data.x1[conn[4]] += corr_d;
                }
            );
        }
        
        // Sphere collision + friction
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int i = 0; i < n_vertices; ++i) {
            if (data.inv_mass[i] <= 0.0) continue; // skip pinned

            Vec3 d = data.x1[i] - data.sphere_center;
            INPUT_FLOAT dist = d.norm();
            if (dist < data.sphere_radius) {
                Vec3 n = d/dist;
                INPUT_FLOAT penetration = data.sphere_radius - dist;
                data.x1[i] += n*penetration;

                // Velocity preview
                const Vec3 vi = (data.x1[i] - data.x0[i])/data.dt;
                const Vec3 vn = vi.dot(n)*n;
                const Vec3 vt = vi - vn;
                const Vec3 vf = data.friction*vt;
                const Vec3 vi_corrected = vn + vt - vf;

                data.x1[i] = data.x0[i] + vi_corrected*data.dt;
            }
        }

        // Update velocities + damping
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int i = 0; i < n_vertices; i++) {
            if (data.inv_mass[i] > 1e-6) {
                data.v[i] = (data.x1[i] - data.x0[i])/data.dt;
            }
            data.v[i] *= data.damping;
        }
        
        // Output
        if (enable_output && time > frame_idx*(1.0/(double)data.fps)) {
            // std::cout << fmt::format("{:.2f}s {}.vtk", time, frame_idx) << std::endl;
            vtkio::VTKFile vtk_file;
            vtk_file.set_points_from_twice_indexable(data.x1);
            vtk_file.set_cells_from_twice_indexable(mesh.triangles, vtkio::CellType::Triangle);
            vtk_file.write(fmt::format(std::string(SYMX_EXAMPLES_OUTPUT_DIR) + "/xpbd_cloth/xpbd_cloth_{:04d}.vtk", frame_idx));
            frame_idx++;
        }
    }
    const double t1 = omp_get_wtime();
    std::cout << fmt::format("Simulation completed in {:.2f} seconds", t1 - t0) << std::endl;
}


void xpbd_cloth_comparison(int n_threads, int res, bool enable_output)
{
    std::cout << "\n=================== NeoHookean_assembly_comparison() ===================" << std::endl;

    if (n_threads == -1) {
        n_threads = omp_get_num_procs()/2;
    }

    xpbd_cloth_sim<double, double>(n_threads, res, enable_output);
    xpbd_cloth_sim<float, float>(n_threads, res, enable_output);
#ifdef SYMX_ENABLE_AVX2
    xpbd_cloth_sim<double, __m256d>(n_threads, res, enable_output);
    xpbd_cloth_sim<float, __m256>(n_threads, res, enable_output);
#endif
}
