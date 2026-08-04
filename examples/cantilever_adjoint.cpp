#include "examples_main.h"

#include <fmt/format.h>
#include <filesystem>
#include <map>
#include <cmath>
#include <fstream>
#include <vtkio>
#include <symx>

#include "elasticity_potentials.h"
#include "examples_utils.h"

using namespace symx;

void write_cube_mesh(const std::string& path, const Eigen::Vector3d& location, const double size)
{
    const double h = 0.5 * size;
    const std::vector<Eigen::Vector3d> vertices = {
        location + Eigen::Vector3d(-h, -h, -h),
        location + Eigen::Vector3d( h, -h, -h),
        location + Eigen::Vector3d( h,  h, -h),
        location + Eigen::Vector3d(-h,  h, -h),
        location + Eigen::Vector3d(-h, -h,  h),
        location + Eigen::Vector3d( h, -h,  h),
        location + Eigen::Vector3d( h,  h,  h),
        location + Eigen::Vector3d(-h,  h,  h)
    };
    const std::vector<int> quads = {
        0, 3, 2, 1,  // -Z
        4, 5, 6, 7,  // +Z
        0, 1, 5, 4,  // -Y
        1, 2, 6, 5,  // +X
        2, 3, 7, 6,  // +Y
        3, 0, 4, 7   // -X
    };

    vtkio::VTKFile vtk_file;
    vtk_file.set_comments("Surface cube marker");
    vtk_file.set_points_from_twice_indexable(vertices);
    vtk_file.set_cells_from_indexable(quads, 4, vtkio::CellType::Quad);
    vtk_file.write(path);
}


// =============================================================================
//  1.  Density identification
// =============================================================================
void cantilever_density_id(FEM_Element fem_element_type)
{
    std::cout << "\n\n==============================================================================" << std::endl;
    std::cout << "Density identification" << std::endl;
    std::cout << "==============================================================================" << std::endl;

    // ----------------------------------------------------------------------
    // Data 
    // ----------------------------------------------------------------------
    struct Data
    {
        // Forward
        double youngs_modulus = 1e6;
        double poissons_ratio = 0.4;
        double bc_stiffness = 1e8;
        double density = 1000.0;
        Eigen::Vector3d gravity = { 0.0, 0.0, -9.81 };

        FEM_Element fem_element_type = FEM_Element::Hex8;
        double length = 1.0;
        double width = 0.2;
        int n_subdivitions = 2; // short axis
        std::vector<Eigen::Vector3d> x; // Current positions
        std::vector<Eigen::Vector3d> X; // Rest positions
        LabelledConnectivity<1> bc_conn{{ "vertex_idx" }};

        // Inverse
        double density_initial_guess = 200.0;
        double shape_matching_weight = 1.0;
        
        std::vector<Eigen::Vector3d> target_vertices;
        LabelledConnectivity<1> all_vertices_conn{{ "vertex_idx" }};

        // Inverse
    } data;

    data.fem_element_type = fem_element_type;

    // ----------------------------------------------------------------------
    // Forward problem
    // ----------------------------------------------------------------------

    // Mesh
    double elem_size = data.width / (double)data.n_subdivitions;
    int n_elems_length = std::round(data.length / elem_size);
    Eigen::Vector3d size = {data.length, data.width, data.width};
    std::array<int, 3> elements_per_axis = {n_elems_length, data.n_subdivitions, data.n_subdivitions};
    FEMMesh<double> mesh = generate_cuboid_mesh(data.fem_element_type, size, elements_per_axis);

    // Print
    std::cout << "> Element type: " << get_name(mesh.element_type) << std::endl;
    std::cout << "> Number of elements: " << mesh.n_elements << std::endl;
    std::cout << "> Number of vertices: " << mesh.vertices.size() << std::endl;

    // Initialize domain
    data.x = mesh.vertices;
    data.X = mesh.vertices;

    //// All vertices
    for (int i = 0; i < (int)data.x.size(); i++) {
        data.all_vertices_conn.push_back({i});
    }

    //// Boundary conditions
    const double left_wall_x = -0.5*data.length + 1e-6;
    for (int i = 0; i < (int)data.x.size(); i++) {
        const Eigen::Vector3d& v = mesh.vertices[i];
        if (v[0] < left_wall_x) {
            data.bc_conn.push_back({i});
        }
    }

    // Define Global Potential
    spGlobalPotential G = GlobalPotential::create();

    // Stable Neo-Hookean material energy (generic on FEM element type)
	G->add_potential("stable_neo_hookean_" + get_name(mesh.element_type), mesh.connectivity, mesh.connectivity_stride,
		[&](MappedWorkspace<double>& mws, Element& elem)
		{
            // Create symbols from data
            std::vector<Vector> xh = mws.make_vectors(data.x, elem);
            std::vector<Vector> Xh = mws.make_vectors(data.X, elem);
            Scalar E = mws.make_scalar(data.youngs_modulus);
            Scalar nu = mws.make_scalar(data.poissons_ratio);
            Vector gravity = mws.make_vector(data.gravity);
            Scalar density = mws.make_scalar(data.density);

            // Define potential
            Scalar P = fem_integrator(mws, mesh.element_type,
				[&](Scalar& w, Vector xi)
				{
                    // Kinematics
                    Vector x = fem_interpolation(mesh.element_type, xh, xi);
					Matrix Jx = fem_jacobian(mesh.element_type, x, xi);
					Matrix JX = fem_jacobian(mesh.element_type, Xh, xi);
					Matrix F = Jx * JX.inv();
                    Scalar dV = JX.det() * w;

                    // Elasticity
					Scalar P_elas = stable_neohookean_energy_density(F, E, nu) * dV;

                    // Gravity potential
                    Scalar P_grav = -density * gravity.dot(x) * dV;

                    return P_elas + P_grav;
                }
            );
            return P;
		}
	);

    // Boundary conditions energy
    G->add_potential("boundary_conditions", data.bc_conn,
        [&](MappedWorkspace<double>& mws, Element& conn)
        {
            // Create symbols from data
            Vector x = mws.make_vector(data.x, conn["vertex_idx"]);
            Vector X = mws.make_vector(data.X, conn["vertex_idx"]);
            Scalar k = mws.make_scalar(data.bc_stiffness);

            // Define potential
            Scalar P = 0.5 * k * (x - X).squared_norm();
            return P;
        }
    );

    // DoF declaration
    G->add_dof(data.x);
    
    // Context
    spContext context = Context::create();
    context->compilation_directory = symx::get_codegen_dir();
    std::string output_folder = std::string(SYMX_EXAMPLES_OUTPUT_DIR) + "/cantilever_id";
    std::filesystem::create_directories(output_folder);
    context->output->open_file(fmt::format("{}/cantilever_id_{}.log", output_folder, get_name(mesh.element_type)));
    
    // Newton Solver
    auto newton = NewtonsMethod::create(G, context);

    // Settings
    newton->settings.projection_mode = ProjectionToPD::ProjectOnDemand;
    newton->settings.residual_tolerance_abs = 1e-6;
    newton->settings.step_tolerance = 1e-8;

    // Prepare connectivity for VTK output
    std::vector<int> output_conn = mesh.connectivity;
    prepare_for_export(output_conn, mesh.element_type);

    // Solve
    newton->solve();

    // Set target for inverse problem
    data.target_vertices = data.x;

    // Write VTKs
    vtkio::VTKFile vtk_file;
    vtk_file.set_cells_from_indexable(output_conn, mesh.connectivity_stride, fem_element_to_vtk_cell_type(mesh.element_type));

    //// Rest
    vtk_file.set_points_from_twice_indexable(data.X);
    vtk_file.write(fmt::format("{}/cantilever_id_rest_{}.vtk", output_folder, get_name(mesh.element_type)));

    //// Deformed
    vtk_file.set_points_from_twice_indexable(data.x);
    vtk_file.write(fmt::format("{}/cantilever_id_sim_{}.vtk", output_folder, get_name(mesh.element_type)));


    // ----------------------------------------------------------------------
    // Inverse problem
    // ----------------------------------------------------------------------

    // Global Loss Definition
    auto G_loss = GlobalPotential::create();

    // Shape matching loss
    const double vertex_vol = 1.0 / (double)data.x.size();
    G_loss->add_potential(
        "shape_match", data.all_vertices_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector x = mws.make_vector(data.x, elem[0]);
            Vector t = mws.make_vector(data.target_vertices, elem[0]);
            Scalar w = mws.make_scalar(data.shape_matching_weight);
            Scalar vol = mws.make_scalar(vertex_vol);

            return 0.5 * w * vol * (x - t).squared_norm();
        });

    // Parameter
    G_loss->add_dof(data.density, "density");

    // Define Adjoint
    LinearSolveSettings lss;
    lss.linear_solver = LinearSolver::DirectLLT;
    auto adjoint = AdjointNewton::create(newton, G_loss, lss);
    // context->output->set_console_verbosity(Verbosity::Medium);
    
    // Define Optimizer
    //// Settings
    auto opt = FirstOrderOptimizerAdjoint::create(adjoint, context);
    
    opt->settings.type = FirstOrderOptType::LBFGS;
    opt->settings.lbfgs.bootstrap_step_length = 100.0;
    opt->settings.gradient_descent.learning_rate = 1e7;
    opt->settings.adam.learning_rate = 1e3;
    
    opt->settings.max_iterations = 200;
    opt->settings.residual_tolerance_rel = 1e-4;
    opt->settings.residual_tolerance_abs = 0.0;
    
    opt->suppress_inner_newton_output = true;

    //// These are not needed here, but they work
    opt->settings.restart_when_non_descent = false;
    opt->settings.enable_armijo_backtracking = false;

    //// Callbacks
    opt->callbacks->add_before_energy_evaluation(
        [&]() {
            // std::cout << fmt::format(" density: {:.2f} | ", data.density);
        }
    );

    // Init inverse problem
    const double density_original = data.density;
    data.density = data.density_initial_guess;
    data.x = data.X;  // Reset to rest for clean Newton convergence

    // Test gradient
    context->output->set_enabled(false);
    const auto value_result = adjoint->run_forward_and_evaluate_L();
    const auto gradient_result = adjoint->evaluate_dL_dp_no_solve();
    const auto fd_result = adjoint->evaluate_dL_dp_finite_differences(1e-3);
    context->output->set_enabled(true);
    if (!value_result.success || !gradient_result.success || !fd_result.success) {
        throw std::runtime_error("Density-identification adjoint evaluation failed.");
    }
    const Eigen::VectorXd& dL = gradient_result.gradient;
    const Eigen::VectorXd& dL_fd = fd_result.gradient;
    const double error = std::abs((dL_fd[0] - dL[0])/dL[0]);

    std::cout << std::endl;
    std::cout << fmt::format("dL:    {:.3e}", dL[0]) << std::endl;
    std::cout << fmt::format("dL_fd: {:.3e}", dL_fd[0]) << std::endl;
    std::cout << fmt::format("error: {:.3e}", error) << std::endl;
    if (error > 1e-5) {
        std::cout << "Gradient test failed!" << std::endl;
        exit(-1);
    }

    // Solve
    {
        const double t0 = omp_get_wtime();
        SolverReturn ret = opt->solve();
        const double t1 = omp_get_wtime();
        opt->print_summary();
        std::cout << "Runtime: " << t1 - t0 << " s." << std::endl;
    }

    // Output
    std::cout << "\nParam: density" << std::endl;
    std::cout << fmt::format("Target:  {:.2f}", density_original) << std::endl;
    std::cout << fmt::format("Initial: {:.2f}", data.density_initial_guess) << std::endl;
    std::cout << fmt::format("Found:   {:.2f}", data.density) << std::endl;
    std::cout << fmt::format("Error:   {:.2e}", (data.density - density_original)/density_original) << std::endl;

    //// VTK Deformed
    vtk_file.set_points_from_twice_indexable(data.x);
    vtk_file.write(fmt::format("{}/cantilever_id_opt_sim_{}.vtk", output_folder, get_name(mesh.element_type)));
}

void cantilever_density_id_comparison()
{
    std::cout << "\n================ cantilever_density_id_comparison() ================" << std::endl;

    cantilever_density_id(FEM_Element::Tet4);
    cantilever_density_id(FEM_Element::Hex8);
}

// =============================================================================
//  2.  Muscle (active spring) control
// =============================================================================
void cantilever_muscle_control(FEM_Element fem_element_type)
{
    std::cout << "\n\n==============================================================================" << std::endl;
    std::cout << "Muscle control" << std::endl;
    std::cout << "==============================================================================" << std::endl;

    // ----------------------------------------------------------------------
    // Data
    // ----------------------------------------------------------------------
    struct Data
    {
        // Forward
        double youngs_modulus = 1e6;
        double poissons_ratio = 0.40;
        double bc_stiffness   = 1e9;
        double k_muscle       = 1e6;

        double length = 1.0;
        double width = 0.2;
        int n_subdivitions = 2; // short axis

        std::vector<Eigen::Vector3d> x;           // Current positions
        std::vector<Eigen::Vector3d> X;           // Rest positions
        LabelledConnectivity<1> bc_conn {{"vertex_idx"}};
        std::vector<double> muscle_strain;

        // Inverse
        LabelledConnectivity<1> tip_conn {{"vertex_idx"}};
        Eigen::Vector3d x_tip_target = {0.0, 0.5, 0.5};
        double tip_target_weight = 1.0;
        bool run_sequence = false;
    } data;

    // ----------------------------------------------------------------------
    // Forward problem
    // ----------------------------------------------------------------------

    // Mesh
    double elem_size = data.width / (double)data.n_subdivitions;
    int n_elems_length = std::round(data.length / elem_size);
    Eigen::Vector3d size = {data.length, data.width, data.width};
    std::array<int, 3> elements_per_axis = {n_elems_length, data.n_subdivitions, data.n_subdivitions};
    FEMMesh<double> mesh = generate_cuboid_mesh(fem_element_type, size, elements_per_axis);
    const int n_verts = (int)mesh.vertices.size();

    // Print
    std::cout << "> Element type: " << get_name(mesh.element_type) << std::endl;
    std::cout << "> Number of elements: " << mesh.n_elements << std::endl;
    std::cout << "> Number of vertices: " << n_verts << std::endl;

    // Initialize domain
    data.x = mesh.vertices;
    data.X = mesh.vertices;

    //// Assign each vertex a fiber ID based on its (y,z) position; all vertices on the same x-aligned line share one strain param
    std::map<std::pair<int,int>, int> fiber_map;
    std::vector<int> fiber_id_per_vertex(n_verts);
    int n_fibers = 0;
    for (int v = 0; v < n_verts; ++v) {
        int ky = (int)std::round(mesh.vertices[v].y() * 1e6);
        int kz = (int)std::round(mesh.vertices[v].z() * 1e6);
        auto [it, inserted] = fiber_map.emplace(std::make_pair(ky, kz), n_fibers);
        if (inserted) ++n_fibers;
        fiber_id_per_vertex[v] = it->second;
    }
    std::cout << "> Muscle fibers: " << n_fibers << std::endl;

    //// Extended connectivity: [element vertices | matching fiber IDs].
    const int nodes_per_element = mesh.connectivity_stride;
    std::vector<int32_t> muscle_conn;
    muscle_conn.reserve(mesh.n_elements * 2 * nodes_per_element);
    for (int e = 0; e < mesh.n_elements; ++e) {
        const int base = e * mesh.connectivity_stride;
        for (int j = 0; j < nodes_per_element; ++j)
            muscle_conn.push_back(mesh.connectivity[base + j]);
        for (int j = 0; j < nodes_per_element; ++j)
            muscle_conn.push_back(fiber_id_per_vertex[mesh.connectivity[base + j]]);
    }

    //// Helper: expand fiber strains to per-vertex (for VTK output)
    auto get_vertex_strains = [&]() {
        std::vector<double> vs(n_verts);
        for (int v = 0; v < n_verts; ++v)
            vs[v] = data.muscle_strain[fiber_id_per_vertex[v]];
        return vs;
    };

    //// Zero initial muscle strain per fiber
    data.muscle_strain.resize(n_fibers, 0.0);

    //// Boundary conditions (clamp left face)
    const double x_min = -size.x() / 2.0 + 1e-6;
    for (int i = 0; i < n_verts; ++i)
        if (mesh.vertices[i].x() < x_min)
            data.bc_conn.push_back({i});

    //// Tip vertex (rightmost, closest to beam axis)
    const double x_max = size.x() / 2.0 - 1e-6;
    double best_dist = 1e30;
    int tip_vertex_idx = -1;
    for (int i = 0; i < n_verts; ++i) {
        if (mesh.vertices[i].x() < x_max) continue;
        double d = mesh.vertices[i].y() * mesh.vertices[i].y()
                 + mesh.vertices[i].z() * mesh.vertices[i].z();
        if (d < best_dist) { best_dist = d; tip_vertex_idx = i; }
    }
    data.tip_conn.data = {{ tip_vertex_idx }};

    // Define Global Potential
    spGlobalPotential G = GlobalPotential::create();

    // Combined passive + active elasticity: stable neo-hookean + GL strain penalty along X
    G->add_potential("elasticity", muscle_conn, 2 * nodes_per_element,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> xh = mws.make_vectors(data.x, elem.slice(0, nodes_per_element));
            std::vector<Vector> Xh = mws.make_vectors(data.X, elem.slice(0, nodes_per_element));
            std::vector<Scalar> eps_h = mws.make_scalars(
                data.muscle_strain, elem.slice(nodes_per_element, 2 * nodes_per_element));
            Scalar E  = mws.make_scalar(data.youngs_modulus);
            Scalar nu = mws.make_scalar(data.poissons_ratio);
            Scalar k  = mws.make_scalar(data.k_muscle);

            return fem_integrator(mws, mesh.element_type,
                [&](Scalar& w, Vector xi)
                {
                    Matrix Jx  = fem_jacobian(mesh.element_type, xh, xi);
                    Matrix JX  = fem_jacobian(mesh.element_type, Xh, xi);
                    Matrix F   = Jx * JX.inv();
                    Matrix C   = F.transpose() * F;
                    Scalar E_XX  = 0.5 * (C(0, 0) - 1.0);
                    Scalar eps_m = fem_interpolation(mesh.element_type, eps_h, xi);
                    Scalar dV  = JX.det() * w;
                    return (stable_neohookean_energy_density(F, E, nu) + 0.5 * k * (E_XX - eps_m).powN(2)) * dV;
                });
        }
    );

    // Boundary conditions energy
    G->add_potential("boundary_conditions", data.bc_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector x = mws.make_vector(data.x, elem[0]);
            Vector X = mws.make_vector(data.X, elem[0]);
            Scalar k = mws.make_scalar(data.bc_stiffness);
            return 0.5 * k * (x - X).squared_norm();
        }
    );

    // DoF declaration
    G->add_dof(data.x);

    // Context
    spContext context = Context::create();
    context->compilation_directory = symx::get_codegen_dir();
    std::string output_folder = std::string(SYMX_EXAMPLES_OUTPUT_DIR)
        + "/cantilever_muscle_" + get_name(mesh.element_type);
    std::filesystem::create_directories(output_folder);
    context->output->open_file(output_folder + "/cantilever_muscle.log");

    // Newton Solver
    auto newton = NewtonsMethod::create(G, context);

    // Settings
    newton->settings.projection_mode        = ProjectionToPD::ProjectOnDemand;
    newton->settings.residual_tolerance_abs = 1e-4;
    newton->settings.step_tolerance         = 1e-6;

    // Prepare connectivity for VTK output
    std::vector<int> output_conn = mesh.connectivity;
    prepare_for_export(output_conn, mesh.element_type);

    // Solve
    newton->solve();

    // Write VTKs
    vtkio::VTKFile vtk_file;
    vtk_file.set_cells_from_indexable(output_conn, mesh.connectivity_stride, fem_element_to_vtk_cell_type(mesh.element_type));

    //// Rest
    vtk_file.set_points_from_twice_indexable(data.X);
    vtk_file.set_point_data_from_indexable("muscle_strain", get_vertex_strains(), vtkio::AttributeType::Scalars);
    vtk_file.write(output_folder + "/cantilever_muscle_rest.vtk");

    //// Deformed (passive)
    vtk_file.set_points_from_twice_indexable(data.x);
    vtk_file.set_point_data_from_indexable("muscle_strain", get_vertex_strains(), vtkio::AttributeType::Scalars);
    vtk_file.write(output_folder + "/cantilever_muscle_passive.vtk");

    // Target marker: load it alongside the simulation and render it red.
    write_cube_mesh(output_folder + "/cantilever_muscle_target.vtk", data.x_tip_target, 0.04);


    // ----------------------------------------------------------------------
    // Inverse problem (single)
    // ----------------------------------------------------------------------

    // Global Loss Definition
    auto G_loss = GlobalPotential::create();

    // Tip position loss
    G_loss->add_potential("tip_distance", data.tip_conn,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector x_tip = mws.make_vector(data.x, elem[0]);
            Vector x_tgt = mws.make_vector(data.x_tip_target);
            Scalar w = mws.make_scalar(data.tip_target_weight);
            return 0.5 * w * (x_tip - x_tgt).squared_norm();
        });

    // Parameters: muscle rest lengths
    G_loss->add_dof(data.muscle_strain, "muscle_strain");

    // Define Adjoint
    LinearSolveSettings lss;
    lss.linear_solver = LinearSolver::DirectLLT;
    auto adjoint = AdjointNewton::create(newton, G_loss, lss);

    // Define Optimizer
    //// Settings
    auto opt = FirstOrderOptimizerAdjoint::create(adjoint, context);

    opt->settings.type = FirstOrderOptType::LBFGS;
    opt->settings.lbfgs.bootstrap_step_length = 0.1;
    opt->settings.gradient_descent.learning_rate = 1e0;
    opt->settings.adam.learning_rate = 1e-2;

    opt->settings.max_iterations = 300;
    // opt->settings.step_cap = 0.1;
    opt->settings.residual_tolerance_abs = 0.0;
    opt->settings.residual_tolerance_rel = 1e-4;

    opt->suppress_inner_newton_output = true;

    //// These are not needed here, but they work
    opt->settings.restart_when_non_descent = false;
    opt->settings.enable_armijo_backtracking = false;

    //// Callbacks
    opt->callbacks->add_before_energy_evaluation(
        [&]() {
            // std::cout << fmt::format(" L0[0]: {:.4f} | ", data.L0[0]);
        }
    );

    // Init inverse problem
    data.x  = data.X;

    // Test gradient
    context->output->set_enabled(false);
    const auto value_result = adjoint->run_forward_and_evaluate_L();
    const auto gradient_result = adjoint->evaluate_dL_dp_no_solve();
    const auto fd_result = adjoint->evaluate_dL_dp_finite_differences(1e-3);
    context->output->set_enabled(true);
    if (!value_result.success || !gradient_result.success || !fd_result.success) {
        throw std::runtime_error("Muscle-control adjoint evaluation failed.");
    }
    const Eigen::VectorXd& dL = gradient_result.gradient;
    const Eigen::VectorXd& dL_fd = fd_result.gradient;
    const double error = std::abs((dL_fd - dL).norm() / dL.norm());

    std::cout << std::endl;
    std::cout << fmt::format("dL[0]:    {:.3e}", dL[0]) << std::endl;
    std::cout << fmt::format("dL_fd[0]: {:.3e}", dL_fd[0]) << std::endl;
    std::cout << fmt::format("error:    {:.3e}", error) << std::endl;
    if (error > 1e-3) {
        std::cout << "Gradient test failed!" << std::endl;
        exit(-1);
    }

    // Solve
    {
        const double t0 = omp_get_wtime();
        SolverReturn ret = opt->solve();
        const double t1 = omp_get_wtime();
        opt->print_summary();
        std::cout << "Runtime: " << t1 - t0 << " s." << std::endl;
    }

    // Output
    const Eigen::Vector3d tip_final = data.x[data.tip_conn[0][0]];
    std::cout << fmt::format("Tip pos: ({:.4f}, {:.4f}, {:.4f})",
                             tip_final.x(),
                             tip_final.y(),
                             tip_final.z()) << std::endl;
    std::cout << fmt::format("Target:  ({:.4f},{:.4f},{:.4f})",
                             data.x_tip_target.x(),
                             data.x_tip_target.y(),
                             data.x_tip_target.z()) << std::endl;

    //// VTK Deformed (optimized)
    vtk_file.set_points_from_twice_indexable(data.x);
    vtk_file.set_point_data_from_indexable("muscle_strain", get_vertex_strains(), vtkio::AttributeType::Scalars);
    vtk_file.write(output_folder + "/cantilever_muscle_opt.vtk");


    // ----------------------------------------------------------------------
    // Inverse problem (sequence)
    // ----------------------------------------------------------------------
    if (!data.run_sequence) return;

    std::cout << "\n\n==============================================================================" << std::endl;
    std::cout << "Muscle control - sequence" << std::endl;
    std::cout << "==============================================================================" << std::endl;
    opt->settings.residual_tolerance_abs = 1e-5;
    opt->settings.lbfgs.bootstrap_step_length = 0.01;
    const Eigen::Vector3d trajectory_aabb_bottom = { 0.0, -0.5, -0.5 };
    const Eigen::Vector3d trajectory_aabb_top    = { 0.5,  0.5,  0.5 };

    // Generate a smooth 3D Lissajous trajectory (1:2:3 frequency ratios).
    const int n_frames = 360;
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<Eigen::Vector3d> targets;
    targets.reserve(n_frames);
    for (int i = 0; i < n_frames; ++i) {
        const double t = 2.0 * pi * i / (double)n_frames;
        targets.push_back({
            trajectory_aabb_bottom.x() + (trajectory_aabb_top.x() - trajectory_aabb_bottom.x()) * 0.5 * (1.0 + std::cos(t)),  // x
            trajectory_aabb_bottom.y() + (trajectory_aabb_top.y() - trajectory_aabb_bottom.y()) * 0.5 * (1.0 + std::sin(2.0 * t)),  // y
            trajectory_aabb_bottom.z() + (trajectory_aabb_top.z() - trajectory_aabb_bottom.z()) * 0.5 * (1.0 + std::sin(3.0 * t))   // z
        });
    }

    // Output subfolder
    std::string seq_folder = output_folder + "/seq";
    std::filesystem::create_directories(seq_folder);

    // Pre-write all target positions as small octahedral spheres so the
    // trajectory can be inspected in a viewer even if the sim loop hangs.
    {
        const double target_sphere_radius = 0.05;

        // Octahedron: 6 vertices at ±r along each axis, 8 triangular faces.
        // VTK_TRIANGLE = 5.
        const std::vector<int> sphere_conn = {
            0, 2, 4,  0, 4, 3,  0, 3, 5,  0, 5, 2,
            1, 4, 2,  1, 3, 4,  1, 5, 3,  1, 2, 5
        };
        vtkio::VTKFile target_vtk;
        target_vtk.set_cells_from_indexable(sphere_conn, 3, vtkio::CellType::Triangle);

        std::vector<Eigen::Vector3d> sphere_verts(6);
        std::cout << "Writing " << n_frames << " target VTKs..." << std::endl;
        for (int frame = 0; frame < n_frames; ++frame) {
            const Eigen::Vector3d& c = targets[frame];
            sphere_verts[0] = c + Eigen::Vector3d( target_sphere_radius, 0, 0);
            sphere_verts[1] = c + Eigen::Vector3d(-target_sphere_radius, 0, 0);
            sphere_verts[2] = c + Eigen::Vector3d(0,  target_sphere_radius, 0);
            sphere_verts[3] = c + Eigen::Vector3d(0, -target_sphere_radius, 0);
            sphere_verts[4] = c + Eigen::Vector3d(0, 0,  target_sphere_radius);
            sphere_verts[5] = c + Eigen::Vector3d(0, 0, -target_sphere_radius);
            target_vtk.set_points_from_twice_indexable(sphere_verts);
            target_vtk.write(fmt::format("{}/cantilever_target_{}.vtk", seq_folder, frame));
        }
        std::cout << "Target VTKs written." << std::endl;
    }

    // Reduce verbosity for the long loop ?
    context->output->set_enabled(true);

    // Reset state to rest for the first frame (warm-start carries forward
    // between frames after that)
    data.x = data.X;
    std::fill(data.muscle_strain.begin(), data.muscle_strain.end(), 0.0);

    std::cout << "Solving " << n_frames << " frames..." << std::endl;
    const double t_seq_start = omp_get_wtime();

    for (int frame = 0; frame < n_frames; ++frame) {
        data.x_tip_target = targets[frame];
        opt->solve();

        // Write per-frame VTK
        vtk_file.set_points_from_twice_indexable(data.x);
        vtk_file.set_point_data_from_indexable("muscle_strain", get_vertex_strains(), vtkio::AttributeType::Scalars);
        vtk_file.write(fmt::format("{}/cantilever_muscle_seq_{}.vtk", seq_folder, frame));

        // Progress report every 10 frames
        if ((frame + 1) % 10 == 0 || frame == 0) {
            const Eigen::Vector3d tip_pos = data.x[data.tip_conn[0][0]];
            const double elapsed = omp_get_wtime() - t_seq_start;
            std::cout << fmt::format("  [{:3d}/{:3d}] target ({:+.3f},{:+.3f},{:+.3f})  tip ({:+.3f},{:+.3f},{:+.3f})  {:.1f}s",
                frame + 1, n_frames,
                data.x_tip_target.x(), data.x_tip_target.y(), data.x_tip_target.z(),
                tip_pos.x(), tip_pos.y(), tip_pos.z(),
                elapsed) << std::endl;
        }
    }

    context->output->set_enabled(true);
    const double t_seq_total = omp_get_wtime() - t_seq_start;
    std::cout << fmt::format("Sequence done: {} frames in {:.1f}s ({:.2f}s/frame)",
        n_frames, t_seq_total, t_seq_total / n_frames) << std::endl;
    std::cout << "Output: " << seq_folder << std::endl;
}

void cantilever_muscle_control_comparison()
{
    std::cout << "\n============= cantilever_muscle_control_comparison() =============" << std::endl;

    cantilever_muscle_control(FEM_Element::Tet4);
    cantilever_muscle_control(FEM_Element::Hex8);
}
