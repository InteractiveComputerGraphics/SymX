#include "examples_main.h"

#include <fmt/format.h>
#include <vtkio>
#include <symx>

#include "elasticity_potentials.h"
#include "examples_utils.h"

using namespace symx;


void rubber_extrusion(FEM_Element fem_element_type, int refinement_level, double poissons_ratio, double extrusion_factor)
{
    struct Data
    {
        double youngs_modulus = 1e6;
        double poissons_ratio = 0.49;
        double bc_stiffness = 1e8;

        std::vector<Eigen::Vector3d> x; // Current positions
        std::vector<Eigen::Vector3d> X; // Rest positions
        LabelledConnectivity<2> bc_connectivity{{ "constraint_idx", "vertex_idx" }};
        std::vector<Eigen::Vector3d> bc_target_positions;
    } data;

    // Create a mesh
    Eigen::Vector3d size(0.65, 1.0, 1.0);
    std::array<int32_t, 3> elements_per_axis = {(int)std::round(extrusion_factor*(int)refinement_level), refinement_level, refinement_level};
    FEMMesh<double> mesh = generate_cuboid_mesh(fem_element_type, size, elements_per_axis);

    // Print
    std::cout << "\n\n==============================================================================" << std::endl;
    std::cout << "==============================================================================" << std::endl;
    std::cout << "> Element type: " << get_name(mesh.element_type) << std::endl;
    std::cout << "> Number of elements: " << mesh.n_elements << std::endl;
    std::cout << "> Number of vertices: " << mesh.vertices.size() << std::endl;

    // Initialize data
    data.x = mesh.vertices;
    data.X = mesh.vertices;
    data.poissons_ratio = poissons_ratio;

    // Set initial guess: affine expansion in extrusion (x) direction
    // Center of block in x is assumed to be 0 (as in mesh generation)
    for (size_t i = 0; i < data.x.size(); ++i) {
        data.x[i].x() = data.X[i].x() * extrusion_factor;
        // y and z remain unchanged
        data.x[i].y() = data.X[i].y();
        data.x[i].z() = data.X[i].z();
    }

    // Apply boundary conditions
    double extrusion_displacement = (extrusion_factor - 1.0) * size.x() / 2.0;
    for (int i = 0; i < mesh.vertices.size(); ++i) {
        const Eigen::Vector3d& v = mesh.vertices[i];
        if (v.x() < -size.x() / 2.0 + 1e-6) {  // Left face
            data.bc_connectivity.numbered_push_back({i});
            data.bc_target_positions.push_back(v - Eigen::Vector3d::UnitX() * extrusion_displacement);
        } 
        else if (v.x() > size.x() / 2.0 - 1e-6) {  // Right face
            data.bc_connectivity.numbered_push_back({i});
            data.bc_target_positions.push_back(v + Eigen::Vector3d::UnitX() * extrusion_displacement);
        }
    }

    // Define Global Potential
    spGlobalPotential G = GlobalPotential::create();

    // Stable Neo-Hookean material energy (generic on FEM element type)
	G->add_potential("stable_neo_hookean_" + get_name(mesh.element_type), mesh.connectivity, mesh.connectivity_stride,
		[&](MappedWorkspace<double>& mws, Element& elem)
		{
            // Create symbols from data
            std::vector<Vector> x = mws.make_vectors(data.x, elem);
            std::vector<Vector> X = mws.make_vectors(data.X, elem);
            Scalar E = mws.make_scalar(data.youngs_modulus);
            Scalar nu = mws.make_scalar(data.poissons_ratio);

            // Define potential
            Scalar P = fem_integrator(mws, mesh.element_type,
				[&](Scalar& w, Vector xi)
				{
					Matrix Jx = fem_jacobian(mesh.element_type, x, xi);
					Matrix JX = fem_jacobian(mesh.element_type, X, xi);
					Matrix F = Jx * JX.inv();
					return stable_neohookean_energy_density(F, E, nu) * JX.det() * w;
				}
            );
            return P;
		}
	);

    // Boundary conditions energy
    G->add_potential("boundary_conditions", data.bc_connectivity,
        [&](MappedWorkspace<double>& mws, Element& conn)
        {
            // Create symbols from data
            Vector x = mws.make_vector(data.x, conn["vertex_idx"]);
            Vector target = mws.make_vector(data.bc_target_positions, conn["constraint_idx"]);
            Scalar k = mws.make_scalar(data.bc_stiffness);

            // Define potential
            Scalar P = 0.5 * k * (x - target).squared_norm();
            return P;
        }
    );

    // DoF declaration
    G->add_dof(data.x);

    // Context
    spContext context = Context::create();
    context->compilation_directory = symx::get_codegen_dir();
    std::string output_folder = std::string(SYMX_EXAMPLES_OUTPUT_DIR) + "/rubber_extrusion";
    std::filesystem::create_directories(output_folder);
    context->output->open_file(fmt::format("{}/rubber_extrusion_{}.log", output_folder, get_name(mesh.element_type)));
    
    // Newton Solver
    NewtonsMethod newton(G, context);  // Compilation occurs here

    // Settings
    newton.settings.project_to_pd_use_mirroring = true;
    newton.settings.projection_mode = ProjectionToPD::ProjectedNewton;
    newton.settings.step_tolerance = 1e-3;
    
    // Prepare connectivity for VTK output
    std::vector<int> output_conn = mesh.connectivity;
    prepare_for_export(output_conn, mesh.element_type);
    
    // Callbacks: Use intermediate state check for output
    newton.callbacks->add_is_intermediate_state_valid(
        [&]()
        {
            // Write output as a VTK file (check with Paraview)
            vtkio::VTKFile vtk_file;
            vtk_file.set_points_from_twice_indexable(data.x);
            vtk_file.set_cells_from_indexable(output_conn, mesh.connectivity_stride, fem_element_to_vtk_cell_type(mesh.element_type));
            vtk_file.write(fmt::format("{}/rubber_extrusion_{}.vtk", output_folder, get_name(mesh.element_type)));

            return true;
        }
    );

    // Solve
    const double t0 = omp_get_wtime();
    SolverReturn ret = newton.solve();
    const double t1 = omp_get_wtime();
    const double runtime = t1 - t0;
    newton.print_summary(runtime);

    // Write yaml log
    context->logger->save_to_disk(fmt::format("{}/rubber_extrusion_{}.yaml", output_folder, get_name(mesh.element_type)));
}
void rubber_extrusion_comparison(int refinement_level, double poissons_ratio, double extrusion_factor)
{
    std::cout << "\n=================== rubber_extrusion_comparison() ===================" << std::endl;

    rubber_extrusion(FEM_Element::Tet4, refinement_level, poissons_ratio, extrusion_factor);
    rubber_extrusion(FEM_Element::Hex8, refinement_level, poissons_ratio, extrusion_factor);
    rubber_extrusion(FEM_Element::Tet10, refinement_level/2, poissons_ratio, extrusion_factor);
    rubber_extrusion(FEM_Element::Hex27, refinement_level/2, poissons_ratio, extrusion_factor);
}
