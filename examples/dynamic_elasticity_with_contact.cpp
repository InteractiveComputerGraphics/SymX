#include "examples_main.h"

#include <fmt/format.h>
#include <vtkio>
#include <symx>

#include "elasticity_potentials.h"
#include "examples_utils.h"

using namespace symx;


void dynamic_elasticity_with_contact()
{
    std::cout << "\n=================== rubber_extrusion_comparison() ===================" << std::endl;

    struct Data
    {
        Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
        Eigen::Vector3d initial_velocity = Eigen::Vector3d(5.0, 0.0, 0.0);
        Eigen::Vector3d initial_translation = Eigen::Vector3d(0.0, 0.0, 0.45);
        double youngs_modulus = 5e4;
        double poissons_ratio = 0.4;
        double density = 1000.0;
        double damping = 0.1;
        double contact_stiffness = 1e6;
        double friction_coefficient = 1.0;
        double time_step_size = 1.0/30.0;

        double duration = 5.0;
        int fps = 30;

        std::vector<std::array<int, 4>> tets;
        std::vector<std::array<int, 1>> all_vertex_indices;
        std::vector<Eigen::Vector3d> X;
        std::vector<Eigen::Vector3d> x1;
        std::vector<Eigen::Vector3d> x0;
        std::vector<Eigen::Vector3d> v;
        std::vector<double> masses;

    } data;

    // Mesh and initial state -------------------------------------------------
    const std::string mesh_path = std::string(SYMX_EXAMPLES_RES_DIR) + "/bunny.vtk";
    vtkio::VTKFile vtk_file;
    vtk_file.read(mesh_path);
    vtk_file.get_points_to_twice_indexable(data.X);
    vtk_file.get_cells_to_twice_indexable(data.tets);
    const int n_vertices = (int)data.X.size();
    const int n_tets = (int)data.tets.size();
    std::cout << "Loaded mesh from " << mesh_path << " with " << n_vertices << " vertices and " << n_tets << " tets." << std::endl;

    // All vertices connectivity
    data.all_vertex_indices.resize(n_vertices);
    for (int i = 0; i < n_vertices; i++) {
        data.all_vertex_indices[i] = {i};
    }

    // Initial conditions
    for (auto& p : data.X) {
        p += data.initial_translation;
    }
    data.x0 = data.X;
    data.x1 = data.X;
    data.v.assign(n_vertices, data.initial_velocity);

    data.masses.assign(n_vertices, 0.0);
    for (const auto& tet : data.tets) {
        const Eigen::Vector3d& X0 = data.X[tet[0]];
        const Eigen::Vector3d& X1 = data.X[tet[1]];
        const Eigen::Vector3d& X2 = data.X[tet[2]];
        const Eigen::Vector3d& X3 = data.X[tet[3]];
        const double volume = std::abs((X1 - X0).dot((X2 - X0).cross(X3 - X0))) / 6.0;
        const double lumped = volume * data.density / 4.0;
        data.masses[tet[0]] += lumped;
        data.masses[tet[1]] += lumped;
        data.masses[tet[2]] += lumped;
        data.masses[tet[3]] += lumped;
    }

    // Global potential -------------------------------------------------------
    spGlobalPotential G = GlobalPotential::create();

    // Backward Euler inertial potential
    G->add_potential("be_inertia", data.all_vertex_indices,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector x1 = mws.make_vector(data.x1, elem[0]);
            Vector x0 = mws.make_vector(data.x0, elem[0]);
            Vector v0 = mws.make_vector(data.v, elem[0]);
            Scalar m = mws.make_scalar(data.masses, elem[0]);
            Vector g = mws.make_vector(data.gravity);
            Scalar damping = mws.make_scalar(data.damping);
            Scalar dt = mws.make_scalar(data.time_step_size);

            Vector xhat = x0 + dt*v0 + dt*dt*g;
            Vector dev = x1 - xhat;
            Vector dev2 = x1 - x0;
            Scalar P = 0.5*m*dev.squared_norm()/dt.powN(2) + 0.5*m*damping*dev2.squared_norm()/dt;
            return P;
        }
    );

    // Stable Neo-Hookean elasticity
    G->add_potential("stable_neo_hookean_tet4", data.tets,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            std::vector<Vector> x1 = mws.make_vectors(data.x1, elem);
            std::vector<Vector> X = mws.make_vectors(data.X, elem);
            Scalar E = mws.make_scalar(data.youngs_modulus);
            Scalar nu = mws.make_scalar(data.poissons_ratio);

            Scalar P = fem_integrator(mws, FEM_Element::Tet4,
                [&](Scalar& w, Vector xi)
                {
                    Matrix Jx = fem_jacobian(FEM_Element::Tet4, x1, xi);
                    Matrix JX = fem_jacobian(FEM_Element::Tet4, X, xi);
                    Matrix F = Jx * JX.inv();
                    return stable_neohookean_energy_density(F, E, nu) * JX.det() * w;
                }
            );
            return P;
        }
    );

    // Cubic contact penalty (ground plane z = 0)
    G->add_potential("ground_contact", data.all_vertex_indices,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector x1 = mws.make_vector(data.x1, elem[0]);
            Scalar k = mws.make_scalar(data.contact_stiffness);

            Scalar penetration = -x1[2];
            Scalar P = k * penetration.powN(3) / 3.0;
            Scalar cond = penetration > 0.0;
            return std::make_pair(P, cond);
        }
    );

    // Coulomb-style friction with lagged pressure, no branching on the slip
    G->add_potential("ground_friction", data.all_vertex_indices,
        [&](MappedWorkspace<double>& mws, Element& elem)
        {
            Vector x1 = mws.make_vector(data.x1, elem[0]);
            Vector x0 = mws.make_vector(data.x0, elem[0]);
            Scalar dt = mws.make_scalar(data.time_step_size);
            Scalar mu = mws.make_scalar(data.friction_coefficient);
            Scalar k = mws.make_scalar(data.contact_stiffness);

            Scalar penetration_lag = -x0[2];
            Scalar pres_lag = k * penetration_lag.powN(2);

            Vector v = (x1 - x0) / dt;
            v[2] *= 0.0;  // Tangential velocity
            Vector xt = v * dt;
            Scalar x_slide = xt.stable_norm(1e-8);

            // Viscous-style friction scaled by lagged normal pressure
            Scalar P = mu * pres_lag * x_slide;
            Scalar cond = x0[2] < 0.0;
            return std::make_pair(P, cond);
        }
    );

    // Degrees of freedom
    G->add_dof(data.x1);

    // Create Newton's Method ---------------------------------------------------------------
    spContext context = Context::create();
    context->compilation_directory = symx::get_codegen_dir();
    NewtonsMethod newton(G, context);
    newton.settings.projection_mode = ProjectionToPD::Progressive;
    newton.settings.step_tolerance = 1e-3;
    newton.settings.residual_tolerance_abs = 1e-6;

    // Time stepping --------------------------------------------------------
    std::filesystem::create_directories(std::string(SYMX_EXAMPLES_OUTPUT_DIR) + "/dynamic_contact");
    auto write_frame = [&](int frame_idx, const std::vector<Eigen::Vector3d>& positions)
    {
        vtkio::VTKFile vtk_file;
        vtk_file.set_points_from_twice_indexable(positions);
        vtk_file.set_cells_from_twice_indexable(data.tets, vtkio::CellType::Tetra);
        vtk_file.write(fmt::format("{}/dynamic_contact/frame_{:04d}.vtk", std::string(SYMX_EXAMPLES_OUTPUT_DIR), frame_idx));
    };

    // Main time stepping loop
    write_frame(0, data.x0);
    const double t0_total = omp_get_wtime();
    double time = 0.0;
    int next_frame = 1;
    for (; time < data.duration; time += data.time_step_size) {

        // Solve the time step
        const double t0 = omp_get_wtime();
        SolverReturn ret = newton.solve();
        const double t1 = omp_get_wtime();
        std::cout << fmt::format("\n[{}.vtk] time {:.3f} s | solve {:.3f} ms\n", next_frame, time, 1000.0 * (t1 - t0));
        if (ret != SolverReturn::Successful) {
            std::cout << "Terminating dynamic simulation early.\n";
            break;
        }

        // Update for next time step
        for (size_t i = 0; i < data.x0.size(); ++i) {
            data.v[i] = (data.x1[i] - data.x0[i]) / data.time_step_size;
            data.x0[i] = data.x1[i];
        }

        // Write output frame
        while ((double)next_frame / (double)data.fps <= time) {
            write_frame(next_frame, data.x0);
            next_frame++;
        }
    }
    const double t1_total = omp_get_wtime();
    const double total_time = t1_total - t0_total;
    newton.print_summary(total_time);
}
