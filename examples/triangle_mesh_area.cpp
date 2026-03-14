#include "examples_main.h"

#include <symx>
using namespace symx;


void triangle_mesh_area()
{
    std::cout << "\n=================== triangle_mesh_area() ===================" << std::endl;

    struct Data
    {
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::array<int, 3>> triangles;
    } data;

    auto mesh = generate_triangle_grid(Eigen::Vector2d{0.5, 0.5}, {10, 10});
    data.vertices = mesh.vertices;
    data.triangles = mesh.triangles;

    // Create MappedWorkspace
    auto [mws, element] = MappedWorkspace<double>::create(data.triangles);

    // Spawn symbols from data
    std::vector<Vector> v = mws->make_vectors(data.vertices, element);

    // Expression
    Vector cross = (v[1] - v[0]).cross3(v[2] - v[0]);
    Scalar area = 0.5*cross.norm();

    // Compilation
    const std::string codegen_dir = symx::get_codegen_dir();
    auto loop = CompiledInLoop<double>::create(mws, {area}, "tri_area", codegen_dir);

    // Evaluation
    double total_area = 0.0;
    const int n_threads = 1;
    loop->run(n_threads, 
        [&](const View<double> out, const int32_t elem_idx, const int32_t thread_id, const View<int32_t> elem_conn)
        {
            total_area += out[0];
        }
    );

    // Print
    std::cout << "Total triangle mesh area: " << total_area << std::endl;
    loop->get_info().print();
}
