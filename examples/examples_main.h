#pragma once

#include <symx>

// hello_world.cpp
void hello_world();

// triangle_mesh_area.cpp
void triangle_mesh_area();

// xpbd_cloth.cpp
template<typename INPUT_FLOAT, typename COMPILED_FLOAT>
void xpbd_cloth_sim(int n_threads = -1, int res = 128, bool enable_output = true);
void xpbd_cloth_comparison(int n_threads = -1, int res = 128, bool enable_output = true);

// NeoHookean.cpp
void NeoHookean_print();
template<typename FLOAT, typename COMPILATION_FLOAT>
void NeoHookean_eval_microbenchmark(int n_threads = -1);
void NeoHookean_eval_comparison();
template<typename COMPILATION_FLOAT, bool PROJECT_TO_PD, bool USE_EIGEN>
void NeoHookean_assembly_microbenchmark(int size, int n_threads);
void NeoHookean_assembly_comparison();

// rubber_extrusion.cpp
void rubber_extrusion(symx::FEM_Element fem_element_type, int refinement_level, double poissons_ratio, double extrusion_factor);
void rubber_extrusion_comparison(int refinement_level = 10, double poissons_ratio = 0.49, double extrusion_factor = 5.0);

// dynamic_elasticity_with_contact.cpp
void dynamic_elasticity_with_contact();

// cantilever_adjoint.cpp
void cantilever_density_id(symx::FEM_Element fem_element_type = symx::FEM_Element::Hex8);
void cantilever_muscle_control(symx::FEM_Element fem_element_type = symx::FEM_Element::Hex8);
void cantilever_density_id_comparison();
void cantilever_muscle_control_comparison();
