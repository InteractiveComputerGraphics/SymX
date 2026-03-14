#include "examples_main.h"


int main()
{
    bool run_all = false;

    bool run_hello_world                     = true;
    bool run_NeoHookean_print                = false;
    bool run_NeoHookean_eval_comparison      = false;
    bool run_triangle_mesh_area              = false;
    bool run_NeoHookean_assembly_comparison  = false;
    bool run_xpbd_cloth_comparison           = false;
    bool run_rubber_extrusion_comparison     = false;
    bool run_dynamic_elasticity_with_contact = false;


    /*
        hello_world()

        Minimal SymX example where a small function is defined and compiled, 
        evaluated, and the result printed.
    */
    if (run_all || run_hello_world) {
        hello_world();
    }

    /*
        NeoHookean_print()

        The second-order derivative (Hessian) of the NeoHookean energy of a
        linear tet element is derived using SymX. A function is compiled,
        evaluated and the results printed.
    */
    if (run_all || run_NeoHookean_print) {
       NeoHookean_print();
    }

    /*
        NeoHookean_eval_comparison()

        The Hessian of the NeoHookean energy for a linear tet element is
        benchmarked for different compilation times and number of threads.
        Only if the compiled objects have changed or are not found,
        differentiation and compilation is carried out.
    */
    if (run_all || run_NeoHookean_eval_comparison) {
       NeoHookean_eval_comparison();
    }

    /*
        triangle_mesh_area()

        The area of all the triangles of a mesh is added using SymX.
        Showcases `MappedWorkspace` and `CompiledInLoop`.
    */
    if (run_all || run_triangle_mesh_area) {
        triangle_mesh_area();
    }

    /*
        xpbd_cloth_comparison()

        A dynamic cloth simulation using XPBD of a piece of cloth falling
        on a sphere. Models strain and bending, plus BC and contact.
        It compares:
            - sequential vs parallel
            - double, float, __m256d, __m256
        The simulation can run with different resolutions.

        Showcases the use of `CompiledInLoop` in a non-FEM setting.
    */
    if (run_all || run_xpbd_cloth_comparison) {
        xpbd_cloth_comparison();
    }

    /*
        NeoHookean_assembly_comparison()

        The global potential, gradient and Hessian of the NeoHookean energy
        for a mesh of linear tets is computed using SymX. The benchmark
        compares 
            - sequential vs parallel
            - double vs __m256d
            - Eigen's vs SymX's assembler
        Projection to PD can be optionally enabled.
    */
    if (run_all || run_NeoHookean_assembly_comparison) {
        NeoHookean_assembly_comparison();
    }

    /*
        rubber_extrusion_comparison()

        Quasistatic simulation of a block of incompressible rubber being
        stretched. Different types of FEM element types are used.
        The simulation can run with different resolutions.
        
        Showcases generic FEM integration in SymX, callbacks, output 
        and logs.
    */
   if (run_all || run_rubber_extrusion_comparison) {
       rubber_extrusion_comparison();
    }
    
    /*
        dynamic_elasticity_with_contact()

        An elastic bunny is thrown in the air, hits the floor, and rolls.
        Showcases dynamic topology potentials (IPC) and time stepping.
    */
    if (run_all || run_dynamic_elasticity_with_contact) {
        dynamic_elasticity_with_contact();
    }
}
