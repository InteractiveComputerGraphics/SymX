#pragma once
#include <string>
#include <limits>

#include "solver_types.h"

namespace symx
{
    // ================================================================
    // SolverSettings
    //
    // Base settings shared by all iterative solvers.
    // ================================================================
    struct SolverSettings
    {
        // Iteration limits
        int max_iterations = std::numeric_limits<int>::max();
        int min_iterations = 0;

        // Convergence criteria
        double residual_tolerance_abs = 1e-6;
        double residual_tolerance_rel = 0.0;
        double step_tolerance = 0.0;
        bool max_iterations_as_success = false;          // Treat hitting max_iterations as convergence

        // Line search
        double step_cap = std::numeric_limits<double>::infinity();  // Clamp step norm to this value
        bool enable_armijo_backtracking = true;
        double line_search_armijo_beta = 1e-4;           // Sufficient decrease parameter
        int max_backtracking_armijo_iterations = 20;
        int max_backtracking_invalid_state_iterations = 8;
        bool print_line_search_upon_failure = false;

        std::string as_string(const std::string& prefix = "") const
        {
            std::string p = prefix;
            std::string out;
            out += "\n" + p + "Iteration limits";
            out += "\n" + p + "    max_iterations: " + std::to_string(max_iterations);
            out += "\n" + p + "    min_iterations: " + std::to_string(min_iterations);
            out += "\n" + p + "Convergence";
            out += "\n" + p + "    residual_tolerance_abs: " + to_string_sci(residual_tolerance_abs);
            out += "\n" + p + "    residual_tolerance_rel: " + to_string_sci(residual_tolerance_rel);
            out += "\n" + p + "    step_tolerance: " + to_string_sci(step_tolerance);
            out += "\n" + p + "    max_iterations_as_success: " + to_string(max_iterations_as_success);
            out += "\n" + p + "Line search";
            out += "\n" + p + "    step_cap: " + to_string_sci(step_cap);
            out += "\n" + p + "    enable_armijo_backtracking: " + to_string(enable_armijo_backtracking);
            out += "\n" + p + "    armijo_beta: " + to_string_sci(line_search_armijo_beta);
            out += "\n" + p + "    max_armijo_iterations: " + std::to_string(max_backtracking_armijo_iterations);
            out += "\n" + p + "    max_invalid_state_iterations: " + std::to_string(max_backtracking_invalid_state_iterations);
            out += "\n" + p + "    print_line_search_upon_failure: " + to_string(print_line_search_upon_failure);
            return out;
        }
    };

    // ================================================================
    // NewtonSettings
    //
    // Settings for Newton's method, extending SolverSettings with
    // Hessian projection and linear solver options.
    // ================================================================
    struct NewtonSettings : public SolverSettings
    {
        // Hessian projection to PD
        ProjectionToPD projection_mode = ProjectionToPD::ProjectedNewton;  // Safe default
        double projection_eps = 1e-10;               // Min eigenvalue after projection
        bool project_to_pd_use_mirroring = false;    // Mirror instead of clamp for negative eigenvalues

        // Project-On-Demand
        int project_on_demand_countdown = 4;         // Iterations to project after failure

        // Progressive Projected Newton (PPN)
        double ppn_tightening_factor = 0.5;          // Tighten threshold on non-descending step
        double ppn_release_factor = 2.0;             // Relax threshold after successful step

        // Linear solver
        LinearSolver linear_solver = LinearSolver::BDPCG;
        int cg_max_iterations = 10000;
        double cg_abs_tolerance = 1e-12;
        double cg_rel_tolerance = 1e-4;
        bool cg_stop_on_indefiniteness = true;
        double bailout_residual = 1e-10;             // Skip step if residual below this (CG stability)

        std::string as_string(const std::string& prefix = "") const
        {
            std::string p = prefix;
            std::string out;
            out += SolverSettings::as_string(prefix);
            out += "\n" + p + "Hessian projection";
            out += "\n" + p + "    projection_mode: " + to_string(projection_mode);
            out += "\n" + p + "    projection_eps: " + to_string_sci(projection_eps);
            out += "\n" + p + "    use_mirroring: " + to_string(project_to_pd_use_mirroring);
            out += "\n" + p + "    on_demand_countdown: " + std::to_string(project_on_demand_countdown);
            out += "\n" + p + "    ppn_tightening_factor: " + to_string_fixed(ppn_tightening_factor);
            out += "\n" + p + "    ppn_release_factor: " + to_string_fixed(ppn_release_factor);
            out += "\n" + p + "Linear solver";
            out += "\n" + p + "    solver: " + to_string(linear_solver);
            out += "\n" + p + "    cg_max_iterations: " + std::to_string(cg_max_iterations);
            out += "\n" + p + "    cg_abs_tolerance: " + to_string_sci(cg_abs_tolerance);
            out += "\n" + p + "    cg_rel_tolerance: " + to_string_sci(cg_rel_tolerance);
            out += "\n" + p + "    cg_stop_on_indefiniteness: " + to_string(cg_stop_on_indefiniteness);
            out += "\n" + p + "    bailout_residual: " + to_string_sci(bailout_residual);
            return out;
        }
    };
}
