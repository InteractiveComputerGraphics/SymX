#pragma once

#include <limits>
#include <string>

#include "solver_types.h"

namespace symx
{
    struct SolverSettings
    {
        int max_iterations = std::numeric_limits<int>::max();
        int min_iterations = 0;
        int iteration_offset = 0;

        double residual_tolerance_abs = 1e-6;
        double residual_tolerance_rel = 0.0;
        double step_tolerance = 0.0;
        bool max_iterations_as_success = false;

        double step_cap = std::numeric_limits<double>::infinity();
        bool enable_armijo_backtracking = true;
        double line_search_armijo_beta = 1e-4;
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
            out += "\n" + p + "    iteration_offset: " + std::to_string(iteration_offset);
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

    struct LinearSolveSettings
    {
        LinearSolver linear_solver = LinearSolver::BDPCG;
        int cg_max_iterations = 10000;
        double cg_abs_tolerance = 1e-12;
        double cg_rel_tolerance = 1e-4;
        bool cg_stop_on_indefiniteness = true;
        double bailout_residual = 1e-10;

        std::string as_string(const std::string& prefix = "") const
        {
            std::string p = prefix;
            std::string out;
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

    struct NewtonSettings : public SolverSettings, public LinearSolveSettings
    {
        ProjectionToPD projection_mode = ProjectionToPD::ProjectedNewton;
        double projection_eps = 1e-10;
        bool project_to_pd_use_mirroring = false;
        int project_on_demand_countdown = 4;
        double ppn_tightening_factor = 0.5;
        double ppn_release_factor = 2.0;

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
            out += LinearSolveSettings::as_string(prefix);
            return out;
        }
    };

    enum class FirstOrderOptType { GradientDescent, Adam, VectorAdam, LBFGS };
    inline std::string to_string(FirstOrderOptType v)
    {
        switch (v)
        {
        case FirstOrderOptType::GradientDescent: return "GradientDescent"; break;
        case FirstOrderOptType::Adam:            return "Adam"; break;
        case FirstOrderOptType::VectorAdam:      return "VectorAdam"; break;
        case FirstOrderOptType::LBFGS:           return "LBFGS"; break;
        default:
            std::cout << "symx::FirstOrderOptType " << (int)v << " does not have a name. Exiting." << std::endl;
            exit(-1);
        }
    }
    struct FirstOrderSettings : public SolverSettings
    {
        struct GradientDescent { double learning_rate = 1e-3; };
        struct Adam { double learning_rate = 1e-3; double beta1 = 0.9; double beta2 = 0.999; double epsilon = 1e-8; };
        struct VectorAdam { double learning_rate = 1e-3; double beta1 = 0.9; double beta2 = 0.999; double epsilon = 1e-8; int vector_block_size = 3; };
        struct LBFGS {
            int history_size = 10;
            double bootstrap_step_length = 1.0;
            double curvature_threshold = 1e-12;
        };

        FirstOrderOptType type = FirstOrderOptType::LBFGS;
        GradientDescent gradient_descent;
        Adam adam;
        VectorAdam vector_adam;
        LBFGS lbfgs;
        bool restart_when_non_descent = true;

        std::string as_string(const std::string& prefix = "") const
        {
            std::string out = SolverSettings::as_string(prefix);
            out += "\n" + prefix + "First-order optimizer";
            out += "\n" + prefix + "    type: " + to_string(type);
            out += "\n" + prefix + "    restart_when_non_descent: " + to_string(restart_when_non_descent);
            out += "\n" + prefix + "GradientDescent";
            out += "\n" + prefix + "    learning_rate: " + to_string_sci(gradient_descent.learning_rate);
            out += "\n" + prefix + "Adam";
            out += "\n" + prefix + "    learning_rate: " + to_string_sci(adam.learning_rate);
            out += "\n" + prefix + "    beta1: " + to_string_fixed(adam.beta1);
            out += "\n" + prefix + "    beta2: " + to_string_fixed(adam.beta2);
            out += "\n" + prefix + "    epsilon: " + to_string_sci(adam.epsilon);
            out += "\n" + prefix + "VectorAdam";
            out += "\n" + prefix + "    learning_rate: " + to_string_sci(vector_adam.learning_rate);
            out += "\n" + prefix + "    beta1: " + to_string_fixed(vector_adam.beta1);
            out += "\n" + prefix + "    beta2: " + to_string_fixed(vector_adam.beta2);
            out += "\n" + prefix + "    epsilon: " + to_string_sci(vector_adam.epsilon);
            out += "\n" + prefix + "    vector_block_size: " + std::to_string(vector_adam.vector_block_size);
            out += "\n" + prefix + "LBFGS";
            out += "\n" + prefix + "    history_size: " + std::to_string(lbfgs.history_size);
            out += "\n" + prefix + "    bootstrap_step_length: " + to_string_sci(lbfgs.bootstrap_step_length);
            out += "\n" + prefix + "    curvature_threshold: " + to_string_sci(lbfgs.curvature_threshold);
            return out;
        }
    };
}
