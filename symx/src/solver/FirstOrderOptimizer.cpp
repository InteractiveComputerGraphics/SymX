#include "FirstOrderOptimizer.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <fmt/format.h>

using namespace symx;

FirstOrderOptimizer::FirstOrderOptimizer(
    spGlobalPotential global_potential,
    spContext context,
    spSolverCallbacks callbacks)
    : global_potential(global_potential), context(context), callbacks(callbacks)
{
    if (!this->global_potential || !this->context) {
        throw std::invalid_argument("FirstOrderOptimizer requires a GlobalPotential and Context.");
    }
    this->output = context->output;
    this->logger = context->logger;
    if (!this->callbacks) {
        this->callbacks = std::make_shared<SolverCallbacks>(context);
    }
    this->compiled = std::make_shared<FirstOrderCompiledGlobal>(this->global_potential, this->context);
}

spFirstOrderOptimizer FirstOrderOptimizer::create(
    spGlobalPotential global_potential,
    spContext context,
    spSolverCallbacks callbacks)
{
    return std::make_shared<FirstOrderOptimizer>(global_potential, context, callbacks);
}

SolverReturn FirstOrderOptimizer::solve()
{
    const int n_dofs = this->global_potential->get_total_n_dofs();
    if (n_dofs <= 0) {
        this->output->print_with_new_line("FirstOrderOptimizer failure: No degrees of freedom.", Verbosity::Summary);
        return SolverReturn::InvalidInitialState;
    }

    this->stats = SolveStats();
    this->dofs.resize(n_dofs);
    this->dofs_before_ls.resize(n_dofs);
    this->dofs_ls.resize(n_dofs);
    this->gradient.resize(n_dofs);
    this->direction.resize(n_dofs);

    std::vector<int32_t> dof_set_sizes;
    dof_set_sizes.reserve(this->global_potential->get_n_dof_sets());
    for (int set_i = 0; set_i < this->global_potential->get_n_dof_sets(); ++set_i) {
        dof_set_sizes.push_back(this->global_potential->get_n_dofs(set_i));
    }
    this->algorithm.prepare(n_dofs, this->global_potential->get_dofs_offsets(), dof_set_sizes, this->settings);

    if (!this->callbacks->run_is_initial_state_valid()) {
        this->output->print_with_new_line("FirstOrderOptimizer failure: Invalid initial state.", Verbosity::Summary);
        return SolverReturn::InvalidInitialState;
    }

    SolverReturn result = SolverReturn::Running;
    double energy = 0.0;
    double initial_residual = std::numeric_limits<double>::max();
    int iteration = -1;
    while (result == SolverReturn::Running) {
        iteration++;

        this->callbacks->run_before_step();
        struct AfterStepGuard {
            spSolverCallbacks& callbacks;
            ~AfterStepGuard() { callbacks->run_after_step(); }
        } after_step{this->callbacks};

        if (iteration == this->settings.max_iterations) {
            result = this->settings.max_iterations_as_success
                ? SolverReturn::Successful : SolverReturn::TooManyIterations;
            break;
        }

        this->output->print_with_new_line(
            fmt::format("{:3d}. ", iteration + this->settings.iteration_offset), Verbosity::Summary);

        this->callbacks->run_before_energy_evaluation();
        this->compiled->evaluate_P__dP_du(energy, this->gradient);
        if (!std::isfinite(energy) || !this->gradient.allFinite()) {
            this->output->print_with_new_line(
                "FirstOrderOptimizer failure: Energy or gradient is NaN/Inf.", Verbosity::Summary);
            result = SolverReturn::InvalidInitialState;
            break;
        }

        const double residual = this->callbacks->compute_residual(this->gradient);
        this->logger->append("energy", energy);
        this->logger->append("residual_norm", residual);
        this->output->print(fmt::format("r0: {:.1e} | ", residual), Verbosity::Summary);
        if (iteration == 0) {
            initial_residual = residual;
        }
        if (iteration >= this->settings.min_iterations
            && (residual < this->settings.residual_tolerance_abs
                || (initial_residual > 0.0 && residual / initial_residual < this->settings.residual_tolerance_rel))) {
            this->output->print("(converged)", Verbosity::Summary);
            result = SolverReturn::Successful;
            break;
        }

        this->global_potential->get_dofs(this->dofs.data());
        const Eigen::VectorXd dofs_prev = this->dofs;
        const Eigen::VectorXd gradient_prev = this->gradient;

        this->direction = this->algorithm.compute_direction(this->gradient, this->settings);
        double gradient_dot_direction = this->gradient.dot(this->direction);
        if (gradient_dot_direction >= 0.0 && this->settings.restart_when_non_descent) {
            this->output->print("reset | ", Verbosity::Summary);
            this->algorithm.reset(n_dofs);
            this->direction = this->algorithm.default_direction(this->gradient, this->settings);
            gradient_dot_direction = this->gradient.dot(this->direction);
        }
        if (gradient_dot_direction >= 0.0) {
            this->output->print("non-descent step", Verbosity::Summary);
            result = SolverReturn::StepDoesNotDescend;
            break;
        }

        const double direction_max = this->direction.cwiseAbs().maxCoeff();
        this->logger->append("step_max_raw", direction_max);
        if (iteration >= this->settings.min_iterations && direction_max < this->settings.step_tolerance) {
            this->output->print("(step converged)", Verbosity::Summary);
            result = SolverReturn::Successful;
            break;
        }
        this->output->print(fmt::format("du_raw: {:.1e} | ", direction_max), Verbosity::Summary);

        result = this->_line_search_inplace(energy, gradient_dot_direction, direction_max);
        if (result != SolverReturn::Running) {
            this->output->print("Line search failure", Verbosity::Summary);
            break;
        }

        if (this->settings.type == FirstOrderOptType::LBFGS) {
            this->global_potential->get_dofs(this->dofs.data());
            double accepted_energy = 0.0;
            Eigen::VectorXd accepted_gradient;
            this->callbacks->run_before_energy_evaluation();
            this->compiled->evaluate_P__dP_du(accepted_energy, accepted_gradient);
            if (!this->algorithm.update_lbfgs_history(
                    this->dofs - dofs_prev, accepted_gradient - gradient_prev, this->settings)) {
                this->stats.lbfgs_history_skips++;
                this->logger->add_and_append("lbfgs_history_skips", 1);
            } else {
                this->logger->add_and_append("lbfgs_history_skips", 0);
            }
        }

        if (iteration >= this->settings.min_iterations && this->callbacks->run_is_converged()) {
            result = SolverReturn::Successful;
        }
    }

    if (result == SolverReturn::Successful && !this->callbacks->run_is_converged_state_valid()) {
        this->output->print_with_new_line("FirstOrderOptimizer failure: Invalid converged state.", Verbosity::Summary);
        result = SolverReturn::InvalidConvergedState;
    }

    this->stats.first_order_iterations = iteration;
    this->logger->add_and_append("first_order_iterations", this->stats.first_order_iterations);
    return result;
}

SolverReturn FirstOrderOptimizer::_line_search_inplace(
    double energy,
    double gradient_dot_direction,
    double direction_max)
{
    this->global_potential->get_dofs(this->dofs_before_ls.data());
    auto apply_direction = [this](double alpha) {
        this->dofs_ls = this->dofs_before_ls + alpha * this->direction;
        this->global_potential->set_dofs(this->dofs_ls.data());
    };
    auto restore_accepted_state = [this]() {
        this->global_potential->set_dofs(this->dofs_before_ls.data());
    };

    double direction_scale = 1.0;
    if (direction_max > this->settings.step_cap) {
        direction_scale = this->settings.step_cap / direction_max;
        this->direction *= direction_scale;
        direction_max = this->settings.step_cap;
        this->stats.ls_cap_iterations++;
        this->logger->add_and_append("ls_cap", 1);
        this->output->print(fmt::format("cap {:.1e} | ", direction_max), Verbosity::Summary);
    } else {
        this->logger->add_and_append("ls_cap", 0);
    }

    double alpha = 1.0;
    const double max_step = this->callbacks->run_max_allowed_step();
    if (max_step < 1.0) {
        alpha = max_step;
        this->stats.ls_max_iterations++;
        this->logger->add_and_append("ls_max", 1);
        this->output->print(fmt::format("ls max {:.1e} | ", alpha), Verbosity::Summary);
    } else {
        this->logger->add_and_append("ls_max", 0);
    }

    apply_direction(alpha);
    int invalid_iterations = 0;
    for (; invalid_iterations < this->settings.max_backtracking_invalid_state_iterations; ++invalid_iterations) {
        if (this->callbacks->run_is_intermediate_state_valid()) {
            break;
        }
        alpha *= 0.5;
        apply_direction(alpha);
    }
    this->stats.ls_inv_iterations += invalid_iterations;
    this->logger->add_and_append("ls_inv", invalid_iterations);
    if (invalid_iterations > 0) {
        this->output->print(fmt::format("ls inv {:d} | ", invalid_iterations), Verbosity::Summary);
    }
    if (invalid_iterations == this->settings.max_backtracking_invalid_state_iterations) {
        this->callbacks->run_on_intermediate_state_invalid();
        restore_accepted_state();
        return SolverReturn::TooManyInvalidIntermediateIterations;
    }

    const double slope = gradient_dot_direction * direction_scale;
    int armijo_iterations = 0;
    if (this->settings.enable_armijo_backtracking) {
        for (; armijo_iterations < this->settings.max_backtracking_armijo_iterations; ++armijo_iterations) {
            double trial_energy = 0.0;
            this->callbacks->run_before_energy_evaluation();
            this->compiled->evaluate_P(trial_energy);
            const double armijo_bound = energy + this->settings.line_search_armijo_beta * alpha * slope;
            if (std::isfinite(trial_energy) && trial_energy <= armijo_bound) {
                break;
            }
            alpha *= 0.5;
            apply_direction(alpha);
        }
    }
    this->stats.ls_bt_iterations += armijo_iterations;
    this->logger->add_and_append("ls_bt", armijo_iterations);
    if (armijo_iterations > 0) {
        this->output->print(fmt::format("ls bt {:d} | ", armijo_iterations), Verbosity::Summary);
    }
    if (armijo_iterations == this->settings.max_backtracking_armijo_iterations) {
        this->callbacks->run_on_armijo_fail();
        restore_accepted_state();
        return SolverReturn::TooManyArmijoIterations;
    }

    this->logger->append("line_search_scale", alpha);
    this->logger->append("step_max_accepted", alpha * this->direction.cwiseAbs().maxCoeff());
    return SolverReturn::Running;
}

void FirstOrderOptimizer::print_summary(double total_time) const
{
    auto* out = this->output.get();
    out->print_with_new_line("");
    out->print_with_new_line(fmt::format("  {:<28} {:>10} {:>8} {:>8} {:>8}", "Solve", "Total", "Avg", "Min", "Max"));
    out->print_with_new_line(fmt::format("  {}", std::string(66, '-')));
    for (const auto& [label, key] : std::vector<std::pair<std::string, std::string>>{
            {"First-order iterations", "first_order_iterations"},
            {"LS cap", "ls_cap"},
            {"LS max step", "ls_max"},
            {"LS invalid state", "ls_inv"},
            {"LS Armijo", "ls_bt"},
            {"L-BFGS history skips", "lbfgs_history_skips"},
        }) {
        const auto summary = this->logger->get_stats(key);
        out->print_with_new_line(fmt::format("  {:<28} {:>10} {:>8.1f} {:>8} {:>8}",
            label, (long long)summary.total, summary.avg, (int)summary.min, (int)summary.max));
    }

    if (total_time < 0.0) {
        total_time = 0.0;
        for (const auto& label : this->logger->get_timer_labels()) {
            total_time += this->logger->get_timer_total(label);
        }
    }
    out->print_with_new_line(fmt::format("  {}", std::string(66, '-')));
    out->print_with_new_line(fmt::format("  Runtime: {:.6f} s", total_time));
    out->print_new_line();
}
