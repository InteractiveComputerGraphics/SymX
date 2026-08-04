#include "FirstOrderOptimizerAdjoint.h"

#include <fmt/format.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace symx;

FirstOrderOptimizerAdjoint::FirstOrderOptimizerAdjoint(spAdjointNewton adjoint, spContext context, spSolverCallbacks callbacks)
    : adjoint(adjoint), context(context)
{
	if (!this->adjoint || !this->context) {
		throw std::invalid_argument("FirstOrderOptimizerAdjoint requires an AdjointNewton and Context.");
	}
    this->global_loss = this->adjoint->get_global_loss();
    this->global_potential = this->adjoint->get_global_potential();
    this->newton_solver = this->adjoint->get_newton();
    this->output = context->output;
    this->logger = context->logger;
    if (callbacks != nullptr) {
        this->callbacks = callbacks;
    } else {
        this->callbacks = std::make_shared<SolverCallbacks>(context);
    }
}

spFirstOrderOptimizerAdjoint FirstOrderOptimizerAdjoint::create(spAdjointNewton adjoint, spContext context, spSolverCallbacks callbacks)
{
    return std::make_shared<FirstOrderOptimizerAdjoint>(adjoint, context, callbacks);
}

SolverReturn FirstOrderOptimizerAdjoint::solve()
{
    const int n_params = global_loss->get_total_n_dofs();

    // Validate
    if (n_params <= 0) {
        this->output->print_with_new_line("FirstOrderOptimizerAdjoint error: no parameters.", Verbosity::Summary);
        return SolverReturn::InvalidInitialState;
    }

    // Resize reusable buffers
    this->params.resize(n_params);
    this->grad.resize(n_params);
    this->dp.resize(n_params);

    // Reset
    this->stats = SolveStats();
    std::vector<int32_t> dof_set_sizes;
    dof_set_sizes.reserve(this->global_loss->get_n_dof_sets());
    for (int set_i = 0; set_i < this->global_loss->get_n_dof_sets(); ++set_i) {
        dof_set_sizes.push_back(this->global_loss->get_n_dofs(set_i));
    }
    this->algorithm.prepare(n_params, this->global_loss->get_dofs_offsets(), dof_set_sizes, this->settings);

    // Check initial state validity
    if (!this->callbacks->run_is_initial_state_valid()) {
        this->output->print_with_new_line("FirstOrderOptimizerAdjoint failure: Invalid initial state.", Verbosity::Summary);
        return SolverReturn::InvalidInitialState;
    }

	// Output control (stored as members so _line_search() can access them).
	// Inner Newton output belongs to AdjointNewton's context and can differ from
	// the outer optimizer's context.
	this->_suppress_inner       = this->suppress_inner_newton_output;

    // Read initial parameter values
    this->global_loss->get_dofs(this->params.data());

    // Initial evaluation: L and dL/dp (runs Newton internally)
    double L = 0.0;
	{
		this->callbacks->run_before_energy_evaluation();
		const spOutputSink inner_output = this->adjoint->get_newton_output();
		const bool inner_output_enabled = inner_output->is_enabled();
		if (this->_suppress_inner) inner_output->set_enabled(false);
		const AdjointValueResult value = this->adjoint->run_forward_and_evaluate_L();
		const AdjointGradientResult gradient = this->adjoint->evaluate_dL_dp_no_solve();
		if (this->_suppress_inner) inner_output->set_enabled(inner_output_enabled);
        if (!value.success) return value.status;
        if (!gradient.success) return gradient.status;
        L = value.value;
        this->grad = gradient.gradient;
    }
    const double initial_residual = this->callbacks->compute_residual(this->grad);

    // Sanity checks on initial state
    if (!std::isfinite(L)) {
        this->output->print_with_new_line("FirstOrderOptimizerAdjoint error: Initial loss is NaN or Inf.", Verbosity::Summary);
        return SolverReturn::InvalidInitialState;
    }
    if (!this->grad.allFinite()) {
        this->output->print_with_new_line("FirstOrderOptimizerAdjoint error: Initial gradient contains NaN or Inf.", Verbosity::Summary);
        return SolverReturn::InvalidInitialState;
    }

    // ============================  MAIN LOOP  =============================
    SolverReturn result = SolverReturn::Running;
    int opt_iter = -1;
	while (result == SolverReturn::Running) {
		opt_iter++;
		this->callbacks->run_before_step();
		struct AfterStepGuard {
			spSolverCallbacks& callbacks;
			~AfterStepGuard() { callbacks->run_after_step(); }
		} after_step{this->callbacks};

        // Max iterations
        if (opt_iter == this->settings.max_iterations) {
            result = this->settings.max_iterations_as_success
                         ? SolverReturn::Successful
                         : SolverReturn::TooManyIterations;
            break;
        }

        // Print iteration header + gradient norm
        const double residual = this->callbacks->compute_residual(this->grad);
		this->output->print_with_new_line(
			fmt::format("{:3d}. ", opt_iter + this->settings.iteration_offset), Verbosity::Summary);
        this->output->print(fmt::format("r0: {:.1e} | ", residual), Verbosity::Summary);

        // Convergence: gradient norm
        if (opt_iter >= this->settings.min_iterations) {
			if (residual < this->settings.residual_tolerance_abs
				|| (initial_residual > 0.0 && residual / initial_residual < this->settings.residual_tolerance_rel)) {
                this->output->print("(converged)", Verbosity::Summary);
                result = SolverReturn::Successful;
                break;
            }
        }

        // Save state for L-BFGS history update
        const double L_prev = L;
        const Eigen::VectorXd params_prev = this->params;
        const Eigen::VectorXd grad_prev   = this->grad;

        // Compute descent direction
        this->dp = this->algorithm.compute_direction(this->grad, this->settings);

        // Check for non-descent direction (L-BFGS can produce ascent directions)
        const double descent_check = this->grad.dot(this->dp);
        if (this->settings.restart_when_non_descent && descent_check >= 0.0) {
            this->output->print("reset | ", Verbosity::Summary);
            this->algorithm.reset(n_params);
            this->dp = this->algorithm.default_direction(this->grad, this->settings);
        }
        if (this->grad.dot(this->dp) >= 0.0) {
            this->output->print("non-descent step", Verbosity::Summary);
            result = SolverReturn::StepDoesNotDescend;
            break;
        }

        // Step size (pre-cap, used for convergence check below)
        const double dp_max = this->dp.cwiseAbs().maxCoeff();

        // Convergence: step size
        if (opt_iter >= this->settings.min_iterations && dp_max < this->settings.step_tolerance) {
            this->output->print("(step converged)", Verbosity::Summary);
            result = SolverReturn::Successful;
            break;
        }

        // Print step size
        this->output->print(fmt::format("dp_raw: {:.1e} | ", dp_max), Verbosity::Summary);

        // Line search: step cap → max step → invalid state → Newton fail → Armijo
        const SolverReturn ls_ret = this->_line_search(L_prev, params_prev, grad_prev);  // <- moves params and dofs
        if (ls_ret != SolverReturn::Running) {
            this->output->print("Line search failure", Verbosity::Summary);
            result = ls_ret;
            break;
        }

        // Get the gradient at the accepted location
		{
			this->callbacks->run_before_energy_evaluation();
			const AdjointValueResult value = this->adjoint->evaluate_L_no_solve();
			const AdjointGradientResult gradient = this->adjoint->evaluate_dL_dp_no_solve();
            if (!value.success) {
                result = value.status;
                break;
            }
            if (!gradient.success) {
                result = gradient.status;
                break;
            }
            L = value.value;
            this->grad = gradient.gradient;
        }

        // Read back parameters (in case adjoint modified them internally)
        this->global_loss->get_dofs(this->params.data());

        // Get inner Newton stats (last accepted solve)
        auto nstats = this->newton_solver->get_last_solve_stats();
        int newton_iters    = nstats.newton_iterations;
        int newton_ls_iters = nstats.all_ls();

        // Accepted step size (true norm of accepted displacement in parameter space)
        const double dp_accepted = (this->params - params_prev).cwiseAbs().maxCoeff();

        // Print iteration stats
        this->output->print(fmt::format("#newton: {:2d} | #newton_ls: {:2d} | dp: {:.1e} | ",
            newton_iters, newton_ls_iters, dp_accepted), Verbosity::Summary);

        // Log stats
        this->logger->add_and_append("newton_iterations", newton_iters);
        this->logger->add_and_append("newton_ls_iterations", newton_ls_iters);

        // Update L-BFGS curvature history
        if (this->settings.type == FirstOrderOptType::LBFGS) {
            const Eigen::VectorXd s  = this->params - params_prev;
            const Eigen::VectorXd y  = this->grad   - grad_prev;
            const double          ys = y.dot(s);

            if (!this->algorithm.update_lbfgs_history(s, y, this->settings)) {
                this->output->print(fmt::format("skip_lbfgs(ys={:.1e}) Non-convex or tiny steps", ys), Verbosity::Medium);
            }

            const int history_skip_count = this->algorithm.get_lbfgs_history_skip_count();
            if (history_skip_count % 5 == 0 && history_skip_count > 0) {
                this->output->print_with_new_line(fmt::format(
                    "[warn] L-BFGS curvature pair rejected {} times in a row (y·s={:.1e}). "
                    "Resetting its history and using steepest descent.",
                    history_skip_count, ys), Verbosity::Summary);
                this->algorithm.reset_lbfgs_history();
            }
        }

        // User convergence callback
        const bool user_converged = this->callbacks->run_is_converged();
        if (opt_iter >= this->settings.min_iterations && user_converged) {
            result = SolverReturn::Successful;
        }

    } // ========================= END MAIN LOOP ==========================

    // Check if converged state is valid
    if (result == SolverReturn::Successful) {
        if (!this->callbacks->run_is_converged_state_valid()) {
            this->output->print_with_new_line(
                "FirstOrderOptimizerAdjoint failure: Invalid converged state.", Verbosity::Summary);
            result = SolverReturn::InvalidConvergedState;
        }
    }

    this->stats.opt_iterations = opt_iter;

    return result;
}
void FirstOrderOptimizerAdjoint::print_summary() const
{
    auto* out = this->output.get();

    out->print_with_new_line("");
    out->print_with_new_line(fmt::format("  {:<28} {:>10} {:>8} {:>8} {:>8}", "Solve", "Total", "Avg", "Min", "Max"));
    out->print_with_new_line(fmt::format("  {}", std::string(66, '-')));
    out->print_with_new_line(fmt::format("  {:<28} {:>10}", "Opt iterations", this->stats.opt_iterations));
    {
        std::vector<std::pair<std::string, std::string>> rows = {
            {"Newton iterations",    "newton_iterations"},
            {"Newton LS iters",      "newton_ls_iterations"},
            {"LS cap",               "ls_cap"},
            {"LS max step",          "ls_max"},
            {"LS inv state",         "ls_inv"},
            {"LS Newton fail",       "ls_nf"},
            {"LS Armijo",            "ls_bt"},
        };
        for (const auto& [label, key] : rows) {
            auto s = logger->get_stats(key);
            out->print_with_new_line(fmt::format("  {:<28} {:>10} {:>8.1f} {:>8} {:>8}",
                label, (long long)s.total, s.avg, (int)s.min, (int)s.max));
        }
    }
    out->print_with_new_line(fmt::format("  {}", std::string(66, '-')));
    out->print_new_line();
}

SolverReturn symx::FirstOrderOptimizerAdjoint::_line_search(
    double L_prev,
    const Eigen::VectorXd& params_prev,
    const Eigen::VectorXd& grad_at_prev)
{
    this->last_safe_dofs.resize(this->global_potential->get_total_n_dofs());
    this->global_potential->get_dofs(this->last_safe_dofs.data());

    auto restore_accepted_state = [&]() {
        this->params = params_prev;
        this->global_loss->set_dofs(this->params.data());
        this->global_potential->set_dofs(this->last_safe_dofs.data());
    };

    // Helper: set params_prev + alpha * dp as the current parameter point
    auto apply_params = [&](double a) {
        this->params = params_prev + a * this->dp;
        this->global_loss->set_dofs(this->params.data());
        this->global_potential->set_dofs(this->last_safe_dofs.data());
    };

    // Helper: run inner Newton solve + adjoint, update L and this->grad
	auto eval_L = [&]() {
		this->callbacks->run_before_energy_evaluation();
		const spOutputSink inner_output = this->adjoint->get_newton_output();
		const bool inner_output_enabled = inner_output->is_enabled();
		if (this->_suppress_inner) inner_output->set_enabled(false);
		const AdjointValueResult value = this->adjoint->run_forward_and_evaluate_L();
		if (this->_suppress_inner) inner_output->set_enabled(inner_output_enabled);
		return value;
	};

    /* ====== 1. Step cap ====== */
    {
        const double dp_max = this->dp.cwiseAbs().maxCoeff();
        if (dp_max > this->settings.step_cap) {
            this->dp *= (this->settings.step_cap / dp_max);
            this->output->print(fmt::format("cap {:.1e} | ", this->settings.step_cap), Verbosity::Summary);
            this->stats.ls_cap_iterations++;
            this->logger->add_and_append("ls_cap", 1);
        } else {
            this->logger->add_and_append("ls_cap", 0);
        }
    }

    /* ====== 2. Max step callback ====== */
    double alpha = 1.0;
    {
        const double max_step = this->callbacks->run_max_allowed_step();
        if (max_step < 1.0) {
            alpha = max_step;
            this->output->print(fmt::format("ls max {:.1e} | ", alpha), Verbosity::Summary);
            this->stats.ls_max_iterations++;
            this->logger->add_and_append("ls_max", 1);
        } else {
            this->logger->add_and_append("ls_max", 0);
        }
    }

    /* ====== 3. Invalid state backtracking ====== */
    apply_params(alpha);
    int ls_inv = 0;
    for (; ls_inv < this->settings.max_backtracking_invalid_state_iterations; ls_inv++) {
        if (this->callbacks->run_is_intermediate_state_valid()) break;
        alpha *= 0.5;
        apply_params(alpha);
    }
    this->stats.ls_inv_iterations += ls_inv;
    this->logger->add_and_append("ls_inv", ls_inv);
    if (ls_inv > 0) {
        this->output->print(fmt::format("ls inv {:d} | ", ls_inv), Verbosity::Summary);
    }
    if (ls_inv == this->settings.max_backtracking_invalid_state_iterations) {
        this->output->print_with_new_line(
            "FirstOrderOptimizerAdjoint failure: Too many invalid state iterations.", Verbosity::Summary);
        this->callbacks->run_on_intermediate_state_invalid();
        restore_accepted_state();
        return SolverReturn::TooManyInvalidIntermediateIterations;
    }

    /* ====== 4. Evaluate L and gradient (inner Newton solve + adjoint) ====== */
    AdjointValueResult value = eval_L();

    /* ====== 5. Newton-fail + Armijo backtracking ====== */
    // Slope recomputed after step cap to use the (possibly scaled) direction
    const double slope = grad_at_prev.dot(this->dp);

    int ls_nf = 0, ls_bt = 0;
    SolverReturn ls_result = SolverReturn::Running;

    {
        const double c1  = this->settings.line_search_armijo_beta;
        const int    max = this->settings.max_backtracking_armijo_iterations;
        int bt;
        for (bt = 0; bt < max; bt++) {
            bool worked = true;
            this->output->print_with_new_line(fmt::format("{:d}. alpha: {:.2e} | ", bt, alpha), Verbosity::Full);
            if (!value.success || !std::isfinite(value.value)) {
                ls_nf++;
                this->output->print("L=Inf/NaN | ", Verbosity::Full);
                worked = false;
            }
            else if (this->settings.enable_armijo_backtracking) {
                const double L_bt = L_prev + c1 * alpha * slope;
                this->output->print(fmt::format("L_bt: {:.2e} | L: {:.2e} | L/L_bt: {:.2e}", L_bt, value.value, value.value / L_bt), Verbosity::Full);

                // Armijo condition
                if (value.value > L_bt) {
                    worked = false;
                    ls_bt++;
                }
            }

            // Exit
            if (worked) {
                break;
            }

            // Half the step size and re-evaluate
            else {
                alpha *= 0.5;
                apply_params(alpha);
                value = eval_L();
            }
        }
        if (bt == max) {
            this->callbacks->run_on_armijo_fail();
            ls_result = SolverReturn::TooManyArmijoIterations;
        }
    }
    if (this->context->output->get_console_verbosity() == Verbosity::Full) {
        this->output->print_with_new_line("      ", Verbosity::Summary);
    }

    if (ls_nf > 0) this->output->print(fmt::format("ls nf {:d} | ", ls_nf), Verbosity::Summary);
    if (ls_bt > 0) this->output->print(fmt::format("ls bt {:d} | ", ls_bt), Verbosity::Summary);

    this->stats.ls_nf_iterations += ls_nf;
    this->stats.ls_bt_iterations += ls_bt;
    this->logger->add_and_append("ls_nf", ls_nf);
    this->logger->add_and_append("ls_bt", ls_bt);

    if (ls_result != SolverReturn::Running) {
        restore_accepted_state();
    }

    return ls_result;
}
