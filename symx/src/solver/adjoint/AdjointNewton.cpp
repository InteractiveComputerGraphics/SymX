#include "AdjointNewton.h"

#include <cmath>
#include <fmt/format.h>

using namespace symx;

namespace
{
    AdjointValueResult failed_value(SolverReturn status)
    {
        return {status, false, std::numeric_limits<double>::infinity()};
    }

    AdjointGradientResult failed_gradient(SolverReturn status)
    {
        return {status, false, Eigen::VectorXd()};
    }
}

AdjointNewton::AdjointNewton(spNewtonsMethod newton, spGlobalPotential global_loss, LinearSolveSettings linear_solve_settings)
    : newton(newton), global_loss(global_loss), linear_solve_settings(linear_solve_settings)
{
    if (!this->newton || !this->global_loss) {
        throw std::invalid_argument("AdjointNewton requires a Newton solver and a global loss.");
    }
    this->global_potential = this->newton->get_global_potential();
    this->context = this->newton->get_context();
    this->linear_solve_context = this->newton->get_linear_solve_context();

    // Get
    const std::vector<std::unique_ptr<Potential>>& potentials = this->global_potential->get_potentials();
    const std::vector<std::unique_ptr<Potential>>& losses = this->global_loss->get_potentials();
    const std::vector<DataMap<double>>& dofs = this->global_potential->get_dof_maps();
    const std::vector<DataMap<double>>& param = this->global_loss->get_dof_maps();
    const int n_potentials = (int)potentials.size();
    const int n_losses = (int)losses.size();

    // Verify inputs
    if (potentials.size() == 0) {
        std::cout << "symx error: AdjointNewton() got GlobalPotential with zero Potentials." << std::endl;
        exit(-1);
    }
    if (losses.size() == 0) {
        std::cout << "symx error: AdjointNewton() got GlobalLoss with zero Potentials." << std::endl;
        exit(-1);
    }
    if (dofs.size() == 0) {
        std::cout << "symx error: AdjointNewton() got GlobalPotential with zero DoF maps." << std::endl;
        exit(-1);
    }
    if (param.size() == 0) {
        std::cout << "symx error: AdjointNewton() got GlobalLoss with zero parameter maps." << std::endl;
        exit(-1);
    }

    // Initialize compiled potentials and losses
    this->compiled_potential.resize(n_potentials);
    this->compiled_loss.resize(n_losses);
    context->output->print_with_new_line("Adjoint Losses:", Verbosity::Minimal);

    // Reuse Newton's already-compiled SecondOrderCompiledGlobal.
    this->compiled_newton_derivatives = newton->get_compiled();
    const std::vector<spCompiledCondition> newton_conditions = this->compiled_newton_derivatives->get_conditions();

    // Parallel init
    DeferredParallelTasks tasks;
    const double t0 = omp_get_wtime();

    // Pass through losses. Try to load / queue compilation
    #pragma omp parallel for num_threads(context->n_threads) schedule(dynamic)
    for (int i = 0; i < n_losses; i++) {
        const Potential& loss = *losses[i];

        this->compiled_loss[i] = std::make_unique<AdjointCompiledLoss>(loss, dofs, param, context->compilation_directory, tasks);

        #pragma omp critical
        {
            if (compiled_loss[i]->was_cached()) {
                context->output->print_with_new_line("    " + in_two_columns(losses[i]->get_name(), "loaded", 70), Verbosity::Minimal);
            }
            else {
                context->output->print_with_new_line("    " + in_two_columns(losses[i]->get_name(), "queued for compilation", 70), Verbosity::Minimal);
            }
        }
    }

    // Pass through potentials, reusing Newton's pre-compiled conditions to avoid
    // checksum collisions with already-loaded dlopen handles.
    for (int i = 0; i < n_potentials; i++) {
        const Potential& potential = *potentials[i];
        spCompiledCondition cond = (i < (int)newton_conditions.size()) ? newton_conditions[i] : nullptr;
        this->compiled_potential[i] = std::make_unique<AdjointCompiledPotential>(potential, cond, dofs, param, context->compilation_directory, tasks);
        // No need to print because SecondOrderCompiledGlobal will have already printed the status of these potentials
    }
    
    // Run compilation tasks
    context->output->print_with_new_line("Compiling... ", Verbosity::Minimal);
    tasks.run(context->n_threads);
    const double t1 = omp_get_wtime();
    context->output->print("done.", Verbosity::Minimal);
    context->output->print_with_new_line("Total time: " + std::to_string(t1 - t0) + " s", Verbosity::Minimal);

    // Print params
    this->context->output->print_with_new_line("\nParameters:", Verbosity::Minimal);
    for (int set_i = 0; set_i < this->global_loss->get_n_dof_sets(); set_i++) {
        const std::string label = this->global_loss->get_dof_label(set_i);
        const std::string msg = fmt::format("{}: {:d}", (label == "") ? "Set " + std::to_string(set_i) : label, this->global_loss->get_n_dofs(set_i));
        this->context->output->print_with_new_line(msg, Verbosity::Minimal);
    }
    this->context->output->print_with_new_line(fmt::format("Total: {:d}", this->global_loss->get_total_n_dofs()), Verbosity::Minimal);
}

std::shared_ptr<AdjointNewton> AdjointNewton::create(spNewtonsMethod newton, spGlobalPotential global_loss, LinearSolveSettings linear_solve_settings)
{
    return std::make_shared<AdjointNewton>(newton, global_loss, linear_solve_settings);
}

void AdjointNewton::force_gradient_to_zero(const std::vector<int> &param_indices)
{
    const int n_params = this->global_loss->get_total_n_dofs();
    for (int param_i : param_indices) {
        if (param_i < 0 || param_i >= n_params) {
            throw std::out_of_range("AdjointNewton fixed-gradient index is outside the parameter range.");
        }
    }
    this->param_indices_to_zero = param_indices;
}

AdjointValueResult AdjointNewton::run_forward_and_evaluate_L()
{
    auto t_ = this->context->logger->time("run_forward_and_evaluate_L");
    const SolverReturn ret = this->newton->solve();
    if (ret != SolverReturn::Successful) {
        return failed_value(ret);
    }
    return this->evaluate_L_no_solve();
}

AdjointGradientResult AdjointNewton::run_forward_and_evaluate_dL_dp()
{
    auto t_ = this->context->logger->time("run_forward_and_evaluate_dL_dp");
    const SolverReturn ret = this->newton->solve();
    if (ret != SolverReturn::Successful) {
        return failed_gradient(ret);
    }
    return this->evaluate_dL_dp_no_solve();
}

AdjointValueResult symx::AdjointNewton::evaluate_L_no_solve()
{
    auto t_ = this->context->logger->time("evaluate_L_no_solve");
    const double value = this->_evaluate_L();
    return std::isfinite(value)
        ? AdjointValueResult{SolverReturn::Successful, true, value}
        : failed_value(SolverReturn::InvalidEvaluation);
}

AdjointGradientResult AdjointNewton::evaluate_dL_dp_no_solve()
{
    auto t_ = this->context->logger->time("evaluate_dL_dp_no_solve");
    Eigen::VectorXd gradient;
    const SolverReturn ret = this->_compute_dL_dp(gradient);
    return ret == SolverReturn::Successful
        ? AdjointGradientResult{SolverReturn::Successful, true, std::move(gradient)}
        : failed_gradient(ret);
}

AdjointGradientResult AdjointNewton::evaluate_dL_dp_finite_differences(const double eps)
{
    auto t_ = this->context->logger->time("evaluate_dL_dp_finite_differences");

    if (!(eps > 0.0) || !std::isfinite(eps)) {
        return failed_gradient(SolverReturn::InvalidEvaluation);
    }

    const int n_params = this->global_loss->get_total_n_dofs();
    const int n_sim_dofs = this->global_potential->get_total_n_dofs();

    // Snapshot of the original parameter values (read once, never modified in the loop)
    Eigen::VectorXd params(n_params);
    this->global_loss->get_dofs(params.data());
    Eigen::VectorXd sim_dofs(n_sim_dofs);
    this->global_potential->get_dofs(sim_dofs.data());

    Eigen::VectorXd dL_dp(n_params);
    SolverReturn status = SolverReturn::Successful;

    for (int i = 0; i < n_params && status == SolverReturn::Successful; i++) {
        // Forward step: perturb only param i
        params[i] += eps;
        this->global_loss->set_dofs(params.data());
        this->global_potential->set_dofs(sim_dofs.data());
        const AdjointValueResult plus = this->run_forward_and_evaluate_L();

        // Backward step: perturb only param i (opposite direction)
        params[i] -= 2 * eps;
        this->global_loss->set_dofs(params.data());
        this->global_potential->set_dofs(sim_dofs.data());
        const AdjointValueResult minus = this->run_forward_and_evaluate_L();

        if (!plus.success) status = plus.status;
        else if (!minus.success) status = minus.status;
        else dL_dp[i] = (plus.value - minus.value) / (2.0 * eps);
        params[i] += eps;  // Restore original value
    }

    // Restore original parameters
    this->global_loss->set_dofs(params.data());
    this->global_potential->set_dofs(sim_dofs.data());

    if (status != SolverReturn::Successful || !dL_dp.allFinite()) {
        return failed_gradient(status == SolverReturn::Successful ? SolverReturn::InvalidEvaluation : status);
    }

    // Zero out selected params
    this->_zero_grad_for_fixed_params(dL_dp);

    return {SolverReturn::Successful, true, std::move(dL_dp)};
}

void AdjointNewton::_evaluate_dL_du__dL_dp()
{
    const int n_threads = this->context->n_threads;
    const std::vector<int>& dof_offsets = this->global_potential->get_dofs_offsets();
    const std::vector<int>& param_offsets = this->global_loss->get_dofs_offsets();
    const int n_dofs = dof_offsets.back();
    const int n_params = param_offsets.back();

	this->parallel_dL_du.begin_accumulation(n_threads, n_dofs);
	this->parallel_dL_dp.begin_accumulation(n_threads, n_params);

    for (const auto& compiled_loss : this->compiled_loss) {
        compiled_loss->evaluate_dL_du__dL_dp(
            this->parallel_dL_du, this->parallel_dL_dp,
            dof_offsets, param_offsets
        );
    }

	this->parallel_dL_du.reduce();
	this->parallel_dL_dp.reduce();
}

const Eigen::VectorXd &AdjointNewton::_evaluate_dot_d2P_dudp_transpose(const Eigen::VectorXd &input)
{
    const int n_threads = this->context->n_threads;
    const std::vector<int>& dof_offsets = this->global_potential->get_dofs_offsets();
    const std::vector<int>& param_offsets = this->global_loss->get_dofs_offsets();
    const int n_dofs = dof_offsets.back();
    const int n_params = param_offsets.back();

    // Input is lambda, which has size n_dofs. Output will be dL_dp, which has size n_params.
	this->parallel_dot_d2P_dudp_transpose.begin_accumulation(n_threads, n_params);

    for (const auto& compiled_potential : this->compiled_potential) {
        compiled_potential->evaluate_d2P_dpdu_dot( // input.T * d2P_dudp = (d2P_dpdu * input).T
            this->parallel_dot_d2P_dudp_transpose, input, dof_offsets, param_offsets
        );
    }

	return this->parallel_dot_d2P_dudp_transpose.reduce();
}

double AdjointNewton::_evaluate_L()
{
	this->parallel_L.begin_accumulation(this->context->n_threads);
    for (const auto& cl : this->compiled_loss) {
        cl->evaluate_L(this->parallel_L);
    }
	return this->parallel_L.reduce();
}

SolverReturn AdjointNewton::_compute_dL_dp(Eigen::VectorXd& dL_dp_total)
{
    /*
        d2P_du2.T * lambda = -dL_du.T
        dL_dp_total = dL_dp + lambda.T * d2P_dudp
    */

    // Evaluate dL_du and dL_dp
    this->_evaluate_dL_du__dL_dp();
	const Eigen::VectorXd& dL_du = this->parallel_dL_du.result();
	const Eigen::VectorXd& dL_dp = this->parallel_dL_dp.result();
    const int n_dofs   = (int)dL_du.size();
    const int n_params = (int)dL_dp.size();

    // Evaluate d2P_du2 at current equilibrium
    double out_P;
    Eigen::VectorXd out_dP_du;
    spElementHessians eh = this->compiled_newton_derivatives->evaluate_P__dP_du__local_d2P_du2(out_P, out_dP_du);
    spBSM d2P_du2 = eh->assemble_global(this->context->n_threads, n_dofs);

    // Solve adjoint: (d2P_du2).T * lambda = -dL_du
    if (!dL_du.allFinite() || !dL_dp.allFinite()) {
        return SolverReturn::InvalidEvaluation;
    }
    const SolverReturn ret = this->_solve_linear_system(d2P_du2, -dL_du);
    if (ret != SolverReturn::Successful) {
        return ret;
    }

    // Total derivative: dL_dp + lambda^T * d2P_dudp
    dL_dp_total = dL_dp + this->_evaluate_dot_d2P_dudp_transpose(this->lambda);
    if (!dL_dp_total.allFinite()) {
        return SolverReturn::InvalidEvaluation;
    }

    // Zero out selected params
    this->_zero_grad_for_fixed_params(dL_dp_total);

    return SolverReturn::Successful;
}

void symx::AdjointNewton::_zero_grad_for_fixed_params(Eigen::VectorXd &dL_dp) const
{
    for (int param_i : this->param_indices_to_zero) {
        dL_dp[param_i] = 0.0;
    }
}

SolverReturn AdjointNewton::_solve_linear_system(const spBSM &d2P_du2, const Eigen::VectorXd &rhs)
{
    auto timer_ls = this->context->logger->time("adjoint_linear_system_solve");

    const int n_dofs = (int)rhs.size();
    this->lambda.resize(n_dofs);

    if (this->linear_solve_settings.linear_solver == LinearSolver::DirectLLT) {
        if (!this->linear_solve_context->solve_direct_llt(
                d2P_du2, this->lambda, rhs, this->context->n_threads)) {
            return SolverReturn::LinearSystemSolveFailure;
        }
    }
    else if (this->linear_solve_settings.linear_solver == LinearSolver::BDPCG) {
        const bsm::PCGInfo info = solve_linear_system_bdpcg(
            d2P_du2, this->lambda, rhs,
            this->linear_solve_settings,
            this->context->n_threads,
            this->pcg_context);
        if (!info.converged) {
            return SolverReturn::LinearSystemSolveFailure;
        }
    }
    else {
        return SolverReturn::LinearSystemSolveFailure;
    }

    return this->lambda.allFinite() ? SolverReturn::Successful : SolverReturn::LinearSystemSolveFailure;
}
