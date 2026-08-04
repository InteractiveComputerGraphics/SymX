#include "FirstOrderCompiledGlobal.h"

#include <fmt/format.h>
#include <omp.h>

#include <stdexcept>

#include "../DeferredParallelTasks.h"
#include "../solver_utils.h"

using namespace symx;

FirstOrderCompiledGlobal::FirstOrderCompiledGlobal(spGlobalPotential global_potential, spContext context)
    : global_potential(global_potential), context(context)
{
    const std::vector<DataMap<double>>& dofs_maps = global_potential->get_dof_maps();
    const std::vector<std::unique_ptr<Potential>>& potentials = global_potential->get_potentials();
    if (dofs_maps.empty() || potentials.empty()) {
        throw std::invalid_argument("FirstOrderOptimizer requires a GlobalPotential with DoFs and potentials.");
    }

    const int n_potentials = (int)potentials.size();
    this->compiled_potentials.resize(n_potentials);
    context->output->print_with_new_line("First Order Potentials:", Verbosity::Minimal);

    DeferredParallelTasks tasks;
    const double start_time = omp_get_wtime();
    #pragma omp parallel for num_threads(context->n_threads) schedule(dynamic)
    for (int i = 0; i < n_potentials; ++i) {
        this->compiled_potentials[i] = std::make_unique<FirstOrderCompiledPotential>(
            *potentials[i], dofs_maps, context->compilation_directory, tasks);
        #pragma omp critical
        {
            const std::string status = this->compiled_potentials[i]->was_cached()
                ? "loaded" : "queued for compilation";
            context->output->print_with_new_line(
                "    " + in_two_columns(potentials[i]->get_name(), status, 70), Verbosity::Minimal);
        }
    }

    context->output->print_with_new_line("Compiling... ", Verbosity::Minimal);
    std::cout << std::flush;
    tasks.run(context->n_threads);
    context->output->print("done.", Verbosity::Minimal);
    context->output->print_with_new_line(
        "Total time: " + std::to_string(omp_get_wtime() - start_time) + " s", Verbosity::Minimal);

    context->output->print_with_new_line("\nDegrees of freedom:", Verbosity::Minimal);
    for (int set_i = 0; set_i < global_potential->get_n_dof_sets(); ++set_i) {
        const std::string label = global_potential->get_dof_label(set_i);
        const std::string name = label.empty() ? "Set " + std::to_string(set_i) : label;
        context->output->print_with_new_line(
            fmt::format("{}: {:d}", name, global_potential->get_n_dofs(set_i)), Verbosity::Minimal);
    }
    context->output->print_with_new_line(
        fmt::format("Total: {:d}", global_potential->get_total_n_dofs()), Verbosity::Minimal);
}

void FirstOrderCompiledGlobal::evaluate_P(double& out_P)
{
    auto timer = context->logger->time("evaluate_P");
    this->assembly.start(global_potential->get_dofs_offsets(), context->n_threads, false, false);
    for (const auto& potential : this->compiled_potentials) {
        potential->evaluate_P(this->assembly);
    }
    this->assembly.stop(false, false);
	out_P = this->assembly.E.result();
}

void FirstOrderCompiledGlobal::evaluate_P__dP_du(double& out_P, Eigen::VectorXd& out_dP_du)
{
    auto timer = context->logger->time("evaluate_P_grad");
    this->assembly.start(global_potential->get_dofs_offsets(), context->n_threads, false, true);
    for (const auto& potential : this->compiled_potentials) {
        potential->evaluate_P__dP_du(this->assembly);
    }
    this->assembly.stop(false, true);
	out_P = this->assembly.E.result();
	out_dP_du = this->assembly.grad.result();
}
