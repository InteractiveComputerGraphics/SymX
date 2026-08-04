#include "AdjointCompiledPotential.h"
#include "../CompiledCondition.h"

using namespace symx;

AdjointCompiledPotential::AdjointCompiledPotential(const Potential& potential, spCompiledCondition condition, const std::vector<DataMap<double>> &dofs, const std::vector<DataMap<double>> &params, const std::string &compilation_directory, DeferredParallelTasks &tasks)
    : mws(potential.get_mws()), condition(condition)
{
    // Get DoFs and Params
    auto [dof_symbols, dof_indices] = this->mws->get_symbols_and_indices(dofs);
    auto [param_symbols, param_indices] = this->mws->get_symbols_and_indices(params);
    const std::vector<Scalar> all_symbols = collect_scalars(dof_symbols, param_symbols);
    this->n_dofs = (int32_t)dof_symbols.size();
    this->n_params = (int32_t)param_symbols.size();
    this->dof_indexing = dof_indices;
    this->param_indexing = param_indices;

    // If there are no DOFs or no params, d2P/du∂p is trivially zero — nothing to compile.
    if (this->n_dofs == 0 || this->n_params == 0) {
        this->cached = true;  // mark as "done" so evaluate_d2P_dpdu_dot is a no-op
        return;
    }

    // Checksum
    std::string pre_hash = get_checksum(all_symbols);
    const std::string checksum = potential.get_checksum(pre_hash);

    // Build condition if not provided externally (handles conditional potentials)
    if (!this->condition) {
        this->condition = std::make_shared<CompiledCondition>(potential, checksum, compilation_directory, tasks);
    }

    // Try to load
    const std::string name = potential.get_name();
    this->cached = this->d2P_dpdu.load_if_cached(this->mws, name + "_d2P_dpdu", compilation_directory, checksum);

    // If not cached, diff and compile
    if (!this->cached) {
        
        // Differentiation
        const Scalar P = potential.get_expression();
		Matrix d2P_dpdu = Matrix::zero({ this->n_dofs, this->n_params }, P);
		DiffCache diff_cache;
        for (int i = 0; i < this->n_params; i++) {
            Scalar dP_dpj = diff(P, param_symbols[i], diff_cache);
            for (int j = 0; j < this->n_dofs; j++) {
                d2P_dpdu(j, i) = diff(dP_dpj, dof_symbols[j], diff_cache);
            }
        }

        // Defer compilation tasks
        tasks.add([this, d2P_dpdu, name, compilation_directory, checksum]() {
            this->d2P_dpdu.compile(this->mws, d2P_dpdu.values(), name + "_d2P_dpdu", compilation_directory, checksum);
        });
    }
}

void symx::AdjointCompiledPotential::evaluate_d2P_dpdu_dot(bsm::ParallelVector<Eigen::VectorXd> &parallel_result, const Eigen::VectorXd &input, const std::vector<int> &dof_offsets, const std::vector<int> &param_offsets)
{
    // Nothing to do if this potential does not depend on any parameters or DOFs
    if (this->n_dofs == 0 || this->n_params == 0) {
        return;
    }

    const int n_threads = parallel_result.get_n_threads();

    // Evaluate element condition
    this->condition->evaluate(n_threads);

    // Evaluate d2P_dpdu * input  [input.T * d2P_dudp  =  (d2P_dudp.T * input).T  =  (d2P_dpdu * input).T]
    this->d2P_dpdu.run(n_threads,
        [&parallel_result, &input, &dof_offsets, &param_offsets, this](const View<double> sol, int32_t element_idx, int32_t thread_id, const View<int32_t> conn)
        {
			Eigen::VectorXd& result = parallel_result.local(thread_id);
            for (int i = 0; i < this->n_params; i++) {
                const DoFIndexing& pi = this->param_indexing[i];
                const int elem_idx_p = (pi.conn_idx >= 0) ? conn[pi.conn_idx] : 0;  // -1 for global param, otherwise use connectivity
                const int param_index = param_offsets[pi.dof_set] + pi.stride * elem_idx_p + pi.offset;

                double val = 0.0;
                for (int j = 0; j < this->n_dofs; j++) {
                    const DoFIndexing& uj = this->dof_indexing[j];
                    const int elem_idx_u = (uj.conn_idx >= 0) ? conn[uj.conn_idx] : 0;  // -1 for global DOF, otherwise use connectivity
                    const int dof_index = dof_offsets[uj.dof_set] + uj.stride * elem_idx_u + uj.offset;
                    
                    const double vij = sol[j * this->n_params + i];
                    const double input_j = input[dof_index];
                    val += vij * input_j;
                }
                result[param_index] += val;
            }
        },
        this->condition->get_element_condition()
    );
}

bool symx::AdjointCompiledPotential::was_cached() const
{
    return this->cached;
}
