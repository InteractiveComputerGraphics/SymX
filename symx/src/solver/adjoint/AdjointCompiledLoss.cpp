#include "AdjointCompiledLoss.h"

symx::AdjointCompiledLoss::AdjointCompiledLoss(const Potential &loss, const std::vector<DataMap<double>> &dofs, const std::vector<DataMap<double>> &params, const std::string &compilation_directory, DeferredParallelTasks &tasks)
    : mws(loss.get_mws())
{
    // Get DoFs and Params
    auto [dof_symbols, dof_indices] = this->mws->get_symbols_and_indices(dofs);
    auto [param_symbols, param_indices] = this->mws->get_symbols_and_indices(params);
    const std::vector<Scalar> all_symbols = collect_scalars(dof_symbols, param_symbols);
    this->n_dofs = (int32_t)dof_symbols.size();
    this->n_params = (int32_t)param_symbols.size();
    this->dof_indexing = dof_indices;
    this->param_indexing = param_indices;

    // Checksum
    std::string pre_hash = get_checksum(all_symbols);
    if (loss.is_factorized()) { pre_hash += "_fem_loss"; }
    std::string checksum = loss.get_checksum(pre_hash);
    if (loss.has_conditional()) {
        checksum += loss.get_condition().get_checksum();
    }
    // Condition
    this->condition = std::make_shared<CompiledCondition>(loss, checksum, compilation_directory, tasks);

    // Try to load
    const std::string name = loss.get_name();
    this->cached = this->L.load_if_cached(this->mws, name + "_L", compilation_directory, checksum);
    this->cached &= this->dL_du__dL_dp.load_if_cached(this->mws, name + "_dL_du__dL_dp", compilation_directory, checksum);

    // If not cached, diff and compile
    if (!this->cached) {

        // Differentiation
        const Scalar L = loss.get_expression();
        const Vector dL_du__dL_dp = gradient(L, all_symbols);

        // Defer compilation tasks
        tasks.add([this, L, name, compilation_directory, checksum]() {
            this->L.compile(this->mws, { L }, name + "_L", compilation_directory, checksum);
        });
        tasks.add([this, dL_du__dL_dp_vals = dL_du__dL_dp.values(), name, compilation_directory, checksum]() {
            this->dL_du__dL_dp.compile(this->mws, dL_du__dL_dp_vals, name + "_dL_du__dL_dp", compilation_directory, checksum);
        });
    }
}

bool symx::AdjointCompiledLoss::was_cached() const
{
    return this->cached;
}

void symx::AdjointCompiledLoss::evaluate_L(bsm::ParallelNumber<double> &parallel_L)
{
    const int n_threads = parallel_L.get_n_threads();

    // Evaluate element condition
    this->condition->evaluate(n_threads);

    // Evaluate L
    this->L.run(n_threads,
        [&parallel_L](const View<double> L_val, int32_t element_idx, int32_t thread_id, const View<int32_t> conn)
        {
			parallel_L.local(thread_id) += L_val[0];
        },
        this->condition->get_element_condition()
    );
}

void symx::AdjointCompiledLoss::evaluate_dL_du__dL_dp(bsm::ParallelVector<Eigen::VectorXd> &parallel_dL_du, bsm::ParallelVector<Eigen::VectorXd> &parallel_dL_dp, const std::vector<int> &dof_offsets, const std::vector<int> &param_offsets)
{
    const int n_threads = parallel_dL_du.get_n_threads();

    // Evaluate element condition
    this->condition->evaluate(n_threads);

    // Evaluate dL_du and dL_dp
    this->dL_du__dL_dp.run(n_threads,
        [&parallel_dL_du, &parallel_dL_dp, this, &dof_offsets, &param_offsets](const View<double> grad_val, int32_t element_idx, int32_t thread_id, const View<int32_t> conn)
        {
            // grad_val layout: [dL/du..., dL/dp...]
            const View<double> dL_du_loc = grad_val.slice(0, this->n_dofs);

            // Write dL/du to global gradient
			Eigen::VectorXd& dL_du = parallel_dL_du.local(thread_id);
            for (int i = 0; i < this->n_dofs; i++) {
                const DoFIndexing& di = this->dof_indexing[i];
                const int elem_idx_u = (di.conn_idx >= 0) ? conn[di.conn_idx] : 0;  // -1 for global DOF, otherwise use connectivity
                const int glob_idx = dof_offsets[di.dof_set] + elem_idx_u * di.stride + di.offset;
                dL_du[glob_idx] += dL_du_loc[i];
            }

            // Write dL/dp to global gradient (only when there are explicit param dependencies)
            if (this->n_params > 0) {
                const View<double> dL_dp_loc = grad_val.slice(this->n_dofs, grad_val.size());
			Eigen::VectorXd& dL_dp = parallel_dL_dp.local(thread_id);
                for (int i = 0; i < this->n_params; i++) {
                    const DoFIndexing& pi = this->param_indexing[i];
                    const int elem_idx_p = (pi.conn_idx >= 0) ? conn[pi.conn_idx] : 0; // -1 for global param, otherwise use connectivity
                    const int glob_idx = param_offsets[pi.dof_set] + elem_idx_p * pi.stride + pi.offset;
                    dL_dp[glob_idx] += dL_dp_loc[i];
                }
            }
         },
        this->condition->get_element_condition()
    );
}
