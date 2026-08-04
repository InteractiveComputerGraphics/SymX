#include "SecondOrderCompiledPotential.h"

#include "../../symbol/diff.h"
#include "project_to_PD.h"

using namespace symx;

SecondOrderCompiledPotential::SecondOrderCompiledPotential(const Potential &potential, const std::vector<DataMap<double>> &dofs_maps, const std::string &compilation_directory, DeferredParallelTasks& tasks)
    : mws(potential.get_mws())
{
    // Identity DoF scalars (mws->get_symbols_and_indices() hardcoded to stride 3)
    std::vector<Scalar> dofs;
    for (int dof_set = 0; dof_set < (int)dofs_maps.size(); dof_set++) {
        const std::vector<DataMap<const double>> original_maps = this->mws->get_original_maps(dofs_maps[dof_set]);  // Get local array maps that are global DoFs
        const std::vector<Scalar> set_dofs = this->mws->get_symbols(dofs_maps[dof_set]); // Get DoF symbols

        // Append dof symbols
        dofs.insert(dofs.end(), set_dofs.begin(), set_dofs.end());

        // DoF indexing
        for (const DataMap<const double>& dof_map : original_maps) {

            // Verify size
            if (dof_map.stride != 3) {
                std::cout << "symx error: SecondOrderCompiledPotential() expected DoF stride of 3, got " << dof_map.stride << "." << std::endl;
                std::cout << "Potential: " << potential.get_name() << std::endl;
                exit(-1);
            }

            DoFInConn map;
            map.dof_set = dof_set;
            map.conn_idx = dof_map.connectivity_index;
            this->dof_in_conn.push_back(map);
        }
    }

    // Store n_dofs
    this->n_dofs = (int32_t)dofs.size();

    // Exit if no dofs
    if (this->n_dofs == 0) {
        std::cout << "symx error: SecondOrderCompiledPotential() got potential with zero DoFs." << std::endl;
        std::cout << "Potential: " << potential.get_name() << std::endl;
        exit(-1);
    }

    // Compute checksum
    std::string pre_hash = potential.get_name();
    for (const Scalar& dof : dofs) {
        pre_hash += std::to_string(dof.get_symbol_idx());
    }
    std::string checksum = potential.get_checksum(pre_hash);
    if (potential.has_conditional()) {
        checksum += potential.get_condition().get_checksum();
    }

    // Condition
    this->condition = std::make_shared<CompiledCondition>(potential, checksum, compilation_directory, tasks);
    
    // Try to load
    const std::string name = potential.get_name();
    this->cached = this->P.load_if_cached(this->mws, name + "_P", compilation_directory, checksum);
    this->cached &= this->P__dP_du.load_if_cached(this->mws, name + "_P__dP_du", compilation_directory, checksum);
    this->cached &= this->P__dP_du__d2P_du2.load_if_cached(this->mws, name + "_P__dP_du__d2P_du2", compilation_directory, checksum);
    this->cached &= this->condition->was_cached();

    // If not cached, compile
    if (!this->cached) {

        // Differentiation
        const Scalar v = potential.get_expression();
        DiffCache diff_cache;
        const Vector g = gradient(v, dofs, diff_cache);
        const Matrix h = gradient(g, dofs, /*symmetric=*/true, diff_cache);

        // Defer compilation tasks
        tasks.add([this, v, name, compilation_directory, checksum]() {
            this->P.compile(this->mws, { v }, name + "_P", compilation_directory, checksum);
        });
        tasks.add([this, v, g_vals = g.values(), name, compilation_directory, checksum]() {
            this->P__dP_du.compile(this->mws, collect_scalars({{ v }, g_vals}), name + "_P__dP_du", compilation_directory, checksum);
        });
        tasks.add([this, v, g_vals = g.values(), h_vals = h.values(), name, compilation_directory, checksum]() {
            this->P__dP_du__d2P_du2.compile(this->mws, collect_scalars({{ v }, g_vals, h_vals}), name + "_P__dP_du__d2P_du2", compilation_directory, checksum);
        });
    }
}

bool SecondOrderCompiledPotential::was_cached() const
{
    return this->cached;
}

spCompiledCondition SecondOrderCompiledPotential::get_condition() const
{
    return this->condition;
}

void SecondOrderCompiledPotential::evaluate_P(Assembly &assembly)
{
    // Evaluate element condition
    this->condition->evaluate(assembly.n_threads);

    // Evaluate potential
    this->P.run(assembly.n_threads,
        [&assembly, this](const View<double> sol, int32_t element_idx, int32_t thread_id, const View<int32_t> conn)
        {
			assembly.E.local(thread_id) += sol[0];
        },
        this->condition->get_element_condition()
    );
}

void SecondOrderCompiledPotential::evaluate_P__dP_du(Assembly &assembly)
{
    // Evaluate element condition
    this->condition->evaluate(assembly.n_threads);

    // Evaluate potential and gradient
    this->P__dP_du.run(assembly.n_threads,
        [&assembly, this](const View<double> sol, int32_t element_idx, int32_t thread_id, const View<int32_t> conn)
        {
            constexpr int BLOCK_SIZE = ElementHessians::BLOCK_SIZE;
            const int n_block_rows = this->n_dofs / BLOCK_SIZE;

            // Energy
			assembly.E.local(thread_id) += sol[0];

            // Gradient
			Eigen::VectorXd& grad = assembly.grad.local(thread_id);
			const View<double> grad_loc = sol.slice(1);

			for (int block_i = 0; block_i < n_block_rows; block_i++) {
				const DoFInConn& map = this->dof_in_conn[block_i];
				const int begin_glob = assembly.dof_set_offsets[map.dof_set] + conn[map.conn_idx] * BLOCK_SIZE;

				const int begin_loc = BLOCK_SIZE * block_i;
				for (int i = 0; i < BLOCK_SIZE; i++) {
					grad[begin_glob + i] += grad_loc[begin_loc + i];
				}
			}
        },
        this->condition->get_element_condition()
    );
}

void SecondOrderCompiledPotential::evaluate_P__dP_du__local_d2P_du2(Assembly &assembly)
{
    // Evaluate element condition
    this->condition->evaluate(assembly.n_threads);
    
    // Evaluate potential, gradient, and Hessian
    this->P__dP_du__d2P_du2.run(assembly.n_threads,
        [&assembly, this](const View<double> sol, int32_t element_idx, int32_t thread_id, const View<int32_t> conn)
        {
            constexpr int BLOCK_SIZE = ElementHessians::BLOCK_SIZE;
            const int n_block_rows = this->n_dofs / BLOCK_SIZE;
            
            // Energy
			assembly.E.local(thread_id) += sol[0];

            // Gradient
			Eigen::VectorXd& grad = assembly.grad.local(thread_id);
            const View<double> grad_loc = sol.slice(1, 1 + this->n_dofs);

            for (int block_i = 0; block_i < n_block_rows; block_i++) {
                const DoFInConn& map = this->dof_in_conn[block_i];
                const int begin_glob = assembly.dof_set_offsets[map.dof_set] + conn[map.conn_idx] * BLOCK_SIZE;

                const int begin_loc = BLOCK_SIZE * block_i;
                for (int i = 0; i < (int)BLOCK_SIZE; i++) {
                    grad[begin_glob + i] += grad_loc[begin_loc + i];
                }
            }

            // Hessian
            View<double> hess_loc = sol.slice(1 + this->n_dofs, sol.size());
            
            // Store element hessian (compute global block indices directly into storage)
            WriteHessian wh = assembly.element_hessians->new_hessian(thread_id, n_block_rows);
            for (int block_i = 0; block_i < n_block_rows; block_i++) {
                const DoFInConn& map = this->dof_in_conn[block_i];
                wh.block_rows[block_i] = assembly.dof_set_offsets[map.dof_set] / BLOCK_SIZE + conn[map.conn_idx];
            }
            std::copy(hess_loc.data(), hess_loc.data() + hess_loc.size(), wh.values);
        },
        this->condition->get_element_condition()
    );
}
