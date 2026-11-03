#include "SecondOrderCompiledPotential.h"

#include "../../symbol/diff.h"
#include "project_to_PD.h"
#include <unordered_map>
#include <unordered_set>

using namespace symx;

namespace
{
    // Inspect only expressions supplied by the FEM split; the workspace graph also contains
    // unrelated expressions and the temporary independent F symbols.
    std::vector<int32_t> referenced_state_nodes(
        const Scalar& scalar, const std::unordered_map<int32_t, int32_t>& state_nodes)
    {
        const auto& graph = scalar.get_expression_graph()->get_expressions();
        std::vector<int32_t> stack{scalar.expr_id};
        std::unordered_set<int32_t> seen;
        std::vector<int32_t> nodes;
        while (!stack.empty()) {
            const int32_t id = stack.back();
            stack.pop_back();
            if (!seen.insert(id).second) { continue; }
            const Expr& expr = graph[id];
            if (expr.type == ExprType::Symbol) {
                const auto it = state_nodes.find(expr.a);
                if (it != state_nodes.end()) { nodes.push_back(it->second); }
            }
            else if (expr.type == ExprType::Branch) {
                stack.push_back(expr.cond);
                stack.push_back(expr.a);
                stack.push_back(expr.b);
            }
            else if (expr.is_operation()) {
                stack.push_back(expr.a);
                if (expr.type == ExprType::Add || expr.type == ExprType::Sub ||
                    expr.type == ExprType::Mul || expr.type == ExprType::PowF) {
                    stack.push_back(expr.b);
                }
            }
        }
        return nodes;
    }
}

SecondOrderCompiledPotential::SecondOrderCompiledPotential(const Potential &potential, const std::vector<DataMap<double>> &dofs_maps, const std::string &compilation_directory, DeferredParallelTasks& tasks)
    : mws(potential.get_mws())
{
    // Shared setup: collect local DoF symbols and their connectivity indices.
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

    // Keep the standard SymX path independent from the specialized FEM path.
    if (potential.is_factorized()) {
        this->_initialize_factorized(potential, dofs, compilation_directory, tasks);
    }
    else {
        this->_initialize_standard(potential, dofs, compilation_directory, tasks);
    }
}

void SecondOrderCompiledPotential::_initialize_standard(const Potential& potential, const std::vector<Scalar>& dofs,
    const std::string& compilation_directory, DeferredParallelTasks& tasks)
{
    // Standard path: checksum and load first; differentiate and queue compilation on a miss.
    const Scalar v = potential.get_expression();
    std::string pre_hash = potential.get_name();
    for (const Scalar& dof : dofs) { pre_hash += std::to_string(dof.get_symbol_idx()); }
    std::string checksum = potential.get_checksum(pre_hash);
    if (potential.has_conditional()) { checksum += potential.get_condition().get_checksum(); }
    const std::string name = potential.get_name();
    this->condition = std::make_shared<CompiledCondition>(
        potential, checksum, compilation_directory, tasks);

    this->cached = this->P.load_if_cached(this->mws, name + "_P", compilation_directory, checksum);
    this->cached &= this->P__dP_du.load_if_cached(this->mws, name + "_P__dP_du", compilation_directory, checksum);
    this->cached &= this->P__dP_du__d2P_du2.load_if_cached(this->mws, name + "_P__dP_du__d2P_du2", compilation_directory, checksum);
    this->cached &= this->condition->was_cached();

    if (!this->cached) {
        tasks.add([this, v, name, compilation_directory, checksum]() {
            this->P.compile(this->mws, {v}, name + "_P", compilation_directory, checksum);
        });
        DiffCache diff_cache;
        const Vector g = gradient(v, dofs, diff_cache);
        const Matrix h = gradient(g, dofs, /*symmetric=*/true, diff_cache);
        tasks.add([this, v, g_vals = g.values(), name, compilation_directory, checksum]() {
            this->P__dP_du.compile(this->mws, collect_scalars({{v}, g_vals}),
                name + "_P__dP_du", compilation_directory, checksum);
        });
        tasks.add([this, v, g_vals = g.values(), h_vals = h.values(), name, compilation_directory, checksum]() {
            this->P__dP_du__d2P_du2.compile(this->mws, collect_scalars({{v}, g_vals, h_vals}),
                name + "_P__dP_du__d2P_du2", compilation_directory, checksum);
        });
    }
}

void SecondOrderCompiledPotential::_initialize_factorized(const Potential& potential, const std::vector<Scalar>& dofs,
    const std::string& compilation_directory, DeferredParallelTasks& tasks)
{
    // Validate the split and node-local map before selecting a cache entry.
    const FactorizationData& fac = potential.get_factorization();
    const int n_nodes = fac.B.rows();
    const int d_ref = fac.B.cols();
    if (n_nodes <= 0 || d_ref <= 0 || fac.x.rows() != n_nodes || fac.x.cols() != 3 ||
        n_nodes * 3 != this->n_dofs || fac.P_dof_w.size() != 3 * d_ref ||
        fac.A_dof_w.rows() != fac.P_dof_w.size() || fac.A_dof_w.cols() != fac.P_dof_w.size()) {
        std::cout << "symx error: factorized potential '" << potential.get_name()
                  << "' has inconsistent local dimensions." << std::endl;
        exit(-1);
    }

    std::unordered_map<int32_t, int32_t> state_nodes;
    for (int i = 0; i < this->n_dofs; i++) {
        if (!state_nodes.emplace(dofs[i].get_symbol_idx(), i / 3).second) {
            std::cout << "symx error: factorized potential has repeated local DoF symbols." << std::endl;
            exit(-1);
        }
    }
    for (const Scalar& b : fac.B.values()) {
        if (!referenced_state_nodes(b, state_nodes).empty()) {
            std::cout << "symx error: factorized potential B depends on state DoFs." << std::endl;
            exit(-1);
        }
    }
    if (!referenced_state_nodes(fac.material_unsubstituted, state_nodes).empty()) {
        std::cout << "symx error: factorized material depends on state DoFs outside F." << std::endl;
        exit(-1);
    }

    DiffCache map_cache;
    std::vector<Scalar> jacobian_values;
    jacobian_values.reserve(n_nodes * 9);
    bool identity = true;
    for (int l = 0; l < n_nodes; l++) {
        for (int a = 0; a < 3; a++) {
            const Scalar& x = fac.x(l, a);
            for (int state_node : referenced_state_nodes(x, state_nodes)) {
                if (state_node != l) {
                    std::cout << "symx error: factorized node map depends on another node's DoFs." << std::endl;
                    exit(-1);
                }
            }
            for (int i = 0; i < 3; i++) {
                const Scalar deriv = diff(x, dofs[3 * l + i], map_cache);
                identity &= (a == i) ? deriv.is_one() : deriv.is_zero();
                jacobian_values.push_back(deriv);
            }
        }
    }
    this->fem_kernel = identity ? FemKernel::Direct : FemKernel::Mapped;

    // Include the selected kernel in the checksum while keeping artifact names user-readable.
    const std::string algorithm_tag = identity ? "_fem_direct" : "_fem_mapped";
    std::string pre_hash = potential.get_name() + algorithm_tag;
    for (const Scalar& dof : dofs) { pre_hash += std::to_string(dof.get_symbol_idx()); }
    std::string checksum = potential.get_checksum(pre_hash);
    if (potential.has_conditional()) { checksum += potential.get_condition().get_checksum(); }
    const Scalar v = potential.get_expression();
    const std::string name = potential.get_name();
    this->condition = std::make_shared<CompiledCondition>(potential, checksum, compilation_directory, tasks);

    this->cached = this->P.load_if_cached(this->mws, name + "_P", compilation_directory, checksum);
    this->cached &= this->P__dP_du.load_if_cached(this->mws, name + "_P__dP_du", compilation_directory, checksum);
    this->cached &= this->P__dP_du__d2P_du2.load_if_cached(this->mws, name + "_P__dP_du__d2P_du2", compilation_directory, checksum);
    this->cached &= this->condition->was_cached();
    if (this->cached) { return; }

    // Cache miss: build the compact gradient and, for mapped FEM, geometric stiffness.
    std::vector<Scalar> compact_gradient;
    std::vector<Scalar> map_hessian_values;
    if (identity) {
        compact_gradient.reserve(n_nodes * 3);
        for (int l = 0; l < n_nodes; l++) {
            for (int a = 0; a < 3; a++) {
                Scalar acc = fac.B(l, 0) * fac.P_dof_w[d_ref * a];
                for (int b = 1; b < d_ref; b++) {
                    acc = acc + fac.B(l, b) * fac.P_dof_w[d_ref * a + b];
                }
                compact_gradient.push_back(acc);
            }
        }
    }
    else {
        compact_gradient.reserve(n_nodes * 3);
        map_hessian_values.reserve(n_nodes * 27);
        for (int l = 0; l < n_nodes; l++) {
            Scalar q[3] = {
                fac.B(l, 0) * fac.P_dof_w[0],
                fac.B(l, 0) * fac.P_dof_w[d_ref],
                fac.B(l, 0) * fac.P_dof_w[2 * d_ref]
            };
            for (int a = 0; a < 3; a++) {
                for (int b = 1; b < d_ref; b++) {
                    q[a] = q[a] + fac.B(l, b) * fac.P_dof_w[d_ref * a + b];
                }
            }
            for (int a = 0; a < 3; a++) {
                for (int i = 0; i < 3; i++) {
                    for (int ip = 0; ip < 3; ip++) {
                        map_hessian_values.push_back(diff(jacobian_values[l * 9 + a * 3 + i], dofs[3 * l + ip], map_cache));
                    }
                }
            }
            for (int i = 0; i < 3; i++) {
                Scalar acc = jacobian_values[l * 9 + i] * q[0];
                for (int a = 1; a < 3; a++) {
                    acc = acc + jacobian_values[l * 9 + a * 3 + i] * q[a];
                }
                compact_gradient.push_back(acc);
            }
        }
    }

    // Preserve independent compilation tasks and capture all symbolic inputs by value.
    tasks.add([this, v, name, compilation_directory, checksum]() {
        this->P.compile(this->mws, {v}, name + "_P", compilation_directory, checksum);
    });
    tasks.add([this, v, compact_gradient, name, compilation_directory, checksum]() {
        this->P__dP_du.compile(this->mws, collect_scalars({{v}, compact_gradient}),
            name + "_P__dP_du", compilation_directory, checksum);
    });
    if (identity) {
        tasks.add([this, energy = v, P = fac.P_dof_w, A = fac.A_dof_w, B = fac.B,
            name, compilation_directory, checksum]() {
            this->P__dP_du__d2P_du2.compile_factorized(this->mws, energy, P, A, B,
                name + "_P__dP_du__d2P_du2", compilation_directory, checksum);
        });
    }
    else {
        Matrix J(jacobian_values, {3 * n_nodes, 3});
        Matrix Hgeo(map_hessian_values, {9 * n_nodes, 3});
        tasks.add([this, energy = v, P = fac.P_dof_w, A = fac.A_dof_w, B = fac.B, J, Hgeo,
            name, compilation_directory, checksum]() {
            this->P__dP_du__d2P_du2.compile_factorized_mapped(this->mws, energy, P, A, B, J, Hgeo,
                name + "_P__dP_du__d2P_du2", compilation_directory, checksum);
        });
    }
}

bool SecondOrderCompiledPotential::was_cached() const
{
    return this->cached;
}

SecondOrderCompiledPotential::FemKernel SecondOrderCompiledPotential::get_fem_kernel() const
{
    return this->fem_kernel;
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
