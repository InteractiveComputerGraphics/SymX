#include "FirstOrderCompiledPotential.h"

#include "../../symbol/diff.h"

using namespace symx;

FirstOrderCompiledPotential::FirstOrderCompiledPotential(
    const Potential& potential,
    const std::vector<DataMap<double>>& dofs_maps,
    const std::string& compilation_directory,
    DeferredParallelTasks& tasks)
    : mws(potential.get_mws())
{
    std::vector<Scalar> dofs;
    int local_offset = 0;
    for (int dof_set = 0; dof_set < (int)dofs_maps.size(); ++dof_set) {
        const std::vector<DataMap<const double>> original_maps = this->mws->get_original_maps(dofs_maps[dof_set]);
        const std::vector<Scalar> set_dofs = this->mws->get_symbols(dofs_maps[dof_set]);
        dofs.insert(dofs.end(), set_dofs.begin(), set_dofs.end());

        for (const DataMap<const double>& dof_map : original_maps) {
            DoFInConn map;
            map.dof_set = dof_set;
            map.conn_idx = dof_map.connectivity_index;
            map.stride = dof_map.stride;
            map.local_offset = local_offset;
            this->dof_in_conn.push_back(map);
            local_offset += dof_map.stride;
        }
    }

    this->n_dofs = (int32_t)dofs.size();
    if (this->n_dofs == 0) {
        std::cout << "symx error: FirstOrderCompiledPotential() got potential with zero DoFs.\n";
        std::cout << "Potential: " << potential.get_name() << std::endl;
        exit(-1);
    }

    std::string pre_hash = potential.get_name();
    if (potential.is_factorized()) { pre_hash += "_factorized"; }
    for (const Scalar& dof : dofs) {
        pre_hash += std::to_string(dof.get_symbol_idx());
    }
    std::string checksum = potential.get_checksum(pre_hash);
    if (potential.has_conditional()) {
        checksum += potential.get_condition().get_checksum();
    }
    // Both solver orders may coexist for the same potential. Keep their artifacts
    // separate: their checksums can differ after symbolic differentiation.
    this->condition = std::make_shared<CompiledCondition>(potential, checksum, compilation_directory, tasks, "_first_order");

    const std::string name = potential.get_name() + "_first_order";
    this->cached = this->P.load_if_cached(this->mws, name + "_P", compilation_directory, checksum);
    this->cached &= this->P__dP_du.load_if_cached(this->mws, name + "_P__dP_du", compilation_directory, checksum);
    this->cached &= this->condition->was_cached();

    if (!this->cached) {
        const Scalar value = potential.get_expression();
        DiffCache diff_cache;
        const Vector gradient_value = gradient(value, dofs, diff_cache);

        tasks.add([this, value, name, compilation_directory, checksum]() {
            this->P.compile(this->mws, { value }, name + "_P", compilation_directory, checksum);
        });
        tasks.add([this, value, gradient_values = gradient_value.values(), name, compilation_directory, checksum]() {
            this->P__dP_du.compile(
                this->mws, collect_scalars({{ value }, gradient_values}), name + "_P__dP_du", compilation_directory, checksum);
        });
    }
}

bool FirstOrderCompiledPotential::was_cached() const
{
    return this->cached;
}

void FirstOrderCompiledPotential::evaluate_P(Assembly& assembly)
{
    this->condition->evaluate(assembly.n_threads);
    this->P.run(assembly.n_threads,
        [&assembly](const View<double> solution, int32_t, int32_t thread_id, const View<int32_t>) {
			assembly.E.local(thread_id) += solution[0];
        },
        this->condition->get_element_condition());
}

void FirstOrderCompiledPotential::evaluate_P__dP_du(Assembly& assembly)
{
    this->condition->evaluate(assembly.n_threads);
    this->P__dP_du.run(assembly.n_threads,
        [&assembly, this](const View<double> solution, int32_t, int32_t thread_id, const View<int32_t> conn) {
			assembly.E.local(thread_id) += solution[0];
			Eigen::VectorXd& gradient = assembly.grad.local(thread_id);
            const View<double> local_gradient = solution.slice(1);

            for (const DoFInConn& map : this->dof_in_conn) {
                const int global_offset = assembly.dof_set_offsets[map.dof_set]
                    + ((map.conn_idx >= 0) ? conn[map.conn_idx] * map.stride : 0);
                for (int i = 0; i < map.stride; ++i) {
                    gradient[global_offset + i] += local_gradient[map.local_offset + i];
                }
            }
        },
        this->condition->get_element_condition());
}
