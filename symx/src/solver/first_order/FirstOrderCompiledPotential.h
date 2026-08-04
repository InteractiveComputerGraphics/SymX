#pragma once

#include "../CompiledCondition.h"
#include "../DeferredParallelTasks.h"
#include "../Potential.h"
#include "../second_order/Assembly.h"
#include "../../compile/CompiledInLoop.h"

namespace symx
{
    class FirstOrderCompiledPotential
    {
    private:
        struct DoFInConn
        {
            int dof_set = -1;
            int conn_idx = -1;
            int stride = -1;
            int local_offset = -1;
        };

        const spMWS<double> mws;
        std::vector<DoFInConn> dof_in_conn;
        CompiledInLoop<double> P;
        CompiledInLoop<double> P__dP_du;
        spCompiledCondition condition;
        int32_t n_dofs = -1;
        bool cached = false;

    public:
        FirstOrderCompiledPotential(const Potential& potential,
                                    const std::vector<DataMap<double>>& dofs_maps,
                                    const std::string& compilation_directory,
                                    DeferredParallelTasks& tasks);

        bool was_cached() const;
        void evaluate_P(Assembly& assembly);
        void evaluate_P__dP_du(Assembly& assembly);
    };
}
