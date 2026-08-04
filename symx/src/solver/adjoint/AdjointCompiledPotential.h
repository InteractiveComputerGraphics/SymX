#pragma once
#include "../Potential.h"
#include "../../compile/CompiledInLoop.h"
#include "../DeferredParallelTasks.h"
#include "../CompiledCondition.h"

#include <Eigen/Dense>
#include <BlockedSparseMatrix/ParallelVector.h>


namespace symx
{
    /*
    * Potential, conditional, gradient and Hessian assumed available in SecondOrderCompiledGlobal.
    */
    class AdjointCompiledPotential
    {
    private:
        /* Fields */
        const spMWS<double> mws;
        CompiledInLoop<double> d2P_dpdu;
        spCompiledCondition condition;
        bool cached = false;

        int n_dofs = -1;
        int n_params = -1;
        std::vector<DoFIndexing> dof_indexing;
        std::vector<DoFIndexing> param_indexing;

    public:
        /* Methods */
        AdjointCompiledPotential(const Potential& potential, spCompiledCondition condition, const std::vector<DataMap<double>>& dofs, const std::vector<DataMap<double>>& params, const std::string &compilation_directory, DeferredParallelTasks& tasks);
		void evaluate_d2P_dpdu_dot(bsm::ParallelVector<Eigen::VectorXd>& parallel_result, const Eigen::VectorXd& input, const std::vector<int>& dof_offsets, const std::vector<int>& param_offsets);
        bool was_cached() const;
    };
}
