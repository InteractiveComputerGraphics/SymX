#pragma once
#include "../Potential.h"
#include "../../compile/CompiledInLoop.h"
#include "../DeferredParallelTasks.h"
#include "../CompiledCondition.h"

#include <Eigen/Dense>
#include <BlockedSparseMatrix/ParallelVector.h>
#include <BlockedSparseMatrix/ParallelNumber.h>


namespace symx
{
    class AdjointCompiledLoss
    {
    private:
        /* Fields */
        const spMWS<double> mws;
        CompiledInLoop<double> L;                // Loss
        CompiledInLoop<double> dL_du__dL_dp;     // Loss, Gradient w.r.t. DoFs, Gradient w.r.t. Params
        spCompiledCondition condition;           // Conditional
        bool cached = false;

        int n_dofs = -1;
        int n_params = -1;
        std::vector<DoFIndexing> dof_indexing;
        std::vector<DoFIndexing> param_indexing;

    public:
        /* Methods */
        AdjointCompiledLoss(const Potential& loss, const std::vector<DataMap<double>>& dofs, const std::vector<DataMap<double>>& params, const std::string &compilation_directory, DeferredParallelTasks& tasks);
        bool was_cached() const;
		void evaluate_L(bsm::ParallelNumber<double>& parallel_L);
		void evaluate_dL_du__dL_dp(bsm::ParallelVector<Eigen::VectorXd>& parallel_dL_du, bsm::ParallelVector<Eigen::VectorXd>& parallel_dL_dp, const std::vector<int>& dof_offsets, const std::vector<int>& param_offsets);
    };
}
