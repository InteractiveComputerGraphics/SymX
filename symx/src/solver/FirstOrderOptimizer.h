#pragma once

#include <memory>

#include "GlobalPotential.h"
#include "first_order/FirstOrderAlgorithm.h"
#include "first_order/FirstOrderCompiledGlobal.h"
#include "solver_utils.h"

namespace symx
{
    // Direct first-order minimizer for a GlobalPotential. It shares only
    // algorithm state with FirstOrderOptimizerAdjoint; evaluation and line
    // search remain direct because no nested equilibrium solve is involved.
    class FirstOrderOptimizer
    {
    public:
        struct SolveStats
        {
            int first_order_iterations = 0;
            int ls_cap_iterations = 0;
            int ls_max_iterations = 0;
            int ls_inv_iterations = 0;
            int ls_bt_iterations = 0;
            int lbfgs_history_skips = 0;
        };

    private:
        spFirstOrderCompiledGlobal compiled;
        spGlobalPotential global_potential;
        spContext context;
        spOutputSink output;
        spLogger logger;

        Eigen::VectorXd dofs;
        Eigen::VectorXd dofs_before_ls;
        Eigen::VectorXd dofs_ls;
        Eigen::VectorXd gradient;
        Eigen::VectorXd direction;
        FirstOrderAlgorithm algorithm;
        SolveStats stats;

    public:
        spSolverCallbacks callbacks;
        FirstOrderSettings settings;

        FirstOrderOptimizer(spGlobalPotential global_potential,
                            spContext context,
                            spSolverCallbacks callbacks = nullptr);
        static std::shared_ptr<FirstOrderOptimizer> create(
            spGlobalPotential global_potential,
            spContext context,
            spSolverCallbacks callbacks = nullptr);

        SolverReturn solve();
        const SolveStats& get_last_solve_stats() const { return this->stats; }
        void print_summary(double total_time = -1.0) const;

    private:
        SolverReturn _line_search_inplace(double energy,
                                          double gradient_dot_direction,
                                          double direction_max);
    };

    using spFirstOrderOptimizer = std::shared_ptr<FirstOrderOptimizer>;
}
