#pragma once
#include "AdjointNewton.h"
#include "../first_order/FirstOrderAlgorithm.h"

namespace symx
{
    /*
     * First-order (and quasi-second-order) optimizer for loss minimisation via AdjointNewton.
     *
     * Supported algorithms (FirstOrderOptType):
     *   GradientDescent – fixed step −lr·g.
     *   Adam            – adaptive-moment optimiser.
     *   VectorAdam      – rotation-aware adaptive moments for vector blocks.
     *   LBFGS           – limited-memory BFGS.
     *
     * At every outer iteration the optimizer:
     *   1. Evaluates L and dL/dp via AdjointNewton (one inner Newton solve + adjoint).
     *   2. Computes a descent direction dp.
     *   3. Updates the parameter vector and repeats.
     *
     * Newton warm-starting: equilibrium DOFs from the previous iteration are reused
     * as the initial guess, reducing the inner Newton cost.
     */
    class FirstOrderOptimizerAdjoint
    {
    public:
        struct SolveStats
        {
            int opt_iterations    = 0;
            int ls_cap_iterations = 0;  // outer step-cap applications
            int ls_max_iterations = 0;  // outer max-step callback activations
            int ls_inv_iterations = 0;  // outer invalid-state backtrack steps
            int ls_nf_iterations  = 0;  // outer Newton-fail backtrack steps
            int ls_bt_iterations  = 0;  // outer Armijo backtrack steps
            inline int all_ls() const {
                return ls_cap_iterations + ls_max_iterations + ls_inv_iterations
                     + ls_nf_iterations  + ls_bt_iterations;
            }
        };

    private:
        /* Fields */
        spAdjointNewton adjoint = nullptr;
        spGlobalPotential global_loss= nullptr;
        spGlobalPotential global_potential = nullptr;
        spNewtonsMethod newton_solver =  nullptr;
        spContext context = nullptr;
        spOutputSink output  = nullptr;
        spLogger logger  = nullptr;

        // Reusable parameter buffers
        Eigen::VectorXd params;          // current parameter vector
        Eigen::VectorXd grad;            // current gradient dL/dp
        Eigen::VectorXd dp;              // descent direction (may be pre-scaled)
        Eigen::VectorXd last_safe_dofs;

        // Output control state (set at start of solve(), shared with _line_search())
        bool _suppress_inner       = false;

        FirstOrderAlgorithm algorithm;

        // Misc
        SolveStats stats;

    public:
        /* Fields */
        // These callbacks govern outer parameter optimization. Callbacks owned by
        // AdjointNewton's Newton solver remain exclusively inner-solver hooks.
        spSolverCallbacks    callbacks = nullptr;
        FirstOrderSettings settings;
        bool suppress_inner_newton_output = true;

        /* Methods */
        FirstOrderOptimizerAdjoint(spAdjointNewton adjoint, spContext context, spSolverCallbacks callbacks = nullptr);
        static std::shared_ptr<FirstOrderOptimizerAdjoint> create(spAdjointNewton adjoint, spContext context, spSolverCallbacks callbacks = nullptr);
        SolverReturn solve();
        void print_summary() const;
        const SolveStats& get_last_solve_stats() const { return stats; }

    private:
        // Outer line search: step cap → max-step callback → invalid-state backtracking
        //   → inner Newton solve → Newton-fail backtracking → Armijo backtracking.
        // On entry:  L = current loss; this->dp = descent direction.
        // On return: this->params and this->grad are at the accepted point; L is updated.
        SolverReturn _line_search(double L_prev,
                                  const Eigen::VectorXd& params_prev,
                                  const Eigen::VectorXd& grad_at_prev);
    };
    using spFirstOrderOptimizerAdjoint = std::shared_ptr<FirstOrderOptimizerAdjoint>;
}
