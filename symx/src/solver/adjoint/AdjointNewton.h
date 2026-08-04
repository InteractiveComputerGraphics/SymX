#pragma once
#include "../second_order/NewtonsMethod.h"
#include "AdjointCompiledPotential.h"
#include "AdjointCompiledLoss.h"

#include <limits>
#include <stdexcept>


namespace symx
{
    struct AdjointValueResult
    {
        SolverReturn status = SolverReturn::Running;
        bool success = false;
        double value = std::numeric_limits<double>::infinity();
    };

    struct AdjointGradientResult
    {
        SolverReturn status = SolverReturn::Running;
        bool success = false;
        Eigen::VectorXd gradient;
    };

    class AdjointNewton
    {
    private:
		/* Fields */
		spGlobalPotential global_potential = nullptr;
        spGlobalPotential global_loss = nullptr;
        spNewtonsMethod newton = nullptr;
		spContext context = nullptr;
        LinearSolveSettings linear_solve_settings;
        std::vector<int> param_indices_to_zero;

        // Compiled
        spSecondOrderCompiledGlobal compiled_newton_derivatives = nullptr;
		std::vector<std::unique_ptr<AdjointCompiledPotential>> compiled_potential;
		std::vector<std::unique_ptr<AdjointCompiledLoss>> compiled_loss;

        // Reusable buffers
        bsm::ParallelNumber<double> parallel_L;
        bsm::ParallelVector<Eigen::VectorXd> parallel_dL_du;
        bsm::ParallelVector<Eigen::VectorXd> parallel_dL_dp;
        bsm::ParallelVector<Eigen::VectorXd> parallel_dot_d2P_dudp_transpose;
        Eigen::VectorXd lambda;
        bsm::PCGContext pcg_context;
        spLinearSolveContext linear_solve_context = nullptr;

    public:
        /* Methods */
        AdjointNewton(spNewtonsMethod newton, spGlobalPotential global_loss, LinearSolveSettings linear_solve_settings = {});
        static std::shared_ptr<AdjointNewton> create(spNewtonsMethod newton, spGlobalPotential global_loss, LinearSolveSettings linear_solve_settings = {});
        void force_gradient_to_zero(const std::vector<int>& param_indices);

        // Solve the inner equilibrium problem before evaluating the requested quantity.
        AdjointValueResult run_forward_and_evaluate_L();
        AdjointGradientResult run_forward_and_evaluate_dL_dp();

        // Evaluate an already-solved equilibrium state.
        AdjointValueResult evaluate_L_no_solve();
        AdjointGradientResult evaluate_dL_dp_no_solve();

        // Parameters and simulation DoFs are restored on success and failure.
        AdjointGradientResult evaluate_dL_dp_finite_differences(double eps);

        // Accessors
        spGlobalPotential get_global_loss()      const { return global_loss; }
        spGlobalPotential get_global_potential() const { return global_potential; }
        spOutputSink      get_newton_output()    const { return newton->get_output(); }
        spNewtonsMethod   get_newton()           const { return newton; }

    private:
        // ---- Internal evaluation helpers (no Newton solve) ----
        double          _evaluate_L();               // Accumulates L over all compiled losses
        SolverReturn _compute_dL_dp(Eigen::VectorXd& dL_dp);
        void _zero_grad_for_fixed_params(Eigen::VectorXd& dL_dp) const;

        void _evaluate_dL_du__dL_dp();
        const Eigen::VectorXd& _evaluate_dot_d2P_dudp_transpose(const Eigen::VectorXd& input);
        SolverReturn _solve_linear_system(const spBSM& d2P_du2, const Eigen::VectorXd& rhs);
    };
    using spAdjointNewton = std::shared_ptr<AdjointNewton>;
}
