#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include <Eigen/Dense>

#include "../solver_settings.h"

namespace symx
{
    // Shared state and direction generation for direct and adjoint first-order
    // optimizers. Evaluation, line search, callbacks, and output deliberately
    // remain in the respective optimizer facades.
    class FirstOrderAlgorithm
    {
    private:
        int adam_t = 0;
        Eigen::VectorXd adam_m;
        Eigen::VectorXd adam_v;

        int vector_adam_t = 0;
        Eigen::VectorXd vector_adam_m;
        Eigen::VectorXd vector_adam_v;
        std::vector<int> vector_adam_block_offsets;

        std::vector<double> lbfgs_alpha;
        std::deque<Eigen::VectorXd> lbfgs_s;
        std::deque<Eigen::VectorXd> lbfgs_y;
        std::deque<double> lbfgs_rho;
        int lbfgs_history_skip_count = 0;

        Eigen::VectorXd direction;

    public:
        void prepare(int n_variables,
                     const std::vector<int32_t>& dof_set_offsets,
                     const std::vector<int32_t>& dof_set_sizes,
                     const FirstOrderSettings& settings);
        void reset(int n_variables);

        const Eigen::VectorXd& compute_direction(const Eigen::VectorXd& gradient,
                                                 const FirstOrderSettings& settings);
        Eigen::VectorXd default_direction(const Eigen::VectorXd& gradient,
                                          const FirstOrderSettings& settings) const;

        // Returns true when the curvature pair was retained.
        bool update_lbfgs_history(const Eigen::VectorXd& s,
                                  const Eigen::VectorXd& y,
                                  const FirstOrderSettings& settings);
        void reset_lbfgs_history();
        int get_lbfgs_history_skip_count() const;

    private:
        void _prepare_vector_adam_blocks(const std::vector<int32_t>& dof_set_offsets,
                                         const std::vector<int32_t>& dof_set_sizes,
                                         const FirstOrderSettings& settings);
        void _compute_gradient_descent(const Eigen::VectorXd& gradient,
                                       const FirstOrderSettings& settings);
        void _compute_adam(const Eigen::VectorXd& gradient,
                           const FirstOrderSettings& settings);
        void _compute_vector_adam(const Eigen::VectorXd& gradient,
                                  const FirstOrderSettings& settings);
        void _compute_lbfgs(const Eigen::VectorXd& gradient,
                            const FirstOrderSettings& settings);
    };
}
