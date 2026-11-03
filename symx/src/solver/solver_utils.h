#pragma once

// solver_utils.h — backward-compatible aggregation header.
// Include the focused headers directly for finer-grained dependencies.

#include <memory>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <BlockedSparseMatrix/solve_pcg.h>

#include "Context.h"
#include "SolverCallbacks.h"
#include "second_order/ElementHessians.h"
#include "solver_settings.h"
#include "solver_types.h"

namespace symx
{
    inline std::string in_two_columns(const std::string& str1, const std::string& str2, size_t width)
    {
        std::string result = str1;
        if (str1.length() + str2.length() >= width) {
            result.append(".." + str2);
        }
        else {
            size_t n_dots = width - str1.length() - str2.length();
            for (size_t i = 0; i < n_dots; i++) {
                result.append(".");
            }
        }
        return result + str2;
    }

    class LinearSolveContext
    {
    public:
        LinearSolveContext();
        ~LinearSolveContext();
        LinearSolveContext(const LinearSolveContext&) = delete;
        LinearSolveContext& operator=(const LinearSolveContext&) = delete;

        // Discard cached matrix storage and symbolic-factorisation state.
        void reset();
        bool solve_direct_llt(
            const ElementHessians::spBSM& hess,
            Eigen::VectorXd& x,
            const Eigen::VectorXd& rhs,
            int n_threads);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
    using spLinearSolveContext = std::shared_ptr<LinearSolveContext>;

    bool solve_linear_system_direct_llt(
        const ElementHessians::spBSM& hess,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& rhs);

    bsm::PCGInfo solve_linear_system_bdpcg(
        const ElementHessians::spBSM& hess,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& rhs,
        const LinearSolveSettings& settings,
        int n_threads,
        bsm::PCGContext& pcg_context,
        double residual_norm = 0.0);
}
