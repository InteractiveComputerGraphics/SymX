#pragma once
#include <functional>
#include <algorithm>
#include <vector>

#include <Eigen/Dense>

#include "solver_types.h"
#include "Context.h"

namespace symx
{
    static auto default_residual = [](const Eigen::VectorXd& r) { return r.cwiseAbs().maxCoeff(); };

    /*
        Holds user-registered callbacks that the Newton solver invokes
        at well-defined points during the solve. Callbacks are registered
        via add_*() and run via run_*() (called internally by the solver).
    */
    class SolverCallbacks
    {
    private:
        /* Fields */
        std::vector<std::function<void()>>     before_energy_evaluation;
        std::vector<std::function<void()>>     before_step;
        std::vector<std::function<void()>>     after_step;
        std::vector<std::function<bool()>>     is_initial_state_valid;
        std::vector<std::function<bool()>>     is_intermediate_state_valid;
        std::vector<std::function<void()>>     on_intermediate_state_invalid;
        std::vector<std::function<void()>>     on_armijo_fail;
        std::vector<std::function<bool()>>     is_converged;
        std::vector<std::function<bool()>>     is_converged_state_valid;
        std::vector<std::function<double()>>   max_allowed_step;
        std::function<double(const Eigen::VectorXd&)> residual = default_residual;
        spContext context = nullptr;

        /* Internal helpers */
        void _run(const std::vector<std::function<void()>>& fs)
        {
            for (auto& f : fs) {
                f();
            }
        }
        bool _run_bool(bool default_bool, const std::vector<std::function<bool()>>& fs)
        {
            bool valid = default_bool;
            for (auto& f : fs) {
                valid = valid && f();
            }
            return valid;
        }

    public:
        /* Construction */
        SolverCallbacks(spContext context) : context(context) {}
        static std::shared_ptr<SolverCallbacks> create(spContext context) { return std::make_shared<SolverCallbacks>(context); }

        // ---- Register callbacks ----

        void set_residual(std::function<double(const Eigen::VectorXd&)> f) { this->residual = f; }

        /// Called once per Newton iteration, before energy/gradient/Hessian evaluation.
        void add_before_energy_evaluation(std::function<void()> f) { this->before_energy_evaluation.push_back(f); }

        /// Called at the very start of each Newton iteration (iteration index is 0-based).
        void add_before_step(std::function<void()> f) { this->before_step.push_back(f); }

        /// Called at the end of each Newton iteration (including failed/early-exit iterations).
        void add_after_step(std::function<void()> f) { this->after_step.push_back(f); }

        void add_is_initial_state_valid(std::function<bool()> f) { this->is_initial_state_valid.push_back(f); }
        void add_is_intermediate_state_valid(std::function<bool()> f) { this->is_intermediate_state_valid.push_back(f); }
        void add_on_intermediate_state_invalid(std::function<void()> f) { this->on_intermediate_state_invalid.push_back(f); }
        void add_on_armijo_fail(std::function<void()> f) { this->on_armijo_fail.push_back(f); }
        void add_is_converged(std::function<bool()> f) { this->is_converged.push_back(f); }
        void add_is_converged_state_valid(std::function<bool()> f) { this->is_converged_state_valid.push_back(f); }
        void add_max_allowed_step(std::function<double()> f) { this->max_allowed_step.push_back(f); }

        // ---- Invoke callbacks (called by the solver) ----

        void run_before_energy_evaluation()
        {
            auto _t = this->context->logger->time("before_energy_evaluation");
            this->_run(this->before_energy_evaluation);
        }
        void run_before_step()
        {
            auto _t = this->context->logger->time("before_step");
            this->_run(this->before_step);
        }
        void run_after_step()
        {
            auto _t = this->context->logger->time("after_step");
            this->_run(this->after_step);
        }
        bool run_is_initial_state_valid()
        {
            auto _t = this->context->logger->time("is_initial_state_valid");
            return this->_run_bool(true, this->is_initial_state_valid);
        }
        bool run_is_intermediate_state_valid()
        {
            auto _t = this->context->logger->time("is_intermediate_state_valid");
            return this->_run_bool(true, this->is_intermediate_state_valid);
        }
        void run_on_intermediate_state_invalid()
        {
            auto _t = this->context->logger->time("on_intermediate_state_invalid");
            this->_run(this->on_intermediate_state_invalid);
        }
        void run_on_armijo_fail()
        {
            auto _t = this->context->logger->time("on_armijo_fail");
            this->_run(this->on_armijo_fail);
        }
        bool run_is_converged()
        {
            auto _t = this->context->logger->time("is_converged");
            bool converged = false;
            for (auto& f : this->is_converged) {
                converged = f() || converged;
            }
            return converged;
        }
        bool run_is_converged_state_valid()
        {
            auto _t = this->context->logger->time("is_converged_state_valid");
            return this->_run_bool(true, this->is_converged_state_valid);
        }
        double run_max_allowed_step()
        {
            auto _t = this->context->logger->time("max_allowed_step");
            double max_step = 1.0;
            for (auto f : this->max_allowed_step) {
                max_step = std::min(max_step, f());
            }
            return max_step;
        }

        // ---- Residual computation ----
        double compute_residual(const Eigen::VectorXd& r) { return this->residual(r); }
    };

    using spSolverCallbacks = std::shared_ptr<SolverCallbacks>;
}
