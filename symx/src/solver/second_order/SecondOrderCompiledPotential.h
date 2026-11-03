#pragma once
#include "../Potential.h"
#include "../../compile/CompiledInLoop.h"
#include "../DeferredParallelTasks.h"
#include "../CompiledCondition.h"
#include "Assembly.h"

namespace symx
{
    class SecondOrderCompiledPotential
    {
	public:
		enum class FemKernel { None, Direct, Mapped };
	private:
		/* Definitions */
		struct DoFInConn
		{
			int dof_set;  // nth set of degrees of freedom
			int conn_idx;  // nth entry in the local element connectivity
		};

        /* Fields */
        const spMWS<double> mws = nullptr;
        std::vector<DoFInConn> dof_in_conn;
        CompiledInLoop<double> P;                 // Potential
        CompiledInLoop<double> P__dP_du;          // Potential, Gradient
        CompiledInLoop<double> P__dP_du__d2P_du2; // Potential, Gradient, Hessian
        spCompiledCondition condition;            // Conditional
        int32_t n_dofs = -1;
        bool cached = false;
        FemKernel fem_kernel = FemKernel::None;

        
    public:
        /* Methods */
        SecondOrderCompiledPotential(const Potential& potential, const std::vector<DataMap<double>>& dofs, const std::string &compilation_directory, DeferredParallelTasks& tasks);
        bool was_cached() const;
        FemKernel get_fem_kernel() const;
        spCompiledCondition get_condition() const;
        void evaluate_P(Assembly& assembly);
		void evaluate_P__dP_du(Assembly& assembly);
		void evaluate_P__dP_du__local_d2P_du2(Assembly& assembly);
        
    private:
        /* Methods */
		void _initialize_standard(const Potential& potential, const std::vector<Scalar>& dofs,
			const std::string& compilation_directory, DeferredParallelTasks& tasks);
		void _initialize_factorized(const Potential& potential, const std::vector<Scalar>& dofs,
			const std::string& compilation_directory, DeferredParallelTasks& tasks);

    };
} // namespace symx
