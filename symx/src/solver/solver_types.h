#pragma once
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace symx
{
    // ================================================================
    // Solver result
    // ================================================================
    enum class SolverReturn
    {
        Successful,
        Running,
        InvalidInitialState,
        TooManyIterations,
        TooManyArmijoIterations,
        LinearSystemSolveFailure,
        TooManyInvalidIntermediateIterations,
        StepDoesNotDescend,
        InvalidConvergedState,
    };

    // ================================================================
    // Linear solver selection
    // ================================================================
    enum class LinearSolver
    {
        DirectLLT,
        BDPCG,  // Block Diagonal Preconditioned Conjugate Gradient
    };
    inline std::string to_string(symx::LinearSolver v)
    {
        switch (v)
        {
        case symx::LinearSolver::BDPCG: return "BDPCG"; break;
        case symx::LinearSolver::DirectLLT: return "DirectLLT"; break;
        default:
            std::cout << "symx::LinearSolver " << (int)v << " does not have a name. Exiting." << std::endl;
            exit(-1);
        }
    }

    // ================================================================
    // Hessian projection strategy
    // ================================================================
    enum class ProjectionToPD
    {
        Newton,           // Pure Newton: no projection
        ProjectedNewton,  // Always project all element Hessians to PD before assembly
        ProjectOnDemand,  // Project only when linear system fails or search direction does not descend
        Progressive,      // PPN: Progressively project based on gradient magnitude threshold
    };
    inline std::string to_string(symx::ProjectionToPD v)
    {
        switch (v)
        {
        case symx::ProjectionToPD::Newton: return "Newton"; break;
        case symx::ProjectionToPD::ProjectedNewton: return "ProjectedNewton"; break;
        case symx::ProjectionToPD::ProjectOnDemand: return "ProjectOnDemand"; break;
        case symx::ProjectionToPD::Progressive: return "Progressive"; break;
        default:
            std::cout << "symx::ProjectionToPD " << (int)v << " does not have a name. Exiting." << std::endl;
            exit(-1);
        }
    }

    // ================================================================
    // Formatting helpers (used by SolverSettings::as_string and other consumers)
    // ================================================================
    inline std::string to_string(bool v) { return v ? "true" : "false"; }
    inline std::string to_string_sci(double v, int prec = 1)
    {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(prec) << v;
        return ss.str();
    }
    inline std::string to_string_fixed(double v, int prec = 2)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(prec) << v;
        return ss.str();
    }
}
