#pragma once

// solver_utils.h — backward-compatible aggregation header.
// The solver internals have been split into focused headers:
//   SolverTypes.h     — SolverReturn, LinearSolver, ProjectionToPD enums + string helpers
//   SolverCallbacks.h — SolverCallbacks class and spSolverCallbacks alias
//   SolverSettings.h  — SolverSettings and NewtonSettings structs
// Include any of those directly for finer-grained dependencies, or include
// this file to get everything at once (existing code remains unchanged).

#include "solver_types.h"
#include "SolverCallbacks.h"
#include "solver_settings.h"

namespace symx
{
    // ================================================================
    // Miscellaneous formatting utility
    // ================================================================

    /// Format two strings in two columns separated by dots, filling to `width` total characters.
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
}

