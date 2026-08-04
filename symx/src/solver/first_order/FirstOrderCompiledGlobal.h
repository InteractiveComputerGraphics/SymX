#pragma once

#include "FirstOrderCompiledPotential.h"
#include "../Context.h"
#include "../GlobalPotential.h"
#include "../second_order/Assembly.h"

namespace symx
{
    class FirstOrderCompiledGlobal
    {
    private:
        std::vector<std::unique_ptr<FirstOrderCompiledPotential>> compiled_potentials;
        spGlobalPotential global_potential;
        spContext context;
        Assembly assembly;

    public:
        FirstOrderCompiledGlobal(spGlobalPotential global_potential, spContext context);
        void evaluate_P(double& out_P);
        void evaluate_P__dP_du(double& out_P, Eigen::VectorXd& out_dP_du);
    };
    using spFirstOrderCompiledGlobal = std::shared_ptr<FirstOrderCompiledGlobal>;
}
