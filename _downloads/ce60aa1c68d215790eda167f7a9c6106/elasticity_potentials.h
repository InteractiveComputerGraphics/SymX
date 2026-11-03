#pragma once

#include <symx>
using namespace symx;


inline Scalar neohookean_energy_density(const Matrix& F, const Scalar& l, const Scalar& m)
{
    Matrix C = F.transpose()*F;
    Scalar Ic = C.trace();
    Scalar logdetF = log(F.det());
    return 0.5*m*(Ic - 3) - m*logdetF + 0.5*l*logdetF.powN(2);
}

inline Scalar stable_neohookean_energy_density(const Matrix& F, const Scalar youngs_modulus, const Scalar poissons_ratio)
{
    Scalar E = youngs_modulus;
    Scalar nu = poissons_ratio;
    Scalar mu = E/(2.0*(1.0 + nu));
    Scalar lambda = (E*nu)/((1.0 + nu)*(1.0 - 2.0*nu));
    Scalar mu_ = 4.0/3.0*mu;
    Scalar lambda_ = lambda + 5.0/6.0*mu;
    Scalar alpha = 1.0 + mu_/lambda_ - mu_/(4.0*lambda_);

    Scalar detF = F.det();
    Scalar Ic = F.frobenius_norm_sq();
    Scalar energy_density = 0.5*mu_*(Ic - 3.0) + 0.5*lambda_*(detF - alpha).powN(2) - 0.5*mu_*log(Ic + 1.0);

    return energy_density;
}
