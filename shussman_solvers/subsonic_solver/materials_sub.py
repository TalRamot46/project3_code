# materials_sub.py
"""
Materials layer of the subsonic self-similar solver (MATLAB Al.m, Au.m, Be.m, Cu.m, Pb.m).

Defines material properties (opacity/EOS exponents and prefactors) required by manager_sub.
No dependencies on other solver modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

HEV_IN_KELVIN = 1_160_500.0
STEFAN_BOLTZMANN = 5.670373e-5


@dataclass(frozen=True, slots=True)
class MaterialSub:
    """Material properties for the subsonic self-similar solver."""

    alpha: float
    beta: float
    lambda_: float
    mu: float
    f: float  # J/g/HeV^beta
    g: float  # g/cm^2/HeV^alpha
    sigma: float  # W/cm^2/K^4
    r: float
    name: str = "Material"


def material_al() -> MaterialSub:
    """Aluminum (Al.m). Yair's data."""
    mat_alpha = 3.1
    mat_beta = 1.2
    mat_lambda = 0.3685
    mat_mu = 0.0
    HeV = HEV_IN_KELVIN
    mat_f = 3.6e11 * (100**mat_beta) / (HeV**mat_beta)
    mat_g = 1.0 / (1487.0 * (HeV**mat_alpha))
    mat_sigma = STEFAN_BOLTZMANN
    mat_r = 0.3
    return MaterialSub(
        mat_alpha, mat_beta, mat_lambda, mat_mu,
        mat_f, mat_g, mat_sigma, mat_r, name="Al"
    )


def material_au() -> MaterialSub:
    """Gold (Au.m)."""
    mat_alpha = 1.5
    mat_beta = 1.6
    mat_lambda = 0.2
    mat_mu = 0.14
    HeV = HEV_IN_KELVIN
    mat_f = 3.4e13 / (HeV**mat_beta)
    mat_g = 1.0 / (7200.0 * (HeV**mat_alpha))
    mat_sigma = STEFAN_BOLTZMANN
    mat_r = 0.25
    return MaterialSub(
        mat_alpha, mat_beta, mat_lambda, mat_mu,
        mat_f, mat_g, mat_sigma, mat_r, name="Au"
    )


def material_be() -> MaterialSub:
    """Beryllium (Be.m)."""
    mat_alpha = 4.893
    mat_beta = 1.0902
    mat_lambda = 0.6726
    mat_mu = 0.0701
    HeV = HEV_IN_KELVIN
    mat_f = 8.8053e13 / (HeV**mat_beta)
    mat_g = 1.0 / (402.8102 * (HeV**mat_alpha))
    mat_sigma = STEFAN_BOLTZMANN
    mat_r = 0.5529
    return MaterialSub(
        mat_alpha, mat_beta, mat_lambda, mat_mu,
        mat_f, mat_g, mat_sigma, mat_r, name="Be"
    )


def material_cu() -> MaterialSub:
    """Copper (Cu.m)."""
    mat_alpha = 2.21
    mat_beta = 1.35
    mat_lambda = 0.29
    mat_mu = 0.14
    HeV = HEV_IN_KELVIN
    mat_f = 5.7e13 / (HeV**mat_beta)
    mat_g = 1.0 / (2237.0 * (HeV**mat_alpha))
    mat_sigma = STEFAN_BOLTZMANN
    mat_r = mat_mu / (mat_beta - 1.0)
    return MaterialSub(
        mat_alpha, mat_beta, mat_lambda, mat_mu,
        mat_f, mat_g, mat_sigma, mat_r, name="Cu"
    )


def material_pb() -> MaterialSub:
    """Lead (Pb.m)."""
    mat_alpha = 2.02
    mat_beta = 1.35
    mat_lambda = 0.23
    mat_mu = 0.14
    HeV = HEV_IN_KELVIN
    mat_f = 3.5e13 / (HeV**mat_beta)
    mat_g = 1.0 / (13333.0 * (HeV**mat_alpha))
    mat_sigma = STEFAN_BOLTZMANN
    mat_r = 0.3442
    return MaterialSub(
        mat_alpha, mat_beta, mat_lambda, mat_mu,
        mat_f, mat_g, mat_sigma, mat_r, name="Pb"
    )
