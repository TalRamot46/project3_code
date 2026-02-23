# materials_super.py
"""
Materials layer of the supersonic (radiation-diffusion) self-similar solver.

Role in solver structure:
    Defines material properties (opacity/EOS exponents and prefactors) required by
    manager_super. No dependencies on other solver modules. Used by manager_super and
    profiles_for_report_super.

Structure:
    - MaterialSuper: dataclass holding alpha, beta, lambda_, mu, rho0, f, g, sigma, r, name.
    - material_al(), material_au(), material_be(), material_pb(), material_sio2():
      factory functions returning a MaterialSuper for each material (from MATLAB Al.m, etc.).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# ---- Unit-conversion constant (MATLAB author) ----
# 1 HeV in Kelvin; used to convert between temperature and energy in opacity/EOS formulas.
KELVIN_PRE_HEV = 1_160_500.0

# Stefan-Boltzmann in W/cm^2/K^4 (MATLAB: 5.670373e-5). Some variants convert to J/ns/cm^2/HeV^4.
STEFAN_BOLTZMANN = 5.670373e-5 * KELVIN_PRE_HEV**4 # To convert to Hev unit system


@dataclass(frozen=True, slots=True)
class MaterialSuper:
    """Material properties for the supersonic (radiation-diffusion) self-similar solver."""

    alpha: float
    beta: float
    lambda_: float  # "lambda" is reserved in Python
    mu: float
    rho0: float  # g/cc
    f: float  # J/g/HeV^beta (opacity/EOS prefactor)
    g: float  # g/cm^2/HeV^alpha (from MATLAB comments)
    sigma: float  # W/cm^2/K^4 (or converted to J/ns/cm^2/HeV^4 in some variants)
    r: Optional[float] = None  # Optional; used in some tau/eta formulas (e.g. Al, Be)
    name: str = "Material"


def material_al() -> MaterialSuper:
    """
    Aluminum (Al.m). Note: MATLAB file header mistakenly says 'Au.m'; content is Yair's data for Al.
    rho0 = 2.78 g/cc.
    """
    alpha = 3.1
    beta = 1.2
    lambda_ = 0.3685
    mu = 0.0
    rho0 = 2.78  # g/cc
    # Unit conversion (MATLAB author): f in J/g/HeV^beta; (100^beta)/(HeV^beta) and (rho0^mu).
    f = 3.6e11 * (100.0**beta) / (KELVIN_PRE_HEV**beta) * (rho0**mu)
    # g in g/cm^2/HeV^alpha (MATLAB: 1/(1487*(HeV^alpha)*(rho0^lambda)))
    g = 1.0 / (1487.0 * (KELVIN_PRE_HEV**alpha) * (rho0**lambda_))
    sigma = STEFAN_BOLTZMANN  # W/cm^2/K^4 (commented conversion to J/ns/cm^2/HeV^4 not applied)
    return MaterialSuper(alpha, beta, lambda_, mu, rho0, f, g, sigma, r=0.3, name="Al")


def material_au() -> MaterialSuper:
    """
    Gold (Au.m). alpha=1.5, beta=1.6; rho0=1.0 g/cc in file (commented 19.32).
    """
    alpha = 1.5
    beta = 1.6
    lambda_ = 0.2
    mu = 0.14
    rho0 = 19.32  # g/cc
    # f: J/g/HeV^beta — MATLAB 3.4e13/((HeV^beta)*(rho0^mu))
    f = 3.4e13 / ((KELVIN_PRE_HEV**beta) * (rho0**mu))
    g = 1.0 / (7200.0 * (KELVIN_PRE_HEV**alpha) * (rho0**lambda_))
    sigma = STEFAN_BOLTZMANN
    return MaterialSuper(alpha, beta, lambda_, mu, rho0, f, g, sigma, r=None, name="Au")


def material_be() -> MaterialSuper:
    """
    Beryllium (Be.m). rho0=1 g/cc (commented 1.85). r=0.5529.
    """
    alpha = 4.893
    beta = 1.0902
    lambda_ = 0.6726
    mu = 0.0701
    rho0 = 1.0  # g/cc
    f = 8.8053e13 / ((KELVIN_PRE_HEV**beta) * (rho0**mu))
    g = 1.0 / (402.8102 * (KELVIN_PRE_HEV**alpha) * (rho0**lambda_))
    sigma = STEFAN_BOLTZMANN
    # Optional sigma conversion (MATLAB commented): sigma*(1160400)*4*(10^-9) for J/ns/cm^2/HeV^4
    return MaterialSuper(alpha, beta, lambda_, mu, rho0, f, g, sigma, r=0.5529, name="Be")


def material_pb() -> MaterialSuper:
    """
    Lead (Pb.m). MATLAB header says 'Au.m' but content is Pb. rho0=1.0 g/cc (commented 10.6).
    """
    alpha = 2.02
    beta = 1.35
    lambda_ = 0.23
    mu = 0.14
    rho0 = 1.0  # g/cc
    f = 3.5e13 / ((KELVIN_PRE_HEV**beta) * (rho0**mu))
    g = 1.0 / (13333.0 * (KELVIN_PRE_HEV**alpha) * (rho0**lambda_))
    sigma = STEFAN_BOLTZMANN
    return MaterialSuper(alpha, beta, lambda_, mu, rho0, f, g, sigma, r=None, name="Pb")


def material_sio2() -> MaterialSuper:
    """
    SiO2 (SiO2.m). rho0=0.05 g/cc.
    """
    alpha = 3.5
    beta = 1.1
    lambda_ = 0.75
    mu = 0.1
    rho0 = 0.05  # g/cc
    f = 8.8e13 / ((KELVIN_PRE_HEV**beta) * (rho0**mu))
    g = 1.0 / (9175.0 * (KELVIN_PRE_HEV**alpha) * (rho0**lambda_))
    sigma = STEFAN_BOLTZMANN
    # Optional conversion (MATLAB commented): sigma*(1160400)*4*(10^-9) J/ns/cm^2/HeV^4
    return MaterialSuper(alpha, beta, lambda_, mu, rho0, f, g, sigma, r=None, name="SiO2")
