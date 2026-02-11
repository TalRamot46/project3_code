# subsonic_solver package
"""
Subsonic self-similar solver (Python conversion of MATLAB subsonic_solver).

Solver structure:
  materials_sub  →  manager_sub(mat, tau)  →  (m0, mw, e0, ew, P0, Pw, V0, Vw, u0, uw, xsi, z, Ptilda, utilda, B, t, x)
       ↑                      │
       │                      └→ solve_normalize_sub(alpha, beta, lambda_, mu, r, tau)
       │                              ├→ F_sub.F(...)   [ODE RHS, 5-component state]
       │                              └→ utils_sub.integrate_ode(...)
       │
       └→ profiles_for_report_sub(mat, tau)  [post-process: m_heat, P_heat, T_heat, u_heat, rho_heat, ...]

Public API:
  - materials_sub: MaterialSub, material_al, material_au, material_be, material_cu, material_pb
  - F_sub: F
  - utils_sub: trapz, mid, integrate_ode
  - solve_for_tau_sub: solve_for_tau
  - solve_normalize_sub: solve_normalize
  - manager_sub: manager_sub
  - profiles_for_report_sub: compute_profiles_for_report
  - run_sub: run_sub
"""

from .materials_sub import (
    HEV_IN_KELVIN,
    STEFAN_BOLTZMANN,
    MaterialSub,
    material_al,
    material_au,
    material_be,
    material_cu,
    material_pb,
)
from .F_sub import F
from .utils_sub import integrate_ode, mid, trapz
from .solve_for_tau_sub import solve_for_tau
from .solve_normalize_sub import solve_normalize
from .manager_sub import manager_sub
from .profiles_for_report_sub import compute_profiles_for_report
from .run_sub import run_sub

__all__ = [
    "HEV_IN_KELVIN",
    "STEFAN_BOLTZMANN",
    "MaterialSub",
    "material_al",
    "material_au",
    "material_be",
    "material_cu",
    "material_pb",
    "F",
    "trapz",
    "mid",
    "integrate_ode",
    "solve_for_tau",
    "solve_normalize",
    "manager_sub",
    "compute_profiles_for_report",
    "run_sub",
]
