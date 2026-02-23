# supersonic_solver package
"""
Supersonic (radiation-diffusion) self-similar solver.

Solver structure (see README_super_python.md for full description):

  materials_super  →  manager_super(mat, tau)  →  (m0, mw, e0, ew, xsi, z, A, t, x)
       ↑                      │
       │                      └→ solve_normalize_super(alpha, beta, tau)
       │                              ├→ F_super.F(...)   [ODE RHS]
       │                              └→ utils_super.integrate_ode(...)
       │
       └→ profiles_for_report_super(mat, tau)  [post-process: m_heat, x_heat, T_heat]

Public API:
  - materials_super: MaterialSuper, material_al, material_au, material_be, material_pb, material_sio2
  - F_super: F
  - utils_super: trapz, mid, integrate_ode
  - solve_normalize_super: solve_normalize
  - manager_super: manager_super
  - profiles_for_report_super: compute_profiles_for_report
  - run_super: run_super
"""

from .materials_super import (
    KELVIN_PRE_HEV,
    STEFAN_BOLTZMANN,
    MaterialSuper,
    material_al,
    material_au,
    material_be,
    material_pb,
    material_sio2,
)
from .F_super import F
from .utils_super import integrate_ode, mid, trapz
from .solve_normalize_super import solve_normalize
from .manager_super import manager_super
from .profiles_for_report_super import compute_profiles_for_report
from .run_super import run_super

__all__ = [
    "KELVIN_PRE_HEV",
    "STEFAN_BOLTZMANN",
    "MaterialSuper",
    "material_al",
    "material_au",
    "material_be",
    "material_pb",
    "material_sio2",
    "F",
    "trapz",
    "mid",
    "integrate_ode",
    "solve_normalize",
    "manager_super",
    "compute_profiles_for_report",
    "run_super",
]
