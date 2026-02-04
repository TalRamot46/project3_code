# problems/__init__.py
"""
Hydrodynamic problem definitions and initialization functions.
"""
from .base_problem import ProblemCase
from .riemann_problem import RiemannCase, RIEMANN_TEST_CASES, init_planar_riemann_case
from .driven_shock_problem import DrivenShockCase, init_planar_driven_shock_case
from .sedov_problem import SedovExplosionCase, SEDOV_TEST_CASES, init_sedov_explosion

__all__ = [
    # Base class
    "ProblemCase",
    # Riemann
    "RiemannCase",
    "RIEMANN_TEST_CASES",
    "init_planar_riemann_case",
    # Driven shock
    "DrivenShockCase",
    "init_planar_driven_shock_case",
    # Sedov
    "SedovExplosionCase",
    "SEDOV_TEST_CASES",
    "init_sedov_explosion",
]
