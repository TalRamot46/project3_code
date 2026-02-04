# simulations/__init__.py
"""
Simulation runners for hydrodynamic problems.
"""
from .riemann_sim import simulate_riemann
from .driven_shock_sim import simulate_driven_shock, ShockHistory
from .riemann_exact import sample_solution, solve_star_region

__all__ = [
    "simulate_riemann",
    "simulate_driven_shock",
    "ShockHistory",
    "sample_solution",
    "solve_star_region",
]
