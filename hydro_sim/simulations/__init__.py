# simulations/__init__.py
"""
Simulation runners for hydrodynamic problems.
"""
from .riemann_sim import simulate_riemann
from .driven_shock_sim import (
    simulate_driven_shock,
    simulate_sedov,
    simulate_lagrangian,
    SimulationHistory,
    ShockHistory,  # Alias for backward compatibility
    SimulationType,
)
from .riemann_exact import sample_solution, solve_star_region

__all__ = [
    # Unified simulation
    "simulate_lagrangian",
    "SimulationHistory",
    "SimulationType",
    # Problem-specific wrappers
    "simulate_riemann",
    "simulate_driven_shock",
    "simulate_sedov",
    # Backward compatibility
    "ShockHistory",
    # Exact solutions
    "sample_solution",
    "solve_star_region",
]
