# problems/__init__.py
"""
Hydrodynamic problem definitions and initialization functions.

Structure:
- simulation_config.py: Numerical solver parameters (N, CFL, etc.) + SIMULATION_CONFIGS
- Hydro_case.py: Base class for physical problem parameters
- presets.py: Predefined test cases with recommended configs
- *_problem.py: Specific problem types with unique physics
"""
from .simulation_config import (
    SimulationConfig, 
    ProblemType, 
    SIMULATION_CONFIGS, 
    get_config,
    make_output_paths,
)
from .Hydro_case import HydroCase
from .presets import get_preset, list_presets, PRESETS, get_preset_names

from .riemann_problem import (
    RiemannCase, 
    RIEMANN_TEST_CASES, 
    init_riemann,
    init_planar_riemann_case,  # Legacy
)
from .driven_shock_problem import (
    DrivenShockCase, 
    DRIVEN_SHOCK_TEST_CASES,
    init_driven_shock,
    init_planar_driven_shock_case,  # Legacy
)
from .sedov_problem import (
    SedovExplosionCase, 
    SEDOV_TEST_CASES, 
    init_sedov,
    init_sedov_explosion,  # Legacy
)

__all__ = [
    # Configuration
    "SimulationConfig",
    "ProblemType",
    "SIMULATION_CONFIGS",
    "get_config",
    "make_output_paths",
    # Presets
    "PRESETS",
    "get_preset",
    "list_presets",
    "get_preset_names",
    # Base class
    "HydroCase",
    # Riemann
    "RiemannCase",
    "RIEMANN_TEST_CASES",
    "init_riemann",
    "init_planar_riemann_case",
    # Driven shock
    "DrivenShockCase",
    "DRIVEN_SHOCK_TEST_CASES",
    "init_driven_shock",
    "init_planar_driven_shock_case",
    # Sedov
    "SedovExplosionCase",
    "SEDOV_TEST_CASES",
    "init_sedov",
    "init_sedov_explosion",
]