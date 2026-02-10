

# ============================================================================
# Preset Configurations
# ============================================================================

# All available presets: maps preset name -> (case, config)
from typing import Dict, Tuple

from project_3.hydro_sim.problems.Hydro_case import HydroCase
from project_3.hydro_sim.core.geometry import planar
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.hydro_sim.problems.simulation_config import (
    SIMULATION_CONFIGS,
    SimulationConfig,
)


SIMPLE_TEST_CASES = {
    "first_attempt" : RadHydroCase(
        # Rosen's opacity parameters
        g = 1.0/7200,
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f = 3.4e13,
        gamma = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0 = 0.86, # Hev units
        tau = 1.0,

        # initial conditions
        rho0 = 1.0,
        p0 = 1.0,
        u0 = 0.0,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0
        
        # grid parameters
        x_min = 0.0,
        x_max = 1.0e-2,
        t_end = 1.0e-3,
        title = "",
        
        # Geometry
        geom = planar()  # Default to planar geometry
        ),
    "constant_pressure_drive": RadHydroCase(
    # Rosen's opacity parameters
        g = 1.0/7200,
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f = 3.4e13,
        gamma = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0 = 0.86, # Hev units
        tau = 1.0,

        # initial conditions
        rho0 = 1e-6,
        p0 = 1.0,
        u0 = 0.0,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0
        
        # grid parameters
        x_min = 0.0,
        x_max = 1.0,
        t_end = 0.5,
        title = "",
        
        # Geometry
        geom = planar()  # Default to planar geometry
    )
}

PRESETS: Dict[str, Tuple[HydroCase, SimulationConfig]] = {
    # -------------------------------------------------------------------------
    # Simple Trial
    # -------------------------------------------------------------------------
    "tau_zero": (
        SIMPLE_TEST_CASES["constant_pressure_drive"],
        SIMULATION_CONFIGS["all_outputs"],
    ),
}