

# ============================================================================
# Preset Configurations
# ============================================================================

# All available presets: maps preset name -> (case, config)
from traitlets import Dict, Tuple

from project_3.hydro_sim.problems.Hydro_case import HydroCase
from project_3.hydro_sim.core.geometry import planar
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.hydro_sim.problems.simulation_config import (
    SIMULATION_CONFIGS,
    SimulationConfig,
)


SIMPLE_TEST_CASES = {
    "first_attempt" : RadHydroCase(
        gamma=1.4,
        kappa=1e-5,
        rho0=1.0,
        p0=1.0,
        T0=300.0,
        E_rad0=1e-5,
        x_min=-1.0, x_max=1.0, t_end=0.25,
        geom=planar(),
        title="First Attempt"
    )
}

PRESETS: Dict[str, Tuple[HydroCase, SimulationConfig]] = {
    # -------------------------------------------------------------------------
    # Simple Trial
    # -------------------------------------------------------------------------
    "tau_zero": (
        SIMPLE_TEST_CASES["first_attempt"],
        SIMULATION_CONFIGS["gif_only"],
    ),
}