# ============================================================================
# Preset Configurations
# ============================================================================
"""
Preset configurations for rad_hydro_sim.

Presets are physical case names (SIMPLE_TEST_CASES keys) - not case+config couples.
Simulation config is a general setting; use get_default_config() and override
N, store_every, png_time_frac manually when running a test.
"""
from typing import Dict, Tuple

from project_3.hydro_sim.core.geometry import planar

# ---------------------------------------------------------------------------
# Preset name constants = SIMPLE_TEST_CASES keys (physical case names)
# ---------------------------------------------------------------------------
PRESET_FIRST_ATTEMPT = "first_attempt"
PRESET_POWER_LAW = "constant_pressure_drive"
PRESET_CONSTANT_PRESSURE = "power_law_pressure_drive"
PRESET_CONSTANT_T_RADIATION = "constant_temperature_drive"
PRESET_RAD_HYDRO_CONSTANT_T = "rad_hydro_constant_temperature_drive"
PRESET_FIG_8 = "fig_8_comparison"
PRESET_FIG_9 = "fig_9_comparison"
PRESET_FIG_10 = "fig_10_comparison"
# Full rad-hydro presets (for grouping in list_presets)
FULL_RAD_HYDRO_PRESET_NAMES = (PRESET_RAD_HYDRO_CONSTANT_T, PRESET_FIG_10)
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.hydro_sim.problems.simulation_config import (
    SIMULATION_CONFIGS,
    SimulationConfig,
)
from project_3.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

KELVIN_PRE_HEV = 1_160_500

# Power-law preset: change this to update both tau and the title
_power_law_tau = -0.45

PRESET_TEST_CASES = {
    PRESET_FIRST_ATTEMPT: RadHydroCase(
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
        T0 = 1, # Hev units
        P0 = None,
        tau = 1.0,

        # initial conditions
        rho0 = 1.0,
        p0 = 1e-6,
        u0 = .00,
        T_initial = None,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0
        
        # grid parameters
        x_min = 0.0,
        x_max = 6e-5,
        t_end = 1.0e-9,

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",

        title="First attempt (hydro only)",
        geom=planar(),
    ),
    PRESET_POWER_LAW: RadHydroCase(
        # Rosen's opacity parameters
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
        T0 = None,
        P0 = 2.71e8,
        tau = _power_law_tau,

        # initial conditions
        rho0 = 19.32,
        p0 = 1e-6,
        u0 = 0.0,

        T_initial = None,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0

        # grid parameters
        x_min = 0.0,
        x_max = 15e-3 / 19.32,
        t_end = 1e-9,

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",
        title=f"Power-law pressure drive (P0 = 2.71 MBar, τ=-0.45)",
        geom=planar(),
    ),
    PRESET_CONSTANT_PRESSURE: RadHydroCase(
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
        T0 = None,
        P0 = 1e12,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = 1e-6,
        u0 = 0.0,

        T_initial = None,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0

        # grid parameters
        x_min = 0.0,
        x_max = 5e-3 / 19.32,
        t_end = 1e-9,

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",
        title=f"Constant pressure drive (P0 = 1 MBar)",
        geom=planar(),
    ),
    PRESET_CONSTANT_T_RADIATION: RadHydroCase(
        # Rosen's opacity parameters (g, f match 1D Diffusion self-similar reference for verification)
        g = 1.0 / 7200,
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f = 3.4e13,
        gamma = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0 = 1.0,
        P0 = None,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial = 300 / KELVIN_PER_HEV, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1
        
        # grid parameters
        x_min = 0,
        x_max = 6e-5, # 6e-5
        t_end = 1.0e-9, # 1.0e-9

        initial_condition="temperature, density",
        scenario="radiation_only",
        title="Radiation-only constant T drive (Au, 300 K)",
        geom=planar(),
    ),
    PRESET_RAD_HYDRO_CONSTANT_T: RadHydroCase(
        # Rosen's opacity parameters
        g = 1.0 / (7200.0 * (KELVIN_PRE_HEV**1.5) * (19.32**0.2)),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f = 3.4e13 / ((KELVIN_PRE_HEV**1.6) * (19.32**0.14)),
        gamma = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0 = 1,
        P0 = None,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial = 300 / KELVIN_PER_HEV, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 3e-4,
        t_end = 1.0e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title="Full rad-hydro constant T drive (Au, 300 K)",
        geom=planar(),
    ),
    PRESET_FIG_8: RadHydroCase(
        # Rosen's opacity parameters
        g = 1.0 / 7200,
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f = 3.4e13,
        gamma = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0 = 1,
        P0 = None,
        tau = 0,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial = 300 / KELVIN_PER_HEV, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 1.5e-3 / 19.32, # m_max = 1.5 mg/cm^2
        t_end = 2e-10,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title="Fig 8 comparison (τ=0, Shussman verification)",
        geom=planar(),
    ),
    PRESET_FIG_9: RadHydroCase(
        # Rosen's opacity parameters
        g = 1.0 / 7200,
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f = 3.4e13,
        gamma = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0 = 1,
        P0 = None,
        tau = 0.123,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial = 300 / KELVIN_PER_HEV, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 1.5e-3 / 19.32, # m_max = 1.5 mg/cm^2
        t_end = 2e-10,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title="Fig 9 comparison (τ=0.123, Shussman verification)",
        geom=planar(),
    ),
    PRESET_FIG_10: RadHydroCase(
        # Rosen's opacity parameters
        g = 1.0 / 7200,
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f = 3.4e13,
        gamma = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0 = 1,
        P0 = None,
        tau = 0.17,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial = 300 / KELVIN_PER_HEV, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 1.5e-3 / 19.32,
        t_end = 1.5e-10,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title="Fig 10 comparison (τ=0.17, Shussman verification)",
        geom=planar(),
    ),
}

# ---------------------------------------------------------------------------
# Default simulation config (slider + PNG at png_time_frac * t_end).
# Override manually when running a specific test.
# ---------------------------------------------------------------------------
DEFAULT_SIMULATION_CONFIG = SIMULATION_CONFIGS["all_outputs"]


def get_default_config() -> SimulationConfig:
    """Return the default simulation config (all_outputs). Override N, png_time_frac, etc. manually."""
    return DEFAULT_SIMULATION_CONFIG


# ---------------------------------------------------------------------------
# PRESETS: preset_name -> (case, config)
# Preset name = physical case key (SIMPLE_TEST_CASES). Config is always all_outputs.
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Tuple[RadHydroCase, SimulationConfig]] = {
    k: (v, DEFAULT_SIMULATION_CONFIG) for k, v in PRESET_TEST_CASES.items()
}