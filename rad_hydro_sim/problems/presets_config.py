

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
from project_3.rad_hydro_sim.simulation.radiation_step import K_per_Hev


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
        x_max = 1.0e-2,
        t_end = 1.0e-3,

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",

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
        T0 = None, 
        P0 = 1.0,
        tau = 0.0,

        # initial conditions
        rho0 = 1.0,
        p0 = 1e-6,
        u0 = 0.0,

        T_initial = None,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0
        
        # grid parameters
        x_min = 0.0,
        x_max = 1.0,
        t_end = 1.0,

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",
        title = "",
        
        # Geometry
        geom = planar()  # Default to planar geometry
    ),
    "power_law_pressure_drive": RadHydroCase(
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
        P0 = 1.0,
        tau = 1.0,

        # initial conditions
        rho0 = 1.0,
        p0 = 1e-6,
        u0 = 0.0,

        T_initial = None,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0
        
        # grid parameters
        x_min = 0.0,
        x_max = 1.0,
        t_end = 1.0,

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",
        title = "",
        
        # Geometry
        geom = planar()  # Default to planar geometry
    ),
    "constant_temperature_drive": RadHydroCase(
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
        T0 = 0.86,
        P0 = None,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial = 300 / K_per_Hev, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1
        
        # grid parameters
        x_min = 0,
        x_max = 3e-4,
        t_end = 1.0e-9,

        # for flags
        initial_condition = "temperature, density",
        scenario = "radiation_only",

        title = "",
        

        # Geometry
        geom = planar()  # Default to planar geometry
    ),
    "rad_hydro_constant_temperature_drive": RadHydroCase(
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
        T0 = 0.86,
        P0 = None,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial = 300 / K_per_Hev, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 3e-4,
        t_end = 1.0e-9,

        # for flags
        initial_condition = "temperature, density",
        scenario = "full_rad_hydro",

        title = "",

        # Geometry
        geom = planar()  # Default to planar geometry
    ),
}

PRESETS: Dict[str, Tuple[HydroCase, SimulationConfig]] = {
    # -------------------------------------------------------------------------
    # Simple Trial
    # -------------------------------------------------------------------------
    "hydro_only_constant_pressure_drive": (
        SIMPLE_TEST_CASES["constant_pressure_drive"],
        SIMULATION_CONFIGS["all_outputs"],
    ),
    "hydro_only_power_law_pressure_drive": (
        SIMPLE_TEST_CASES["power_law_pressure_drive"],
        SIMULATION_CONFIGS["all_outputs"],
    ),
    "radiation_only_constant_temperature_drive": (
        SIMPLE_TEST_CASES["constant_temperature_drive"],
        SIMULATION_CONFIGS["all_outputs"],
    ),
    "rad_hydro_constant_temperature_drive": (
        SIMPLE_TEST_CASES["rad_hydro_constant_temperature_drive"],
        SIMULATION_CONFIGS["all_outputs"],
    ),
}