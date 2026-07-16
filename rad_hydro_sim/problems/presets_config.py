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

import numpy as np
from project3_code.hydro_sim.core.geometry import planar

# ---------------------------------------------------------------------------
# Preset name constants = SIMPLE_TEST_CASES keys (physical case names)
# ---------------------------------------------------------------------------
PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE = "power_law_pressure_drive"
PRESET_CONSTANT_PRESSURE = "constant_pressure_drive"
PRESET_CONSTANT_T_RADIATION_ONLY = "constant_temperature_drive"
PRESET_COPPER_CONST_TEMPERATURE = "copper_const_temperature"
PRESET_ALUMINUM_CONST_TEMPERATURE = "aluminum_const_temperature"
PRESET_OPAQUE_ALUMINUM_CONST_TEMPERATURE = "aluminum_opaque_const_temperature"
PRESET_FIG_8_CONSTANT_TEMPERATURE = "fig_8_comparison"
PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK = "fig_8_comparison_marshak"
CONSTANT_TEMPERATURE_OMEGA_0_5_HYDRO_ONLY = "constant_temperature_omega_0_5_hydro_only"
CONSTANT_TEMPERATURE_OMEGA_0_5_RADIATION_ONLY = "constant_temperature_omega_0_5_radiation_only"
CONSTANT_TEMPERATURE_OMEGA_0_5_FULL = "constant_temperature_omega_0_5_full"
PRESET_FIG_9_CONSTANT_FLUX = "fig_9_comparison"
PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE = "fig_10_comparison"
PRESET_MATLAB = "matlab_comparison"
PRESET_MALKA_HEIZLER = "malka_heizler_comparison"
PRESET_MENAHEM_ABLATION_COMPARISON = "menahem_ablation_comparison"
PRESET_SUPERSONIC_INSTANTANEOUS_ANALYTIC = "supersonic_instantaneous_analytic"
from project3_code.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project3_code.hydro_sim.problems.simulation_config import (
    SIMULATION_CONFIGS,
    SimulationConfig,
)
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

KELVIN_PRE_HEV = 1_160_500

# Power-law preset: change this to update both tau and the title
_power_law_tau = -43/96

PRESET_TEST_CASES = {
    PRESET_CONSTANT_PRESSURE: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0/7200,
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13,
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0_Kelvin = None,
        P0_Barye = 1e12,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = 1e-6,
        u0 = 0.0,

        T_initial_Kelvin = None,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0

        # grid parameters
        x_min = 0.0,
        x_max = 5e-3 / 19.32,
        t_sec_end = 1e-9,

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",
        title=f"Constant pressure drive (P0 = 1 MBar)",
        geom=planar(),
    ),
    PRESET_CONSTANT_T_RADIATION_ONLY: RadHydroCase(
        # Rosen's opacity parameters (g uses alpha=1.5, lambda_=0.2 for KELVIN and rho exponents)
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0_Kelvin = 1 * KELVIN_PER_HEV,  # 1,160,500 K
        P0_Barye = None,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300,  # 300 K

        # adiabatic index
        r = 0.25,  # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 6e-5,
        t_sec_end = 1.0e-9,

        initial_condition="temperature, density",
        scenario="radiation_only",
        title="Radiation-only constant T drive (Au, 300 K)",
        geom=planar(),
        bc_type="Dirichlet",
    ),
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0_Kelvin = None,
        P0_Barye = 2.71e12,
        tau = _power_law_tau,

        # initial conditions
        rho0 = 19.32,
        p0 = 1e-6,
        u0 = 0.0,

        T_initial_Kelvin = None,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0

        # grid parameters
        x_min = 0.0,
        x_max = 25e-3 / 19.32,
        t_sec_end = 2.0e-9, # should be 1ns to compare to fig7

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",
        title=f"Power-law pressure drive (P0 = 2.71 MBar, τ=-0.447)",
        geom=planar(),
    ),
    PRESET_FIG_8_CONSTANT_TEMPERATURE: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14, # ensure

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.00,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 2.5e-2 / 19.32,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Fig 8 comparison ($T_0 = 1$ HeV, $\tau = 0$, $Au$, early time)",
        geom=planar(),
        times_for_png=np.array([0.05e-9, 0.1e-9, 0.15e-9], dtype=float),
    ),
    PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14, # ensure

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.00,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 2.5e-2 / 19.32,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Fig 8 comparison ($T_0 = 1$ HeV, $\tau = 0$, $Au$, early time)",
        geom=planar(),
        times_for_png=np.array([0.05e-9, 0.1e-9, 0.15e-9], dtype=float),
        bc_type="Marshak",
    ),
    CONSTANT_TEMPERATURE_OMEGA_0_5_HYDRO_ONLY: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = None,
        P0_Barye = 2.71e12,
        tau = _power_law_tau,

        # initial conditions
        rho0 = 19.32,
        p0 = 1e-6,
        u0 = 0.0,

        T_initial_Kelvin = None,

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # Initial conditions0

        # grid parameters
        x_min = 0.0,
        x_max = 3e-2 / 19.32,
        t_sec_end = 2.0e-9, # should be 1ns to compare to fig7

        initial_condition="pressure, velocity, density",
        scenario="hydro_only",
        title=r"Ablation-driven pressure at shock region, non-homogeneous media ($P_0 = 2.71~MBar$, $\tau=-\frac{43}{96}$, $\omega=0.5$, $Au$, $2~ns$)",
        geom=planar(),
        times_for_png=np.array([1e-9, 1.5e-9, 2e-9], dtype=float),
        bc_type="Marshak",
        omega=0.5
    ),
    CONSTANT_TEMPERATURE_OMEGA_0_5_RADIATION_ONLY: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14, # ensure

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.00,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 1e-12,
        x_max = 3e-2 / 19.32,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="radiation_only",
        title=r"Constant temperature radiation only ($\omega=0.5$, $Au$, $2~ns$)",
        geom=planar(),
        times_for_png=np.array([0.05e-9, 0.1e-9, 0.15e-9], dtype=float),
        bc_type="Marshak",
        omega=0.01
    ),
    CONSTANT_TEMPERATURE_OMEGA_0_5_FULL: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14, # ensure

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.00,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 2.5e-2 / 19.32,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Constant temperature radiation only ($\omega=0.5$, $Au$, $2~ns$)",
        geom=planar(),
        times_for_png=np.array([0.05e-9, 0.1e-9, 0.15e-9], dtype=float),
        bc_type="Marshak",
        omega=0.5
    ),
    PRESET_COPPER_CONST_TEMPERATURE: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (2237 * KELVIN_PER_HEV**2.21),
        alpha = 2.21,
        lambda_ = 0.29,

        # Rosen's specific energy parameters
        f_Kelvin = 5.7e13 / (KELVIN_PER_HEV**1.35),
        beta_Rosen = 1.35,
        mu = 0.14, # ensure

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.00,

        # initial conditions
        rho0 = 8.96,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 14.0/35.0, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 1e-1 / 8.96,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Cupper constant temprature ($T_0 = 1$ HeV, $\tau = 0$, $Cu$)",
        geom=planar(),
        times_for_png=np.array([1e-9, 1.5e-9, 2e-9], dtype=float),
        bc_type="Marshak"
    ),
    PRESET_ALUMINUM_CONST_TEMPERATURE: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (1487 * KELVIN_PER_HEV**3.1),
        alpha = 3.1,
        lambda_ = 0.3685,

        # Rosen's specific energy parameters
        f_Kelvin = 9.04e13 / (KELVIN_PER_HEV**1.2),
        beta_Rosen = 1.2,
        mu = 0, # ensure

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.00,

        # initial conditions
        rho0 = 2.78,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.3, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 0.015 / 2.78,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Aluminum constant temprature ($T_0 = 1$ HeV, $\tau = 0$, $Al$)",
        geom=planar(),
        times_for_png=np.array([1e-9, 1.5e-9, 2e-9], dtype=float),
        bc_type="Marshak"
    ),
    PRESET_OPAQUE_ALUMINUM_CONST_TEMPERATURE: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (1487 * KELVIN_PER_HEV**3.1),
        alpha = 3.1,
        lambda_ = 0.3685,

        # Rosen's specific energy parameters
        f_Kelvin = 9.04e13 / (KELVIN_PER_HEV**1.2),
        beta_Rosen = 1.2,
        mu = 0, # ensure

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.00,

        # initial conditions
        rho0 = 278,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.3, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 0.005 / 278,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Opaque Aluminum constant temprature ($T_0 = 1$ HeV, $\tau = 0$, $Al$)",
        geom=planar(),
        times_for_png=np.array([1e-9, 1.5e-9, 2e-9], dtype=float),
        bc_type="Marshak"
    ),
    PRESET_FIG_9_CONSTANT_FLUX: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.122957198444,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 1.6e-3 / 19.32,
        t_sec_end = 1.5e-10,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Fig 9 comparison ($T_0 = 1$ HeV, $\tau = 0.123$, $Au$, early times)",
        geom=planar(),
        times_for_png=np.array([0.05e-9, 0.1e-9, 0.15e-9], dtype=float),
        bc_type="Marshak"
    ),
    PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1000,

        # Boundary conditions
        T0_Kelvin = 1* KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.17,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 1.5e-3 / 19.32,
        t_sec_end = 1.5e-10,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Fig 10 comparison ($T_0 = 1$ HeV, $\tau = 0.17$, $Au$, early times)",
        geom=planar(),
        times_for_png=np.array([0.05e-9, 0.1e-9, 0.15e-9], dtype=float),
    ),
    PRESET_MATLAB: RadHydroCase(
    # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1e3,

        # Boundary conditions
        T0_Kelvin = 1 * KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 15e-3 / 19.32, # m_max = 15 mg/cm^2
        t_sec_end = 1e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title="Prset Matlab (T0=1 HeV, τ=0, t=1 ns, Shussman verification)",
        geom=planar(),
        force_black = None,
        times_for_png=np.array([0.05e-9, 0.1e-9, 0.15e-9], dtype=float),
    ),
    PRESET_MALKA_HEIZLER: RadHydroCase(
    # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1e3,

        # Boundary conditions
        T0_Kelvin = 1 * KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.0,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 1.25e-3 / 19.32, # m_max = 15 mg/cm^2
        t_sec_end = 1e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title="Prset Intermediate (T0=1 HeV, t=1 ns, Malka & Heizler verification)",
        geom=planar(),
        force_black = None
    ),
    PRESET_MENAHEM_ABLATION_COMPARISON: RadHydroCase(
    # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1e3,

        # Boundary conditions
        T0_Kelvin = 1 * KELVIN_PER_HEV,
        P0_Barye = None,
        tau = 0.123,

        # initial conditions
        rho0 = 19.32,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 0,
        x_max = 1e-3, 
        t_sec_end = 2.061e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title="Prset Menahem Ablation Comparison (T0=Tb HeV, t=1 ns, Malka & Heizler verification)",
        geom=planar(),
        force_black = None
    ),
    PRESET_SUPERSONIC_INSTANTANEOUS_ANALYTIC: RadHydroCase(
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5),
        alpha = 1.5,
        lambda_ = 0.2,

        # Rosen's specific energy parameters
        f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0.14,

        # coupling factor
        chi = 1e3,

        # Boundary conditions
        T0_Kelvin = 1 * KELVIN_PER_HEV,
        P0_Barye = None,
        tau = -0.14534769833496573,

        # initial conditions
        rho0 = 19.32,
        omega = 0.3,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K

        # adiabatic index
        r = 0.25,

        # grid parameters
        x_min = 1e-12,
        x_max = 5e-3 / 19.32,
        t_sec_end = 1e-9,

        initial_condition="temperature, density",
        scenario="radiation_only",
        title=r"Supersonic Instantaneous Radiation Wave ($\omega=0.3$, $Au$, radiation_only)",
        geom=planar(),
        bc_type="Dirichlet",
    )
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