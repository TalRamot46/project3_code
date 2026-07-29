# ============================================================================
# Preset Configurations
# ============================================================================
"""
Preset configurations for rad_hydro_sim.

Presets are physical case names (SIMPLE_TEST_CASES keys) - not case+config couples.
Simulation config is a general setting; use get_default_config() and override
N, store_every, png_time_frac manually when running a test.
"""
from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
from hydro_sim.core.geometry import planar

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
CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL = "constant_temperature_omega_minus_0_5_full"
PRESET_FIG_9_CONSTANT_FLUX = "fig_9_comparison"
PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE = "fig_10_comparison"
PRESET_MATLAB = "matlab_comparison"
PRESET_MALKA_HEIZLER = "malka_heizler_comparison"
PRESET_MENAHEM_ABLATION_COMPARISON = "menahem_ablation_comparison"
PRESET_SUPERSONIC_INSTANTANEOUS_ANALYTIC = "supersonic_instantaneous_analytic"
from rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from hydro_sim.problems.simulation_config import (
    SIMULATION_CONFIGS,
    SimulationConfig,
)
from rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

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
        # Just past the shock front at t_end, so the domain is spent on the
        # perturbed region instead of far field the wave never reaches.
        # Menahem's PistonShock puts the front at 5.6419e-5 cm at 2 ns
        # (AblationSolver(**_ablation_kwargs_from_case(case))
        #  .shock_solver.shock_position(time=t_sec_end)).
        # The margin is 10% of the enclosed *mass*, which is what the outer
        # boundary actually has to hold back. Since m ~ x^(1-omega), that is
        # 1.1**(1/(1-omega)) = 1.1**2 in position -- a 10% margin in x would
        # have left only 5% in mass.
        x_max = 1.1**2 * 5.6419e-5,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Constant temperature radiation only ($\omega=0.5$, $Au$, $2~ns$)",
        geom=planar(),
        # Late times: the paper quotes Test 1 profiles at 1.0/1.5/2.0 ns, and
        # these are where the ablation layer is best resolved. The former
        # 0.05/0.1/0.15 ns sampled only the run's least accurate window.
        times_for_png=np.array([1.0e-9, 1.5e-9, 2.0e-9], dtype=float),
        bc_type="Marshak",
        omega=0.5
    ),
    CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL: RadHydroCase(
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
        # Just past the shock front at t_end, so the domain is spent on the
        # perturbed region instead of far field the wave never reaches.
        # Menahem's PistonShock puts the front at 5.6419e-5 cm at 2 ns
        # (AblationSolver(**_ablation_kwargs_from_case(case))
        #  .shock_solver.shock_position(time=t_sec_end)).
        # The margin is 10% of the enclosed *mass*, which is what the outer
        # boundary actually has to hold back. Since m ~ x^(1-omega), that is
        # 1.1**(1/(1-omega)) = 1.1**2 in position -- a 10% margin in x would
        # have left only 5% in mass.
        x_max = 1.1**2 * 5.6419e-3,
        t_sec_end = 2e-9,

        initial_condition="temperature, density",
        scenario="full_rad_hydro",
        title=r"Constant temperature radiation only ($\omega=-0.5$, $Au$, $2~ns$)",
        geom=planar(),
        # Late times: the paper quotes Test 1 profiles at 1.0/1.5/2.0 ns, and
        # these are where the ablation layer is best resolved. The former
        # 0.05/0.1/0.15 ns sampled only the run's least accurate window.
        times_for_png=np.array([1.0e-9, 1.5e-9, 2.0e-9], dtype=float),
        bc_type="Marshak",
        omega=-0.5
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
        # Rosen's opacity parameters
        g_Kelvin = 1.0 / (KELVIN_PER_HEV**2),
        alpha = 2,
        lambda_ = 1,

        # Rosen's specific energy parameters
        f_Kelvin = 1.0 / (KELVIN_PER_HEV**1.6),
        beta_Rosen = 1.6,
        mu = 0, # ensure

        # coupling factor
        chi = 1,

        # Boundary conditions
        T0_Kelvin = 26560.843307827196,
        P0_Barye = None,
        tau =  -0.13888889,

        # initial conditions
        rho0 = 1,
        p0 = None,
        u0 = None,
        T_initial_Kelvin = 300, # 300 K in Hev

        # adiabatic index
        r = 0.25, # r = \gamma_adiabatic - 1

        # grid parameters
        x_min = 1e-12,
        x_max = 0.1333,
        t_sec_end = 9e-25,

        t_sec_start = 9e-28,   # 1e-3 * t_sec_end; front starts at ~0.018 cm

        initial_condition="analytic_supersonic_instantaneous",
        scenario="radiation_only",
        title=r"Constant temperature radiation only ($\omega=0.2$)",
        geom=planar(),
        times_for_png=np.array([0.1, 0.25, 0.5], dtype=float) * 9e-25,
        bc_type="Dirichlet",
        omega=0.2,
        # Krief (2021) Eq. (1) is pure nonlinear conduction: energy is stored in
        # the matter (u = f T^beta rho^(1-mu)) and radiation enters only as the
        # flux potential a T^4. The default two-temperature model and the
        # "black" mode both store energy in a T^4 instead, which for these
        # benchmark coefficients is ~1e11 times larger -- a different PDE.
        force_black="conduction",
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
# Per-preset numerical overrides.
#
# Most presets run fine on DEFAULT_SIMULATION_CONFIG. A few do not, and pairing
# every preset with the shared default silently hands those an unusable setting.
# Timings below are measured on this repo, running to the preset's own t_end.
# ---------------------------------------------------------------------------
PRESET_CONFIG_OVERRIDES: Dict[str, SimulationConfig] = {
    # rho ~ r^-1/2 concentrates the mass at small r, so resolving the ablation
    # layer (~1e-3 g/cm^2, i.e. the first ~1e-9 cm) costs cells that are minute
    # in *space*, and the acoustic CFL -- which binds on 99% of steps -- scales
    # with them. Measured to 2 ns: N=25 finishes in ~2 min; N=50 reaches only
    # 3.9% of t_end in 5 min; N=100 was still running after 74 min.
    # N=25 is not a compromise on accuracy: against Menahem's AblationSolver the
    # shocked-plateau pressure agrees to 0.3-3% from 0.5 ns onwards (12% at
    # 0.05 ns, while the layer still spans only ~3 cells) and the surface holds
    # 1.0003 HeV against the 1 HeV drive.
    CONSTANT_TEMPERATURE_OMEGA_0_5_FULL: replace(
        DEFAULT_SIMULATION_CONFIG, N=25, store_every=200
    ),
    CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL: replace(
        DEFAULT_SIMULATION_CONFIG, N=100, store_every=200
    ),
    # ~2.5e5 steps over 1397 cells. At store_every=10 the history buffer alone
    # is ~2.6 GB and the run dies in np.stack after the solve has succeeded.
    CONSTANT_TEMPERATURE_OMEGA_0_5_HYDRO_ONLY: replace(
        DEFAULT_SIMULATION_CONFIG, store_every=500
    ),
}


# ---------------------------------------------------------------------------
# PRESETS: preset_name -> (case, config)
# Preset name = physical case key (SIMPLE_TEST_CASES). Config is all_outputs
# unless the preset appears in PRESET_CONFIG_OVERRIDES.
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Tuple[RadHydroCase, SimulationConfig]] = {
    k: (v, PRESET_CONFIG_OVERRIDES.get(k, DEFAULT_SIMULATION_CONFIG))
    for k, v in PRESET_TEST_CASES.items()
}