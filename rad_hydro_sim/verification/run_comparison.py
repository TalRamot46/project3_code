# verification/run_comparison.py
"""
Unified runner for rad_hydro_sim verification comparisons.

Compares:
  1. Radiation-only: run_rad_hydro (radiation_only_constant_temperature_drive)
     vs 1D Diffusion self similar in gold (constant temperature drive)
     vs Supersonic solver (radiation self-similar, same physics).
  2. Hydro-only: run_rad_hydro (hydro_only_power_law_pressure_drive)
     vs hydro_sim run_hydro (matching driven shock case).
  3. Full rad_hydro: run_rad_hydro (constant temperature drive) vs Shussman piecewise
     reference (subsonic solver to shock front; shock solver driven by front pressure).

Usage:
  # In main(), set MODE and run:
  python -m project3_code.rad_hydro_sim.verification.run_comparison

  Or from repo root:
  python project3_code/rad_hydro_sim/verification/run_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from dataclasses import replace

import numpy as np
# Ensure project3_code is on path (when run as script): add parent of repo root so "project3_code" package resolves
_REPO_PARENT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from project3_code.rad_hydro_sim.output_paths import get_rad_hydro_npz_path
from project3_code.rad_hydro_sim.plotting.gif import save_history_gif
from project3_code.rad_hydro_sim.verification.verification_config import (
    VerificationMode,
    get_preset_for_mode,
    get_output_prefix_for_mode,
    make_verification_output_paths,
)
from project3_code.rad_hydro_sim.verification.radiation_data import (
    RadiationData,
    rad_hydro_history_to_radiation_data,
    diffusion_output_to_radiation_data,
    supersonic_output_to_radiation_data,
)
from project3_code.rad_hydro_sim.verification.compare_radiation_plots import (
    plot_radiation_comparison_single_time,
    plot_radiation_comparison_slider,
)

from project3_code.rad_hydro_sim.problems.RadHydroCase import RadHydroCase

KELVIN_PER_HEV = 1_160_500
a_Kelvin = 7.5657e-15 / (KELVIN_PER_HEV**4)

# =============================================================================
# Radiation-only: run rad_hydro + 1D Diffusion + Supersonic solver, then compare T and E_rad
# =============================================================================

def run_supersonic_solver_reference(
    case,
    n_times: int = 30,
) -> RadiationData | None:
    """
    Run the supersonic (radiation self-similar) solver with parameters matching the RadHydroCase.

    Builds a MaterialSuper from the case (alpha, beta_Rosen, lambda_, mu, rho0, f, g, sigma),
    uses case.tau (0 for constant temperature drive), and maps dimensionless time to
    physical time via t_end. Returns RadiationData in the same format as diffusion reference.
    """
    try:
        from project3_code.shussman_solvers.supersonic_solver import (
            MaterialSuper,
            STEFAN_BOLTZMANN_KELVIN,
            compute_profiles_for_report,
        )
    except ImportError as e:
        print(f"  Could not import supersonic solver: {e}, skipping.")
        return None

    # Material matching the preset (same opacity/EOS as rad_hydro)
    mat = MaterialSuper(
        alpha=float(case.alpha),
        beta=float(case.beta_Rosen),
        lambda_=float(case.lambda_),
        mu=float(case.mu),
        rho0=float(case.rho0),
        f=float(case.f_Kelvin) / float(case.rho0)**float(case.mu),
        g=float(case.g_Kelvin) / float(case.rho0)**float(case.lambda_),
        sigma=STEFAN_BOLTZMANN_KELVIN,
        r=float(case.r),
        name="Au_rad_hydro",
    )
    tau = float(case.tau)  # 0 for constant temperature drive
    T0_Kelvin = float(case.T0_Kelvin)

    # Dimensionless times 0.05..0.95 (solver expects these; supersonic_output_to_radiation_data maps to physical time via t_end)
    times_ns = np.linspace(0.05, 0.95, n_times) * 1e9 * case.t_sec_end
    profiles = compute_profiles_for_report(
        mat,
        T0_phys_HeV=float(case.T0_Kelvin) / KELVIN_PER_HEV,
        tau=tau,
        times_ns=times_ns
    )
    times_sec = times_ns / 1e9
    return supersonic_output_to_radiation_data(
        profiles,
        times_sec,
        label="Supersonic solver (reference)",
        color="green",   
        linestyle=":",
    )


def run_radiation_only_comparison(
    skip_rad_hydro: bool = False,
    skip_diffusion: bool = False,
    skip_supersonic: bool = False,
    show_plot: bool = True,
    save_png: bool = True,
    save_gif: bool = True,
) -> None:
    """Run rad_hydro (radiation_only preset), 1D Diffusion, and Supersonic solver; compare T, E_rad."""
    from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
    from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project3_code.rad_hydro_sim.verification.run_diffusion_1d import run_diffusion_1d

    preset_name = get_preset_for_mode(VerificationMode.RADIATION_ONLY)
    case, config = get_preset(preset_name)
    case_title = case.title or preset_name
    output_prefix = get_output_prefix_for_mode(VerificationMode.RADIATION_ONLY)
    png_path, gif_path = make_verification_output_paths(f"{output_prefix}_{case_title}")

    sim_data = None
    if not skip_rad_hydro:
        print("Running rad_hydro (radiation_only_constant_temperature_drive)...")
        x_cells, state, meta, history = simulate_rad_hydro(
            rad_hydro_case=case,
            simulation_config=config,
        )
        sim_data = rad_hydro_history_to_radiation_data(history)
        print(f"Stored {len(sim_data.times)} time steps.")

        # Save and load from rad_hydro_sim/data/ (same path for round-trip)
        # import matplotlib.pyplot as plt
        # plt.plot(sim_data.x[-1], sim_data.T[-1], label="Sim data")
        # # plt.plot(ref_data.x[-1], ref_data.T[-1], label="Ref data")
        # plt.show()

    ref_data = None
    if not skip_diffusion:
        print("Running 1D Diffusion self-similar reference...")
        times_sec, z, T_list, E_rad_list = run_diffusion_1d(
            x_max=float(case.x_max),
            t_end=float(case.t_sec_end),
            T_bath_Kelvin=float(case.T0_Kelvin),
            rho0=float(case.rho0),
            n_times=40,
            Nz=config.N,
            f_Kelvin=float(case.f_Kelvin),
            g_Kelvin=float(case.g_Kelvin),
            T_right_Kelvin=float(case.T_right_Kelvin or 0.0),
        )
        ref_data = diffusion_output_to_radiation_data(times_sec, z, T_list, E_rad_list)
        print(f"Stored {len(ref_data.times)} time steps.")
    
    if not skip_rad_hydro:
        sim_npz = get_rad_hydro_npz_path(case_title, prefix="sim_data")
        np.savez(str(sim_npz), times=sim_data.times, x=sim_data.x, T=sim_data.T, E_rad=sim_data.E_rad)
        print(f"Saved sim_data to {sim_npz}")
    if not skip_diffusion:
        ref_npz = get_rad_hydro_npz_path(case_title, prefix="ref_data")
        np.savez(str(ref_npz), times=ref_data.times, x=ref_data.x, T=ref_data.T, E_rad=ref_data.E_rad)
        print(f"Saved ref_data to {ref_npz}")

    def _to_list_of_arrays(arr):
        """Convert saved array (2D or object array of arrays) to list of 1D arrays."""
        a = np.asarray(arr)
        if a.dtype == object:
            return [np.asarray(v, dtype=float) for v in a.tolist()]
        if a.ndim == 2:
            return [a[i, :].astype(float, copy=False) for i in range(a.shape[0])]
        return [a.astype(float, copy=False)]

    if not skip_rad_hydro:
        loaded = np.load(str(sim_npz), allow_pickle=True)
        sim_data = RadiationData(
            times=np.asarray(loaded["times"], dtype=float),
            x=_to_list_of_arrays(loaded["x"]),
            T=_to_list_of_arrays(loaded["T"]),
            E_rad=_to_list_of_arrays(loaded["E_rad"]),
            label="Rad-Hydro (radiation only)",
            color="blue",
            linestyle="-",
        )
        print(f"Loaded sim_data from {sim_npz}")
    if not skip_diffusion:
        loaded = np.load(str(ref_npz), allow_pickle=True)
        ref_data = RadiationData(
            times=np.asarray(loaded["times"], dtype=float),
            x=_to_list_of_arrays(loaded["x"]),
            T=_to_list_of_arrays(loaded["T"]),
            E_rad=_to_list_of_arrays(loaded["E_rad"]),
            label="1D Diffusion (reference)",
            color="red",
            linestyle="--",
        )
        print(f"Loaded ref_data from {ref_npz}")

    super_data = None
    if not skip_supersonic:
        print("Running Supersonic solver (radiation self-similar) reference...")
        super_data = run_supersonic_solver_reference(
            case,
            n_times=100,
        )
        # convert T to Kelvin
        if super_data is not None:
            print(f"Stored {len(super_data.times)} time steps.")

    # if any of the data among sim_data, ref_data and super_data is None, 
    # set it to the other data. Be aware of not setting None to None.
    if sim_data is None and ref_data is not None and super_data is not None:
        sim_data = ref_data
    elif sim_data is not None and ref_data is None and super_data is not None:
        ref_data = super_data
    elif sim_data is not None and ref_data is not None and super_data is None:
        super_data = sim_data
    elif sim_data is not None and ref_data is None and super_data is None:
        ref_data = sim_data
        super_data = sim_data
    elif sim_data is None and ref_data is None and super_data is not None:
        sim_data = super_data
        ref_data = super_data
    elif sim_data is None and ref_data is None and super_data is None:
        print("Need at least one data for comparison.")
        return
    else:
        print("All data is set.")
    title = "Radiation-only: Rad-Hydro vs 1D Diffusion" + (" + Supersonic solver" if super_data is not None else "")

    print("\nPlotting radiation comparison (T, E_rad vs x)...")
    if show_plot:
        plot_radiation_comparison_slider(
            sim_data, ref_data, super_data,
            show=True,
            title=title,
        )
    if save_png:
        time_mid = config.png_time_frac * float(case.t_sec_end)
        plot_radiation_comparison_single_time(
            sim_data, ref_data,
            time=time_mid,
            savepath=str(png_path),
            show=False,
            title=title,
            extra_ref_data=[super_data] if super_data is not None else None,
        )
        print(f"Saved PNG: {png_path}")
    if save_gif:
        # Also save an animated GIF of the Rad-Hydro history for this case
        save_history_gif(
            history,
            case,
            gif_path=str(gif_path),
            fps=10,
            stride=max(1, len(history.t) // 50),
        )
        print(f"Saved GIF: {gif_path}")
    print("Radiation-only comparison done.")


# =============================================================================
# Hydro-only: run rad_hydro + hydro_sim + shock solver (matching P0*t^tau), compare rho, p, u, e
# =============================================================================

def _rad_hydro_case_to_shock_material(case: RadHydroCase) -> "Material":
    """Build Shussman shock Material from RadHydroCase. Uses f_Kelvin, g_Kelvin."""
    from project3_code.shussman_solvers.shock_solver.materials_shock import (
        Material,
        HEV_IN_KELVIN,
    )
    alpha = float(case.alpha)
    beta = float(case.beta_Rosen)
    rho0 = float(case.rho0) if case.rho0 is not None else 19.32
    V0 = 1.0 / rho0
    # f_Kelvin: e = f_Kelvin * T_Kelvin^beta_Rosen * rho^(-mu) [erg/g]
    # Shock solver expects f such that e = f * T_Hev^beta * rho^(-mu) => f = f_Kelvin / HEV^beta
    f = float(case.f_Kelvin) / (HEV_IN_KELVIN**beta)
    g = float(case.g_Kelvin) / (HEV_IN_KELVIN**alpha)
    return Material(
        alpha=alpha,
        beta=beta,
        lambda_=float(case.lambda_),
        mu=float(case.mu),
        f=f,
        g=g,
        sigma=5.670373e-5,
        r=float(case.r),
        V0=V0,
        name="Au_shock",
    )


def run_shock_solver_hydro_reference(
    case_rh,
    times_sec: np.ndarray,
) -> "SimulationData | None":
    """
    Run shock solver with P(t) = P0 * t^tau boundary conditions matching the RadHydroCase.

    Units: P0_Barye [Barye], tau [dimensionless], times [s].
    Drive: p_drive(t_ns) = P0_Barye * (t_ns)^tau with t_ns = t_sec * 1e9.
    """
    try:
        from project3_code.shussman_solvers.shock_solver.profiles_for_report_shock import (
            compute_shock_profiles,
        )
        from project3_code.hydro_sim.verification.compare_shock_plots import SimulationData
    except ImportError as e:
        print(f"Could not import shock solver: {e}, skipping.")
        return None

    P0_Barye = float(case_rh.P0_Barye)
    tau = float(case_rh.tau)
    # Shock solver expects times in nanoseconds
    times_ns = np.asarray(times_sec, dtype=float) * 1e9

    mat = _rad_hydro_case_to_shock_material(case_rh)
    data = compute_shock_profiles(
        mat=mat,
        P0_phys_Barye=P0_Barye,
        tau=tau,
        Pw=None,
        times_ns=times_ns,
        patching_method=False,
        save_npz=None,
    )
    # Convert to SimulationData (times_sec, m, x, rho, p, u, e in cgs)
    times_out = np.asarray(data["times_sec"], dtype=float)
    if times_out.ndim > 1:
        times_out = times_out.ravel()
    return SimulationData(
        times=times_out,
        m=list(data["m_shock"]),
        x=list(data["x_shock"]),
        rho=list(data["rho_shock"]),
        p=list(data["P_shock"]),
        u=list(data["u_shock"]),
        e=list(data["e_shock"]),
        label="Shock solver (P0*t^τ)",
        color="green",
        linestyle="-.",
    )


def run_hydro_only_comparison(
    skip_rad_hydro: bool = False,
    skip_hydro_sim: bool = False,
    skip_shock_solver: bool = False,
    show_plot: bool = True,
    save_png: bool = True,
    save_gif: bool = True,
) -> None:
    """Run rad_hydro and hydro_sim and shock solver with matching P0*t^tau; compare rho, p, u, e."""
    from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
    from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project3_code.hydro_sim.problems.driven_shock_problem import DrivenShockCase
    from project3_code.hydro_sim.core.geometry import planar
    from project3_code.hydro_sim.simulations.lagrangian_sim import (
        simulate_lagrangian,
        SimulationType,
    )
    from project3_code.hydro_sim.verification.compare_shock_plots import (
        plot_comparison_single_time,
        plot_comparison_slider,
        load_rad_hydro_history,
        load_hydro_history,
    )
    from project3_code.hydro_sim.verification.compare_shock_plots import (
        plot_comparison_single_time,
        plot_comparison_slider,
    )

    preset_name = get_preset_for_mode(VerificationMode.HYDRO_ONLY)
    case_rh, config = get_preset(preset_name)
    case_title = case_rh.title or preset_name
    output_prefix = get_output_prefix_for_mode(VerificationMode.HYDRO_ONLY)
    png_path, gif_path = make_verification_output_paths(f"{output_prefix}_{case_title}")

    # Matching driven shock case for hydro_sim (same physics: P0=1, tau=1, gamma=1.25, etc.)
    driven_case = DrivenShockCase(
        gamma=float(case_rh.r + 1),
        x_min=float(case_rh.x_min),
        x_max=float(case_rh.x_max),
        t_end=float(case_rh.t_sec_end),
        rho0=float(case_rh.rho0),
        p0=float(case_rh.p0),
        u0=float(case_rh.u0),
        P0=float(case_rh.P0_Barye),
        tau=float(case_rh.tau),
        geom=planar(),
        title=f"Power-law pressure drive (τ={case_rh.tau})",
    )
    sim_data = None
    if not skip_rad_hydro:
        print(f"Running rad_hydro ({preset_name})...")
        x_cells, state, meta, history_rh = simulate_rad_hydro(
            rad_hydro_case=case_rh,
            simulation_config=config,
        )
        sim_data = load_rad_hydro_history(history_rh, label="Rad-Hydro (hydro only)")
        print(f"Stored {len(sim_data.times)} time steps.")

    ref_data = None
    if not skip_hydro_sim:
        print("Running hydro_sim (matching driven shock)...")
        # Use coarser history for faster verification (e.g. ~100 frames)
        store_every = max(getattr(config, "store_every", 10), config.N // 10)
        x_cells, state, meta, history_h = simulate_lagrangian(
            driven_case,
            sim_type=SimulationType.DRIVEN_SHOCK,
            Ncells=config.N,
            gamma=driven_case.gamma,
            CFL=config.CFL,
            sigma_visc=config.sigma_visc,
            store_every=max(1, store_every),
            geom=driven_case.geom,
        )
        ref_data = load_hydro_history(history_h)
        ref_data.label = "Hydro (run_hydro)"
        ref_data.color = "red"
        ref_data.linestyle = "--"
        print(f"Stored {len(ref_data.times)} time steps.")

    shock_data = None
    if not skip_shock_solver and (sim_data is not None or ref_data is not None):
        times_source = sim_data.times if sim_data is not None else ref_data.times
        print("Running shock solver (P0*t^τ)...")
        shock_data = run_shock_solver_hydro_reference(case_rh, times_source)
        if shock_data is not None:
            print(f"Stored {len(shock_data.times)} time steps.")

    if sim_data is None or ref_data is None:
        print("Need both rad_hydro and hydro_sim data for comparison.")
        return

    title_base = "Hydro-only: Rad-Hydro vs run_hydro" + (" + Shock solver" if shock_data is not None else "")
    print("\nPlotting hydro comparison (rho, P, u, e vs x)...")
    if show_plot:
        plot_comparison_slider(
            sim_data, ref_data,
            xaxis="m",
            show=True,
            title=title_base,
            shock_data=shock_data,
        )
    if save_png:
        time_mid = config.png_time_frac * case_rh.t_sec_end
        plot_comparison_single_time(
            sim_data, ref_data,
            time=time_mid,
            xaxis="m",
            savepath=str(png_path),
            show=False,
            title=title_base,
            shock_data=shock_data,
        )
        print(f"Saved PNG: {png_path}")
    if save_gif:
          # Also save an animated GIF of the Rad-Hydro history for this hydro-only case
        save_history_gif(
            history_rh,
            case_rh,
            gif_path=str(gif_path),
            fps=10,
            stride=max(1, len(history_rh.t) // 50),
        )
        print(f"Saved GIF: {gif_path}")
    print("Hydro-only comparison done.")


# =============================================================================
# Full rad_hydro: constant T drive vs Shussman (subsonic + shock)
# =============================================================================


def _pad_rad_hydro_data_to_min_frames(data, min_frames: int = 2, t_end_ns: float | None = None):
    """Duplicate frames so data has at least min_frames entries (for slider)."""
    from project3_code.rad_hydro_sim.verification.hydro_data import RadHydroData

    n = len(data.times)
    if n >= min_frames or n == 0:
        return data
    times = np.asarray(data.times, dtype=float)
    t0 = float(times[-1]) if n > 0 else 0.0
    pad = min_frames - n
    t_pad = t_end_ns if t_end_ns is not None else t0 * 1.001
    new_times = np.concatenate([times, np.full(pad, t_pad)])
    new_m = list(data.m) + [np.array(data.m[-1], copy=True) for _ in range(pad)]
    new_x = list(data.x) + [np.array(data.x[-1], copy=True) for _ in range(pad)]
    new_rho = list(data.rho) + [np.array(data.rho[-1], copy=True) for _ in range(pad)]
    new_p = list(data.p) + [np.array(data.p[-1], copy=True) for _ in range(pad)]
    new_u = list(data.u) + [np.array(data.u[-1], copy=True) for _ in range(pad)]
    new_e = list(data.e) + [np.array(data.e[-1], copy=True) for _ in range(pad)]
    new_T = list(data.T) + [np.array(data.T[-1], copy=True) for _ in range(pad)] if data.T else []
    E_list = list(data.E_rad or [])
    new_E = E_list + [np.array(E_list[-1], copy=True) for _ in range(pad)] if E_list else []
    Tm_list = list(data.T_material or [])
    new_Tm = Tm_list + [np.array(Tm_list[-1], copy=True) for _ in range(pad)] if Tm_list else []
    return RadHydroData(
        times=new_times, m=new_m, x=new_x, rho=new_rho, p=new_p, u=new_u, e=new_e,
        T=new_T, E_rad=new_E if new_E else None,
        T_material=new_Tm,
        label=data.label, color=data.color, linestyle=data.linestyle,
    )


def run_full_rad_hydro_comparison(
    skip_rad_hydro: bool = False,
    skip_shussman: bool = False,
    show_plot: bool = True,
    save_png: bool = True,
    save_gif: bool = True,
) -> None:
    """
    Run rad_hydro with constant temperature drive and compare to piecewise Shussman
    reference: subsonic solver (profiles until shock front) + shock solver (driven by
    pressure at front from subsonic). Shock front is diagnosed from rad_hydro solution.
    """
    from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
    from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project3_code.rad_hydro_sim.verification.hydro_data import (
        RadHydroData,
    )
    from project3_code.rad_hydro_sim.verification.shussman_comparison import (
        run_shussman_piecewise_reference,
    )
    from project3_code.hydro_sim.verification.compare_shock_plots import (
        plot_comparison_single_time,
        plot_comparison_slider,
        load_rad_hydro_history,
        load_hydro_history,
    )

    preset_name = get_preset_for_mode(VerificationMode.FULL_RAD_HYDRO)
    case, config = get_preset(preset_name)
    case_title = case.title or preset_name
    output_prefix = get_output_prefix_for_mode(VerificationMode.FULL_RAD_HYDRO)
    png_path, gif_path = make_verification_output_paths(f"{output_prefix}_{case_title}")

    sim_data = None
    if not skip_rad_hydro:
        print(f"Running rad_hydro ({preset_name})...")
        _, _, _, history_rh = simulate_rad_hydro(
            rad_hydro_case=case,
            simulation_config=config,
        )
        sim_data = load_rad_hydro_history(history_rh)
        print(f"Stored {len(sim_data.times)} time steps.")

        # save the sim_data to rad_hydro_sim/data/
        sim_npz = get_rad_hydro_npz_path(case_title, prefix="sim_data")
        np.savez(
            str(sim_npz),
            times=sim_data.times, m=sim_data.m, x=sim_data.x,
            rho=sim_data.rho, p=sim_data.p, u=sim_data.u, e=sim_data.e,
            T=sim_data.T, E_rad=sim_data.E_rad,
            T_material=sim_data.T_material if sim_data.T_material else [],
        )
        print(f"Saved sim_data to {sim_npz}")

        # load sim_data back from the file (round-trip for debugging)
        loaded = np.load(str(sim_npz), allow_pickle=True)
        def _to_list_of_arrays(arr):
            a = np.asarray(arr)
            if a.dtype == object:
                return [np.asarray(v, dtype=float) for v in a.tolist()]
            if a.ndim == 2:
                return [a[i, :].astype(float, copy=False) for i in range(a.shape[0])]
            return [a.astype(float, copy=False)]

        T_raw = _to_list_of_arrays(loaded["T"]) if "T" in loaded else []
        T_hev = [t_arr / KELVIN_PER_HEV for t_arr in T_raw]
        Tm_raw = _to_list_of_arrays(loaded["T_material"]) if "T_material" in loaded else []
        Tm_hev = [t_arr / KELVIN_PER_HEV for t_arr in Tm_raw]
        sim_data = RadHydroData(
            times=np.asarray(loaded["times"], dtype=float),
            m=_to_list_of_arrays(loaded["m"]),
            x=_to_list_of_arrays(loaded["x"]),
            rho=_to_list_of_arrays(loaded["rho"]),
            p=_to_list_of_arrays(loaded["p"]),
            u=_to_list_of_arrays(loaded["u"]),
            e=_to_list_of_arrays(loaded["e"]),
            T=T_hev,
            E_rad=_to_list_of_arrays(loaded["E_rad"]) if "E_rad" in loaded else [],
            T_material=Tm_hev,
            label="Rad-Hydro (full)",
            color="blue",
            linestyle="-",
        )

    ref_data = None
    if not skip_shussman:
        print("Building Shussman piecewise reference (subsonic + shock)...")
        if skip_rad_hydro:
            # 0.01 ns to t_end (1 ns), 100 frames for slider
            t_min = max(case.t_sec_end / 100, 1e-15)
            times_sec = np.linspace(t_min, case.t_sec_end, 100)
        else:
            times_sec = np.linspace(0, case.t_sec_end, 10000)
        time_ns = np.atleast_1d(times_sec * 1e9).ravel()
        T0_HeV = float(case.T0_Kelvin)/KELVIN_PER_HEV  # pyright: ignore[reportArgumentType]
        ref_data = run_shussman_piecewise_reference(case, times_ns=time_ns, T0_HeV=T0_HeV)

    if skip_rad_hydro:
        sim_data = ref_data
        # Ensure at least 2 frames so the slider functions; pad frame uses t_end (1 ns)
        if sim_data is not None and len(sim_data.times) < 2:
            t_end_ns = float(case.t_sec_end) * 1e9
            sim_data = _pad_rad_hydro_data_to_min_frames(sim_data, min_frames=2, t_end_ns=t_end_ns)
        ref_data = sim_data
        print(f"Copied ref_data to sim_data")

    print("\nPlotting full rad_hydro comparison (rho, P, u, e vs x)...")
    if show_plot:
        plot_comparison_slider(
            sim_data,
            ref_data,
            xaxis="m",
            show=True,
            title="Full rad_hydro vs Shussman (subsonic + shock)",
        )

    if save_png:
        time_mid = config.png_time_frac * float(case.t_sec_end)
        plot_comparison_single_time(
            sim_data,
            ref_data,
            time=time_mid,
            xaxis="m",
            savepath=str(png_path),
            show=False,
            title="Full rad_hydro vs Shussman (subsonic + shock)",
        )
        print(f"Saved PNG: {png_path}")
    if save_gif:
        # Also save an animated GIF of the Rad-Hydro history with Shussman overlay for verification
        save_history_gif(
            history_rh,
            case,
            gif_path=str(gif_path),
            fps=10,
            stride=max(1, len(history_rh.t) // 50),
            ref_data=ref_data,
        )
        print(f"Saved GIF: {gif_path}")
    print("Full rad_hydro comparison done.")


def run_full_rad_hydro_force_black_comparison(
    show_plot: bool = True,
    save_png: bool = True,
    save_gif: bool = True,
) -> None:
    """
    Run full rad_hydro force_black variants from PRESET_MATLAB and compare on one figure:
    - force_black=None
    - force_black="gray corrected"
    - force_black="full black"
    """
    from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
    from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project3_code.hydro_sim.verification.compare_shock_plots import (
        plot_comparison_single_time,
        plot_comparison_slider,
        load_rad_hydro_history,
    )

    preset_name = get_preset_for_mode(VerificationMode.FULL_RAD_HYDRO)
    case, config = get_preset(preset_name)
    case_title = f"{case.title or preset_name}_force_black_compare"
    output_prefix = get_output_prefix_for_mode(VerificationMode.FULL_RAD_HYDRO)
    png_path, gif_path = make_verification_output_paths(f"{output_prefix}_{case_title}")

    case_none = replace(case, force_black=None)
    print(f'Running rad_hydro ({preset_name}, force_black=None)...')
    _, _, _, history_none = simulate_rad_hydro(
        rad_hydro_case=case_none,
        simulation_config=config,
    )
    sim_data_none = load_rad_hydro_history(history_none, label='Rad-Hydro (force_black=None)')
    sim_data_none.color = "red"
    sim_data_none.linestyle = "--"
    print(f"Stored {len(sim_data_none.times)} time steps.")

    case_gray = replace(case, force_black="gray corrected")
    print(f'Running rad_hydro ({preset_name}, force_black="gray corrected")...')
    _, _, _, history_gray = simulate_rad_hydro(
        rad_hydro_case=case_gray,
        simulation_config=config,
    )
    sim_data_gray = load_rad_hydro_history(history_gray, label='Rad-Hydro (force_black="gray corrected")')
    sim_data_gray.color = "blue"
    sim_data_gray.linestyle = "-"
    print(f"Stored {len(sim_data_gray.times)} time steps.")

    case_full_black = replace(case, force_black="full black")
    print(f'Running rad_hydro ({preset_name}, force_black="full black")...')
    _, _, _, history_full_black = simulate_rad_hydro(
        rad_hydro_case=case_full_black,
        simulation_config=config,
    )
    sim_data_full_black = load_rad_hydro_history(
        history_full_black, label='Rad-Hydro (force_black="full black")'
    )
    sim_data_full_black.color = "green"
    sim_data_full_black.linestyle = ":"
    print(f"Stored {len(sim_data_full_black.times)} time steps.")

    title = 'Full rad_hydro: force_black=None vs "gray corrected" vs "full black"'
    print("\nPlotting full rad_hydro force_black comparison (rho, P, u, e vs x)...")
    if show_plot:
        plot_comparison_slider(
            sim_data_none,
            sim_data_gray,
            shock_data=sim_data_full_black,
            xaxis="m",
            show=True,
            title=title,
        )
    if save_png:
        time_mid = config.png_time_frac * float(case.t_sec_end)
        plot_comparison_single_time(
            sim_data_none,
            sim_data_gray,
            shock_data=sim_data_full_black,
            time=time_mid,
            xaxis="m",
            savepath=str(png_path),
            show=False,
            title=title,
        )
        print(f"Saved PNG: {png_path}")
    if save_gif:
        save_history_gif(
            history_none,
            case_none,
            gif_path=str(gif_path),
            fps=10,
            stride=max(1, len(history_none.t) // 50),
            ref_data=sim_data_gray,
        )
        print(f"Saved GIF: {gif_path}")
    print("Full rad_hydro force_black comparison done.")


def run_full_rad_hydro_radiation_coeff_scheme_comparison(
    show_plot: bool = True,
    save_png: bool = True,
    save_gif: bool = True,
) -> None:
    """
    Run full rad_hydro with force_black=None and compare radiation coefficient schemes:
    - radiation_coeff_scheme="legacy"
    - radiation_coeff_scheme="face_weighted"
    """
    from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
    from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project3_code.hydro_sim.verification.compare_shock_plots import (
        plot_comparison_single_time,
        plot_comparison_slider,
        load_rad_hydro_history,
    )

    preset_name = get_preset_for_mode(VerificationMode.FULL_RAD_HYDRO)
    case, config = get_preset(preset_name)
    case_title = f"{case.title or preset_name}_coeff_scheme_compare"
    output_prefix = get_output_prefix_for_mode(VerificationMode.FULL_RAD_HYDRO)
    png_path, gif_path = make_verification_output_paths(f"{output_prefix}_{case_title}")

    case_legacy = replace(case, force_black=None, radiation_coeff_scheme="legacy")
    print(f'Running rad_hydro ({preset_name}, force_black=None, radiation_coeff_scheme="legacy")...')
    _, _, _, history_legacy = simulate_rad_hydro(
        rad_hydro_case=case_legacy,
        simulation_config=config,
    )
    sim_data_legacy = load_rad_hydro_history(
        history_legacy,
        label='Rad-Hydro (force_black=None, coeff="legacy")',
    )
    sim_data_legacy.color = "red"
    sim_data_legacy.linestyle = "--"
    print(f"Stored {len(sim_data_legacy.times)} time steps.")

    case_face = replace(case, force_black=None, radiation_coeff_scheme="face_weighted")
    print(f'Running rad_hydro ({preset_name}, force_black=None, radiation_coeff_scheme="face_weighted")...')
    _, _, _, history_face = simulate_rad_hydro(
        rad_hydro_case=case_face,
        simulation_config=config,
    )
    sim_data_face = load_rad_hydro_history(
        history_face,
        label='Rad-Hydro (force_black=None, coeff="face_weighted")',
    )
    sim_data_face.color = "blue"
    sim_data_face.linestyle = "-"
    print(f"Stored {len(sim_data_face.times)} time steps.")

    title = 'Full rad_hydro: force_black=None, coeff "legacy" vs "face_weighted"'
    print("\nPlotting full rad_hydro radiation coefficient comparison (rho, P, u, e vs x)...")
    if show_plot:
        plot_comparison_slider(
            sim_data_legacy,
            sim_data_face,
            xaxis="m",
            show=True,
            title=title,
        )
    if save_png:
        time_mid = config.png_time_frac * float(case.t_sec_end)
        plot_comparison_single_time(
            sim_data_legacy,
            sim_data_face,
            time=time_mid,
            xaxis="m",
            savepath=str(png_path),
            show=False,
            title=title,
        )
        print(f"Saved PNG: {png_path}")
    if save_gif:
        save_history_gif(
            history_legacy,
            case_legacy,
            gif_path=str(gif_path),
            fps=10,
            stride=max(1, len(history_legacy.t) // 50),
            ref_data=sim_data_face,
        )
        print(f"Saved GIF: {gif_path}")
    print("Full rad_hydro radiation coefficient comparison done.")


# =============================================================================
# Main
# =============================================================================

def run_comparison(
    mode: VerificationMode,
    *,
    skip_rad_hydro: bool = False,
    skip_diffusion: bool = False,
    skip_supersonic: bool = False,
    skip_hydro_sim: bool = False,
    skip_shock_solver: bool = False,
    skip_shussman: bool = False,
    compare_force_black_cases: bool = False,
    compare_radiation_coeff_schemes: bool = False,
    show_plot: bool = True,
    save_png: bool = True,
    save_gif: bool = True,
) -> None:
    """Run the verification comparison for the given mode."""
    preset_name = get_preset_for_mode(mode)
    print("=" * 60)
    print("Rad-Hydro Verification Comparison")
    print("=" * 60)
    print(f"Mode: {mode.value} (preset: {preset_name})")
    print()

    if mode == VerificationMode.RADIATION_ONLY:
        run_radiation_only_comparison(
            skip_rad_hydro=skip_rad_hydro,
            skip_diffusion=skip_diffusion,
            skip_supersonic=skip_supersonic,
            show_plot=show_plot,
            save_png=save_png,
            save_gif=save_gif,
        )
    elif mode == VerificationMode.HYDRO_ONLY:
        run_hydro_only_comparison(
            skip_rad_hydro=skip_rad_hydro,
            skip_hydro_sim=skip_hydro_sim,
            skip_shock_solver=skip_shock_solver,
            show_plot=show_plot,
            save_png=save_png,
            save_gif=save_gif,
        )
    elif mode == VerificationMode.FULL_RAD_HYDRO:
        if compare_force_black_cases:
            run_full_rad_hydro_force_black_comparison(
                show_plot=show_plot,
                save_png=save_png,
                save_gif=save_gif,
            )
        elif compare_radiation_coeff_schemes:
            run_full_rad_hydro_radiation_coeff_scheme_comparison(
                show_plot=show_plot,
                save_png=save_png,
                save_gif=save_gif,
            )
        else:
            run_full_rad_hydro_comparison(
                skip_rad_hydro=skip_rad_hydro,
                skip_shussman=skip_shussman,
                show_plot=show_plot,
                save_png=save_png,
                save_gif=save_gif,
            )

def main() -> None:
    """Entry point: select mode and run comparison."""
    MODE = VerificationMode.FULL_RAD_HYDRO
    # MODE = VerificationMode.RADIATION_ONLY
    # MODE = VerificationMode.HYDRO_ONLY
    
    run_comparison(
        MODE,
        skip_rad_hydro=False,
        skip_diffusion=False,
        skip_supersonic=False,
        skip_hydro_sim=False,
        skip_shock_solver=False,
        skip_shussman=False,
        compare_force_black_cases=False,
        compare_radiation_coeff_schemes=False,
        show_plot=True,
        save_png=True,
        save_gif=True
    )

if __name__ == "__main__":
    main()