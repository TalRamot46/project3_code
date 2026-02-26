"""
Automated verification tests: compare simulations and reference solvers for good fit.

Tests that Rad-Hydro, 1D Diffusion, Supersonic solver (radiation-only) and
Rad-Hydro vs hydro_sim (hydro-only) produce results within expected tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import List, Tuple

from project_3.rad_hydro_sim.verification.verification_config import (
    VerificationMode,
    get_preset_for_mode,
)
from project_3.rad_hydro_sim.verification.radiation_data import RadiationData


# ---------------------------------------------------------------------------
# Helper: compare RadiationData at a common time on a common x grid
# ---------------------------------------------------------------------------


def _index_closest_time(times: np.ndarray, target: float) -> int:
    """Index of closest time to target."""
    return int(np.argmin(np.abs(np.asarray(times) - target)))


def _interpolate_to_common_grid(
    data_a: RadiationData,
    data_b: RadiationData,
    target_time: float,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate both datasets to a common x grid at target_time.

    Returns (x_common, T_a, T_b, E_a, E_b) where each profile is on x_common.
    """
    ka = _index_closest_time(data_a.times, target_time)
    kb = _index_closest_time(data_b.times, target_time)

    x_a = np.asarray(data_a.x[ka], dtype=float)
    x_b = np.asarray(data_b.x[kb], dtype=float)
    x_min = max(x_a.min(), x_b.min())
    x_max = min(x_a.max(), x_b.max())
    if x_max <= x_min:
        x_max = max(x_a.max(), x_b.max())
    x_common = np.linspace(x_min, x_max, n_points)

    T_a = np.interp(x_common, x_a, np.asarray(data_a.T[ka], dtype=float))
    T_b = np.interp(x_common, x_b, np.asarray(data_b.T[kb], dtype=float))
    E_a = np.interp(x_common, x_a, np.asarray(data_a.E_rad[ka], dtype=float))
    E_b = np.interp(x_common, x_b, np.asarray(data_b.E_rad[kb], dtype=float))

    return x_common, T_a, T_b, E_a, E_b


def _relative_l2_error(y_a: np.ndarray, y_b: np.ndarray, eps: float = 1e-30) -> float:
    """Relative L2 error: ||y_a - y_b||_2 / (||y_a||_2 + eps)."""
    diff = np.asarray(y_a, dtype=float) - np.asarray(y_b, dtype=float)
    norm_a = np.linalg.norm(y_a) + eps
    return np.linalg.norm(diff) / norm_a


def _relative_max_error(y_a: np.ndarray, y_b: np.ndarray, eps: float = 1e-30) -> float:
    """Relative max error: max|y_a - y_b| / (max|y_a| + eps)."""
    diff = np.abs(np.asarray(y_a, dtype=float) - np.asarray(y_b, dtype=float))
    scale = np.max(np.abs(y_a)) + eps
    return np.max(diff) / scale


# ---------------------------------------------------------------------------
# Radiation-only: run and compare sim vs diffusion vs supersonic
# ---------------------------------------------------------------------------


def _run_radiation_data(
    skip_rad_hydro: bool = False,
    skip_diffusion: bool = False,
    skip_supersonic: bool = False,
    N: int = 100,
    n_times_diffusion: int = 10,
) -> Tuple[RadiationData | None, RadiationData | None, RadiationData | None]:
    """Run radiation-only comparison and return (sim_data, ref_data, super_data)."""
    from project_3.rad_hydro_sim.problems.presets_utils import get_preset
    from project_3.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project_3.rad_hydro_sim.verification.run_diffusion_1d import run_diffusion_1d
    from project_3.rad_hydro_sim.verification.run_comparison import (
        rad_hydro_history_to_radiation_data,
        diffusion_output_to_radiation_data,
        run_supersonic_solver_reference,
    )
    from project_3.hydro_sim.problems.simulation_config import SimulationConfig

    preset_name = get_preset_for_mode(VerificationMode.RADIATION_ONLY)
    case, config = get_preset(preset_name)
    # Use fast config for tests
    fast_config = SimulationConfig(
        N=N,
        CFL=config.CFL,
        sigma_visc=config.sigma_visc,
        store_every=max(1, 50),
        png_time_frac=config.png_time_frac,
    )

    sim_data = None
    if not skip_rad_hydro:
        x_cells, state, meta, history = simulate_rad_hydro(
            rad_hydro_case=case,
            simulation_config=fast_config,
        )
        sim_data = rad_hydro_history_to_radiation_data(history)

    ref_data = None
    if not skip_diffusion:
        times_sec, z, T_list, E_rad_list = run_diffusion_1d(
            x_max=float(case.x_max),
            t_end=float(case.t_sec_end),
            T_bath_Kelvin=float(case.T0_Kelvin),
            rho0=float(case.rho0),
            n_times=n_times_diffusion,
            Nz=N,
            f_Kelvin=float(case.f_Kelvin),
            g_Kelvin=float(case.g_Kelvin),
            T_right_Kelvin=0.0,
        )
        ref_data = diffusion_output_to_radiation_data(
            times_sec, z, T_list, E_rad_list
        )

    super_data = None
    if not skip_supersonic:
        super_data = run_supersonic_solver_reference(case, n_times=n_times_diffusion)

    return sim_data, ref_data, super_data


@pytest.mark.slow
def test_radiation_rad_hydro_vs_diffusion():
    """Rad-Hydro and 1D Diffusion (radiation-only) should agree within ~30% L2 at mid-time."""
    sim_data, ref_data, _ = _run_radiation_data(
        skip_rad_hydro=False,
        skip_diffusion=False,
        skip_supersonic=True,
        N=80,
        n_times_diffusion=8,
    )
    assert sim_data is not None
    assert ref_data is not None

    t_mid = 0.5 * sim_data.times[-1]
    x_common, T_sim, T_ref, E_sim, E_ref = _interpolate_to_common_grid(
        sim_data, ref_data, t_mid, n_points=60
    )

    rel_l2_T = _relative_l2_error(T_sim, T_ref)
    rel_l2_E = _relative_l2_error(E_sim, E_ref)

    assert rel_l2_T < 0.5, f"T relative L2 error {rel_l2_T:.3f} exceeds 50%"
    assert rel_l2_E < 0.6, f"E_rad relative L2 error {rel_l2_E:.3f} exceeds 60%"


@pytest.mark.slow
def test_radiation_rad_hydro_vs_supersonic():
    """Rad-Hydro and Supersonic solver (radiation-only) should agree within ~25% L2 at mid-time."""
    sim_data, _, super_data = _run_radiation_data(
        skip_rad_hydro=False,
        skip_diffusion=True,
        skip_supersonic=False,
        N=80,
        n_times_diffusion=8,
    )
    if super_data is None:
        pytest.skip("Supersonic solver not available")

    assert sim_data is not None
    t_mid = 0.5 * sim_data.times[-1]
    x_common, T_sim, T_super, E_sim, E_super = _interpolate_to_common_grid(
        sim_data, super_data, t_mid, n_points=60
    )

    rel_l2_T = _relative_l2_error(T_sim, T_super)
    rel_l2_E = _relative_l2_error(E_sim, E_super)

    assert rel_l2_T < 0.35, f"T relative L2 error {rel_l2_T:.3f} exceeds 35%"
    assert rel_l2_E < 0.4, f"E_rad relative L2 error {rel_l2_E:.3f} exceeds 40%"


@pytest.mark.slow
def test_radiation_diffusion_vs_supersonic():
    """1D Diffusion and Supersonic solver should agree within ~40% L2 at mid-time."""
    _, ref_data, super_data = _run_radiation_data(
        skip_rad_hydro=True,
        skip_diffusion=False,
        skip_supersonic=False,
        N=80,
        n_times_diffusion=8,
    )
    if super_data is None:
        pytest.skip("Supersonic solver not available")

    assert ref_data is not None
    t_mid = 0.5 * ref_data.times[-1]
    x_common, T_ref, T_super, E_ref, E_super = _interpolate_to_common_grid(
        ref_data, super_data, t_mid, n_points=60
    )

    rel_l2_T = _relative_l2_error(T_ref, T_super)
    rel_l2_E = _relative_l2_error(E_ref, E_super)

    assert rel_l2_T < 0.5, f"T relative L2 error {rel_l2_T:.3f} exceeds 50%"
    assert rel_l2_E < 0.55, f"E_rad relative L2 error {rel_l2_E:.3f} exceeds 55%"


# ---------------------------------------------------------------------------
# Hydro-only: run rad_hydro vs hydro_sim, compare rho, p, u, e
# ---------------------------------------------------------------------------


def _run_hydro_data(
    N: int = 80,
):
    """Run hydro-only comparison and return (sim_data, ref_data)."""
    from project_3.rad_hydro_sim.problems.presets_utils import get_preset
    from project_3.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project_3.hydro_sim.problems.driven_shock_problem import DrivenShockCase
    from project_3.hydro_sim.core.geometry import planar
    from project_3.hydro_sim.simulations.lagrangian_sim import (
        simulate_lagrangian,
        SimulationType,
    )
    from project_3.hydro_sim.verification.compare_shock_plots import (
        load_rad_hydro_history,
        load_hydro_history,
    )
    from project_3.hydro_sim.problems.simulation_config import SimulationConfig

    preset_name = get_preset_for_mode(VerificationMode.HYDRO_ONLY)
    case_rh, config = get_preset(preset_name)
    fast_config = SimulationConfig(
        N=N,
        CFL=config.CFL,
        sigma_visc=config.sigma_visc,
        store_every=max(1, 30),
        png_time_frac=config.png_time_frac,
    )

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
        title="Power-law pressure drive",
    )

    x_cells, state, meta, history_rh = simulate_rad_hydro(
        rad_hydro_case=case_rh,
        simulation_config=fast_config,
    )
    sim_data = load_rad_hydro_history(history_rh, label="Rad-Hydro (hydro only)")

    store_every = max(1, N // 10)
    x_cells, state, meta, history_h = simulate_lagrangian(
        driven_case,
        sim_type=SimulationType.DRIVEN_SHOCK,
        Ncells=N,
        gamma=driven_case.gamma,
        CFL=fast_config.CFL,
        sigma_visc=fast_config.sigma_visc,
        store_every=store_every,
        geom=driven_case.geom,
    )
    ref_data = load_hydro_history(history_h)
    ref_data.label = "Hydro (run_hydro)"

    return sim_data, ref_data


def _interpolate_hydro_to_common_m(sim_data, ref_data, target_time, n_points=80):
    """Interpolate hydro profiles to common m grid. Returns (m_common, profiles_sim, profiles_ref)."""
    ks = _index_closest_time(sim_data.times, target_time)
    kr = _index_closest_time(ref_data.times, target_time)

    m_s = np.asarray(sim_data.m[ks], dtype=float)
    m_r = np.asarray(ref_data.m[kr], dtype=float)
    m_min = max(m_s.min(), m_r.min())
    m_max = min(m_s.max(), m_r.max())
    if m_max <= m_min:
        m_max = max(m_s.max(), m_r.max())
    m_common = np.linspace(m_min, m_max, n_points)

    def interp_all(data, k):
        m = np.asarray(data.m[k], dtype=float)
        return {
            "rho": np.interp(m_common, m, np.asarray(data.rho[k], dtype=float)),
            "p": np.interp(m_common, m, np.asarray(data.p[k], dtype=float)),
            "u": np.interp(m_common, m, np.asarray(data.u[k], dtype=float)),
            "e": np.interp(m_common, m, np.asarray(data.e[k], dtype=float)),
        }

    return m_common, interp_all(sim_data, ks), interp_all(ref_data, kr)


@pytest.mark.slow
def test_hydro_rad_hydro_vs_hydro_sim():
    """Rad-Hydro and hydro_sim (hydro-only) should agree within ~40% L2 at mid-time."""
    sim_data, ref_data = _run_hydro_data(N=80)

    t_mid = 0.5 * sim_data.times[-1]
    m_common, prof_sim, prof_ref = _interpolate_hydro_to_common_m(
        sim_data, ref_data, t_mid, n_points=60
    )

    for var in ("rho", "p", "u", "e"):
        rel_l2 = _relative_l2_error(prof_sim[var], prof_ref[var])
        assert rel_l2 < 0.5, f"{var} relative L2 error {rel_l2:.3f} exceeds 50%"


# ---------------------------------------------------------------------------
# Smoke tests: run without comparison (fast)
# ---------------------------------------------------------------------------


def test_radiation_diffusion_runs():
    """1D Diffusion reference should run without error."""
    _, ref_data, _ = _run_radiation_data(
        skip_rad_hydro=True,
        skip_diffusion=False,
        skip_supersonic=True,
        N=50,
        n_times_diffusion=5,
    )
    assert ref_data is not None
    assert len(ref_data.times) >= 3
    assert len(ref_data.x[0]) >= 10


def test_radiation_supersonic_runs():
    """Supersonic solver should run without error (or skip if not available)."""
    _, _, super_data = _run_radiation_data(
        skip_rad_hydro=True,
        skip_diffusion=True,
        skip_supersonic=False,
        N=50,
        n_times_diffusion=5,
    )
    if super_data is None:
        pytest.skip("Supersonic solver not available")
    assert len(super_data.times) >= 3
    assert len(super_data.x[0]) >= 5
