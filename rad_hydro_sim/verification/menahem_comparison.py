# verification/menahem_comparison.py
"""
Menahem's analytic / semi-analytic solvers wrapped for rad_hydro_sim verification.

The Menahem solver package lives as a plain folder ``project3_code/menahem_new``
(no ``__init__.py``) containing three modules:

- ``subsonic_heat_wave.SubsonicHeatWave``: subsonic ablative heat wave
  (equivalent in role to Shussman's subsonic solver).
- ``piston_shock.PistonShock``: power-law driven piston shock
  (equivalent in role to Shussman's shock solver).
- ``ablation_solver.AblationSolver``: piecewise patching of the two above
  (equivalent in role to Shussman's subsonic + shock piecewise reference).

This module translates a ``RadHydroCase`` into the kwargs Menahem expects and
builds the same data containers the existing verification plots consume:

- ``run_menahem_subsonic_reference``  -> ``RadiationSimData``  (radiation-only).
- ``run_menahem_shock_reference``     -> ``HydroSimData``      (hydro-only).
- ``run_menahem_piecewise_reference`` -> ``RadHydroData``      (full rad-hydro).

Time-amplitude convention
-------------------------
Menahem consumes time in seconds in the power-law drives
``T(0, t) = Tb * t**tau`` and ``P(m=0, t) = p0 * t**tau``. The rest of the
project (Shussman references, rad_hydro driver) instead uses nanoseconds as the
time unit in the power-law (``T(t_ns) = T0_HeV * t_ns**tau``). To keep the three
curves physically identical for a given ``RadHydroCase`` we pre-scale Menahem's
boundary amplitudes:

    Tb   = T0_Kelvin  * (1e9) ** tau
    p0   = P0_Barye   * (1e9) ** tau

For tau == 0 (constant drive) both reduce to the raw ``T0_Kelvin`` / ``P0_Barye``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Menahem modules are a flat directory (no __init__.py); add it to sys.path.
_MENAHEM_DIR = Path(__file__).resolve().parents[2] / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

from project3_code.rad_hydro_sim.verification.radiation_data import RadiationSimData
from project3_code.rad_hydro_sim.verification.hydro_data import RadHydroData
from project3_code.hydro_sim.verification.compare_shock_plots import HydroSimData
from project3_code.rad_hydro_sim.simulation.radiation_step import (
    a_Kelvin,
    KELVIN_PER_HEV,
)


# Default styling so the Menahem curves are visually distinguishable from
# rad_hydro (blue/solid) and Shussman (green/dash-dot).
MENAHEM_COLOR = "magenta"
MENAHEM_LINESTYLE = (0, (3, 1, 1, 1))  # dash-dot-ish dashed


# ---------------------------------------------------------------------------
# Parameter translation (case -> Menahem kwargs)
# ---------------------------------------------------------------------------

def _ns_amplitude_rescale(amp: float, tau: float) -> float:
    """Convert a power-law amplitude defined with t in ns to one with t in seconds.

    If ``A_phys(t_ns) = amp * t_ns**tau`` and we want ``A_phys(t_sec) = amp' * t_sec**tau``
    with the same physical curve, then ``amp' = amp * (1e9)**tau``.
    """
    return float(amp) * (1.0e9 ** float(tau))


def _heat_kwargs_from_case(case) -> dict:
    """Kwargs for ``SubsonicHeatWave`` built from a ``RadHydroCase``."""
    tau = float(case.tau or 0.0)
    Tb = _ns_amplitude_rescale(float(case.T0_Kelvin), tau)
    return dict(
        Tb=Tb,
        tau=tau,
        g=float(case.g_Kelvin),
        alpha=float(case.alpha),
        lambdap=float(case.lambda_),
        f=float(case.f_Kelvin),
        beta=float(case.beta_Rosen),
        mu=float(case.mu),
        gamma=float(case.r) + 1.0,
    )


def _shock_kwargs_from_case(case) -> dict:
    """Kwargs for ``PistonShock`` (standalone hydro drive, time in seconds)."""
    tau = float(case.tau or 0.0)
    p0 = _ns_amplitude_rescale(float(case.P0_Barye), tau)
    return dict(
        rho0=float(case.rho0),
        omega=0.0,
        p0=p0,
        tau=tau,
        gamma=float(case.r) + 1.0,
    )


def _ablation_kwargs_from_case(case) -> dict:
    """Kwargs for ``AblationSolver`` (heat + shock with the same EOS)."""
    heat = _heat_kwargs_from_case(case)
    return dict(
        Tb=heat["Tb"],
        tau=heat["tau"],
        g=heat["g"],
        alpha=heat["alpha"],
        lambdap=heat["lambdap"],
        f_heat=heat["f"],
        beta_heat=heat["beta"],
        mu_heat=heat["mu"],
        gamma_heat=heat["gamma"],
        rho0=float(case.rho0),
        omega=0.0,
        f_shock=heat["f"],
        beta_shock=heat["beta"],
        mu_shock=heat["mu"],
        gamma_shock=heat["gamma"],
    )


# ---------------------------------------------------------------------------
# Lagrangian mass grid used by all three builders
# ---------------------------------------------------------------------------

def _build_mass_grid(
    case,
    num_cells: int = 400,
    m_min_factor: float = 1e-10,
) -> np.ndarray:
    """Return a monotonically increasing Lagrangian mass coordinate array.

    Based on the discretisation used by Menahem's own tests
    (see ``menahem_new/ablation_solver.py::test_profiles``): build a dense spatial
    grid on ``[0, x_max]``, turn it into cumulative mass at ``case.rho0``
    (uniform initial density), and prepend a tiny cell so the solver has a
    well-defined minimum mass strictly greater than zero.
    """
    x_max = float(case.x_max)
    coordinate = np.linspace(0.0, x_max, int(num_cells) + 1)
    dx = np.diff(coordinate)
    density = np.full_like(dx, float(case.rho0))
    mass_cells = density * dx
    mass = np.cumsum(mass_cells)
    # prepend two tiny coordinates so the first real cell has m > 0 with room
    tiny_m = m_min_factor * mass[0]
    return np.concatenate([[tiny_m * 1e-5, tiny_m], mass])


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def run_menahem_subsonic_reference(
    case,
    times_sec: np.ndarray,
    num_cells: int = 400,
    label: str = "Menahem (subsonic)",
    color: str = MENAHEM_COLOR,
    linestyle=MENAHEM_LINESTYLE,
) -> Optional[RadiationSimData]:
    """Build radiation-only reference from Menahem's ``SubsonicHeatWave``.

    Returns ``RadiationSimData`` with ``T`` in Kelvin and ``E_rad`` in erg/cm^3
    (same conventions as the 1D Diffusion / Supersonic references).
    """
    try:
        from subsonic_heat_wave import SubsonicHeatWave  # type: ignore
    except ImportError as exc:
        print(f"  Could not import Menahem SubsonicHeatWave: {exc}; skipping.")
        return None

    times = np.asarray(times_sec, dtype=float).ravel()
    times = times[times > 0.0]
    if times.size == 0:
        print("  Menahem subsonic: no positive times provided; skipping.")
        return None

    kwargs = _heat_kwargs_from_case(case)
    print(f"  Menahem SubsonicHeatWave: Tb={kwargs['Tb']:g} K, tau={kwargs['tau']:g}")
    solver = SubsonicHeatWave(**kwargs).find_xsi_f()

    mass = _build_mass_grid(case, num_cells=num_cells)

    x_list: list[np.ndarray] = []
    T_list: list[np.ndarray] = []
    E_list: list[np.ndarray] = []
    for t in times:
        sol = solver.solve(mass=mass, time=float(t))
        rho = np.asarray(sol["density"], dtype=float)
        T_K = np.asarray(sol["temperature"], dtype=float)
        # Rebuild Eulerian x from mass/rho so x starts at 0 at the boundary
        # (Menahem's raw ``position`` sits at negative x because the ablation
        # boundary has moved into -x; the rad_hydro domain uses x>=0).
        dm = np.diff(mass, prepend=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.cumsum(np.abs(dm) / (rho + 1e-30))
        # Clip non-physical / NaN entries beyond the heat front
        T_K = np.where(np.isfinite(T_K), T_K, 0.0)
        E_rad = a_Kelvin * T_K ** 4
        x_list.append(x)
        T_list.append(T_K)
        E_list.append(E_rad)

    return RadiationSimData(
        times=times,
        x=x_list,
        T=T_list,
        E_rad=E_list,
        label=label,
        color=color,
        linestyle=linestyle,
    )


def run_menahem_shock_reference(
    case,
    times_sec: np.ndarray,
    num_cells: int = 400,
    label: str = "Menahem (piston shock)",
    color: str = MENAHEM_COLOR,
    linestyle=MENAHEM_LINESTYLE,
) -> Optional[HydroSimData]:
    """Build hydro-only reference from Menahem's ``PistonShock``."""
    try:
        from piston_shock import PistonShock  # type: ignore
    except ImportError as exc:
        print(f"  Could not import Menahem PistonShock: {exc}; skipping.")
        return None

    times = np.asarray(times_sec, dtype=float).ravel()
    times = times[times > 0.0]
    if times.size == 0:
        print("  Menahem shock: no positive times provided; skipping.")
        return None

    kwargs = _shock_kwargs_from_case(case)
    print(f"  Menahem PistonShock: p0={kwargs['p0']:g} Ba, tau={kwargs['tau']:g}")
    solver = PistonShock(**kwargs)

    mass = _build_mass_grid(case, num_cells=num_cells)

    m_list, x_list = [], []
    rho_list, p_list, u_list, e_list = [], [], [], []
    for t in times:
        sol = solver.solve(mass=mass, time=float(t))
        m_list.append(mass.copy())
        x_list.append(np.asarray(sol["position"], dtype=float))
        rho_list.append(np.asarray(sol["density"], dtype=float))
        p_list.append(np.asarray(sol["pressure"], dtype=float))
        u_list.append(np.asarray(sol["velocity"], dtype=float))
        e_list.append(np.asarray(sol["sie"], dtype=float))

    return HydroSimData(
        times=times,
        m=m_list,
        x=x_list,
        rho=rho_list,
        p=p_list,
        u=u_list,
        e=e_list,
        label=label,
        color=color,
        linestyle=linestyle,
    )


def run_menahem_piecewise_reference(
    case,
    times_sec: np.ndarray,
    num_cells: int = 400,
    label: str = "Menahem (ablation solver)",
    color: str = MENAHEM_COLOR,
    linestyle=MENAHEM_LINESTYLE,
) -> Optional[RadHydroData]:
    """Build full-rad-hydro reference from Menahem's ``AblationSolver``.

    The solver patches the subsonic heat wave and the piston shock internally;
    no manual splicing is needed.
    """
    try:
        from ablation_solver import AblationSolver  # type: ignore
    except ImportError as exc:
        print(f"  Could not import Menahem AblationSolver: {exc}; skipping.")
        return None

    times = np.asarray(times_sec, dtype=float).ravel()
    times = times[times > 0.0]
    if times.size == 0:
        print("  Menahem ablation: no positive times provided; skipping.")
        return None

    kwargs = _ablation_kwargs_from_case(case)
    print(
        f"  Menahem AblationSolver: Tb={kwargs['Tb']:g} K, "
        f"tau={kwargs['tau']:g}, rho0={kwargs['rho0']:g}"
    )
    solver = AblationSolver(**kwargs)

    mass = _build_mass_grid(case, num_cells=num_cells)

    m_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    rho_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    u_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    T_list: list[np.ndarray] = []
    E_list: list[np.ndarray] = []

    for t in times:
        sol = solver.solve(mass=mass, time=float(t))
        rho = np.asarray(sol["density"], dtype=float)
        p = np.asarray(sol["pressure"], dtype=float)
        u = np.asarray(sol["velocity"], dtype=float)
        e = np.asarray(sol["sie"], dtype=float)
        T_K = np.asarray(sol["temperature"], dtype=float)
        # Eulerian positions: Menahem anchors the ablation boundary at a
        # negative x; shift so the boundary sits at x=0 (rad_hydro convention).
        pos = np.asarray(sol["position"], dtype=float)
        x = pos - float(sol["boundary_position"])

        T_HeV = T_K / KELVIN_PER_HEV
        T_HeV = np.where(np.isfinite(T_HeV), T_HeV, 0.0)
        E_rad = a_Kelvin * np.where(np.isfinite(T_K), T_K, 0.0) ** 4

        m_list.append(mass.copy())
        x_list.append(x)
        rho_list.append(rho)
        p_list.append(p)
        u_list.append(u)
        e_list.append(e)
        T_list.append(T_HeV)
        E_list.append(E_rad)

    return RadHydroData(
        times=times,
        m=m_list,
        x=x_list,
        rho=rho_list,
        p=p_list,
        u=u_list,
        e=e_list,
        T=T_list,
        E_rad=E_list,
        T_material=list(T_list),  # Planck equilibrium; plotter reuses same array
        label=label,
        color=color,
        linestyle=linestyle,
    )


if __name__ == "__main__":
    # Quick smoke test: heat/ablation presets need T0_Kelvin; piston shock needs P0_Barye.
    from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
    from project3_code.rad_hydro_sim.problems.presets_config import (
        PRESET_FIG_8,
        PRESET_CONSTANT_PRESSURE,
    )

    case_heat, _ = get_preset(PRESET_FIG_8)
    case_shock, _ = get_preset(PRESET_CONSTANT_PRESSURE)
    t_heat = np.array([0.25, 0.5, 0.75, 1.0]) * float(case_heat.t_sec_end)
    t_shock = np.array([0.25, 0.5, 0.75, 1.0]) * float(case_shock.t_sec_end)
    
    # plot the results
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(t_heat, run_menahem_subsonic_reference(case_heat, t_heat).T, label="subsonic")
    ax.plot(t_shock, run_menahem_shock_reference(case_shock, t_shock).T, label="shock")
    ax.plot(t_heat, run_menahem_piecewise_reference(case_heat, t_heat).T, label="piecewise")
    ax.legend()
    plt.show()
