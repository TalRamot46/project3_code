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
import scipy.integrate

if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(scipy.integrate, "cumtrapz"):
    scipy.integrate.cumtrapz = scipy.integrate.cumulative_trapezoid
if not hasattr(np, "trapz"):
    if hasattr(scipy.integrate, "trapezoid"):
        np.trapz = scipy.integrate.trapezoid
    elif hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid

# Menahem modules are a flat directory (no __init__.py); add it to sys.path.
_MENAHEM_DIR = Path(__file__).resolve().parents[2] / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

# Monkeypatch PistonShock to handle NumPy array inputs gracefully
def _patch_piston_shock(piston_shock_module):
    if hasattr(piston_shock_module, "PistonShock"):
        cls = piston_shock_module.PistonShock
        original_fxsi_s = cls.fxsi_s
        def patched_fxsi_s(self, xsi_s):
            if hasattr(xsi_s, "item"):
                xsi_s = float(xsi_s.item())
            elif isinstance(xsi_s, (list, np.ndarray)) and len(xsi_s) == 1:
                xsi_s = float(xsi_s[0])
            else:
                xsi_s = float(xsi_s)
            return original_fxsi_s(self, xsi_s)
        cls.fxsi_s = patched_fxsi_s

try:
    import project3_code.menahem_new.piston_shock_og as ps_og1
    _patch_piston_shock(ps_og1)
except ImportError:
    pass

try:
    import piston_shock_og as ps_og2
    _patch_piston_shock(ps_og2)
except ImportError:
    pass

# Ensure project3_code is on path (when run as script): add parent of repo root so "project3_code" package resolves
_REPO_PARENT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
from rad_hydro_sim.verification.radiation_data import RadiationSimData
from rad_hydro_sim.verification.hydro_data import RadHydroSimData
from hydro_sim.verification.compare_shock_plots import HydroSimData
from rad_hydro_sim.simulation.radiation_step import (
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
        omega=float(getattr(case, "omega", 0.0)),
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
        omega=float(case.omega),
        f_shock=heat["f"],
        beta_shock=heat["beta"],
        mu_shock=heat["mu"],
        gamma_shock=heat["gamma"],
    )


# ---------------------------------------------------------------------------
# Lagrangian mass grid used by all three builders
# ---------------------------------------------------------------------------

from hydro_sim.core.grid import make_nodes

def _build_mass_grid(
    case,
    num_cells: int = 400,
) -> np.ndarray:
    """Return a monotonically increasing Lagrangian mass coordinate array.

    Uses make_nodes to build the spatial grid (handling both uniform and
    gradually refined non-uniform omega != 0 grids), converts it into cumulative
    mass, and prepends a tiny cell for solver mass boundary indexing.
    """
    x_min = float(getattr(case, "x_min", 0.0))
    x_max = float(case.x_max)
    omega = float(getattr(case, "omega", 0.0))

    coordinate = make_nodes(x_min, x_max, num_cells, omega=omega)
    dx = coordinate[1:] - coordinate[:-1]

    if omega != 1.0:
        density = (case.rho0 / (1.0 - omega)) * (coordinate[1:]**(1.0 - omega) - coordinate[:-1]**(1.0 - omega)) / dx
    else:
        coord_non_zero = np.where(coordinate == 0.0, 1e-30, coordinate)
        density = case.rho0 * np.log(coord_non_zero[1:] / coord_non_zero[:-1]) / dx

    mass_cells = density * dx
    mass_from_cells = np.cumsum(mass_cells)
    mass = np.array([1e-30, 1e-7 * mass_from_cells[0]] + list(mass_from_cells))
    return mass


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
        from subsonic_heat_wave_og import SubsonicHeatWave  # type: ignore
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
    p_list = []
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
        p_list.append(sol["pressure"])

    return times, p_list, np.array(x_list) / float(case.rho0)

    # return RadiationSimData(
    #     times=times,
    #     x=x_list,
    #     T=T_list,
    #     E_rad=E_list,
    #     label=label,
    #     color=color,
    #     linestyle=linestyle,
    # )


def run_menahem_shock_reference(
    case,
    times_sec: np.ndarray,
    num_cells: int = 400,
    label: str = "Menahem (piston shock)",
    color: str = MENAHEM_COLOR,
    linestyle=MENAHEM_LINESTYLE,
    use_master_grid: bool = True,
) -> Optional[HydroSimData]:
    """Build hydro-only reference from Menahem's ``PistonShock``."""
    try:
        from piston_shock_og import PistonShock  # type: ignore
    except ImportError as exc:
        print(f"  Could not import Menahem PistonShock: {exc}; skipping.")
        return None

    times = np.asarray(times_sec, dtype=float).ravel()
    times = times[times > 0.0]
    if times.size == 0:
        print("  Menahem shock: no positive times provided; skipping.")
        return None

    kwargs = _shock_kwargs_from_case(case)
    kwargs["use_master_grid"] = use_master_grid
    print(f"  Menahem PistonShock: p0={kwargs['p0']:g} Ba, tau={kwargs['tau']:g}, use_master_grid={use_master_grid}")
    solver = PistonShock(**kwargs)

    mass = _build_mass_grid(case, num_cells=num_cells)

    m_list, x_list = [], []
    rho_list, p_list, u_list, e_list = [], [], [], []

    from tqdm import tqdm
    for t in tqdm(times, desc="Menahem PistonShock"):
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
    config,
    times_sec: np.ndarray,
) -> Optional[RadHydroSimData]:
    """Build full-rad-hydro reference from Menahem's ``AblationSolver``.

    The solver patches the subsonic heat wave and the piston shock internally;
    no manual splicing is needed.
    """

    num_cells = config.N
    label = "Analytic"
    color = MENAHEM_COLOR
    linestyle = MENAHEM_LINESTYLE

    try:
        from ablation_solver_og import AblationSolver  # type: ignore
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
        # Eulerian positions: Keep in absolute laboratory frame (no shift),
        # matching the simulation's absolute laboratory frame history.x.
        pos = np.asarray(sol["position"], dtype=float)
        x = pos

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

    return RadHydroSimData(
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


def run_menahem_supersonic_instantaneous_reference(
    case,
    times_sec: np.ndarray,
    num_cells: int = 400,
    label: str = "Menahem (supersonic instantaneous)",
    color: str = MENAHEM_COLOR,
    linestyle=MENAHEM_LINESTYLE,
) -> Optional[RadiationSimData]:
    """Build radiation-only reference from ``SupersonicInstantaneousAnalytic``.

    Returns ``RadiationSimData`` with ``T`` in Kelvin and ``E_rad`` in erg/cm^3.
    """
    try:
        from supersonic_instantaneous_analytic import SupersonicInstantaneousAnalytic
    except ImportError as exc:
        print(f"  Could not import SupersonicInstantaneousAnalytic: {exc}; skipping.")
        return None

    times = np.asarray(times_sec, dtype=float).ravel()
    times = times[times > 0.0]
    if times.size == 0:
        print("  Menahem supersonic instantaneous: no positive times provided; skipping.")
        return None

    kwargs = dict(
        g=float(case.g_Kelvin),
        alpha=float(case.alpha),
        lambdap=float(case.lambda_),
        f=float(case.f_Kelvin),
        beta=float(case.beta_Rosen),
        mu=float(case.mu),
        rho0=float(case.rho0),
        omega=float(getattr(case, "omega", 0.0)),
        T0_Kelvin=float(case.T0_Kelvin),
    )

    solver = SupersonicInstantaneousAnalytic(**kwargs)

    x_grid = np.linspace(float(case.x_min or 1e-12), float(case.x_max), num_cells)

    x_list: list[np.ndarray] = []
    T_list: list[np.ndarray] = []
    E_list: list[np.ndarray] = []

    for t in times:
        T_K = solver.temperature_profile(x_grid, float(t))
        E_rad = a_Kelvin * (T_K ** 4)

        x_list.append(x_grid.copy())
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



if __name__ == "__main__":
    # Quick smoke test: heat/ablation presets need T0_Kelvin; piston shock needs P0_Barye.
    from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
    from project3_code.rad_hydro_sim.problems.presets_config import (
        PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK,
        PRESET_CONSTANT_PRESSURE,
        PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE
    )

    case_heat, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK)
    case_shock, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    t_heat = np.array([0.25, 0.5, 0.75]) * float(case_heat.t_sec_end)
    t_shock = np.array([0.25, 0.5, 0.75]) * float(case_shock.t_sec_end)
    colors = ["royalblue", "darkorange", "crimson"]
    # plot the results
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    fig, ax = plt.subplots(figsize=(8, 6))

    # data = run_menahem_piecewise_reference(case_heat, t_heat)
    
    # times,p_list, m_list = run_menahem_subsonic_reference(case_heat,t_heat)
    
    # for i in range(len(times)):
    #     ax.plot(m_list[i]*1000,p_list[i]/1e12, color=colors[i], label=f"t={times[i]*1e9:.1f} ns")
    # ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm²]", fontsize=12)
    # ax.set_xlim(0, 20)
    # ax.set_ylim(0,6)
    # ax.set_ylabel("Pressure (MBar)")
    # ax.grid(True)
    # ax.legend(loc="upper right")

    # # add a zoomed version in x=[0,2] and y=[0,4]
    # # on the same figure, with zooming illustration
    # axins = inset_axes(ax, width="40%", height="40%", loc='center')
    # for i in range(len(times)):
    #     axins.plot(m_list[i]*1000,p_list[i]/1e12, color=colors[i], label=f"t={times[i]*1e9:.1f} ns")
    # axins.set_xlim(0, 1.5)
    # axins.set_ylim(0, 4)
    # axins.grid(True)
    
    # # Draw a link between the main axes and the zoomed inset axes
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # plt.show()


    data = run_menahem_shock_reference(case_shock, t_heat)
    
    for i in range(len(data.times)):
        ax.plot(data.m[i]*1000,data.p[i]/1e12, color=colors[i], label=f"t={data.times[i]*1e9:.1f} ns")
    ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm²]", fontsize=12)
    ax.set_xlim(0, 20)
    ax.set_ylim(0,6)
    ax.set_ylabel("Pressure (MBar)")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.show()
