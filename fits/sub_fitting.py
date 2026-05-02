"""
Analytic ablation stack helpers: build AblationSolver from a rad_hydro preset,
plot self-similar dimensionless profiles, and compare to Shussman MATLAB ports.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast
from unittest.mock import patch

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
_MENAHEM_SOLVERS_DIR = Path(__file__).resolve().parents[1] / "menahem_solvers"
if str(_MENAHEM_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_SOLVERS_DIR))

from project3_code.fits.menahem_reproduction import (
    _ablation_kwargs_from_case,
    _build_mass_grid_uniform,
    _ns_amplitude_rescale,
)
from project3_code.menahem_solvers.ablation_solver import AblationSolver
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_MENAHEM_ABLATION_COMPARISON,
)
from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.shussman_solvers.shock_solver.manager_shock import manager_shock
from project3_code.shussman_solvers.shock_solver.materials_shock import (
    STEFAN_BOLTZMANN_KELVIN as SIGMA_SHOCK,
    Material,
)
from project3_code.shussman_solvers.subsonic_solver.manager_sub import manager_sub
from project3_code.shussman_solvers.subsonic_solver.materials_sub import (
    HEV_IN_KELVIN,
    STEFAN_BOLTZMANN_KELVIN as SIGMA_SUB,
    MaterialSub,
)
from project3_code.shussman_solvers.subsonic_solver.profiles_for_report_sub import (
    compute_profiles_for_report,
)
from project3_code.shussman_solvers.shock_solver.profiles_for_report_shock import (
    compute_shock_profiles,
)

T = TypeVar("T")


def ablation_solver_from_preset(
    preset: str = PRESET_MENAHEM_ABLATION_COMPARISON,
) -> AblationSolver:
    """
    Load a rad_hydro preset and construct AblationSolver with the same parameters
    as in ``menahem_reproduction`` (no grid simulation).
    """
    case, _config = get_preset(preset)
    if case.rho0 is None:
        raise ValueError("Preset case must define rho0 for AblationSolver.")
    if case.r is None:
        raise ValueError("Preset case must define EOS r (gamma-1).")
    return AblationSolver(**_ablation_kwargs_from_case(case))


def _beta(case: Any) -> float:
    return float(case.beta_Rosen)


def material_shock_from_case(case: Any) -> Material:
    """Shussman shock ``Material`` aligned with rad_hydro case EOS / density."""
    return Material(
        alpha=float(case.alpha),
        beta=_beta(case),
        lambda_=float(case.lambda_),
        mu=float(case.mu),
        f=float(case.f_Kelvin),
        g=float(case.g_Kelvin),
        sigma=float(SIGMA_SHOCK),
        r=float(case.r),
        V0=1.0 / float(case.rho0),
        name="rad_hydro_case",
    )


def material_sub_from_case(case: Any) -> MaterialSub:
    """Shussman subsonic ``MaterialSub`` aligned with the same case."""
    return MaterialSub(
        alpha=float(case.alpha),
        beta=_beta(case),
        lambda_=float(case.lambda_),
        mu=float(case.mu),
        f=float(case.f_Kelvin),
        g=float(case.g_Kelvin),
        sigma=float(SIGMA_SUB),
        r=float(case.r),
        name="rad_hydro_case",
    )


def _run_shussman_quiet(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run Shussman managers without tqdm bars or inner prints."""
    buf = io.StringIO()

    def _silent_trange(n: Any, **_kw: Any) -> range:
        return range(int(n))

    def _silent_tqdm(it: Any, **_kw: Any) -> Any:
        return it

    with (
        patch("tqdm.trange", _silent_trange),
        patch("tqdm.tqdm", _silent_tqdm),
        redirect_stdout(buf),
    ):
        return fn(*args, **kwargs)


def shussman_subsonic_dimensionless(
    case: Any,
    tau: float,
    *,
    iternum: int = 300,
    xsi0: float = 1.0,
    P0: float = 4.0,
) -> dict[str, np.ndarray]:
    """
    Shussman subsonic self-similar profiles vs ``xi`` (same ordering as Menahem
    ``dimensionless``: rho=1/V, P, U from state ``[V,V',P,P',u]``, and
    ``T = (P*V^(1-mu))^(1/beta)``).
    """
    mat = material_sub_from_case(case)
    out = _run_shussman_quiet(
        manager_sub,
        mat,
        float(tau),
        iternum=int(iternum),
        xsi0=float(xsi0),
        P0=float(P0),
    )
    *_, t_arr, x_arr = out
    t_arr = np.asarray(t_arr, dtype=float)
    x_arr = np.asarray(x_arr, dtype=float)
    V, P, U = x_arr[:, 0], x_arr[:, 2], x_arr[:, 4]
    beta, mu = mat.beta, mat.mu
    rho = 1.0 / V
    T_dim = (P * V ** (1.0 - mu)) ** (1.0 / beta)
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    return {
        "xsi": t_arr,
        "rho": rho[order],
        "P": P[order],
        "U": U[order],
        "T": T_dim[order],
    }


def shussman_shock_dimensionless(
    case: Any,
    tau: float,
    *,
    iternum: int = 24,
    xi_f0: float = 4.0,
) -> dict[str, np.ndarray]:
    """
    Shussman piston-shock profiles: ``x = [Vtilde, Ptilde, utilde]`` vs ``xi``,
    plus ``T`` from the same EOS combination as Menahem shock when ``f`` is set.
    """
    mat = material_shock_from_case(case)
    out = _run_shussman_quiet(
        manager_shock,
        mat,
        float(tau),
        iternum=int(iternum),
        xi_f0=float(xi_f0),
    )
    *_, t_arr, x_arr = out
    t_arr = np.asarray(t_arr, dtype=float)
    x_arr = np.asarray(x_arr, dtype=float)
    V, P, U = x_arr[:, 0], x_arr[:, 1], x_arr[:, 2]
    rho = 1.0 / V
    r, f, beta, mu = float(mat.r), float(mat.f), float(mat.beta), float(mat.mu)
    T_dim = ((P * rho ** (mu - 1.0)) / (r * f)) ** (1.0 / beta)
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    return {
        "xsi": t_arr,
        "rho": rho[order],
        "P": P[order],
        "U": U[order],
        "T": T_dim[order],
    }


def _menahem_heat_dimensionless(
    solver: AblationSolver,
    case: Any,
    *,
    time: float,
    num_cells: int,
) -> dict[str, np.ndarray]:
    """
    Sample on the same Lagrangian mass grid as ``grid_comparison_rad_hydro`` when
    ``SAME_GRID`` is false: ``_build_mass_grid_uniform(case, omega=0, num_cells)``.
    ``solve`` maps mass → ``xsi_vec`` internally (see ``SubsonicHeatWave.solve``).
    """
    heat = solver.heat_solver
    t = max(float(time), 1e-300)
    mass = np.asarray(
        _build_mass_grid_uniform(case, omega=0.0, num_cells=int(num_cells)),
        dtype=float,
    )
    res_h = heat.solve(mass=mass, time=t)
    return cast(dict[str, np.ndarray], res_h["dimensionless"])


def _menahem_shock_dimensionless(
    solver: AblationSolver,
    case: Any,
    *,
    time: float,
    num_cells: int,
) -> dict[str, np.ndarray]:
    """Same mass grid as heat; ``PistonShock.solve`` infers ``xsi`` from mass and time."""
    shock = solver.shock_solver
    t = max(float(time), 1e-300)
    mass = np.asarray(
        _build_mass_grid_uniform(case, omega=0.0, num_cells=int(num_cells)),
        dtype=float,
    )
    res_s = shock.solve(mass=mass, time=t)
    return cast(dict[str, np.ndarray], res_s["dimensionless"])


def _menahem_heat_physical(
    solver: AblationSolver,
    case: Any,
    *,
    time: float,
    num_cells: int,
) -> dict[str, np.ndarray]:
    """Physical profiles from ``SubsonicHeatWave.solve`` on the uniform mass grid."""
    heat = solver.heat_solver
    t = max(float(time), 1e-300)
    mass = np.asarray(
        _build_mass_grid_uniform(case, omega=0.0, num_cells=int(num_cells)),
        dtype=float,
    )
    res = heat.solve(mass=mass, time=t)
    return {
        "mass": mass,
        "rho": np.asarray(res["density"], dtype=float),
        "P": np.asarray(res["pressure"], dtype=float),
        "u": np.asarray(res["velocity"], dtype=float),
        "T": np.asarray(res["temperature"], dtype=float),
    }


def _menahem_shock_physical(
    solver: AblationSolver,
    case: Any,
    *,
    time: float,
    num_cells: int,
) -> dict[str, np.ndarray]:
    """Physical profiles from ``PistonShock.solve`` on the same mass grid."""
    shock = solver.shock_solver
    t = max(float(time), 1e-300)
    mass = np.asarray(
        _build_mass_grid_uniform(case, omega=0.0, num_cells=int(num_cells)),
        dtype=float,
    )
    res = shock.solve(mass=mass, time=t)
    rho = np.asarray(res["density"], dtype=float)
    p = np.asarray(res["pressure"], dtype=float)
    r = float(shock.r)
    if shock.eos_mu is not None:
        assert shock.eos_beta is not None and shock.eos_f is not None
        mu_b = float(shock.eos_mu)
        beta_b = float(shock.eos_beta)
        f_b = float(shock.eos_f)
        T = np.full_like(rho, np.nan, dtype=float)
        ok = np.isfinite(p) & np.isfinite(rho) & (rho > 0) & (p > 0)
        T[ok] = ((p[ok] * rho[ok] ** (mu_b - 1.0)) / (r * f_b)) ** (1.0 / beta_b)
    else:
        v = 1.0 / rho
        T = np.where(np.isfinite(p) & np.isfinite(rho) & (rho > 0), p * v / r, np.nan)
    return {
        "mass": mass,
        "rho": rho,
        "P": p,
        "u": np.asarray(res["velocity"], dtype=float),
        "T": T,
    }


def shussman_subsonic_physical(
    case: Any,
    tau: float,
    *,
    time_sec: float,
) -> dict[str, np.ndarray]:
    """
    Dimensional heat profiles at one time using ``compute_profiles_for_report``
    (same scaling as ``profiles_for_report_sub`` / MATLAB).
    ``T`` is returned in Kelvin for comparison to Menahem.
    """
    mat = material_sub_from_case(case)
    Tb_k = _ns_amplitude_rescale(float(case.T0_Kelvin), float(tau))
    T0_heV = Tb_k / float(HEV_IN_KELVIN)
    t_ns = float(time_sec) * 1e9
    data = compute_profiles_for_report(
        mat,
        T0_heV,
        float(tau),
        np.asarray([t_ns], dtype=float),
    )
    return {
        "mass": np.asarray(data["m_heat"][0, :], dtype=float),
        "rho": np.asarray(data["rho_heat"][0, :], dtype=float),
        "P": np.asarray(data["P_heat"][0, :], dtype=float),
        "u": np.asarray(data["u_heat"][0, :], dtype=float),
        "T": np.asarray(data["T_heat"][0, :], dtype=float) * float(HEV_IN_KELVIN),
    }


def shussman_shock_physical(
    case: Any,
    tau: float,
    *,
    P0_barye: float,
    time_sec: float,
) -> dict[str, np.ndarray]:
    """
    Dimensional shock profiles at one time (``compute_shock_profiles``).
    ``T`` is returned in Kelvin (Shussman report uses HeV then we convert).
    """
    mat = material_shock_from_case(case)
    t_ns = float(time_sec) * 1e9
    data = compute_shock_profiles(
        mat,
        float(P0_barye),
        float(tau),
        None,
        np.asarray([t_ns], dtype=float),
        patching_method=False,
        save_npz=None,
    )
    T_heV = np.asarray(data["T_shock"][0], dtype=float)
    return {
        "mass": np.asarray(data["m_shock"][0], dtype=float),
        "rho": np.asarray(data["rho_shock"][0], dtype=float),
        "P": np.asarray(data["P_shock"][0], dtype=float),
        "u": np.asarray(data["u_shock"][0], dtype=float),
        "T": T_heV * float(HEV_IN_KELVIN),
    }


def plot_menahem_vs_shussman_dimensional(
    solver: AblationSolver,
    case: Any,
    *,
    time: float = 2e-9,
    num_cells: int = 400,
    heat_tau: float | None = None,
    save_paths: tuple[Optional[Path], Optional[Path]] = (None, None),
    show: bool = True,
) -> tuple[Figure, Figure]:
    """
    Two figures (heat and shock). Each is a 2×2 grid overlaying Menahem (solid)
    vs Shussman (dashed) for ``rho``, ``P``, ``u``, ``T`` vs Lagrangian mass ``m``,
    with matching CGS / Kelvin units on shared axes.
    """
    tau_h = float(case.tau) if heat_tau is None else float(heat_tau)
    tau_s = float(solver.shock_solver.tau)
    P0_shock = float(solver.shock_solver.p0)

    mh = _menahem_heat_physical(solver, case, time=time, num_cells=num_cells)
    ms = _menahem_shock_physical(solver, case, time=time, num_cells=num_cells)
    sh = _run_shussman_quiet(
        shussman_subsonic_physical, case, tau_h, time_sec=float(time)
    )
    ss = _run_shussman_quiet(
        shussman_shock_physical,
        case,
        tau_s,
        P0_barye=P0_shock,
        time_sec=float(time),
    )

    xlabel = r"$m$ [g/cm$^2$]"
    panels: tuple[tuple[str, str], ...] = (
        (r"$\rho$ [g/cm$^3$]", "rho"),
        (r"$P$ [Ba]", "P"),
        (r"$u$ [cm/s]", "u"),
        (r"$T$ [K]", "T"),
    )

    def _overlay_dimensional_fig(
        title: str, m: dict[str, np.ndarray], s: dict[str, np.ndarray]
    ) -> Figure:
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4))
        fig.suptitle(title, fontsize=12)
        for ax, (ylab, key) in zip(axs.ravel(), panels):
            xm = np.asarray(m["mass"], dtype=float)
            ym = np.asarray(m[key], dtype=float)
            xs = np.asarray(s["mass"], dtype=float)
            ys = np.asarray(s[key], dtype=float)
            mm = np.isfinite(xm) & np.isfinite(ym)
            ms = np.isfinite(xs) & np.isfinite(ys)
            ax.plot(xm[mm], ym[mm], color="#c0392b", lw=1.8, label="Menahem")
            ax.plot(xs[ms], ys[ms], color="#1f618d", lw=1.8, ls="--", label="Shussman")
            ax.set_ylabel(ylab)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig


    fig_h = _overlay_dimensional_fig(
        rf"Subsonic heat — dimensional ($t={time:g}$ s)",
        mh,
        sh,
    )
    fig_s = _overlay_dimensional_fig(
        rf"Piston shock — dimensional ($t={time:g}$ s)",
        ms,
        ss,
    )

    for fig, pth in zip((fig_h, fig_s), save_paths):
        if pth is not None:
            pth = Path(pth)
            pth.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pth, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_h)
        plt.close(fig_s)

    return fig_h, fig_s


def plot_menahem_vs_shussman_dimensionless(
    solver: AblationSolver,
    case: Any,
    *,
    time: float = 2e-9,
    num_cells: int = 400,
    heat_tau: float | None = None,
    shuss_sub_iternum: int = 300,
    shuss_shock_iternum: int = 24,
    save_paths: tuple[Optional[Path], Optional[Path]] = (None, None),
    show: bool = True,
) -> tuple[Figure, Figure]:
    """
    Two 2x2 figures: each panel overlays Menahem (solid) vs Shussman (dashed)
    dimensionless profiles for the heat wave and the shock region.
    """
    tau_h = float(case.tau) if heat_tau is None else float(heat_tau)
    tau_s = float(solver.shock_solver.tau)

    dh_m = _menahem_heat_dimensionless(solver, case, time=time, num_cells=num_cells)
    ds_m = _menahem_shock_dimensionless(solver, case, time=time, num_cells=num_cells)
    dh_s = shussman_subsonic_dimensionless(
        case, tau_h, iternum=shuss_sub_iternum
    )
    ds_s = shussman_shock_dimensionless(case, tau_s, iternum=shuss_shock_iternum)

    panels: tuple[tuple[str, str], ...] = (
        (r"$\tilde{\rho}$", "rho"),
        (r"$\tilde{P}$", "P"),
        (r"$\tilde{U}$", "U"),
        (r"$\tilde{T}$", "T"),
    )
    xlabel = r"$\xi$"

    def _overlay_fig(title: str, m: dict[str, np.ndarray], s: dict[str, np.ndarray]) -> Figure:
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4))
        fig.suptitle(title, fontsize=12)
        for ax, (ylab, key) in zip(axs.ravel(), panels):
            xm, ym = np.asarray(m["xsi"], dtype=float), np.asarray(m[key], dtype=float)
            xs, ys = np.asarray(s["xsi"], dtype=float), np.asarray(s[key], dtype=float)
            mm = np.isfinite(xm) & np.isfinite(ym)
            ms = np.isfinite(xs) & np.isfinite(ys)
            ax.plot(xm[mm], ym[mm], color="#c0392b", lw=1.8, label="Menahem")
            ax.plot(xs[ms], ys[ms], color="#1f618d", lw=1.8, ls="--", label="Shussman")
            ax.set_ylabel(ylab)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig

    fig_h = _overlay_fig(
        r"Subsonic heat wave: Menahem vs Shussman ($\tau=T$-drive exponent from case)",
        dh_m,
        dh_s,
    )
    fig_s = _overlay_fig(
        r"Piston shock: Menahem vs Shussman ($\tau=P$-drive from Menahem shock solver)",
        ds_m,
        ds_s,
    )

    for fig, pth in zip((fig_h, fig_s), save_paths):
        if pth is not None:
            pth = Path(pth)
            pth.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pth, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_h)
        plt.close(fig_s)

    return fig_h, fig_s

if __name__ == "__main__":
    from project3_code.rad_hydro_sim.output_paths import get_self_similar_fitting_figures_dir

    _case, _cfg = get_preset(PRESET_MENAHEM_ABLATION_COMPARISON)
    _solver = ablation_solver_from_preset()
    _out = get_self_similar_fitting_figures_dir()
    _out.mkdir(parents=True, exist_ok=True)

    plot_menahem_vs_shussman_dimensionless(
        _solver,
        _case,
        time=2e-9,
        num_cells=1000,
        show=False,
        save_paths=(
            _out / "dimensionless_heat_menahem_vs_shussman.png",
            _out / "dimensionless_shock_menahem_vs_shussman.png",
        ),
    )

    plot_menahem_vs_shussman_dimensional(
        _solver,
        _case,
        time=2e-9,
        num_cells=1000,
        show=False,
        save_paths=(
            _out / "dimensional_heat_menahem_vs_shussman.png",
            _out / "dimensional_shock_menahem_vs_shussman.png",
        ),
    )

    print(
        "Saved:\n"
        f"  {_out / 'dimensionless_heat_menahem_vs_shussman.png'}\n"
        f"  {_out / 'dimensionless_shock_menahem_vs_shussman.png'}\n"
        f"  {_out / 'dimensional_heat_menahem_vs_shussman.png'}\n"
        f"  {_out / 'dimensional_shock_menahem_vs_shussman.png'}\n"
    )
