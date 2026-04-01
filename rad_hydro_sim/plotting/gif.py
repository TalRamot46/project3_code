# ============================================================================
# GIF Animation (Generic for time-history data)
# ============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from . import mpl_style  # noqa: F401 - apply project style
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from project3_code.hydro_sim.simulations.lagrangian_sim import HydroHistory
from project3_code.hydro_sim.plotting.hydro_plots import _create_7panel_vertical_figure
from project3_code.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

if TYPE_CHECKING:
    from project3_code.rad_hydro_sim.verification.hydro_data import RadHydroData


def save_history_gif(
    history: "HydroHistory",
    case: RadHydroCase,
    gif_path: str = "simulation.gif",
    fps: int = 20,
    stride: int = 1,
    ref_data: Optional["RadHydroData"] = None,
    subtitle: str | None = None,
):
    """
    Save an animated GIF of the time-history data.

    Optionally overlays the piecewise Shussman solver result at each time step
    for verification (simulation vs reference comparison).

    Parameters:
        history: HydroHistory object with time-series data
        case: Problem case (for title info)
        gif_path: Output file path
        fps: Frames per second
        stride: Frame stride (skip frames for smaller file)
        ref_data: Optional RadHydroData (e.g. Shussman piecewise reference) to overlay
        subtitle: Second line under preset title (e.g. verification mode description)
    """
    # Display scaling: stored p [Barye], u [cm/s], e [erg/g]
    p_scale, u_scale, e_scale = 1e12, 1e5, 1e9

    has_T_material = hasattr(history, "T_material") and history.T_material is not None
    has_ref = ref_data is not None and len(ref_data.times) > 0

    fig, axes = _create_7panel_vertical_figure()
    k0 = 0
    m0 = history.m[k0] if hasattr(history, "m") else history.x[k0]
    lines = []
    lines.append(axes[0].plot(m0, history.rho[k0], lw=2, label="Simulation", color="blue")[0])
    lines.append(axes[1].plot(m0, history.p[k0] / p_scale, lw=2, label="Simulation", color="blue")[0])
    lines.append(axes[2].plot(m0, history.u[k0] / u_scale, lw=2, label="Simulation", color="blue")[0])
    lines.append(axes[3].plot(m0, history.e[k0] / e_scale, lw=2, label="Simulation", color="blue")[0])
    lines.append(axes[4].plot(m0, history.T_material[k0] if has_T_material else history.T[k0], lw=2, label="Simulation", color="blue")[0])
    lines.append(axes[5].plot(m0, history.T[k0], lw=2, label="Simulation", color="blue")[0])
    lines.append(axes[6].plot(m0, history.E_rad[k0], lw=2, label="Simulation", color="blue")[0])

    ref_lines = []
    if has_ref:
        ref_color = getattr(ref_data, "color", "green")
        ref_ls = getattr(ref_data, "linestyle", "-.")
        ref_label = getattr(ref_data, "label", "Shussman (piecewise)")
        ref_lines.append(axes[0].plot([], [], lw=1.5, color=ref_color, linestyle=ref_ls, label=ref_label)[0])
        ref_lines.append(axes[1].plot([], [], lw=1.5, color=ref_color, linestyle=ref_ls, label=ref_label)[0])
        ref_lines.append(axes[2].plot([], [], lw=1.5, color=ref_color, linestyle=ref_ls, label=ref_label)[0])
        ref_lines.append(axes[3].plot([], [], lw=1.5, color=ref_color, linestyle=ref_ls, label=ref_label)[0])
        ref_lines.append(axes[4].plot([], [], lw=1.5, color=ref_color, linestyle=ref_ls, label=ref_label)[0])
        ref_lines.append(axes[5].plot([], [], lw=1.5, color=ref_color, linestyle=ref_ls, label=ref_label)[0])
        ref_lines.append(axes[6].plot([], [], lw=1.5, color=ref_color, linestyle=ref_ls, label=ref_label)[0])
        for ax in axes:
            ax.legend(loc="upper right", fontsize=8)

    x_mass = r"Mass coordinate $m$ [g/cm²]"
    axes[0].set_ylabel(r"$\rho$ [g/cm³]")
    axes[1].set_ylabel(r"$P$ [MBar]")
    axes[2].set_ylabel(r"$u$ [km/s]")
    axes[3].set_ylabel(r"$e_{\mathrm{mat}}$ [hJ/g]")
    axes[4].set_ylabel(r"$T_{\mathrm{mat}}$ [K]")
    axes[5].set_ylabel(r"$T_{\mathrm{rad}}$ [K]")
    axes[6].set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    for ax in axes:
        ax.set_xlabel(x_mass)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.3)
    title = fig.suptitle("", fontweight="medium")
    frame_ids = np.arange(0, len(history.t), stride)
    
    def init():
        return lines
    
    def update(frame_idx):
        k = int(frame_ids[frame_idx])
        mk = history.m[k] if hasattr(history, "m") else history.x[k]
        t = history.t[k]
        lines[0].set_data(mk, history.rho[k])
        lines[1].set_data(mk, history.p[k] / p_scale)
        lines[2].set_data(mk, history.u[k] / u_scale)
        lines[3].set_data(mk, history.e[k] / e_scale)
        lines[4].set_data(mk, history.T_material[k] if has_T_material else history.T[k])
        lines[5].set_data(mk, history.T[k])
        lines[6].set_data(mk, history.E_rad[k])

        if has_ref:
            ref_idx = int(np.argmin(np.abs(ref_data.times - t)))
            mr = ref_data.m[ref_idx]
            ref_lines[0].set_data(mr, ref_data.rho[ref_idx])
            ref_lines[1].set_data(mr, ref_data.p[ref_idx] / p_scale)
            ref_lines[2].set_data(mr, ref_data.u[ref_idx] / u_scale)
            ref_lines[3].set_data(mr, ref_data.e[ref_idx] / e_scale)
            # Shussman T is in HeV; convert to Kelvin for display
            T_ref_K = (ref_data.T[ref_idx] * KELVIN_PER_HEV) if (ref_data.T and ref_idx < len(ref_data.T)) else np.array([])
            ref_lines[4].set_data(mr, T_ref_K)
            ref_lines[5].set_data(mr, T_ref_K)
            E_ref = ref_data.E_rad[ref_idx] if (ref_data.E_rad and ref_idx < len(ref_data.E_rad)) else np.array([])
            ref_lines[6].set_data(mr, E_ref)

        case_title = case.title if hasattr(case, "title") and case.title else "Simulation"
        header = f"{case_title}\n{subtitle}" if subtitle else case_title
        if case.T0_Kelvin is not None and case.tau is not None:
            title.set_text(
                f"{header}\n"
                f"$T(0,t)=T_0 t^{{\\tau}},\\; T_0={case.T0_Kelvin},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        elif getattr(case, "P0", None) is not None and getattr(case, "tau", None) is not None:
            title.set_text(
                f"{header}\n"
                f"$P(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0_Barye},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        else:
            title.set_text(f"{header}\n$t={t:.3e}$ s")
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        return lines
    
    anim = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=False)

    from pathlib import Path
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)

    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF to {gif_path}")
