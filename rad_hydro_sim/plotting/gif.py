# ============================================================================
# GIF Animation (Generic for time-history data)
# ============================================================================

import numpy as np
from . import mpl_style  # noqa: F401 - apply project style
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from project_3.hydro_sim.simulations.lagrangian_sim import HydroHistory
from project_3.hydro_sim.plotting.hydro_plots import _create_6panel_vertical_figure
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase

def save_history_gif(
    history: "HydroHistory",
    case: RadHydroCase,
    gif_path: str = "simulation.gif",
    fps: int = 20,
    stride: int = 1,
):
    """
    Save an animated GIF of the time-history data.
    
    Parameters:
        history: HydroHistory object with time-series data
        case: Problem case (for title info)
        gif_path: Output file path
        fps: Frames per second
        stride: Frame stride (skip frames for smaller file)
    """
    fig, axes = _create_6panel_vertical_figure()
    k0 = 0
    m0 = history.m[k0] if hasattr(history, "m") else history.x[k0]
    lines = []
    lines.append(axes[0].plot(m0, history.rho[k0], lw=2)[0])
    lines.append(axes[1].plot(m0, history.p[k0], lw=2)[0])
    lines.append(axes[2].plot(m0, history.u[k0], lw=2)[0])
    lines.append(axes[3].plot(m0, history.e[k0], lw=2)[0])
    lines.append(axes[4].plot(m0, history.T[k0], lw=2)[0])
    lines.append(axes[5].plot(m0, history.E_rad[k0], lw=2)[0])
    axes[0].set_ylabel(r"$\rho$ [g/cm³]")
    axes[1].set_ylabel(r"$p$ [MBar]")
    axes[2].set_ylabel(r"$u$ [km/s]")
    axes[3].set_ylabel(r"$e$ [hJ/g]")
    axes[4].set_ylabel(r"$T$ [HeV]")
    axes[5].set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    axes[5].set_xlabel(r"Mass coordinate $m$ [g/cm²]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    title = fig.suptitle("", fontweight="medium")
    frame_ids = np.arange(0, len(history.t), stride)
    
    def init():
        return lines
    
    def update(frame_idx):
        k = int(frame_ids[frame_idx])
        mk = history.m[k] if hasattr(history, "m") else history.x[k]
        lines[0].set_data(mk, history.rho[k])
        lines[1].set_data(mk, history.p[k])
        lines[2].set_data(mk, history.u[k])
        lines[3].set_data(mk, history.e[k])
        lines[4].set_data(mk, history.T[k])
        lines[5].set_data(mk, history.E_rad[k])
        t = history.t[k]
        case_title = case.title if hasattr(case, "title") and case.title else "Simulation"
        if case.T0_Kelvin is not None and case.tau is not None:
            title.set_text(
                f"{case_title}\n"
                f"$T(0,t)=T_0 t^{{\\tau}},\\; T_0={case.T0_Kelvin},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        elif getattr(case, "P0", None) is not None and getattr(case, "tau", None) is not None:
            title.set_text(
                f"{case_title}\n"
                f"$P(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0_Barye},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        else:
            title.set_text(f"{case_title}, t={t:.3e}")
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

