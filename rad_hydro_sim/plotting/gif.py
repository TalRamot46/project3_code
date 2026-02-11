# ============================================================================
# GIF Animation (Generic for time-history data)
# ============================================================================

from . import mpl_style  # noqa: F401 - apply project style
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
    lines = []
    lines.append(axes[0].plot(history.m[k0], history.rho[k0], lw=2)[0])
    lines.append(axes[1].plot(history.m[k0], history.p[k0], lw=2)[0])
    lines.append(axes[2].plot(history.m[k0], history.u[k0], lw=2)[0])
    lines.append(axes[3].plot(history.m[k0], history.e[k0], lw=2)[0])
    
    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_ylabel(r"$p$")
    axes[2].set_ylabel(r"$u$")
    axes[3].set_ylabel(r"$e$")
    axes[3].set_xlabel(r"$x$")
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    title = fig.suptitle("", fontsize=12)
    frame_ids = np.arange(0, len(history.t), stride)
    
    def init():
        return lines
    
    def update(frame_idx):
        k = int(frame_ids[frame_idx])
        lines[0].set_data(history.m[k], history.rho[k])
        lines[1].set_data(history.m[k], history.p[k])
        lines[2].set_data(history.m[k], history.u[k])
        lines[3].set_data(history.m[k], history.e[k])
        
        t = history.t[k]
        case_title = case.title if hasattr(case, 'title') else "Simulation"
        if hasattr(case, 'P0') and hasattr(case, 'tau'):
            title.set_text(
                f"{case_title}\n"
                f"$p(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        else:
            title.set_text(f"{case_title}, t={t:.3e}")
        
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        
        return lines
    
    anim = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=False)

    # make directory if not exists
    import os
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF to {gif_path}")

