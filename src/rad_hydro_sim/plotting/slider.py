# ============================================================================
# Interactive Slider Plot (Generic for time-history data)
# ============================================================================

from . import mpl_style  # noqa: F401 - apply project style
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from project3_code.hydro_sim.simulations.lagrangian_sim import HydroHistory
from project3_code.hydro_sim.plotting.hydro_plots import _create_7panel_vertical_figure
from project3_code.rad_hydro_sim.problems.RadHydroCase import RadHydroCase

def plot_history_slider(
    history: "HydroHistory",
    case: RadHydroCase,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Create an interactive slider plot for time-history data.
    
    Parameters:
        history: SimulationHistory object with time-series data
        case: Problem case (for title info)
        savepath: Optional path to save static figure
        show: Whether to display the figure
    """
    k0 = 0  # Initial frame
    # Use mass coordinate for x-axis when available (consistent with update)
    x_axis = history.x[k0] if hasattr(history, "x") else history.m[k0]
    
    has_T_material = hasattr(history, "T_material") and history.T_material is not None

    fig, axes = _create_7panel_vertical_figure()
    plt.subplots_adjust(bottom=0.12)
    
    # Initial lines (same abscissa as in update)
    lines = []
    lines.append(axes[0].plot(x_axis, history.rho[k0], lw=2)[0])
    lines.append(axes[1].plot(x_axis, history.p[k0], lw=2)[0])
    lines.append(axes[2].plot(x_axis, history.u[k0], lw=2)[0])
    lines.append(axes[3].plot(x_axis, history.e[k0], lw=2)[0])
    lines.append(axes[4].plot(x_axis, history.T_material[k0] if has_T_material else history.T[k0], lw=2)[0])
    lines.append(axes[5].plot(x_axis, history.T[k0], lw=2)[0])
    lines.append(axes[6].plot(x_axis, history.E_rad[k0], lw=2)[0])
    axes[0].set_ylabel(r"$\rho$ [g/cm³]")
    axes[1].set_ylabel(r"$p$ [Barye]")
    axes[2].set_ylabel(r"$u$ [cm/s]")
    axes[3].set_ylabel(r"$e_{\mathrm{mat}}$ [erg/g]")
    axes[4].set_ylabel(r"$T_{\mathrm{mat}}$ [K]")
    axes[5].set_ylabel(r"$T_{\mathrm{rad}}$ [K]")
    axes[6].set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    axes[6].set_xlabel(r"Mass coordinate $m$ [g/cm²]")
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    title = fig.suptitle("", fontweight="medium")
    
    def set_title(k):
        t = history.t[k]
        case_title = case.title if hasattr(case, 'title') else "Simulation"
        if case.T0_Kelvin is not None and case.tau is not None:
            title.set_text(
                f"{case_title}\n"
                f"$T(0,t)=T_0 t^{{\\tau}},\\; T_0={case.T0_Kelvin},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        elif case.P0_Barye is not None and case.tau is not None:
            title.set_text(
                f"{case_title}\n"
                f"$P(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0_Barye},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        else:
            title.set_text(f"{case_title}, t={t:.3e}")
    
    set_title(k0)
    
    # Slider
    ax_slider = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(ax_slider, "frame", 0, len(history.t) - 1, valinit=k0, valstep=1)
    
    def update(val):
        k = int(slider.val)
        lines[0].set_data(history.x[k], history.rho[k])
        lines[1].set_data(history.x[k], history.p[k])
        lines[2].set_data(history.x[k], history.u[k])
        lines[3].set_data(history.x[k], history.e[k])
        lines[4].set_data(history.x[k], history.T_material[k] if has_T_material else history.T[k])
        lines[5].set_data(history.x[k], history.T[k])
        lines[6].set_data(history.x[k], history.E_rad[k])
        set_title(k)
        
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Use subplots_adjust to avoid tight_layout warning with slider axes
    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.08, hspace=0.35)
    
    if savepath:
        from pathlib import Path
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved slider figure (static) to {savepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes

