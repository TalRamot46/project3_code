# ============================================================================
# Interactive Slider Plot (Generic for time-history data)
# ============================================================================

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from project_3.hydro_sim.simulations.lagrangian_sim import HydroHistory
from project_3.hydro_sim.plotting.hydro_plots import _create_6panel_vertical_figure
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase

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
    
    fig, axes = _create_6panel_vertical_figure()
    plt.subplots_adjust(bottom=0.12)
    
    # Initial lines
    lines = []
    lines.append(axes[0].plot(history.m[k0], history.rho[k0], lw=2)[0])
    lines.append(axes[1].plot(history.m[k0], history.p[k0], lw=2)[0])
    lines.append(axes[2].plot(history.m[k0], history.u[k0], lw=2)[0])
    lines.append(axes[3].plot(history.m[k0], history.e[k0], lw=2)[0])
    lines.append(axes[4].plot(history.m[k0], history.T[k0], lw=2)[0])
    lines.append(axes[5].plot(history.m[k0], history.E_rad[k0], lw=2)[0])
    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_ylabel(r"$p$")
    axes[2].set_ylabel(r"$u$")
    axes[3].set_ylabel(r"$e$")
    axes[4].set_ylabel(r"$T$")
    axes[5].set_ylabel(r"$E_\mathrm{rad}$")
    axes[5].set_xlabel(r"$x$")
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    title = fig.suptitle("", fontsize=12)
    
    def set_title(k):
        t = history.t[k]
        case_title = case.title if hasattr(case, 'title') else "Simulation"
        if case.T0 is not None and case.tau is not None:
            title.set_text(
                f"{case_title}\n"
                f"$T(0,t)=T_0 t^{{\\tau}},\\; T_0={case.T0},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        elif case.P0 is not None and case.tau is not None:
            title.set_text(
                f"{case_title}\n"
                f"$P(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0},\\; \\tau={case.tau},\\; t={t:.3e}$"
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
        lines[4].set_data(history.x[k], history.T[k])
        lines[5].set_data(history.x[k], history.E_rad[k])
        set_title(k)
        
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved slider figure (static) to {savepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes


