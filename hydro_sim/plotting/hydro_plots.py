# problems/hydro_plots.py
"""
Unified plotting routines for hydrodynamic simulations.
Supports Riemann, Driven Shock, and Sedov explosion problems.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.state import HydroState
    from driven_shock_sim import ShockHistory


# ============================================================================
# Common Styling
# ============================================================================

def _style_axis(ax, ylabel: str, x_min: float = None, x_max: float = None, grid: bool = True):
    """Apply common styling to an axis."""
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)


def _create_4panel_figure(sharex: bool = True, figsize: tuple = (10, 8)):
    """Create a standard 4-panel figure for hydro variables."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=sharex)
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_e = axes[1, 0], axes[1, 1]
    return fig, (ax_rho, ax_p, ax_u, ax_e)


def _create_4panel_vertical_figure(sharex: bool = True, figsize: tuple = (7, 10)):
    """Create a vertical 4-panel figure for hydro variables."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=sharex)
    return fig, axes


# ============================================================================
# Riemann Problem Plotting
# ============================================================================

def plot_riemann_results(
    *,
    x_cells: np.ndarray,
    numerical: dict,
    exact: dict,
    meta: dict,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot Riemann problem results comparing numerical and exact solutions.
    
    Parameters:
        x_cells: Cell center positions
        numerical: Dict with keys 'rho', 'p', 'u', 'e' for numerical solution
        exact: Dict with keys 'rho', 'p', 'u', 'e' for exact solution
        meta: Dict with 'test_id', 't_end', 'Ncells', 'gamma', 'x_min', 'x_max', 'title_extra'
        savepath: Optional path to save figure
        show: Whether to display the figure
    """
    fig, (ax_rho, ax_p, ax_u, ax_e) = _create_4panel_figure()
    
    x_min, x_max = meta["x_min"], meta["x_max"]
    
    # Density
    ax_rho.plot(x_cells, exact["rho"], linewidth=2, label="Exact")
    ax_rho.plot(x_cells, numerical["rho"], linestyle="None", marker="+", label="Numerical")
    _style_axis(ax_rho, r"$\rho$", x_min, x_max)
    ax_rho.legend()
    
    # Pressure
    ax_p.plot(x_cells, exact["p"], linewidth=2)
    ax_p.plot(x_cells, numerical["p"], linestyle="None", marker="+")
    _style_axis(ax_p, r"$p$", x_min, x_max)
    
    # Velocity
    ax_u.plot(x_cells, exact["u"], linewidth=2)
    ax_u.plot(x_cells, numerical["u"], linestyle="None", marker="+")
    _style_axis(ax_u, r"$u$", x_min, x_max)
    ax_u.set_xlabel(r"$x$")
    
    # Energy
    ax_e.plot(x_cells, exact["e"], linewidth=2)
    ax_e.plot(x_cells, numerical["e"], linestyle="None", marker="+")
    _style_axis(ax_e, r"$e$", x_min, x_max)
    ax_e.set_xlabel(r"$x$")
    
    # Title
    title_extra = f" — {meta['title_extra']}" if meta.get('title_extra') else ""
    fig.suptitle(
        f"Riemann Test {meta['test_id']} at t={meta['t_end']:.4g}, "
        f"N={meta['Ncells']}, γ={meta['gamma']}{title_extra}"
    )
    fig.tight_layout()
    
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved figure to {savepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# ============================================================================
# Driven Shock Plotting
# ============================================================================

def plot_shock_results(
    *,
    x_cells: np.ndarray,
    state: "HydroState",
    case,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot driven shock simulation results.
    
    Parameters:
        x_cells: Cell center positions
        state: Final HydroState
        case: DrivenShockCase instance
        savepath: Optional path to save figure
        show: Whether to display the figure
    """
    fig, axes = _create_4panel_vertical_figure()
    
    # Cell-centered velocity from node values
    u_cells = 0.5 * (state.u[:-1] + state.u[1:])
    m_cells = x_cells * case.rho0

    axes[0].plot(x_cells, state.rho, lw=2)
    _style_axis(axes[0], r"$\rho$")
    
    axes[1].plot(x_cells, state.p, lw=2)
    _style_axis(axes[1], r"$p$")
    
    axes[2].plot(x_cells, u_cells, lw=2)
    _style_axis(axes[2], r"$u$")
    
    axes[3].plot(x_cells, state.e, lw=2)
    _style_axis(axes[3], r"$e$")
    axes[3].set_xlabel(r"$x$")
    
    # Title with case info
    title = case.title if hasattr(case, 'title') else "Driven Shock"
    if hasattr(case, 'P0') and hasattr(case, 'tau'):
        fig.suptitle(
            f"{title}\n"
            f"$p(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0},\\; \\tau={case.tau},\\; t={state.t:.3e}$",
            fontsize=12
        )
    else:
        fig.suptitle(f"{title}, t={state.t:.3e}", fontsize=12)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved figure to {savepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# ============================================================================
# Sedov Explosion Plotting
# ============================================================================

def plot_sedov_results(
    *,
    x_cells: np.ndarray,
    state: "HydroState",
    case,
    exact: dict | None = None,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot Sedov explosion simulation results.
    
    Parameters:
        x_cells: Cell center positions (radii)
        state: Current HydroState
        case: SedovExplosionCase instance
        exact: Optional dict with exact solution {'rho', 'p', 'u', 'e'}
        savepath: Optional path to save figure
        show: Whether to display the figure
    """
    fig, (ax_rho, ax_p, ax_u, ax_e) = _create_4panel_figure()
    
    # Cell-centered velocity from node values
    u_cells = 0.5 * (state.u[:-1] + state.u[1:])
    
    # Numerical solution
    ax_rho.plot(x_cells, state.rho, 'b-', lw=2, label="Numerical")
    ax_p.plot(x_cells, state.p, 'b-', lw=2)
    ax_u.plot(x_cells, u_cells, 'b-', lw=2)
    ax_e.plot(x_cells, state.e, 'b-', lw=2)
    
    # Exact solution if provided
    if exact is not None:
        ax_rho.plot(x_cells, exact["rho"], 'r--', lw=1.5, label="Exact")
        ax_p.plot(x_cells, exact["p"], 'r--', lw=1.5)
        ax_u.plot(x_cells, exact["u"], 'r--', lw=1.5)
        ax_e.plot(x_cells, exact["e"], 'r--', lw=1.5)
        ax_rho.legend()
    
    _style_axis(ax_rho, r"$\rho$")
    _style_axis(ax_p, r"$p$")
    _style_axis(ax_u, r"$u$")
    _style_axis(ax_e, r"$e$")
    
    ax_u.set_xlabel(r"$r$")
    ax_e.set_xlabel(r"$r$")
    
    # Title
    title = case.title if hasattr(case, 'title') else "Sedov Explosion"
    fig.suptitle(
        f"{title}\n"
        f"$E_0={case.E0},\\; \\rho_0={case.rho0},\\; t={state.t:.3e}$",
        fontsize=12
    )
    fig.tight_layout()
    
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved figure to {savepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# ============================================================================
# Interactive Slider Plot (Generic for time-history data)
# ============================================================================

def plot_history_slider(
    history: "ShockHistory",
    case,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Create an interactive slider plot for time-history data.
    
    Parameters:
        history: ShockHistory object with time-series data
        case: Problem case (for title info)
        savepath: Optional path to save static figure
        show: Whether to display the figure
    """
    k0 = 0  # Initial frame
    
    fig, axes = _create_4panel_vertical_figure()
    plt.subplots_adjust(bottom=0.12)
    
    # Initial lines
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
    
    def set_title(k):
        t = history.t[k]
        case_title = case.title if hasattr(case, 'title') else "Simulation"
        if hasattr(case, 'P0') and hasattr(case, 'tau'):
            title.set_text(
                f"{case_title}\n"
                f"$p(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0},\\; \\tau={case.tau},\\; t={t:.3e}$"
            )
        else:
            title.set_text(f"{case_title}, t={t:.3e}")
    
    set_title(k0)
    
    # Slider
    ax_slider = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(ax_slider, "frame", 0, len(history.t) - 1, valinit=k0, valstep=1)
    
    def update(val):
        k = int(slider.val)
        lines[0].set_data(history.m[k], history.rho[k])
        lines[1].set_data(history.m[k], history.p[k])
        lines[2].set_data(history.m[k], history.u[k])
        lines[3].set_data(history.m[k], history.e[k])
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


# ============================================================================
# GIF Animation (Generic for time-history data)
# ============================================================================

def save_history_gif(
    history: "ShockHistory",
    case,
    gif_path: str = "simulation.gif",
    fps: int = 20,
    stride: int = 1,
):
    """
    Save an animated GIF of the time-history data.
    
    Parameters:
        history: ShockHistory object with time-series data
        case: Problem case (for title info)
        gif_path: Output file path
        fps: Frames per second
        stride: Frame stride (skip frames for smaller file)
    """
    fig, axes = _create_4panel_vertical_figure()
    
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


# ============================================================================
# Legacy compatibility wrappers
# ============================================================================

def plot_riemann_comparison(
    *,
    x_cells,
    rho_num, p_num, u_num, e_num,
    rho_ex, p_ex, u_ex, e_ex,
    test_id: int,
    t_end: float,
    Ncells: int,
    gamma: float,
    x_min: float,
    x_max: float,
    title_extra: str = "",
    savepath: str | None = None,
    show: bool = True,
):
    """Legacy wrapper for plot_riemann_results."""
    numerical = {"rho": rho_num, "p": p_num, "u": u_num, "e": e_num}
    exact = {"rho": rho_ex, "p": p_ex, "u": u_ex, "e": e_ex}
    meta = {
        "test_id": test_id,
        "t_end": t_end,
        "Ncells": Ncells,
        "gamma": gamma,
        "x_min": x_min,
        "x_max": x_max,
        "title_extra": title_extra,
    }
    return plot_riemann_results(
        x_cells=x_cells,
        numerical=numerical,
        exact=exact,
        meta=meta,
        savepath=savepath,
        show=show,
    )
