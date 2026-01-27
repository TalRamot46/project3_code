# problems/riemann_plots.py
from __future__ import annotations
import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    ax_rho, ax_p, ax_u, ax_e = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    # density
    ax_rho.plot(x_cells, rho_ex, linewidth=2)
    ax_rho.plot(x_cells, rho_num, linestyle="None", marker="+")
    _style(ax_rho, "density", x_min, x_max)

    # pressure
    ax_p.plot(x_cells, p_ex, linewidth=2)
    ax_p.plot(x_cells, p_num, linestyle="None", marker="+")
    _style(ax_p, "pressure", x_min, x_max)

    # velocity
    ax_u.plot(x_cells, u_ex, linewidth=2)
    ax_u.plot(x_cells, u_num, linestyle="None", marker="+")
    _style(ax_u, "velocity", x_min, x_max)

    # energy
    ax_e.plot(x_cells, e_ex, linewidth=2)
    ax_e.plot(x_cells, e_num, linestyle="None", marker="+")
    _style(ax_e, "energy", x_min, x_max)

    ax_u.set_xlabel("x")
    ax_e.set_xlabel("x")

    extra = f" — {title_extra}" if title_extra else ""
    fig.suptitle(f"Planar Riemann Test {test_id} at t={t_end}, N={Ncells}, gamma={gamma}{extra}")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)

    if show:
        plt.show()

    return fig, axes

def _style(ax, ylabel: str, x_min: float, x_max: float):
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
