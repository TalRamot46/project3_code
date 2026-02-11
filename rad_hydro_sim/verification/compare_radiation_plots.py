# verification/compare_radiation_plots.py
"""
Comparison plots for radiation-only verification: rad_hydro vs 1D Diffusion and/or Supersonic solver.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from project_3.rad_hydro_sim.plotting import mpl_style  # noqa: F401 - apply project style
from matplotlib.widgets import Slider

from project_3.rad_hydro_sim.verification.radiation_data import RadiationData


def _interpolate_to_time(data: RadiationData, target_time: float) -> int:
    """Index of closest time."""
    return int(np.argmin(np.abs(data.times - target_time)))


def plot_radiation_comparison_single_time(
    sim_data: RadiationData,
    ref_data: RadiationData,
    time: float,
    savepath: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
    extra_ref_data: Optional[List[RadiationData]] = None,
):
    """Plot T and E_rad vs x at a single time (2 panels). Optionally add extra reference curves."""
    sim_k = _interpolate_to_time(sim_data, time)
    ref_k = _interpolate_to_time(ref_data, time)
    t_sim = sim_data.times[sim_k]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ax_t, ax_e = axes[0], axes[1]

    ax_t.plot(
        sim_data.x[sim_k], sim_data.T[sim_k],
        color=sim_data.color, linestyle=sim_data.linestyle, lw=2,
        label=f"{sim_data.label} t={t_sim:.2e}s",
    )
    ax_t.plot(
        ref_data.x[ref_k], ref_data.T[ref_k],
        color=ref_data.color, linestyle=ref_data.linestyle, lw=2,
        label=f"{ref_data.label} t={ref_data.times[ref_k]:.2e}s",
    )
    if extra_ref_data:
        for extra in extra_ref_data:
            ek = _interpolate_to_time(extra, time)
            ax_t.plot(
                extra.x[ek], extra.T[ek],
                color=extra.color, linestyle=extra.linestyle, lw=2,
                label=f"{extra.label} t={extra.times[ek]:.2e}s",
            )
    ax_t.set_ylabel(r"Temperature [Hev]")
    ax_t.legend(loc="best", fontsize=9)
    ax_t.grid(True, alpha=0.3)

    ax_e.plot(
        sim_data.x[sim_k], sim_data.E_rad[sim_k],
        color=sim_data.color, linestyle=sim_data.linestyle, lw=2,
    )
    ax_e.plot(
        ref_data.x[ref_k], ref_data.E_rad[ref_k],
        color=ref_data.color, linestyle=ref_data.linestyle, lw=2,
    )
    if extra_ref_data:
        for extra in extra_ref_data:
            ek = _interpolate_to_time(extra, time)
            ax_e.plot(
                extra.x[ek], extra.E_rad[ek],
                color=extra.color, linestyle=extra.linestyle, lw=2,
            )
    ax_e.set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    ax_e.set_xlabel(r"Position $x$ [cm]")
    ax_e.grid(True, alpha=0.3)

    if title is None:
        title = f"Radiation comparison at t ≈ {time:.2e} s"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def plot_radiation_comparison_slider(
    sim_data: RadiationData,
    ref_data: RadiationData,
    show: bool = True,
    title: str | None = None,
    extra_ref_data: Optional[List[RadiationData]] = None,
):
    """Interactive 2-panel (T, E_rad) comparison with time slider. Optionally add extra reference curves."""
    all_times = sim_data.times
    n = len(all_times)
    extra_ref_data = extra_ref_data or []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    plt.subplots_adjust(bottom=0.15)
    ax_t, ax_e = axes[0], axes[1]

    k0 = 0
    ref_k0 = _interpolate_to_time(ref_data, all_times[k0])

    line_sim_t, = ax_t.plot(
        sim_data.x[k0], sim_data.T[k0],
        color=sim_data.color, lw=2, label=sim_data.label,
    )
    line_ref_t, = ax_t.plot(
        ref_data.x[ref_k0], ref_data.T[ref_k0],
        color=ref_data.color, linestyle="--", lw=2, label=ref_data.label,
    )
    line_sim_e, = ax_e.plot(
        sim_data.x[k0], sim_data.E_rad[k0],
        color=sim_data.color, lw=2,
    )
    line_ref_e, = ax_e.plot(
        ref_data.x[ref_k0], ref_data.E_rad[ref_k0],
        color=ref_data.color, linestyle="--", lw=2,
    )
    lines_extra_t: List[Any] = []
    lines_extra_e: List[Any] = []
    for extra in extra_ref_data:
        ek = _interpolate_to_time(extra, all_times[k0])
        lines_extra_t.append(ax_t.plot(
            extra.x[ek], extra.T[ek],
            color=extra.color, linestyle=extra.linestyle, lw=2, label=extra.label,
        )[0])
        lines_extra_e.append(ax_e.plot(
            extra.x[ek], extra.E_rad[ek],
            color=extra.color, linestyle=extra.linestyle, lw=2,
        )[0])

    ax_t.set_ylabel(r"$T$ [Hev]")
    ax_t.legend(loc="best")
    ax_t.grid(True, alpha=0.3)
    ax_e.set_ylabel(r"$E_{\mathrm{rad}}$")
    ax_e.set_xlabel(r"$x$ [cm]")
    ax_e.grid(True, alpha=0.3)

    title_text = fig.suptitle("", fontsize=12)

    def update(val):
        k = int(val)
        ref_k = _interpolate_to_time(ref_data, all_times[k])
        line_sim_t.set_data(sim_data.x[k], sim_data.T[k])
        line_ref_t.set_data(ref_data.x[ref_k], ref_data.T[ref_k])
        line_sim_e.set_data(sim_data.x[k], sim_data.E_rad[k])
        line_ref_e.set_data(ref_data.x[ref_k], ref_data.E_rad[ref_k])
        for i, extra in enumerate(extra_ref_data):
            ek = _interpolate_to_time(extra, all_times[k])
            lines_extra_t[i].set_data(extra.x[ek], extra.T[ek])
            lines_extra_e[i].set_data(extra.x[ek], extra.E_rad[ek])
        t_sim = sim_data.times[k]
        t_ref = ref_data.times[ref_k]
        base = title or "Radiation comparison"
        title_text.set_text(f"{base}\nSim: t={t_sim:.2e}s, Ref: t={t_ref:.2e}s")
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n - 1, valinit=k0, valstep=1)
    slider.on_changed(update)
    update(k0)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes, slider
