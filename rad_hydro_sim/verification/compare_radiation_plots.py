# verification/compare_radiation_plots.py
"""
Comparison plots for radiation-only verification: rad_hydro vs 1D Diffusion and/or Supersonic solver.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from project_3.rad_hydro_sim.plotting import mpl_style  # noqa: F401 - apply project style
from matplotlib.widgets import Slider

from project_3.rad_hydro_sim.verification.radiation_data import RadiationData


def _interpolate_to_time(data: RadiationData, target_time: float) -> int:
    """Index of closest time."""
    return int(np.argmin(np.abs(data.times - target_time)))


def _format_time_legend(t_sec: float) -> str:
    """Compact time for legend: e.g. 5e-10 -> '0.5 ns'."""
    if t_sec >= 1e-6:
        return f"{t_sec*1e9:.1f} ns"
    if t_sec >= 1e-9:
        return f"{t_sec*1e9:.2f} ns"
    return f"{t_sec:.2e} s"


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

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    ax_t, ax_e = axes[0], axes[1]

    # Collect handles/labels for a single figure legend
    handles, labels = [], []

    def add_curve(ax_t_plot, ax_e_plot, data, k, color, linestyle, label):
        ht, = ax_t_plot.plot(
            data.x[k], data.T[k],
            color=color, linestyle=linestyle, lw=2.25,
            label=label,
        )
        ax_e_plot.plot(
            data.x[k], data.E_rad[k],
            color=color, linestyle=linestyle, lw=2.25,
        )
        return ht

    ht_sim = add_curve(
        ax_t, ax_e, sim_data, sim_k,
        sim_data.color, sim_data.linestyle,
        f"{sim_data.label}  t = {_format_time_legend(t_sim)}",
    )
    handles.append(ht_sim)
    labels.append(ht_sim.get_label())

    ht_ref = add_curve(
        ax_t, ax_e, ref_data, ref_k,
        ref_data.color, ref_data.linestyle,
        f"{ref_data.label}  t = {_format_time_legend(ref_data.times[ref_k])}",
    )
    handles.append(ht_ref)
    labels.append(ht_ref.get_label())

    # Plot extra references (e.g. Supersonic solver)
    if extra_ref_data:
        for extra in extra_ref_data:
            extra_k = _interpolate_to_time(extra, time)
            ht_extra = add_curve(
                ax_t, ax_e, extra, extra_k,
                extra.color, extra.linestyle,
                f"{extra.label}  t = {_format_time_legend(extra.times[extra_k])}",
            )
            handles.append(ht_extra)
            labels.append(ht_extra.get_label())

    ax_t.set_ylabel(r"Temperature $T$ [K]")
    ax_t.grid(True, alpha=0.35, linestyle="--")
    ax_t.tick_params(axis="both")

    # Clear x-axis tick overlap: limit number of ticks and use scientific notation
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
        sf = ScalarFormatter(useMathText=True)
        sf.set_scientific(True)
        sf.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(sf)
    ax_e.set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    ax_e.set_xlabel(r"Position $x$ [cm]")
    ax_e.grid(True, alpha=0.35, linestyle="--")
    ax_e.tick_params(axis="both")
    # E_rad axis: make exponent clear (e.g. "2×10⁹")
    ax_e.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_e.yaxis.get_offset_text().set_fontsize(10)

    if title is None:
        title = f"Radiation comparison at t ≈ {_format_time_legend(time)}"
    fig.suptitle(title, fontsize=12, fontweight="medium", y=1.02)
    # Single legend above both panels, no overlap with data
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(3, len(handles)),
        frameon=True,
        fontsize=10,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def plot_radiation_comparison_slider(
    sim_data: RadiationData,
    ref_data: RadiationData,
    super_data: Optional[RadiationData] = None,
    show: bool = True,
    title: str | None = None,
):
    """Interactive 2-panel (T, E_rad) comparison with time slider. Optionally add extra reference curves."""
    all_times = sim_data.times
    n = len(all_times)

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
    if super_data is not None:
        super_k0 = _interpolate_to_time(super_data, all_times[k0])
        line_super_t, = ax_t.plot(
            super_data.x[super_k0], super_data.T[super_k0],
            color=super_data.color, linestyle="--", lw=2, label=super_data.label,
        )
        line_super_e, = ax_e.plot(
            super_data.x[super_k0], super_data.E_rad[super_k0],
            color=super_data.color, linestyle="--", lw=2,
        )
    else:
        line_super_t, = ax_t.plot([], [], color="green", linestyle="--", lw=2)
        line_super_e, = ax_e.plot([], [], color="green", linestyle="--", lw=2)

    line_sim_e, = ax_e.plot(
        sim_data.x[k0], sim_data.E_rad[k0],
        color=sim_data.color, lw=2,
    )
    line_ref_e, = ax_e.plot(
        ref_data.x[ref_k0], ref_data.E_rad[ref_k0],
        color=ref_data.color, linestyle="--", lw=2,
    )

    ax_t.set_ylabel(r"Temperature $T$ [K]")
    ax_t.legend(loc="best")
    ax_t.grid(True, alpha=0.3)
    ax_e.set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    ax_e.set_xlabel(r"Position $x$ [cm]")
    ax_e.grid(True, alpha=0.3)

    title_text = fig.suptitle("")

    def update(val):
        k = int(val)
        line_sim_t.set_data(sim_data.x[k], sim_data.T[k])
        line_sim_e.set_data(sim_data.x[k], sim_data.E_rad[k])
        ref_k = _interpolate_to_time(ref_data, all_times[k])
        line_ref_t.set_data(ref_data.x[ref_k], ref_data.T[ref_k])
        line_ref_e.set_data(ref_data.x[ref_k], ref_data.E_rad[ref_k])
        base = title or "Radiation comparison"
        t_sim = all_times[k]
        t_ref = ref_data.times[ref_k]
        if super_data is not None:
            super_k = _interpolate_to_time(super_data, all_times[k])
            line_super_t.set_data(super_data.x[super_k], super_data.T[super_k])
            line_super_e.set_data(super_data.x[super_k], super_data.E_rad[super_k])
            title_text.set_text(f"{base}\nSim: t={t_sim:.2e}s, Ref: t={t_ref:.2e}s, Super: t={super_data.times[super_k]:.2e}s")
        else:
            title_text.set_text(f"{base}\nSim: t={t_sim:.2e}s, Ref: t={t_ref:.2e}s")
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    ax_slider = fig.add_axes((0.15, 0.05, 0.7, 0.03))
    slider = Slider(ax_slider, "Frame", 0, n - 1, valinit=k0, valstep=1)
    slider.on_changed(update)
    update(k0)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes, slider
