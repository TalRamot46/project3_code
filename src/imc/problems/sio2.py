#!/usr/bin/env python3
"""The Back SiO2 problem on the imc package (milestone M0).

Same problem, constants and outputs as the archived original script,
src/imc/archive/sio2_original/sio2_1mm_gray_imc_check_og.py:
  * 1D slab, L = 1 mm, 100 cells, unit transverse area
  * rho = 0.05 g/cm^3 SiO2, gray pure absorption, Hammer-Rosen opacity and EOS
  * initial T_m = 300 K, no radiation
  * left blackbody bath from back_pulse.txt (synthetic), right vacuum

Units: cm, ns, MJ, and temperature theta = T / (100 eV), so a = E100.

Differences from the frozen script, none of which changes the solved equation:
  * bath birth times follow the drive inside the step (--bath-times uniform
    restores the script's choice);
  * the census is combed with a stratified comb instead of a systematic one
    (it is not triggered at the default settings);
  * face crossings move the packet onto the face and change its cell index,
    instead of nudging it by 1e-10 dx.

    uv run python -m imc.problems.sio2 --prefix results/sio2_pkg
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from imc.core import sources, stepper
from imc.core.material import PowerLawMaterial
from imc.core.mesh import Mesh

C = 29.9792458                        # speed of light [cm/ns]
EV_TO_K = 11604.518121550082          # 1 eV expressed in kelvin
A_RAD = 7.56573325003e-28             # radiation constant a [MJ/(cm^3 K^4)]
E100 = A_RAD * (100.0 * EV_TO_K) ** 4  # a (100 eV)^4: a in theta units
SIGMA_SB_K = A_RAD * C / 4.0          # Stefan-Boltzmann constant in kelvin units

L_CM = 0.1
N_CELLS = 100
RHO = 0.05
T0_K = 300.0
THETA0 = T0_K / EV_TO_K / 100.0
DRIVE_PATH = Path(__file__).resolve().parents[1] / "data" / "back_pulse.txt"

MATERIAL = PowerLawMaterial(
    f=8.78, beta=1.10, mu=0.09,        # u_m = f theta^beta rho^(1-mu)  [MJ/cm^3]
    g=1.0 / 9175.0, alpha=3.53, lam=0.75,  # k = rho^(1+lambda) / (g theta^alpha)  [1/cm]
    a=E100, c=C, T_floor=1.0e-12,
)


def make_bath(drive_path=DRIVE_PATH):
    drive = sources.TabulatedDrive(drive_path, temperature_scale=0.01, T_pad=THETA0)  # eV -> theta

    def flux(t):
        return 0.25 * C * E100 * drive.temperature(t) ** 4      # sigma_SB T^4

    return sources.BathSource(flux)


class UniformTimeBath(sources.BathSource):
    """The frozen script's choice: birth times uniform in the step."""

    def sample_times(self, t0, t1, n):
        return t0 + (t1 - t0) * np.random.random(n)


def run_one(seed, dt, end_time, settings, bath, progress=False):
    stepper.seed_all(seed)
    mesh = Mesh.uniform(L_CM, N_CELLS)
    state = stepper.initial_state(np.full(N_CELLS, THETA0), mesh, RHO, MATERIAL)
    steps = int(np.ceil(end_time / dt - 1e-9))
    time_mid = np.empty(steps)
    outgoing = np.empty(steps)
    energy_error = np.empty(steps)
    cell_residual = np.empty(steps)
    census_count = np.empty(steps, dtype=np.int64)
    capped = 0
    for step in range(steps):
        local_dt = min(dt, end_time - state.t)
        state, tl = stepper.imc_step(state, mesh, RHO, MATERIAL, local_dt, bath, settings)
        time_mid[step] = 0.5 * (tl.t0 + tl.t1)
        outgoing[step] = tl.escaped_right / tl.dt
        energy_error[step] = state.ledger.relative_error(
            stepper.total_energy(state, mesh, RHO, MATERIAL))
        scale = max(tl.census_after.max(), tl.emitted.max(), tl.deposited.max(), 1e-300)
        cell_residual[step] = np.abs(tl.cell_balance_residual()).max() / scale
        census_count[step] = state.census.size
        capped += tl.n_capped
        if progress and (step % max(1, steps // 20) == 0 or step == steps - 1):
            print(f"seed={seed:6d} step={step+1:5d}/{steps} t={state.t:7.4f} ns "
                  f"Tmax={100*state.T.max():7.2f} eV census={state.census.size:6d} "
                  f"energy err={energy_error[step]:+.2e}", flush=True)
    return {
        "time_ns": time_mid, "Fout": outgoing, "energy_error": energy_error,
        "cell_residual": cell_residual, "census_count": census_count,
        "Tm_final_eV": 100.0 * state.T, "n_capped": capped,
    }


def block_average(t, y, width):
    """Bin a noisy series into blocks of the given width (as the frozen script)."""
    if width <= 0.0:
        return t.copy(), y.copy()
    edges = np.arange(t[0] - 0.5 * (t[1] - t[0]), t[-1] + width, width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(centers.size, np.nan)
    for i in range(centers.size):
        mask = (t >= edges[i]) & (t < edges[i + 1])
        if np.any(mask):
            out[i] = np.mean(y[mask])
    good = np.isfinite(out)
    return centers[good], out[good]


def summarize(t, fluxes, smooth):
    """Breakout (normalized flux >= 0.01) and peak of the seed-averaged smoothed flux."""
    norm = 0.25 * C * E100 * 1.65**4             # sigma_SB (165 eV)^4
    mean = fluxes.mean(axis=0)
    ts, ms = block_average(t, mean / norm, smooth)
    stderr = (fluxes.std(axis=0, ddof=1) / np.sqrt(fluxes.shape[0])
              if fluxes.shape[0] > 1 else np.zeros_like(mean))
    _, ss = block_average(t, stderr / norm, smooth)
    crossing = np.flatnonzero(ms >= 0.01)
    k = int(np.argmax(ms))
    return {
        "norm": norm, "smooth_t": ts, "smooth_mean": ms, "smooth_stderr": ss,
        "breakout": ts[crossing[0]] if crossing.size else np.nan,
        "peak_time": ts[k], "peak": ms[k], "peak_stderr": ss[k],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--end", type=float, default=3.0)
    p.add_argument("--boundary-packets", type=int, default=4000)
    p.add_argument("--material-packets", type=int, default=2500)
    p.add_argument("--max-census", type=int, default=120000)
    p.add_argument("--seeds", type=str, default="1729,2718,3141")
    p.add_argument("--smooth", type=float, default=0.02)
    p.add_argument("--fleck-mode", choices=["implicit", "explicit"], default="implicit")
    p.add_argument("--bath-times", choices=["flux", "uniform"], default="flux")
    p.add_argument("--prefix", type=str, default="sio2_1mm_gray_imc_pkg")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    settings = stepper.IMCSettings(
        n_emission=args.material_packets, n_bath=args.boundary_packets,
        max_census=args.max_census, fleck_mode=args.fleck_mode,
    )
    bath = make_bath()
    if args.bath_times == "uniform":
        bath = UniformTimeBath(bath.flux)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    wall0 = time.time()
    runs = [run_one(s, args.dt, args.end, settings, bath, not args.no_progress) for s in seeds]
    wall = time.time() - wall0

    t = runs[0]["time_ns"]
    fluxes = np.vstack([r["Fout"] for r in runs])
    s = summarize(t, fluxes, args.smooth)
    max_err = max(np.max(np.abs(r["energy_error"])) for r in runs)
    max_cell = max(np.max(r["cell_residual"]) for r in runs)

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(prefix) + ".npz", time_ns=t, Fout_runs=fluxes, Fout_mean=fluxes.mean(axis=0),
        normalization_flux=s["norm"], smooth_time_ns=s["smooth_t"],
        smooth_normalized_mean=s["smooth_mean"], smooth_normalized_stderr=s["smooth_stderr"],
        energy_error=np.vstack([r["energy_error"] for r in runs]),
        cell_residual=np.vstack([r["cell_residual"] for r in runs]),
        final_material_temperature_eV=np.vstack([r["Tm_final_eV"] for r in runs]),
        seeds=np.asarray(seeds), dt_ns=args.dt, wall_seconds=wall,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for k, seed in enumerate(seeds):
        rt, ry = block_average(t, fluxes[k] / s["norm"], args.smooth)
        ax.plot(rt, ry, alpha=0.35, lw=0.9, label=f"seed {seed}" if len(seeds) <= 4 else None)
    ax.plot(s["smooth_t"], s["smooth_mean"], color="black", lw=2.0, label="IMC mean")
    if len(seeds) > 1:
        ax.fill_between(s["smooth_t"], s["smooth_mean"] - 2 * s["smooth_stderr"],
                        s["smooth_mean"] + 2 * s["smooth_stderr"], alpha=0.18, label=r"mean $\pm2$ s.e.")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"$F_{\rm out}/[\sigma_{\rm SB}(165\,{\rm eV})^4]$")
    ax.set_title(f"imc package, SiO$_2$: L=1 mm, 100 cells, dt={args.dt:g} ns ({args.fleck_mode})")
    ax.set_xlim(0.8, args.end)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(prefix) + "_Fout.png", dpi=200)
    plt.close(fig)

    with open(str(prefix) + "_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("imc package, SiO2 L=1 mm, 100 cells\n")
        for key, val in [
            ("dt_ns", args.dt), ("seeds", seeds), ("fleck_mode", args.fleck_mode),
            ("bath_times", args.bath_times), ("boundary_packets_per_step", args.boundary_packets),
            ("target_material_packets_per_step", args.material_packets),
            ("smoothing_width_ns", args.smooth),
            ("breakout_normalized_0p01_ns", f"{s['breakout']:.12g}"),
            ("peak_time_ns", f"{s['peak_time']:.12g}"), ("peak_normalized", f"{s['peak']:.12g}"),
            ("peak_normalized_stderr", f"{s['peak_stderr']:.12g}"),
            ("maximum_absolute_energy_error", f"{max_err:.12g}"),
            ("maximum_relative_cell_residual", f"{max_cell:.12g}"),
            ("capped_histories", sum(r["n_capped"] for r in runs)),
            ("wall_seconds", f"{wall:.3f}"),
        ]:
            fh.write(f"{key} {val}\n")
    print(f"Breakout {s['breakout']:.4f} ns, peak {s['peak']:.4f} +- {s['peak_stderr']:.4f} "
          f"at {s['peak_time']:.3f} ns, max energy error {max_err:.2e}, wall {wall:.1f} s")
    return s, runs


if __name__ == "__main__":
    main()
