"""Figures, table and slider data of the Al runs to 50 ns (al_long.py, md/misalignment/01_surface_equilibrium.md section 3).

    python -m test_misalignment.al_long_figures            # figures, table, slider data
"""
from __future__ import annotations

import base64
import json
import logging

import numpy as np

logging.disable(logging.INFO)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from test_misalignment import light_analytic_start_figures as LF
from test_misalignment.al_long import FIGURES, OUT, T_SEED, case_and_config

K = ivd.K_PER_HEV
INK = "#1a1a1a"
COLORS = {"2T": "#2a78d6", "chi1000": "#eb6834", "imc": "#1baf7a"}
LABELS = {"2T": "2T diffusion, χ = 1", "chi1000": "2T diffusion, χ = 1000", "imc": "IMC (mean of seeds)"}
QTY = ("rho", "p", "u", "T", "Trad", "A")
GRID = np.geomspace(1e-5, 0.185, 320)          # mass grid of the slider [g/cm^2]


def solver():
    from dataclasses import replace
    from test_misalignment.al_long import T_TARGET
    s, case = LF.solver("al")
    return s, replace(case, T_initial_Kelvin=T_TARGET)


def gamma():
    case, _ = case_and_config("cold_2T")
    return case.r + 1.0


# --------------------------------------------------------------------------- loading
def load(start: str, phys: str):
    """dict(t, m, rho, p, u, T_material, T_rad, series...) ; IMC: list of the seeds' runs."""
    if phys == "imc":
        runs = [np.load(p) for p in sorted(OUT.glob(f"{start}_imc[0-9].npz"))]
        return runs or None
    p = OUT / f"{start}_{phys}.npz"
    return [np.load(p)] if p.exists() else None


def frame(runs, t):
    """The run's profile nearest t (IMC: mean over the seeds), cell centres."""
    out = None
    for r in runs:
        k = int(np.argmin(np.abs(np.log(r["t"] / t))))
        m = r["m"]
        mc = m - 0.5 * np.diff(m, prepend=0.0)
        f = dict(t=float(r["t"][k]), mc=mc, rho=r["rho"][k].astype(float), p=r["p"][k].astype(float),
                 u=r["u"][k].astype(float), T=r["T_material"][k].astype(float), Trad=r["T_rad"][k].astype(float))
        if out is None:
            out = f
        else:
            for key in ("rho", "p", "u", "T", "Trad"):
                out[key] = out[key] + f[key]
    for key in ("rho", "p", "u", "T", "Trad"):
        out[key] = out[key] / len(runs)
    return out


def on_grid(f, g):
    """Interpolate a profile onto the mass grid (log for positive quantities)."""
    res = {}
    floor = {"rho": 1e-300, "p": 1e-300, "T": 1.0, "Trad": 1.0}       # IMC leaves E_rad = 0 in cold cells
    for key in ("rho", "p", "T", "Trad"):
        res[key] = np.exp(np.interp(np.log(GRID), np.log(f["mc"]), np.log(np.maximum(f[key], floor[key]))))
    res["u"] = np.interp(np.log(GRID), np.log(f["mc"]), f["u"])
    res["A"] = res["p"] / res["rho"]**g
    return res


def solver_on_grid(s, case, t, g):
    hs, ps = s.heat_solver, s.shock_solver
    mf = float(hs.ablated_mass(time=t))
    ms = float(np.ravel(ps.shocked_mass(time=t))[0])
    res = {key: np.full(GRID.size, np.nan) for key in QTY}
    ok = (GRID > 2e-3 * mf) & (GRID < 0.999 * ms)
    sol = s.solve(mass=GRID[ok], time=t)
    for key, name in (("rho", "density"), ("p", "pressure"), ("u", "velocity"), ("T", "temperature")):
        res[key][ok] = np.asarray(sol[name])
    ahead = GRID >= ms
    T0 = case.T_initial_Kelvin
    res["rho"][ahead] = case.rho0
    res["T"][ahead] = T0
    res["u"][ahead] = 0.0
    res["p"][ahead] = case.r * case.rho0 * case.f_Kelvin * T0**case.beta_Rosen * case.rho0**(-case.mu)
    res["Trad"] = res["T"].copy()
    res["A"] = res["p"] / res["rho"]**g
    return res, mf, ms


# --------------------------------------------------------------------------- slider data
def slider_data(path, n_cold=110, n_seeded=80):
    """JSON lines for the slider page: one record per (start, run, quantity) with the frames
    as base64 float32 [n_times x n_grid] (log10 for rho, p, T, Trad, A; u in km/s)."""
    s, case = solver()
    g = gamma()
    times = {"cold": np.geomspace(1e-11, 5e-8, n_cold), "seeded": np.geomspace(T_SEED * 1.002, 5e-8, n_seeded)}
    records = [dict(kind="grid", m_mg=(GRID * 1e3).round(6).tolist(), gamma=g)]

    def pack(arr, key):
        a = np.asarray(arr, dtype=float)
        a = a * 1e-5 if key == "u" else np.log10(np.maximum(a, 1e-300))
        a = np.where(np.isfinite(a), a, np.nan).astype(np.float32)
        return base64.b64encode(a.tobytes()).decode()

    fronts = {}
    for start, tt in times.items():
        sol = [solver_on_grid(s, case, t, g) for t in tt]
        records.append(dict(kind="times", start=start, t_ns=(tt * 1e9).round(6).tolist(),
                            mf_mg=[round(x[1] * 1e3, 6) for x in sol], ms_mg=[round(x[2] * 1e3, 6) for x in sol]))
        for key in QTY:
            records.append(dict(kind="data", start=start, run="solver", qty=key,
                                b64=pack(np.stack([x[0][key] for x in sol]), key)))
        fronts[(start, "solver")] = [(x[1], x[2]) for x in sol]
        for phys in ("2T", "chi1000", "imc"):
            runs = load(start, phys)
            if runs is None:
                continue
            fr = [on_grid(frame(runs, t), g) for t in tt]
            for key in QTY:
                records.append(dict(kind="data", start=start, run=phys, qty=key,
                                    b64=pack(np.stack([x[key] for x in fr]), key)))
            # fronts at the display times (mean over the seeds), over the solver's
            rows = []
            for t, (mf, ms) in zip(tt, fronts[(start, "solver")]):
                vals = [np.interp(np.log(t), np.log(r["series_t"][1:]), r[key][1:])
                        for r in runs for key in ("m_heat", "m_shock")]
                mh = np.nanmean(vals[0::2])
                msh = np.nanmean(vals[1::2])
                rows.append([round(mh / mf, 4), round(msh / ms, 4)])
            records.append(dict(kind="fronts", start=start, run=phys, rows=rows))
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(json.dumps(r) for r in records) + "\n]\n")
    print(f"  wrote   {path} ({path.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- static figures
def fig_profiles():
    s, case = solver()
    g = gamma()
    times_ns = (2.0, 10.0, 50.0)
    keys = (("rho", r"$\rho$ [g/cm$^3$]", True), ("p", r"$P$ [Mbar]", True), ("u", r"$u$ [km/s]", False),
            ("T", r"$T$ [HeV] (dashed: radiation)", True), ("A", r"entropy $P/\rho^\gamma$ / solver's", False))
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.2))
    tcol = {2.0: 0.35, 10.0: 0.65, 50.0: 1.0}
    for row, start in enumerate(("cold", "seeded")):
        for t_ns in times_ns:
            t = t_ns * 1e-9
            ref, mf, ms = solver_on_grid(s, case, t, g)
            for col, (key, lab, log) in enumerate(keys):
                ax = axes[row, col]
                unit = {"rho": 1, "p": 1e-12, "u": 1e-5, "T": 1 / K, "A": 1}[key]
                if key != "A":
                    ax.plot(GRID * 1e3, ref[key] * unit, color=INK, lw=1.3, alpha=tcol[t_ns],
                            label="solver" if t_ns == 50.0 else None)
                for phys in ("2T", "chi1000", "imc"):
                    runs = load(start, phys)
                    if runs is None:
                        continue
                    f = on_grid(frame(runs, t), g)
                    if key == "A":
                        shocked = (GRID > 1.5 * mf) & (GRID < 0.97 * ms)
                        y = np.where(shocked, f["A"] / ref["A"], np.nan)
                    else:
                        y = f[key] * unit
                    ax.plot(GRID * 1e3, y, color=COLORS[phys], lw=1.2, alpha=tcol[t_ns],
                            label=LABELS[phys] if t_ns == 50.0 else None)
                    if key == "T":
                        ax.plot(GRID * 1e3, f["Trad"] * unit, color=COLORS[phys], lw=0.9, ls="--", alpha=tcol[t_ns])
                ax.set_xscale("log")
                ax.set_xlim(1e-2, 190)
                if log:
                    ax.set_yscale("log")
                ax.set_ylabel(lab)
                ax.set_xlabel(r"$m$ [mg/cm$^2$]")
                ivd.style_axes(ax)
        axes[row, 3].set_ylim(1e-5, 3.0)
        axes[row, 4].axhline(1.0, color="#888", lw=0.8)
        axes[row, 4].set_ylim(0.8, 1.3)
        axes[row, 0].set_title(("cold start (t = 0)" if start == "cold" else "solver's profile seeded at 2 ns")
                               + "; 2, 10, 50 ns (light to dark)", loc="left", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Al (Test 6), 30 K target, paper bath", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "profiles.png", dpi=130)
    plt.close(fig)


def fig_fronts():
    s, case = solver()
    hs, ps = s.heat_solver, s.shock_solver
    p0 = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for start, ls in (("cold", "-"), ("seeded", "--")):
        for phys in ("2T", "chi1000", "imc"):
            runs = load(start, phys)
            if runs is None:
                continue
            for i, r in enumerate(runs):
                t = r["series_t"][1:]
                ok = t > 1e-11
                t = t[ok]
                mf = np.array([float(hs.ablated_mass(time=x)) for x in t])
                ms = np.array([float(np.ravel(ps.shocked_mass(time=x))[0]) for x in t])
                kw = dict(color=COLORS[phys], ls=ls, lw=1.2, alpha=1.0 if i == 0 else 0.5,
                          label=f"{LABELS[phys]}, {start}" if i == 0 else None)
                axes[0].plot(t * 1e9, r["m_heat"][1:][ok] / mf, **kw)
                axes[1].plot(t * 1e9, r["m_shock"][1:][ok] / ms, **kw)
                axes[2].plot(t * 1e9, r["p_dense"][1:][ok] / (p0 * t**hs.c3), **kw)
                axes[3].plot(t * 1e9, r["T_surface"][1:][ok] / r["Trad_surface"][1:][ok], **kw)
    for ax, lab in zip(axes, ("heat front mass / solver's", "shock mass / solver's",
                              "front pressure (dense side) / solver's", r"$T_m / T_{rad}$, first cell")):
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 50)
        ax.axhline(1.0, color="#888", lw=0.8)
        ax.axvline(2.0, color="#bbb", lw=0.8, ls=":")
        ax.set_xlabel("t [ns]")
        ax.set_ylabel(lab)
        ivd.style_axes(ax)
    axes[0].set_ylim(0.8, 1.3)
    axes[1].set_ylim(0.8, 1.1)
    axes[2].set_ylim(0.7, 1.3)
    axes[3].set_ylim(0.4, 1.2)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fronts.png", dpi=130)
    plt.close(fig)


def table():
    s, case = solver()
    g = gamma()
    hs, ps = s.heat_solver, s.shock_solver
    p0 = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3
    lines = ["| start | run | t [ns] | heat front | shock | front pressure | ρ, shocked: median (range) | "
             "entropy, shocked: median (range) | T_m/T_r, mid-layer |", "|---|---|---|---|---|---|---|---|---|"]
    for start in ("cold", "seeded"):
        for phys in ("2T", "chi1000", "imc"):
            runs = load(start, phys)
            if runs is None:
                continue
            for t_ns in (2.0, 5.0, 10.0, 20.0, 50.0):
                t = t_ns * 1e-9
                if t < runs[0]["t"][0] * 0.999:
                    continue
                f = frame(runs, t)
                mf = float(hs.ablated_mass(time=f["t"]))
                ms = float(np.ravel(ps.shocked_mass(time=f["t"]))[0])
                v = [np.interp(np.log(f["t"]), np.log(r["series_t"][1:]), r[key][1:])
                     for r in runs for key in ("m_heat", "m_shock", "p_dense")]
                mh, msh, pd = np.nanmean(v[0::3]), np.nanmean(v[1::3]), np.nanmean(v[2::3])
                sh = (f["mc"] > 1.5 * mf) & (f["mc"] < 0.97 * ms)
                ref = ps.solve(mass=f["mc"][sh], time=f["t"])
                rr = f["rho"][sh] / ref["density"]
                aa = (f["p"][sh] / f["rho"][sh]**g) / (ref["pressure"] / ref["density"]**g)
                i_mid = int(np.searchsorted(f["mc"], 0.5 * mh))
                lines.append(f"| {start} | {LABELS[phys]} | {f['t'] * 1e9:.1f} | {mh / mf:.3f} | {msh / ms:.3f} | "
                             f"{pd / (p0 * f['t']**hs.c3):.3f} | {np.median(rr):.3f} ({rr.min():.3f}–{rr.max():.3f}) | "
                             f"{np.median(aa):.3f} ({aa.min():.3f}–{aa.max():.3f}) | "
                             f"{f['T'][i_mid] / f['Trad'][i_mid]:.3f} |")
    text = "\n".join(lines)
    (FIGURES / "table.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    table()
    fig_profiles()
    fig_fronts()
    slider_data(FIGURES / "slider" / "slider_data.json")
