"""Figures and tables of md/misalignment/01_surface_equilibrium.md, 03_startup.md, 05_shocked_region.md.

    python -m test_misalignment.test3_transparency_figures
"""
from __future__ import annotations

import logging

import numpy as np

logging.disable(logging.INFO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from test_misalignment import test3_hydro_only as h
from test_misalignment import test3_on_the_solver as tos
from test_misalignment.test3_transparency import FIGURES, OUT

K = ivd.K_PER_HEV
GAMMA = 1.25
# categorical slots in fixed order (dataviz reference palette); the solver is ink
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#1a1a1a"
C8 = C + ["#6250d6", "#e34948"]
TIMES_PS = (0.01, 0.1, 1.0, 10.0, 100.0, 2000.0)
EXTRA = {"cold_2T_30K_chi100": (r"2T diffusion, $\chi$ = 100", "#6250d6"),
         "cold_IMC_30K_seed1": ("IMC, seed 1", "#e34948")}


def extra_dense():
    return {n: (load_dense(n), lab, col) for n, (lab, col) in EXTRA.items() if (OUT / f"dense_{n}.npz").exists()}


def _style(ax):
    ivd.style_axes(ax)


def case():
    return ivd._hydro_case(ivd.HYDRO_CASES["test3_smooth"])


def opacity(rho, T, c=None):
    """Rosseland mass opacity kappa [cm^2/g] of the test's law: sigma = kappa rho = rho^(1+lambda) / (g T^alpha)."""
    c = c or case()
    return rho**c.lambda_ / (c.g_Kelvin * np.maximum(T, 1.0)**c.alpha)


# --------------------------------------------------------------------------- the solver
def solver_profile(s, t, n=600, beyond=20.0):
    """(m, rho, T) of the solver at t from 2e-3 m_f to beyond x m_f (or the shock)."""
    mf = float(s.heat_solver.ablated_mass(time=t))
    ms = float(np.ravel(s.shock_solver.shocked_mass(time=t))[0])
    m = np.concatenate((np.geomspace(2e-3 * mf, 0.9999 * mf, n), np.geomspace(1.0001 * mf, min(beyond * mf, 0.99 * ms), n)))
    sol = s.solve(mass=m, time=t)
    return mf, m, np.asarray(sol["density"]), np.asarray(sol["temperature"]), np.asarray(sol["pressure"])


def tau_from_front(m, kappa_dm_or_kappa, mf, cells=False):
    """|int_{m_f}^{m} kappa dm'| at each point: the optical depth between the front and m."""
    if cells:                      # per-cell kappa dm, m = cell-centre mass
        d = kappa_dm_or_kappa
        edges = np.concatenate(([0.0], np.cumsum(d)))
        mc_edges = np.concatenate(([0.0], np.cumsum(np.diff(m, prepend=0.0) * 0 + 1)))  # unused
        cum = 0.5 * (edges[:-1] + edges[1:])          # tau from the surface to the cell centre
        cum_f = np.interp(mf, m, cum)
        return np.abs(cum - cum_f)
    from scipy.integrate import cumulative_trapezoid
    cum = cumulative_trapezoid(kappa_dm_or_kappa, m, initial=0.0)
    return np.abs(cum - np.interp(mf, m, cum))


def solver_tau_from_front(m, kappa, mf):
    """int_m^{m_f} kappa dm (m < m_f) and int_{m_f}^m kappa dm (m > m_f), each side on its own (the
    opacity jumps by ~10^5 across the front)."""
    from scipy.integrate import cumulative_trapezoid
    tau = np.empty_like(m)
    hot = m < mf
    cum = cumulative_trapezoid(kappa[hot], m[hot], initial=0.0)
    tau[hot] = cum[-1] - cum
    tau[~hot] = cumulative_trapezoid(kappa[~hot], m[~hot], initial=0.0)
    return tau


HOT = 0.5      # the hot plasma: T > HOT x the surface temperature (T_s for the solver)


def solver_layer_tau(s, t, c=None, level=None):
    """Optical depth of the hot plasma (T > HOT T_s), or of the whole layer (level=0)."""
    mf, m, rho, T, _ = solver_profile(s, t)
    level = HOT if level is None else level
    hot = (m < mf) & (T > level * K)          # T_s = 1 HeV
    return float(np.trapz(opacity(rho[hot], T[hot], c), m[hot]))


# --------------------------------------------------------------------------- the simulation
def load_dense(name="cold_2T_30K"):
    return np.load(OUT / f"dense_{name}.npz")


def sim_front(m_edges, T):
    """(heat front mass, index of the first cold cell): T falls below 0.3 T(cell 0)."""
    mc = m_edges - 0.5 * np.diff(m_edges, prepend=0.0)
    level = ivd.HEAT_FRONT_LEVEL * T[0]
    below = np.flatnonzero(T < level)
    if not below.size or below[0] == 0:
        return np.nan, 0
    j = int(below[0])
    return float(np.interp(level, [T[j], T[j - 1]], [mc[j], mc[j - 1]])), j


def sim_tau(d, k, c=None):
    """(m_c, tau from the front, m_f, layer optical depth) of frame k of a dense run."""
    m = d["m"]
    dm = np.diff(m, prepend=0.0)
    mc = m - 0.5 * dm
    rho, T = d["rho"][k].astype(float), d["T_material"][k].astype(float)
    mf, j = sim_front(m, T)
    kdm = opacity(rho, T, c) * dm
    cum = np.cumsum(kdm) - 0.5 * kdm
    tau = np.abs(cum - np.interp(mf, mc, cum))
    hot = T > HOT * T[0]
    hot[j:] = False
    layer = float(np.sum(kdm[hot]))
    return mc, tau, mf, layer


# --------------------------------------------------------------------------- figures
def fig_tau_profiles(s, d):
    c = case()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    # times log-spaced, coloured along a rainbow (turbo) map by log t, with a colour bar
    times_ps = np.geomspace(0.01, 2000.0, 12)
    cmap = plt.get_cmap("turbo")
    norm = matplotlib.colors.LogNorm(vmin=times_ps[0], vmax=times_ps[-1])
    for tps in times_ps:
        t = tps * 1e-12
        col = cmap(norm(tps))
        mf, m, rho, T, _ = solver_profile(s, t, beyond=3.0)
        tau = solver_tau_from_front(m, opacity(rho, T, c), mf)
        axes[0].plot(m / mf, tau, color=col, lw=2)
        k = int(np.argmin(np.abs(d["t"] - t)))
        if abs(np.log(d["t"][k] / t)) < 0.1:
            mc, tau_s, mf_s, _ = sim_tau(d, k, c)
            keep = mc < 3 * mf_s
            axes[0].plot(mc[keep] / mf_s, tau_s[keep], color=col, lw=1.1, ls="--")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes[0], pad=0.02)
    cb.set_label("t [ps]")
    axes[0].plot([], [], color="#555", lw=2, label="solver")
    axes[0].plot([], [], color="#555", lw=1.1, ls="--", label=r"2T diffusion, $\chi$ = 1")
    axes[0].axhline(1.0, color="#888", lw=0.8)
    axes[0].axvline(1.0, color="#888", lw=0.8, ls=":")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.0, 2.0)
    axes[0].set_ylim(1e-3, 1e7)
    axes[0].set_xlabel(r"$m / m_f(t)$")
    axes[0].set_ylabel(r"$\tau(m,t)=\left|\int_{m_f}^{m}\kappa\,dm'\right|=\left|\int\sigma\,dx\right|$")
    axes[0].set_title("optical depth measured from the ablation front", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    _style(axes[0])

    # the layer's optical depth against time
    tt = np.geomspace(1e-15, 2e-9, 40)
    lay = np.array([solver_layer_tau(s, t, c) for t in tt])
    axes[1].plot(tt * 1e12, lay, color=INK, lw=2, label=r"solver, $T > T_s/2$")
    lay_all = np.array([solver_layer_tau(s, t, c, level=0.0) for t in tt])
    axes[1].plot(tt * 1e12, lay_all, color=INK, lw=1.2, ls="--", label=r"solver, the whole layer $\int_0^{m_f}\kappa\,dm$")
    ks = np.unique(np.searchsorted(d["t"], np.geomspace(max(d["t"][1], 1e-16), 2e-9, 150)).clip(1, d["t"].size - 1))
    sim = np.array([sim_tau(d, k, c)[3] for k in ks])
    axes[1].plot(d["t"][ks] * 1e12, sim, color=C[0], lw=1.6, label=r"2T diffusion ($\chi$ = 1), $T_m > T_m(0)/2$")
    for n, (dd, lab, col) in extra_dense().items():
        kk = np.unique(np.searchsorted(dd["t"], np.geomspace(max(dd["t"][1], 1e-16), 2e-9, 150)).clip(1, dd["t"].size - 1))
        axes[1].plot(dd["t"][kk] * 1e12, [sim_tau(dd, k, c)[3] for k in kk], color=col, lw=1.3, label=lab)
    axes[1].axhline(1.0, color="#888", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("t [ps]")
    axes[1].set_ylabel(r"optical depth of the hot plasma, $\mathcal{T}$")
    axes[1].set_title("optical depth of the ablated plasma", fontsize=10)
    axes[1].legend(frameon=False, fontsize=9)
    _style(axes[1])
    fig.tight_layout()
    fig.savefig(FIGURES / "tau_profiles.png", dpi=150)
    plt.close(fig)
    return tt, lay, d["t"][ks], sim


def fig_surface(s, d):
    """T at the surface (matter, radiation) vs T_bath(t) and T_s; and equilibrium in the layer."""
    t = d["t"]
    ok = t > 0
    T_bath = np.array([float(s.heat_solver.Tbath(time=x)) for x in t[ok]])
    Tm0, Tr0 = d["T_material"][ok, 0], d["T_rad"][ok, 0]
    # mid-layer: the cell at half the heat front mass
    mid_m, mid_r = [], []
    for k in np.flatnonzero(ok):
        mf, j = sim_front(d["m"], d["T_material"][k])
        i = int(np.searchsorted(d["m"], 0.5 * mf)) if np.isfinite(mf) else 0
        mid_m.append(d["T_material"][k, i])
        mid_r.append(d["T_rad"][k, i])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    tp = t[ok] * 1e12
    ax = axes[0]
    ax.plot(tp, T_bath / K, color=INK, lw=1.2, ls="--", label=r"$T_{\rm bath}(t)$ (paper bath)")
    ax.axhline(1.0, color=INK, lw=1.6, label=r"solver: $T(0,t)=T_s$ = 1 HeV")
    ax.plot(tp, Tr0 / K, color=C[1], lw=1.6, label="radiation, first cell")
    ax.plot(tp, Tm0 / K, color=C[0], lw=1.6, label="matter, first cell")
    for n, (dd, lab, col) in extra_dense().items():
        ok2 = dd["t"] > 0
        ax.plot(dd["t"][ok2] * 1e12, dd["T_material"][ok2, 0] / K, color=col, lw=1.2, label=f"matter, first cell, {lab}")
        axes[1].plot(dd["t"][ok2] * 1e12, dd["T_material"][ok2, 0] / dd["T_rad"][ok2, 0], color=col, lw=1.2,
                     label=f"first cell, {lab}")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 2000)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel("t [ps]")
    ax.set_ylabel("T [HeV]")
    ax.set_title(r"surface temperature, 2T diffusion ($\chi$ = 1) vs the solver", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    ax = axes[1]
    ax.plot(tp, Tm0 / Tr0, color=C[0], lw=1.6, label="first cell")
    ax.plot(tp, np.array(mid_m) / np.array(mid_r), color=C[2], lw=1.6, label=r"mid-layer ($m = m_f/2$)")
    ax.axhline(1.0, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 2000)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"$T_{\rm matter} / T_{\rm rad}$")
    ax.set_title("equilibrium in the ablated layer", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "surface_temperature.png", dpi=150)
    plt.close(fig)
    return dict(t=t[ok], T_bath=T_bath, Tm0=Tm0, Tr0=Tr0, Tm_mid=np.array(mid_m), Tr_mid=np.array(mid_r))


def front_series(s, d, beyond=2.0):
    """Sim / solver against t: the heat front mass, the pressure on the dense side, and rho, T of the
    just-shocked matter (at beyond x m_f of the solver) and of the mid-layer (0.5 m_f)."""
    c = case()
    rows = []
    m = d["m"]
    mc = m - 0.5 * np.diff(m, prepend=0.0)
    for k in range(1, d["t"].size):
        t = float(d["t"][k])
        T = d["T_material"][k].astype(float)
        mf_sim, j = sim_front(m, T)
        if not np.isfinite(mf_sim):
            continue
        p_dense = float(d["p"][k][j:j + 30].max())
        mf = float(s.heat_solver.ablated_mass(time=t))
        pts = np.array([0.5 * mf, beyond * mf])
        if mc[0] > 0.5 * mf:
            continue
        sol = s.solve(mass=np.concatenate((pts, [1.2 * beyond * mf, 1.4 * beyond * mf])), time=t)
        rho_s, T_s, p_s = (np.asarray(sol[key])[:2] for key in ("density", "temperature", "pressure"))
        sim = {key: np.interp(pts, mc, d[key][k].astype(float)) for key in ("rho", "T_material", "p")}
        p0, c3 = h.solver_pressure_law(s)
        rows.append((t, mf_sim / mf, p_dense / (p0 * t**c3),
                     sim["rho"][0] / rho_s[0], sim["T_material"][0] / T_s[0], sim["p"][0] / p_s[0],
                     sim["rho"][1] / rho_s[1], sim["T_material"][1] / T_s[1], sim["p"][1] / p_s[1]))
    return np.array(rows)


def fig_front(rows, extra=None):
    names = [("heat front mass $m_f$", 1, C[0]), ("pressure on the dense side of the front", 2, C[1]),
             (r"$\rho$ just beyond the front ($m = 2m_f$)", 6, C[2]),
             (r"$T$ just beyond the front ($m = 2m_f$)", 7, C[3]),
             (r"$\rho$, mid-layer ($m = m_f/2$)", 3, C[4]), (r"$T$, mid-layer ($m = m_f/2$)", 4, C[5])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, group in zip(axes, ((0, 1), (2, 3), (4, 5))):
        for g in group:
            label, col, color = names[g]
            ax.plot(rows[:, 0] * 1e12, rows[:, col], color=color, lw=1.6, label=label)
        ax.axhline(1.0, color="#888", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 2000)
        ax.set_xlabel("t [ps]")
        ax.set_ylabel("2T diffusion ($\\chi$ = 1) / solver")
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
    for n, (r2, lab, col) in (extra or {}).items():
        axes[0].plot(r2[:, 0] * 1e12, r2[:, 1], color=col, lw=1.0, ls="--", label=f"$m_f$, {lab}")
        axes[0].plot(r2[:, 0] * 1e12, r2[:, 2], color=col, lw=1.0, label=f"pressure, {lab}")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylim(0.2, 5.0)
    axes[1].set_yscale("log")
    axes[1].set_title(r"that matter was shocked at $t_s \approx 1.3\times10^{-4}\,t$", fontsize=10)
    axes[0].set_ylim(0, 3)
    axes[2].set_yscale("log")
    fig.suptitle("Test 3 at the ablation front: the simulation over the solver against time", x=0.02, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "front_vs_time.png", dpi=150)
    plt.close(fig)


def shock_time_of_mass(s, m):
    """t_s(m): when the solver's shock reached m (m_s is a power law of t)."""
    m1 = float(np.ravel(s.shock_solver.shocked_mass(time=1e-9))[0])
    m2 = float(np.ravel(s.shock_solver.shocked_mass(time=2e-9))[0])
    a = np.log(m2 / m1) / np.log(2.0)
    return 1e-9 * (m / m1)**(1.0 / a), a


def fig_entropy(s, rows, runs=("cold_2T_30K", "cold_2T_30K_chi100", "cold_1T_30K")):
    """The entropy of each element against the time its shock passed, with the drive at that time."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ps = s.shock_solver
    ax = axes[0]
    labels = {"cold_2T_30K": r"2T diffusion, $\chi$ = 1", "cold_2T_30K_chi100": r"2T diffusion, $\chi$ = 100",
              "cold_1T_30K": "1T conduction"}
    out = {}
    for i, name in enumerate(runs):
        path = tos.OUT / f"{name}.npz"
        if not path.exists():
            continue
        r = np.load(path)
        for k, ls in zip(range(r["t"].size), (":", "-.", "--", "-")):
            t = float(r["t"][k])
            mc = r["m"][k] - 0.5 * np.diff(r["m"][k], prepend=0.0)
            mf = float(s.heat_solver.ablated_mass(time=t))
            msh = float(np.ravel(ps.shocked_mass(time=t))[0])
            keep = (mc > 1.5 * mf) & (mc < 0.97 * msh)
            ref = ps.solve(mass=mc[keep], time=t)
            A = (r["p"][k][keep] / r["rho"][k][keep]**GAMMA) / (ref["pressure"] / ref["density"]**GAMMA)
            if name == "cold_2T_30K" or k == r["t"].size - 1:
                ax.plot(mc[keep] * 1e3, A, color=C[i], lw=1.5 if k == r["t"].size - 1 else 1.0, ls=ls,
                        label=f"{labels[name]}, {t * 1e9:.1f} ns" if (name == "cold_2T_30K" or True) else None)
            out[(name, k)] = (mc[keep], A)
    ax.axhline(1.0, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 300)
    ax.set_ylim(0.75, 1.15)
    ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
    ax.set_ylabel(r"entropy $P/\rho^\gamma$ / solver's")
    ax.set_title("each element keeps its entropy (2T, $\\chi$ = 1: 0.5, 1, 1.5, 2 ns)", fontsize=10, pad=34)
    ax.legend(frameon=False, fontsize=8, ncol=1)
    sec = ax.secondary_xaxis("top", functions=(lambda mm: shock_time_of_mass(s, np.maximum(mm, 1e-9) * 1e-3)[0] * 1e12,
                                               lambda tp: _mass_of_shock_time(s, tp * 1e-12) * 1e3))
    sec.set_xlabel(r"time the shock passed, $t_s(m)$ [ps]")
    _style(ax)

    # the drive at the time the element was shocked against its entropy deficit at 2 ns
    ax = axes[1]
    r = np.load(tos.OUT / "cold_2T_30K.npz")
    mc, A = out[("cold_2T_30K", r["t"].size - 1)]
    ts, _ = shock_time_of_mass(s, mc)
    ax.plot(ts * 1e12, A, color=C[0], lw=1.6, label="entropy at 2 ns / solver's, against $t_s(m)$")
    ax.plot(rows[:, 0] * 1e12, rows[:, 2], color=C[1], lw=1.4, label="front pressure (dense side) / solver's, at t")
    ax.plot(rows[:, 0] * 1e12, rows[:, 1], color=C[2], lw=1.2, ls="--", label="heat front mass / solver's, at t")
    ax.axhline(1.0, color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 2000)
    ax.set_ylim(0.6, 1.6)
    ax.set_xlabel("t [ps]")
    ax.set_ylabel("2T diffusion ($\\chi$ = 1) / solver")
    ax.set_title("the entropy deficit is the drive deficit at the time of shocking", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "entropy_memory.png", dpi=150)
    plt.close(fig)
    return out


def _mass_of_shock_time(s, t):
    m1 = float(np.ravel(s.shock_solver.shocked_mass(time=1e-9))[0])
    _, a = shock_time_of_mass(s, np.array([m1]))
    return m1 * (np.maximum(t, 1e-30) / 1e-9)**a


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    s = h.solver()
    d = load_dense()
    tt, lay, ts, sim = fig_tau_profiles(s, d)
    np.savez(OUT / "tau_layer.npz", t_solver=tt, tau_solver=lay,
             t_sim=ts, tau_sim=sim)
    surf = fig_surface(s, d)
    rows = front_series(s, d)
    np.save(OUT / "front_rows.npy", rows)
    extra = {}
    for n, (dd, lab, col) in extra_dense().items():
        r2 = front_series(s, dd)
        np.save(OUT / f"front_rows_{n}.npy", r2)
        extra[n] = (r2, lab, col)
    fig_front(rows, extra)
    fig_entropy(s, rows)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- task 2: p_f(t)-driven runs
def pdrive_compare(pairs=(("cold", "p_solver"), ("cold", "p_2T"), ("seeded", "p_2T"), ("seeded", "p_solver"))):
    """rad / hydro twins: the shocked matter (1.5 m_f(t) to 0.97 m_s(t) of the solver) at each snapshot."""
    s = h.solver()
    ps = s.shock_solver
    lines = ["| start | drive | t [ns] | ρ rad/hydro: median (range) | entropy rad/hydro | u rad/hydro | "
             "entropy hydro/solver: median (range) | max T shocked [eV] | max T_rad/T_m shocked |",
             "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for start, drive in pairs:
        pr, ph = OUT / f"pdrive_{start}_rad_{drive}.npz", OUT / f"pdrive_{start}_hydro_{drive}.npz"
        if not (pr.exists() and ph.exists()):
            continue
        R, H = np.load(pr), np.load(ph)
        m = R["m"]
        mc = m - 0.5 * np.diff(m, prepend=0.0)
        for k in range(min(R["t"].size, H["t"].size)):
            t = float(R["t"][k])
            mf = float(s.heat_solver.ablated_mass(time=t))
            msh = float(np.ravel(ps.shocked_mass(time=t))[0])
            keep = (mc > 1.5 * mf) & (mc < 0.97 * msh)
            A = lambda r: r["p"][k][keep] / r["rho"][k][keep]**GAMMA
            ref = ps.solve(mass=mc[keep], time=t)
            Aref = ref["pressure"] / ref["density"]**GAMMA
            rr = R["rho"][k][keep] / H["rho"][k][keep]
            ra = A(R) / A(H)
            ru = R["u"][k][keep] / H["u"][k][keep]
            ah = A(H) / Aref
            fmt = lambda a: f"{np.median(a):.5f} ({a.min():.5f}–{a.max():.5f})"
            lines.append(f"| {start} | {drive} | {t * 1e9:.2f} | {fmt(rr)} | {fmt(ra)} | {fmt(ru)} | "
                         f"{np.median(ah):.3f} ({ah.min():.3f}–{ah.max():.3f}) | "
                         f"{R['T_material'][k][keep].max() / 11604.5:.3f} | "
                         f"{(R['T_rad'][k][keep] / R['T_material'][k][keep]).max():.4f} |")
            out[(start, drive, k)] = (mc[keep], rr, ra, ah)
    text = "\n".join(lines)
    (FIGURES / "pdrive_table.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return out


def fig_pdrive(out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    i = 0
    for (start, drive, k), (mc, rr, ra, ah) in out.items():
        t_ns = (0.5, 1.0, 1.5, 2.0)[k]
        axes[0].plot(mc * 1e3, ra, color=C8[i % 8], lw=1.3, label=f"{start} start, {drive.replace('_', ' ')}, {t_ns:g} ns")
        axes[1].plot(mc * 1e3, ah, color=C8[i % 8], lw=1.3)
        i += 1
    axes[0].set_ylabel("entropy: rad-hydro / hydro only (same drive)")
    axes[1].set_ylabel("entropy: hydro only / solver's")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
        ax.axhline(1.0, color="#888", lw=0.8)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "pdrive_entropy.png", dpi=150)
    plt.close(fig)
