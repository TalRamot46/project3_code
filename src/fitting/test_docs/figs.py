import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fits, subfit
from compute import fit_funcs, YC, NS, KPH

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 160})
C_REF, C_FIT = "#222222", "#d62728"

def selfsimilar(R, path):
    fig, ax = plt.subplots(2, 8, figsize=(17, 4.6), gridspec_kw=dict(height_ratios=[3, 1.3]), sharex="col")
    sp, sf = R["sub_prof"], R["sub_fit"]
    names = dict(V="V", U="U", P="P", T=r"\hat T")
    for i, v in enumerate("VUPT"):
        y = sp["y"]; F = sp[v]; Ff = subfit.ev(sf[v], y)
        yl = np.geomspace(2e-3, YC, 60); Fl = R["h"].get_self_similar_profiles(xsi_vec=yl * R["h"].xsi_f)[v]
        a = ax[0, i]; a.plot(np.r_[yl, y], np.r_[Fl, F], c=C_REF, lw=1.6, label="solver")
        a.plot(y, Ff, "--", c=C_FIT, lw=1.3, label="fit"); a.plot(yl, subfit.ev(sf[v], yl), ":", c=C_FIT, lw=1)
        a.axvspan(0, YC, color="0.85", alpha=0.6); a.set_title(f"sub ${names[v]}(y)$")
        if v == "V": a.set_yscale("log")
        ax[1, i].semilogy(y, np.abs(Ff / F - 1) + 1e-12, c=C_FIT); ax[1, i].set_ylim(1e-6, 1e-1); ax[1, i].set_xlabel("$y=m/m_f$")
    ax[0, 0].legend(fontsize=7); ax[1, 0].set_ylabel("rel. error")
    sp, sf = R["shock_prof"], R["shock_fit"]
    fam = dict(V="piston", U="blend", P="blend", T="piston")
    yh = [R["sc"]["yhat"] * (t / NS)**float(R["E"]["yhat"]) for t in (0.1e-9, 2e-9)]
    for i, v in enumerate("VUPT"):
        y = sp["y"]; F = sp[v]; Ff = fits.ev(fam[v], sf[v], y)
        a = ax[0, 4 + i]; a.plot(y, F, c=C_REF, lw=1.6); a.plot(y, Ff, "--", c=C_FIT, lw=1.3)
        a.axvspan(min(yh), max(yh), color="#9ecae1", alpha=0.4); a.set_xscale("log"); a.set_title(f"shock ${names[v]}_s(y_s)$")
        if v in "VT": a.set_yscale("log")
        ax[1, 4 + i].loglog(y, np.abs(Ff / F - 1) + 1e-12, c=C_FIT); ax[1, 4 + i].set_ylim(1e-7, 1e-1); ax[1, 4 + i].set_xlabel("$y_s=m/m_s$")
    fig.suptitle(f"{R['meta']['name']}: self-similar profiles (black) and fits (red). Grey: surface layer $y<{YC}$ (fit not valid). Blue: $\\hat y(t)=m_f/m_s$ for 0.1–2 ns", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def dimensional(R, path_m, path_x, times=(0.5e-9, 1e-9, 2e-9)):
    F = fit_funcs(R); s = R["s"]
    figm, axm = plt.subplots(1, 4, figsize=(17, 3.8)); figx, axx = plt.subplots(1, 4, figsize=(17, 3.8))
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for t, c in zip(times, cols):
        mf = s.heat_solver.ablated_mass(time=t); ms = s.shock_solver.shocked_mass(time=t)
        m_sub = mf * np.linspace(YC, 0.999, 200)
        m_sh = np.unique(np.concatenate([mf * (1 + np.geomspace(1e-4, 1e-1, 40)), np.geomspace(mf * 1.1, ms * 0.999, 250)]))
        m = np.concatenate([m_sub, m_sh]); ref = s.solve(mass=m, time=t)
        fs, fh = F["sub"](m_sub, t), F["shock"](m_sh, t)
        fit = {k: np.concatenate([fs[k], fh[k]]) for k in ("v", "u", "p", "T", "x")}
        rv = dict(rho=ref["density"], u=ref["velocity"] * 1e-5, p=ref["pressure"] * 1e-12, T=ref["temperature"] / KPH)
        fv = dict(rho=1 / fit["v"], u=fit["u"], p=fit["p"], T=fit["T"])
        for j, key in enumerate(("rho", "u", "p", "T")):
            for axs, X, Xf in ((axm, m, m), (axx, ref["position"] * 1e4, fit["x"] * 1e4)):
                axs[j].plot(X, rv[key], c=c, lw=1.8, alpha=0.6, label=f"solver {t/NS:g} ns")
                axs[j].plot(Xf, fv[key], "--", c="k", lw=0.9, label="fit" if t == times[0] else None)
    labels = dict(rho=r"$\rho$ [g/cm$^3$]", u="$u$ [km/s]", p="$p$ [MBar]", T="$T$ [HeV]")
    for j, key in enumerate(("rho", "u", "p", "T")):
        axm[j].set_ylabel(labels[key]); axm[j].set_xlabel(r"$m$ [g/cm$^2$]"); axm[j].set_xscale("log")
        axx[j].set_ylabel(labels[key]); axx[j].set_xlabel(r"$x$ [$\mu$m]")
        if key == "rho": axm[j].set_yscale("log"); axx[j].set_yscale("log")
    axm[0].legend(fontsize=7); axx[0].legend(fontsize=7)
    figm.suptitle(f"{R['meta']['name']}: Lagrangian profiles, solver (colour) vs. fit formulas (black dashed); sub region shown for $y\\geq{YC}$", fontsize=10)
    figx.suptitle(f"{R['meta']['name']}: Eulerian profiles, solver (colour) vs. fit formulas (black dashed)", fontsize=10)
    for f_, p_ in ((figm, path_m), (figx, path_x)):
        f_.tight_layout(); f_.savefig(p_); plt.close(f_)

def fronts(R, path):
    F = fit_funcs(R); s = R["s"]
    tc = NS * (1 / R["sc"]["yhat"])**(1 / float(R["E"]["yhat"]))  # m_f = m_s here
    t = np.geomspace(max(0.05e-9, 1.5 * tc), 2e-9, 40)
    ref = {k: [] for k in ("x_b", "x_af", "x_p", "x_s")}
    for ti in t:
        mf = s.heat_solver.ablated_mass(time=ti); ms = s.shock_solver.shocked_mass(time=ti)
        r = s.solve(mass=np.array([mf * 0.5, mf * 0.9, mf + 0.2 * (ms - mf), mf + 0.5 * (ms - mf), mf + 0.9 * (ms - mf)]), time=ti)
        ref["x_b"].append(r["boundary_position"] + r["heat_position"]); ref["x_af"].append(r["heat_position"])
        ref["x_p"].append(r["piston_position"]); ref["x_s"].append(r["shock_position"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    for key, c, lab in (("x_b", "#9467bd", "boundary $x_b$"), ("x_af", "#d62728", "ablation front $x_{af}$"), ("x_p", "#2ca02c", "piston $x_p$"), ("x_s", "#1f77b4", "shock $x_s$")):
        ax[0].plot(t / NS, np.array(ref[key]) * 1e4, c=c, lw=2, alpha=0.5, label=lab)
        ax[0].plot(t / NS, F[key](t) * 1e4, "--", c="k", lw=0.8)
    ax[0].plot(t / NS, F["x_yc"](t) * 1e4, ":", c="#9467bd", label=f"element $y={YC}$ (fit)")
    ax[0].set_xlabel("t [ns]"); ax[0].set_ylabel(r"x [$\mu$m]"); ax[0].legend(fontsize=7)
    for key, c in (("x_af", "#d62728"), ("x_p", "#2ca02c"), ("x_s", "#1f77b4")):
        ax[2].plot(t / NS, np.array(ref[key]) * 1e4, c=c, lw=2, alpha=0.5); ax[2].plot(t / NS, F[key](t) * 1e4, "--", c="k", lw=0.8)
    ax[2].set_xlabel("t [ns]"); ax[2].set_ylabel(r"x [$\mu$m]"); ax[2].set_title("zoom: ablation front, piston, shock", fontsize=9)
    ax = [ax[0], ax[1]]
    mf = [s.heat_solver.ablated_mass(time=ti) for ti in t]; ms = [s.shock_solver.shocked_mass(time=ti) for ti in t]
    ax[1].loglog(t / NS, mf, c="#d62728", lw=2, alpha=0.5, label="$m_f$"); ax[1].loglog(t / NS, ms, c="#1f77b4", lw=2, alpha=0.5, label="$m_s$")
    ax[1].loglog(t / NS, R["sc"]["mf"] * (t / NS)**float(R["E"]["mf"]), "--k", lw=0.8)
    ax[1].loglog(t / NS, R["sc"]["ms"] * (t / NS)**float(R["E"]["ms"]), "--k", lw=0.8)
    ax[1].set_xlabel("t [ns]"); ax[1].set_ylabel(r"m [g/cm$^2$]"); ax[1].legend(fontsize=7)
    fig.suptitle(f"{R['meta']['name']}: fronts, solver (colour) vs. formulas (black dashed)", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
