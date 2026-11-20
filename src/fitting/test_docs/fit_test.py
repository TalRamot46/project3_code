import sys, json, numpy as np, logging
logging.disable(logging.CRITICAL)
from load import load
import fits

def sub_profiles(h, y):
    r = h.get_self_similar_profiles(xsi_vec=y * h.xsi_f)
    assert r["imin"] == 0
    return dict(V=r["V"], U=r["U"], P=r["P"], T=r["T"], S=r["S"])

def shock_profiles(sh, y):
    V, U, P = sh.get_self_similar_profiles(xsi_vec=y * sh.xsi_s)
    return dict(V=V, U=U, P=P)

def sub_grid():
    return np.unique(np.concatenate([np.geomspace(1e-8, 1e-2, 80), np.linspace(1e-2, 0.98, 600),
                                     1 - np.geomspace(2e-2, 1e-6, 150)]))
def shock_grid():
    return np.unique(np.concatenate([np.geomspace(1e-8, 1e-2, 150), np.linspace(1e-2, 1 - 1e-10, 600)]))

def fit_all(k, tol=2e-3):
    s = load(k); h, sh = s.heat_solver, s.shock_solver
    out = {}
    # ---- subsonic region -------------------------------------------------
    y = sub_grid(); pr = sub_profiles(h, y)
    d = 1.0 / h.q                     # V ~ (1-y)^(1/q) at the front (exact asymptotics)
    eT = h.k * d / h.beta             # T ~ (1-y)^((1-mu)/(q beta))
    out["sub"] = dict(
        V=fits.auto_fit("front", y, pr["V"], F0=h.V0, e=d, tol=tol),
        U=fits.auto_fit("front", y, pr["U"], F0=h.U0, e=d, tol=tol),
        P=fits.auto_fit("pfront", y, pr["P"], F0=h.P0, Pf=h.Pf, e=d, tol=tol),
        T=fits.auto_fit("front", y, pr["T"], F0=1.0, e=eT, tol=tol),
    )
    # ---- shock region ------------------------------------------------------
    ys = shock_grid(); ps = shock_profiles(sh, ys)
    tau, om, g = sh.tau, sh.omega, sh.gamma
    dm = (tau + 2) * (1 - om) / (2 - om)
    eV = (tau / g + om * (tau + 2) / (2 - om)) / dm   # V ~ y^eV at the piston (entropy layer)
    mu, beta = s.mu_shock, s.beta_shock
    That = (ps["P"] * ps["V"]**(1 - mu))**(1 / beta)
    eTs = (1 - mu) * eV / beta
    Ths = (sh.Ps * sh.Vs**(1 - mu))**(1 / beta)
    out["shock"] = dict(
        V=fits.auto_fit("piston", ys, ps["V"], F1=sh.Vs, e=eV, tol=tol),
        U=fits.auto_fit("blend", ys, ps["U"], F0=sh.U0, F1=sh.Us, tol=tol),
        P=fits.auto_fit("blend", ys, ps["P"], F0=1.0, F1=sh.Ps, tol=tol),
        T=fits.auto_fit("piston", ys, That, F1=Ths, e=eTs, tol=tol),
    )
    out["exps"] = dict(d_front=d, eT_front=eT, eV_piston=eV, eT_piston=eTs)
    return s, out

if __name__ == "__main__":
    for k in map(int, sys.argv[1:]):
        s, out = fit_all(k)
        print(f"== Test {k}")
        for reg in ("sub", "shock"):
            for v, p in out[reg].items():
                print(f"  {reg:5s} {v}: fam={p['fam']:6s} K={p['K']} max={p['max_err']*100:.3f}% avg={p['avg_err']*100:.4f}%  a={np.round(p['a'],4).tolist()} {'d=%.4f'%p['d'] if 'd' in p else ''}")
