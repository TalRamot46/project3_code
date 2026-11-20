"""Generic fit families for the dimensionless self-similar profiles.

All families are exact at one or both ends by construction and use the
analytically known end-point power laws, so only a short polynomial
correction inside an exponential is fitted (linear least squares in log space).

Families
--------
front(F0, e, a)     : F(y) = F0 (1-y)^e exp(sum_k a_k y^k)          , y in [0,1)  (sub region)
pfront(Pf, D0, e, a): P(y) = Pf - D0 (1-y)^e exp(sum_k a_k y^k)     , D0 = Pf - P(0)
piston(F1, e, a)    : F(y) = F1 y^e exp(sum_k a_k (1-y)^k)          , y in (0,1]  (shock region)
blend(F0, F1, d, a) : F(y) = F0 + (F1-F0) y^d exp(sum_k a_k (1-y)^k)  (finite at both ends)
"""
import numpy as np
from scipy.optimize import least_squares

def _poly(z, a):
    return sum(ak * z**(k + 1) for k, ak in enumerate(a)) if len(a) else 0.0 * z

def ev(fam, p, y):
    y = np.asarray(y, float)
    if fam == "front":
        return p["F0"] * (1 - y)**p["e"] * np.exp(_poly(y, p["a"]))
    if fam == "pfront":
        return p["Pf"] - p["D0"] * (1 - y)**p["e"] * np.exp(_poly(y, p["a"]))
    if fam == "sing":
        return p["C"] * y**(-p["s"]) * (1 - y)**p["e"] * np.exp(_poly(y, p["a"]))
    if fam == "piston":
        return p["F1"] * y**p["e"] * np.exp(_poly(1 - y, p["a"]))
    if fam == "blend":
        return p["F0"] + (p["F1"] - p["F0"]) * y**p["d"] * np.exp(_poly(1 - y, p["a"]))
    raise ValueError(fam)

def relerr(fit, ref):
    return np.abs(fit / ref - 1.0)

def _lin(z, target, K):
    """least squares target = sum_k a_k z^k (k=1..K)"""
    M = np.stack([z**(k + 1) for k in range(K)], axis=1)
    a, *_ = np.linalg.lstsq(M, target, rcond=None)
    return list(a)

def fit_family(fam, y, F, *, K, F0=None, F1=None, e=None, Pf=None, s_exp=None):
    y = np.asarray(y, float); F = np.asarray(F, float)
    if fam == "front":
        tgt = np.log(F / F0) - e * np.log(1 - y)
        p = dict(F0=F0, e=e, a=_lin(y, tgt, K))
    elif fam == "pfront":
        D0 = Pf - F0
        tgt = np.log((Pf - F) / D0) - e * np.log(1 - y)
        p = dict(Pf=Pf, D0=D0, e=e, a=_lin(y, tgt, K))
        # refine on relative error of P itself
        def res(a):
            q = dict(p, a=list(a)); return ev("pfront", q, y) / F - 1
        p["a"] = list(least_squares(res, p["a"]).x)
    elif fam == "sing":
        tgt = np.log(F / F0) + s_exp * np.log(y) - e * np.log(1 - y)   # F0: sign/scale reference
        M = np.stack([np.ones_like(y)] + [y**(k + 1) for k in range(K)], axis=1)
        sol, *_ = np.linalg.lstsq(M, tgt, rcond=None)
        p = dict(C=F0 * np.exp(sol[0]), s=s_exp, e=e, a=list(sol[1:]))
        def res(x):
            q = dict(p, C=F0 * np.exp(x[0]), a=list(x[1:])); return ev("sing", q, y) / F - 1
        x = least_squares(res, sol).x
        p["C"] = F0 * np.exp(x[0]); p["a"] = list(x[1:])
    elif fam == "piston":
        tgt = np.log(F / F1) - e * np.log(y)
        p = dict(F1=F1, e=e, a=_lin(1 - y, tgt, K))
    elif fam == "blend":
        # nonlinear: d and a free
        g = (F - F0) / (F1 - F0)
        ok = g > 0
        d0 = np.polyfit(np.log(y[ok][: max(5, ok.sum() // 10)]), np.log(g[ok][: max(5, ok.sum() // 10)]), 1)[0]
        d0 = float(np.clip(d0, 1e-3, 5))
        def res(x):
            q = dict(F0=F0, F1=F1, d=x[0], a=list(x[1:]))
            return ev("blend", q, y) / F - 1
        x = least_squares(res, [d0] + [0.0] * K, bounds=([1e-4] + [-np.inf] * K, [20] + [np.inf] * K)).x
        p = dict(F0=F0, F1=F1, d=float(x[0]), a=list(x[1:]))
    else:
        raise ValueError(fam)
    if fam in ("front", "piston"):
        # refine on relative error of F itself (log-space solution is the start)
        z = y if fam == "front" else 1 - y
        def res(a):
            q = dict(p, a=list(a)); return ev(fam, q, y) / F - 1
        p["a"] = list(least_squares(res, p["a"]).x)
    p["fam"] = fam
    return p

def auto_fit(fam, y, F, *, tol=2e-3, Kmax=5, ygrid_err=None, Fgrid_err=None, **kw):
    best = None
    for K in range(1, Kmax + 1):
        p = fit_family(fam, y, F, K=K, **kw)
        yy = y if ygrid_err is None else ygrid_err
        FF = F if Fgrid_err is None else Fgrid_err
        err = relerr(ev(fam, p, yy), FF)
        p["K"] = K; p["max_err"] = float(err.max()); p["avg_err"] = float(err.mean())
        if best is None or p["max_err"] < best["max_err"] * 0.8:
            best = p
        if p["max_err"] < tol:
            best = p
            break
    return best
