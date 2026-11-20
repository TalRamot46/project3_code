"""Fit family for the subsonic (ablation) region, y = m/m_f in [y_c, 1).

    F(y) = A * y^p * (1-y)^e * exp( a_1 y + ... + a_K y^K + b_1 (1-y)^delta + ... + b_J (1-y)^(J delta) )

e     : exact front exponent (delta = 1/q for V, U and Pf-P; (1-mu) delta / beta for T)
delta : 1/q; the front corrections are a series in (1-y)^delta
p     : free power at the surface side (V ~ y^(-1/k), U ~ y^(1-1/k) in the unregularized limit)
Everything is linear in log F, so the fit is a linear least-squares problem refined on the
relative error of F.
"""
import numpy as np
from scipy.optimize import least_squares

COMBOS = [(2, 1), (3, 1), (2, 2), (3, 2), (4, 2)]   # (K, J); L (=free y^p) always on

def design(y, K, J, delta):
    cols = [np.ones_like(y), np.log(y)] + [y**(i + 1) for i in range(K)] + [(1 - y)**((j + 1) * delta) for j in range(J)]
    return np.stack(cols, 1)

def unpack(sol, K, J, e, delta, sign):
    return dict(fam="gen", sign=sign, A=float(np.exp(sol[0])), p=float(sol[1]), e=float(e), delta=float(delta),
                a=[float(v) for v in sol[2:2 + K]], b=[float(v) for v in sol[2 + K:2 + K + J]], K=K, J=J)

def ev(p, y):
    y = np.asarray(y, float)
    z = sum(ak * y**(i + 1) for i, ak in enumerate(p["a"])) + sum(bj * (1 - y)**((j + 1) * p["delta"]) for j, bj in enumerate(p["b"]))
    return p["sign"] * p["A"] * y**p["p"] * (1 - y)**p["e"] * np.exp(z)

def fit(y, F, *, e, delta, tol=5e-3, y_err_max=0.999, combos=COMBOS):
    sign = 1.0 if np.all(F > 0) else -1.0
    G = sign * F
    assert np.all(G > 0)
    b = np.log(G) - e * np.log(1 - y)
    m_in = y <= y_err_max
    best = None
    for K, J in combos:
        M = design(y, K, J, delta)
        sol = np.linalg.lstsq(M, b, rcond=None)[0]
        sol = least_squares(lambda s: np.exp(M @ s - b) - 1, sol).x
        err = np.abs(np.exp(M @ sol - b) - 1)
        p = unpack(sol, K, J, e, delta, sign)
        p.update(max_err=float(err[m_in].max()), avg_err=float(err[m_in].mean()), max_err_all=float(err.max()), n=len(sol))
        if best is None or p["max_err"] < 0.7 * best["max_err"]:
            best = p
        if p["max_err"] < tol:
            return p
    return best
