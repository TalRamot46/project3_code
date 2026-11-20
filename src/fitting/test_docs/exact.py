"""Exact (rational) exponents of the subsonic ablation + piston shock self-similar solution."""
from fractions import Fraction as Fr

def fr(x, maxden=100000):
    return Fr(str(round(float(x), 12))).limit_denominator(maxden)

def exponents(alpha, beta, lam, mu, r, tau, omega):
    al, be, la, mu, r, tau, om = map(fr, (alpha, beta, lam, mu, r, tau, omega))
    tau = tau.limit_denominator(10000)
    D, D2 = 4 + 2*la - 4*mu, 2 + la - 2*mu
    E = {}
    E["n"] = (4 + al) / be; E["k"] = 1 - mu; E["q"] = E["k"]*E["n"] + la - 1
    E["delta"] = 1 / E["q"]; E["eT"] = E["k"] * E["delta"] / be
    E["a"] = (2*be - be*la - 8 - 2*al + mu*(4 + al)) / D
    E["b"] = (mu - 2) / D
    E["c"] = (3*mu - 2*la - 2 + tau*(2*(be - al - 4) - be*la + mu*(4 + al))) / D
    E["av"] = (2*be - 4 - al) / D2; E["bv"] = Fr(-1) / D2; E["cv"] = (1 - tau*(4 + al - 2*be)) / D2
    E["au"] = (be*(2 + la) - mu*(4 + al)) / D; E["bu"] = -mu / D; E["cu"] = (tau*(be*(2 + la) - mu*(4 + al)) + mu) / D
    E["ap"] = (be*la + 4 + al - mu*(4 + al)) / D2; E["bp"] = (1 - mu) / D2
    E["cp"] = (tau*(be*la + 4 + al - mu*(4 + al)) + mu - 1) / D2
    E["aS"] = E["a"] + (E["q"] + 1)*E["av"] + E["n"]*E["ap"]
    E["bS"] = 1 + E["b"] + (E["q"] + 1)*E["bv"] + E["n"]*E["bp"]
    E["cS"] = E["c"] + (E["q"] + 1)*E["cv"] + E["n"]*E["cp"]
    E["eta_bath"] = E["cS"] - 4*tau
    E["mf"] = -E["c"]; E["xb"] = E["cu"] + 1; E["Esub"] = 2*E["cu"] - E["c"]
    E["aE"] = 2*E["au"] - E["a"]; E["bE"] = 2*E["bu"] - E["b"]
    ts = E["cp"]; E["tau_s"] = ts
    E["ms"] = (ts + 2)*(1 - om)/(2 - om); E["xs"] = (ts + 2)/(2 - om)
    E["us"] = (ts + om)/(2 - om); E["vs"] = om*(ts + 2)/(2 - om); E["ps"] = ts
    E["Ts"] = (ts + (1 - mu)*E["vs"]) / be
    E["Eshock"] = (ts*(3 - om) + 2)/(2 - om)
    E["yhat"] = E["mf"] - E["ms"]
    g = 1 + r
    E["eV_piston"] = (ts/g + om*(ts + 2)/(2 - om)) / E["ms"]
    E["eT_piston"] = (1 - mu)*E["eV_piston"]/be
    E["tau"] = tau; E["omega"] = om; E["gamma"] = g
    E["Tsub"] = tau
    return E
