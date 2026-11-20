"""All numbers, fits and checks for one paper test (cloud generator for the per-test formula docs)."""
import numpy as np, logging
logging.disable(logging.CRITICAL)
from load import load, load_heat_tight
from exact import exponents
import fits, subfit
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV as KPH

NS = 1e-9
YC = 0.05                     # sub fits are made on y in [YC, 1)
T_CHECK = (0.5e-9, 1e-9, 2e-9)

TESTS = {
    1: dict(name="Test 1", short="test1_Au_constT", material="Au", drive=r"$T_s = 1\,\mathrm{HeV}$ ($\tau=0$)"),
    2: dict(name="Test 2", short="test2_Au_constF", material="Au", drive=r"$T_s=(t/\mathrm{ns})^{158/1285}\,\mathrm{HeV}$ (constant surface flux)"),
    3: dict(name="Test 3", short="test3_Au_omega+0.5", material="Au", drive=r"$T_s = 1\,\mathrm{HeV}$ ($\tau=0$), $\rho_0\propto x^{-1/2}$"),
    4: dict(name="Test 4", short="test4_Au_omega-0.5", material="Au", drive=r"$T_s = 1\,\mathrm{HeV}$ ($\tau=0$), $\rho_0\propto x^{+1/2}$"),
    5: dict(name="Test 5", short="test5_Cu_constT", material="Cu", drive=r"$T_s = 1\,\mathrm{HeV}$ ($\tau=0$)"),
    6: dict(name="Test 6", short="test6_Al_constT", material="Al", drive=r"$T_s = 1\,\mathrm{HeV}$ ($\tau=0$)"),
}

def sub_grid(yc=YC):
    return np.unique(np.concatenate([np.linspace(yc, 0.99, 500), 1 - np.geomspace(1e-2, 1e-6, 50)]))

def shock_grid():
    return np.unique(np.concatenate([np.geomspace(1e-6, 1e-2, 120), np.linspace(1e-2, 1 - 1e-10, 600)]))

def compute(k):
    s = load(k); sh = s.shock_solver
    h = load_heat_tight(k); s.heat_solver = h          # tight-tolerance heat solution (see notes)
    E = exponents(h.alpha, h.beta, h.lambdap, h.mu, h.r, h.tau, sh.omega)
    R = dict(k=k, meta=TESTS[k], E=E, h=h, sh=sh, s=s, YC=YC)
    d = 1.0 / h.q
    # ---------------- subsonic fits ----------------
    y = sub_grid(); pr = h.get_self_similar_profiles(xsi_vec=y * h.xsi_f); assert pr["imin"] == 0
    R["sub_fit"] = dict(
        V=subfit.fit(y, pr["V"], e=d, delta=d),
        U=subfit.fit(y, pr["U"], e=d, delta=d),
        P=subfit.fit(y, pr["P"], e=0.0, delta=d),
        T=subfit.fit(y, pr["T"], e=float(E["eT"]), delta=d),
    )
    # errors of the extrapolated fits inside the surface layer y < YC (regularisation dependent)
    yl = np.geomspace(1e-3, YC, 60); pl = h.get_self_similar_profiles(xsi_vec=yl * h.xsi_f)
    R["sub_layer_err"] = {v: float(np.max(np.abs(subfit.ev(R["sub_fit"][v], yl) / pl[v] - 1))) for v in "VUPT"}
    R["sub_prof"] = dict(y=y, **{v: pr[v] for v in "VUPTS"})
    # ---------------- shock fits ----------------
    ys = shock_grid(); V, U, P = sh.get_self_similar_profiles(xsi_vec=ys * sh.xsi_s)
    mu, beta = s.mu_shock, s.beta_shock
    Th = (P * V**(1 - mu))**(1 / beta); Ths = (sh.Ps * sh.Vs**(1 - mu))**(1 / beta)
    R["shock_fit"] = dict(
        V=fits.auto_fit("piston", ys, V, F1=sh.Vs, e=float(E["eV_piston"]), tol=1e-3),
        U=fits.auto_fit("blend", ys, U, F0=sh.U0, F1=sh.Us, tol=1e-3),
        P=fits.auto_fit("blend", ys, P, F0=1.0, F1=sh.Ps, tol=1e-3),
        T=fits.auto_fit("piston", ys, Th, F1=Ths, e=float(E["eT_piston"]), tol=1e-3),
    )
    R["shock_prof"] = dict(y=ys, V=V, U=U, P=P, T=Th)
    # ---------------- dimensional scales at t = 1 ns ----------------
    A, B = h.A, h.B
    sc = {}
    sc["mf"] = h.ablated_mass(time=NS)
    sc["v"] = A**h.a1 * B**h.b1 * NS**h.c1
    sc["u"] = A**h.a2 * B**h.b2 * NS**h.c2 * 1e-5
    sc["p"] = A**h.a3 * B**h.b3 * NS**h.c3 * 1e-12
    sc["T"] = h.Tb * NS**h.tau / KPH
    sc["Cpos"] = h._position_temporal_factor(time=NS)
    sc["Esub"] = h._energy_temporal_factor(time=NS)
    sc["Fs"] = h.S0 * A**h.aS * B**h.bS * NS**h.cS
    sc["bathB"] = h.bath_B * NS**h.bath_power
    om, ts, v0, p0 = sh.omega, sh.tau, sh.v0, sh.p0
    sc["ms"] = sh.shocked_mass(time=NS)
    sc["vs"] = (v0 * v0 * (p0 * NS**(ts + 2))**om)**(1 / (2 - om))
    sc["us"] = (v0 * p0 * NS**(om + ts))**(1 / (2 - om)) * 1e-5
    sc["ps"] = p0 * NS**ts * 1e-12
    sc["Ts"] = (p0 * NS**ts * sc["vs"]**(1 - mu) / (sh.r * s.f_shock))**(1 / beta) / KPH
    sc["Xs"] = sh._position_temporal_factor(time=NS)
    sc["Eshock"] = sh._energy_temporal_factor(time=NS)
    sc["yhat"] = sc["mf"] / sc["ms"]
    sc["q1"], sc["q2"] = sh.q1, sh.q2
    R["sc"] = sc
    # ---------------- energies ----------------
    yy = np.unique(np.concatenate([np.geomspace(1e-10, 1e-2, 2000), np.linspace(1e-2, 1 - 1e-7, 6000)]))
    pe = h.get_self_similar_profiles(xsi_vec=yy * h.xsi_f)
    trap = np.trapz if not hasattr(np, "trapezoid") else np.trapezoid
    Ik = trap(0.5 * pe["U"]**2, yy) * h.xsi_f; Ii = trap(pe["P"] * pe["V"] / h.r, yy) * h.xsi_f
    It = (h.S0 + h.P0 * h.U0) / float(E["Esub"])
    yf = np.linspace(YC, 1 - 1e-7, 6000)
    Uf, Vf, Pf = (subfit.ev(R["sub_fit"][v], yf) for v in "UVP")
    R["energy_sub"] = dict(Ik=Ik, Ii=Ii, It=It, Ik_fit_core=trap(0.5 * Uf**2, yf) * h.xsi_f,
                           Ii_fit_core=trap(Pf * Vf / h.r, yf) * h.xsi_f,
                           Ik_core=trap(np.where(yy >= YC, 0.5 * pe["U"]**2, 0), yy) * h.xsi_f,
                           Ii_core=trap(np.where(yy >= YC, pe["P"] * pe["V"] / h.r, 0), yy) * h.xsi_f)
    # shock: closed-form partial integral I(y) = int_0^{xi} (PV/r + U^2/2) dxi
    def I_closed(yv, Vv, Uv, Pv):
        e = Pv * Vv / sh.r + 0.5 * Uv**2
        return ((1 - om) * (2 + ts) * sh.xsi_s * yv * e - (2 - om) * (Pv * Uv - sh.U0)) / (ts * (3 - om) + 2)
    ys2 = np.unique(np.concatenate([np.geomspace(1e-8, 1e-2, 3000), np.linspace(1e-2, 1 - 1e-10, 6000)]))
    V2, U2, P2 = sh.get_self_similar_profiles(xsi_vec=ys2 * sh.xsi_s)
    e2 = P2 * V2 / sh.r + 0.5 * U2**2
    from scipy.integrate import cumulative_trapezoid
    cum = cumulative_trapezoid(e2, ys2 * sh.xsi_s, initial=0)
    Ic = I_closed(ys2, V2, U2, P2)
    R["energy_shock"] = dict(Ik=sh.energy_kinetic_intgeral, Ii=sh.energy_internal_intgeral, It=sh.energy_total_intgeral,
                             closed_check=float(np.max(np.abs(Ic[ys2 > 1e-3] - cum[ys2 > 1e-3])) / sh.energy_total_intgeral))
    R["I_closed"] = I_closed
    return R

# ------------------------------------------------------------------ dimensional fit formulas
def fit_funcs(R):
    E, sc, h, sh = R["E"], R["sc"], R["h"], R["sh"]
    SF, HF = R["sub_fit"], R["shock_fit"]
    tn = lambda t: t / NS
    def sub(m, t):
        y = m / (sc["mf"] * tn(t)**float(E["mf"]))
        V = subfit.ev(SF["V"], y); U = subfit.ev(SF["U"], y)
        return dict(y=y, v=sc["v"] * tn(t)**float(E["cv"]) * V, u=sc["u"] * tn(t)**float(E["cu"]) * U,
                    p=sc["p"] * tn(t)**float(E["cp"]) * subfit.ev(SF["P"], y),
                    T=sc["T"] * tn(t)**float(E["tau"]) * subfit.ev(SF["T"], y),
                    x=x_af(t) + sc["Cpos"] * tn(t)**float(E["xb"]) * (U - h.c * h.xsi_f * y * V))
    def shock(m, t):
        ys = m / (sc["ms"] * tn(t)**float(E["ms"]))
        V = fits.ev("piston", HF["V"], ys); U = fits.ev("blend", HF["U"], ys)
        return dict(y=ys, v=sc["vs"] * tn(t)**float(E["vs"]) * V, u=sc["us"] * tn(t)**float(E["us"]) * U,
                    p=sc["ps"] * tn(t)**float(E["ps"]) * fits.ev("blend", HF["P"], ys),
                    T=sc["Ts"] * tn(t)**float(E["Ts"]) * fits.ev("piston", HF["T"], ys),
                    x=sc["Xs"] * tn(t)**float(E["xs"]) * (sc["q1"] * sh.xsi_s * ys * V + sc["q2"] * U))
    def yhat(t):
        return sc["yhat"] * tn(t)**float(E["yhat"])
    def x_af(t):
        yh = yhat(t)
        return sc["Xs"] * tn(t)**float(E["xs"]) * (sc["q1"] * sh.xsi_s * yh * fits.ev("piston", HF["V"], yh) + sc["q2"] * fits.ev("blend", HF["U"], yh))
    def x_p(t):
        return sc["Xs"] * tn(t)**float(E["xs"]) * sc["q2"] * sh.U0
    def x_s(t):
        return sc["Xs"] * tn(t)**float(E["xs"]) * (sc["q1"] * sh.xsi_s * sh.Vs + sc["q2"] * sh.Us)
    def x_b(t):
        return x_af(t) + sc["Cpos"] * tn(t)**float(E["xb"]) * h.U0
    def x_yc(t):  # Eulerian position of the element y = YC (edge of the fitted sub region)
        return x_af(t) + sc["Cpos"] * tn(t)**float(E["xb"]) * (subfit.ev(SF["U"], YC) - h.c * h.xsi_f * YC * subfit.ev(SF["V"], YC))
    return dict(sub=sub, shock=shock, yhat=yhat, x_af=x_af, x_p=x_p, x_s=x_s, x_b=x_b, x_yc=x_yc)

def verify(R):
    """dimensional fit formulas vs the (tight-tolerance) patched solver at T_CHECK"""
    F = fit_funcs(R); s, sc, E = R["s"], R["sc"], R["E"]
    out = []
    for t in T_CHECK:
        mf = s.heat_solver.ablated_mass(time=t); ms = s.shock_solver.shocked_mass(time=t)
        m_sub = mf * np.linspace(YC, 0.995, 300)
        m_sh = np.unique(np.concatenate([mf * (1 + np.geomspace(1e-4, 1e-1, 40)), np.linspace(mf * 1.1, ms * 0.995, 300)]))
        m_sh = m_sh[(m_sh > mf * 1.0001) & (m_sh < ms * 0.999)]
        m = np.concatenate([m_sub, m_sh])
        ref = s.solve(mass=m, time=t)
        fs, fh = F["sub"](m_sub, t), F["shock"](m_sh, t)
        n = len(m_sub)
        row = dict(t=t)
        for reg, fv, sl in (("sub", fs, slice(0, n)), ("shock", fh, slice(n, None))):
            for key, rk, conv in (("rho", "density", None), ("u", "velocity", 1e-5), ("p", "pressure", 1e-12), ("T", "temperature", 1 / KPH), ("x", "position", 1.0)):
                rv = ref[rk][sl]
                fvv = 1 / fv["v"] if key == "rho" else fv[key]
                if conv: rv = rv * conv
                if key == "x":
                    err = np.abs(fvv - rv) / (np.max(np.abs(rv)) if reg == "sub" else (np.max(rv) - np.min(rv)))  # x: relative to the region width/extent
                else:
                    err = np.abs(fvv / rv - 1)
                row[(reg, key)] = float(np.max(err))
        # fronts
        row["x_s"] = float(F["x_s"](t) / ref["shock_position"] - 1)
        row["x_af"] = float(F["x_af"](t) / ref["heat_position"] - 1)
        xb_ref = ref["boundary_position"] + ref["heat_position"]   # solver returns x_b relative to the ablation front
        row["x_b"] = float(F["x_b"](t) / xb_ref - 1)
        row["x_b_ref"] = xb_ref; row["x_af_ref"] = ref["heat_position"]; row["x_s_ref"] = ref["shock_position"]
        row["x_p_ref"] = ref["piston_position"]
        out.append(row)
    return out
