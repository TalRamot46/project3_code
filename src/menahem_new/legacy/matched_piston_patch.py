"""LEGACY: analytic patching of the subsonic heat wave to the shocked target (fix C).

Superseded (md/phase_c/legacy/analytic_patching_impulse.md): the impulse match trades the
original's shock error for a front-pressure error and is not a consistent improvement.  The
current reference is the original AblationSolver (md/phase_c/bath_calibration.md).

``AblationSolver`` patches two self-similar solutions: the heat wave with a rigid
ablation front (u = 0 at m_f) and ``PistonShock``, a piston at m = 0 pushing with the
ablation pressure p_f(t) and cut at m_f (md/phase_c/why_the_solver_differs.md, Sec. 3).
In that construction the slab 0 < m < m_f decelerates with the piston and adds ~m_f|du/dt|
to the pressure the target feels: to first order it is a "massive" ablation front.

``MatchedPistonPatch`` keeps both pieces self-similar -- no field is integrated in time --
and changes only how they are joined: PistonShock gets an effective amplitude p_eff(t)
chosen for a "massless" front (md/phase_c/legacy/analytic_patching_impulse.md), and the heat layer is
shifted to the shock solution's position and velocity at the cut, so x and u are
continuous at m_f (p too for the pressure match, not for the impulse match).  The numerical moving-front target (fix B) that motivated it is in
``menahem_new.legacy.moving_front_shock``.
"""
from __future__ import annotations

import numpy as np


class MatchedPistonPatch:
    """Semi-analytic patch (fix C): PistonShock with an effective amplitude.

    ``match="impulse"`` (massless front only) chooses the amplitude by the target's momentum
    balance instead, which keeps the drive's history (see ``_march``); ``match="pressure"``
    is the local condition below.

    The shocked region stays a self-similar PistonShock (piston at m = 0), but its amplitude
    p_eff(t) is chosen so that the solution *at the cut* m_f carries the pressure the
    ablation front exerts on the target,

        p_PS(m_f; p_eff, t) = p_b(t),   p_b = p_f              ("massless")
                                         p_b = p_f - m_f a_f    ("massive", a_f = du_PS(m_f)/dt),

    instead of putting p_f at m = 0.  The heat layer is shifted to the shock solution's
    position *and velocity* at m_f (and, for "massive", its pressure by -m a_f), so x and u
    are continuous at the front, and with ``match="pressure"`` p as well.  With
    ``match="impulse"`` the target starts at p_PS(m_f; p_eff), which differs from the heat
    layer's p_f (Test 4: 10% below it).  p_eff/p_f depends on m_f/m_s, which changes
    in time, so this is quasi-static: each time uses the self-similar solution of the
    amplitude it needs at that time.
    """

    def __init__(self, ablation_solver, *, front="massless", match="pressure", t_end=None, n_t=60):
        import copy

        assert front in ("massive", "massless")
        assert match in ("pressure", "impulse")
        self.base, self.front, self.match = ablation_solver, front, match
        self.hs, ps = ablation_solver.heat_solver, ablation_solver.shock_solver
        self._ps, self._copy = ps, copy.copy
        p0s = self.hs.Pf * self.hs.A**self.hs.a3 * self.hs.B**self.hs.b3
        self.p_f = lambda t: p0s * t**self.hs.c3
        self.m_f = lambda t: float(self.hs.ablated_mass(time=t))
        self._cache = {}
        if match == "impulse":
            assert front == "massless", "impulse matching is the massless front's momentum balance"
            self._march(t_end, n_t)

    def _shock(self, p0):
        ps = self._copy(self._ps)
        ps.p0 = p0
        return ps

    def _at_cut(self, ps, t, m_f, dt_rel=1e-3):
        """(p, u, x, du/dt) of the shock solution at the Lagrangian mass m_f."""
        m = np.array([m_f, 1.0001 * m_f])
        r = ps.solve(mass=m, time=t)
        u1 = ps.solve(mass=m, time=t * (1 + dt_rel))["velocity"][0]
        u0 = ps.solve(mass=m, time=t * (1 - dt_rel))["velocity"][0]
        return r["pressure"][0], r["velocity"][0], r["position"][0], (u1 - u0) / (2 * dt_rel * t)

    def _layer_momentum(self, ps, t, m_f, n=200):
        """int_0^m_f u dm of the shock solution: the slab the cut discards (u = 0 beyond m_s)."""
        m_top = min(m_f, float(np.ravel(ps.shocked_mass(time=t))[0]) * (1 - 1e-9))
        m = m_top * np.linspace(1e-6, 1.0, n)
        return np.trapz(ps.solve(mass=m, time=t)["velocity"], m)

    def _march(self, t_end, n_t):
        """Impulse matching: p_eff(t) such that the shock solution's momentum beyond the cut,

            P_eff t^(1+c3) / (1+c3) - int_0^m_f u_PS dm,

        equals what the moving front has delivered to the target, int_0^t (p_f - m_f' u_f) dt'
        with u_f the shock solution's velocity at the cut; marched in time on a log grid.

        Early on the heat front runs ahead of any shock it could drive (m_f > m_s): the matter
        the front swallows takes nearly all the impulse.  There the loss m_f' u_f dt is capped
        at the push p_f dt, and p_eff is bounded below by the amplitude that puts the shock
        exactly on the cut (m_s = m_f, nothing delivered).  have - target is then monotone in
        p_eff, so the root is unique.
        """
        import scipy.optimize

        c3, w = self.hs.c3, self._ps.omega
        ts = np.geomspace(1e-3 * t_end, t_end, n_t)
        delivered, t_prev, u_prev = 0.0, 0.0, 0.0
        p0s = self.p_f(1.0)
        self.march = []
        for t in ts:
            m_f = self.m_f(t)
            dm = m_f - (self.m_f(t_prev) if t_prev > 0 else 0.0)
            push = p0s * (t**(1 + c3) - t_prev**(1 + c3)) / (1 + c3)

            def gain(ps):
                u_f = self._at_cut(ps, t, m_f)[1]
                return push - min(0.5 * (u_prev + u_f) * dm, push), u_f

            def mismatch(log_ratio):
                ps = self._shock(np.exp(log_ratio) * self._ps.p0)
                have = ps.p0 * t**(1 + c3) / (1 + c3) - self._layer_momentum(ps, t, m_f)
                return have - (delivered + gain(ps)[0])

            # m_s ~ p0^((1-w)/(2-w)): the amplitude with the shock on the cut
            m_s0 = float(np.ravel(self._ps.shocked_mass(time=t))[0])
            lo = np.log((m_f / m_s0)**((2 - w) / (1 - w)) * (1 + 1e-6))
            lr = lo if mismatch(lo) >= 0 else scipy.optimize.brentq(mismatch, lo, max(lo + np.log(1e4), np.log(10.0)), xtol=1e-7)
            ps = self._shock(np.exp(lr) * self._ps.p0)
            g, u_f = gain(ps)
            delivered += g
            u_prev, t_prev = u_f, t
            self.march.append((t, np.exp(lr)))
        self.march = np.array(self.march)

    def amplitude(self, t):
        """p_eff(t) / t^c3, the PistonShock amplitude matched at the cut."""
        import scipy.optimize

        if self.match == "impulse":
            return float(np.exp(np.interp(np.log(t), np.log(self.march[:, 0]), np.log(self.march[:, 1])))) * self._ps.p0
        if t in self._cache:
            return self._cache[t]
        m_f, pf, tau = self.m_f(t), self.p_f(t), self.hs.c3

        def mismatch(log_ratio):
            ps = self._shock(np.exp(log_ratio) * self._ps.p0)
            p, _, _, a = self._at_cut(ps, t, m_f)
            p_b = pf - (m_f * a if self.front == "massive" else 0.0)
            return p / p_b - 1.0

        lr = scipy.optimize.brentq(mismatch, np.log(0.3), np.log(3.0), xtol=1e-6)
        self._cache[t] = np.exp(lr) * self._ps.p0
        return self._cache[t]

    def front_values(self, t):
        ps = self._shock(self.amplitude(t))
        m_f = self.m_f(t)
        p, u, x, a = self._at_cut(ps, t, m_f)
        return dict(m_heat=m_f, m_shock=float(np.ravel(ps.shocked_mass(time=t))[0]), p_abl=p,
                    x_front=x, u_front=u, a_front=a, p_eff_over_pf=ps.p0 / self._ps.p0)

    def solve(self, *, mass, time):
        fv = self.front_values(time)
        ps = self._shock(self.amplitude(time))
        m_f = fv["m_heat"]
        mass_heat, mass_shock = mass[mass <= m_f], mass[mass > m_f]
        sh = ps.solve(mass=mass_shock, time=time)
        base = self.base
        sh["temperature"] = ((sh["pressure"] * sh["density"]**(base.mu_shock - 1.)) / (ps.r * base.f_shock))**(1. / base.beta_shock)
        sh["radiation_energy_flux"] = np.zeros_like(sh["sie"])
        heat = self.hs.solve(mass=mass_heat, time=time)
        heat["position"] = heat["position"] + fv["x_front"]
        heat["velocity"] = heat["velocity"] + fv["u_front"]
        if self.front == "massive":
            heat["pressure"] = heat["pressure"] - mass_heat * fv["a_front"]
            heat["sie"] = heat["pressure"] / (heat["density"] * self.hs.r)
        out = {k: np.concatenate((np.asarray(heat[k], dtype=float), np.asarray(sh[k], dtype=float)))
               for k in ("density", "pressure", "velocity", "temperature", "sie", "radiation_energy_flux", "position")}
        out.update(heat_position=fv["x_front"], ablated_mass=m_f, boundary_position=heat["boundary_position"] + fv["x_front"],
                   shock_position=float(ps.shock_position(time=time)), **fv)
        return out
