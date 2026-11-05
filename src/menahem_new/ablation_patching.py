"""Better patching of the subsonic heat wave to the shocked target (proof of concept).

``AblationSolver`` patches two self-similar solutions: the heat wave with a rigid
ablation front (u = 0 at m_f) and ``PistonShock``, a piston at m = 0 pushing with the
ablation pressure p_f(t) and cut at m_f (md/phase_c/why_the_solver_differs.md, Sec. 3).
This module replaces the shocked region by a Lagrangian hydro solution of the target
driven at the *moving* ablation front m_f(t) (fix B of Sec. 3.3), with two closures for
the front:

- ``front="massless"``: pressure p_f at m_f; matter crossing m_f leaves the target with
  its momentum, which the rigid heat solution does not pick up.
- ``front="massive"``: the ablated layer is carried along with the front, u = u_f(t) +
  u_heat(m, t).  Its momentum is then  int u_heat dm + m_f u_f,  and momentum balance of
  the layer gives the pressure on the target

      p_b = p_f - m_f du_f/dt,

  i.e. the front is a piston of mass m_f pushed by p_f (fix C's frame acceleration,
  applied to all orders instead of to first order).  In the heat layer the same frame
  term adds  -m du_f/dt  to the rigid solution's pressure, so p is continuous at m_f.

The heat region is the rigid self-similar solution, shifted to the front's position and
velocity (and, for "massive", pressure).  Only the target is integrated numerically.
"""
from __future__ import annotations

import numpy as np


class MovingFrontShock:
    """1-D planar Lagrangian hydro of rho0 x^-omega ideal gas, driven at the mass m_f(t).

    Staggered von Neumann-Richtmyer scheme: node velocities, cell volumes and energies,
    quadratic + linear artificial viscosity, the energy update implicit in p for the
    ideal gas.  Cells with mass coordinate below m_f(t) are removed as the front sweeps
    them; the first remaining node is the front.
    """

    def __init__(self, *, rho0, omega, gamma, m_front, p_front, m_max, n_cells=3000,
                 front="massive", cfl=0.4, q_quad=1.5, q_lin=0.3):
        assert front in ("massive", "massless")
        self.rho0, self.omega, self.gamma = float(rho0), float(omega), float(gamma)
        self.m_front, self.p_front, self.front = m_front, p_front, front
        self.cfl, self.q_quad, self.q_lin = cfl, q_quad, q_lin
        self.M = np.linspace(0.0, m_max, n_cells + 1)               # node mass coordinates
        self.dm = np.diff(self.M)

    def initial_position(self, m):
        w = self.omega
        return ((1.0 - w) * m / self.rho0) ** (1.0 / (1.0 - w))

    def _front_node(self, t):
        return int(np.clip(np.rint(np.interp(self.m_front(t), self.M, np.arange(self.M.size))),
                           0, self.M.size - 3))

    def run(self, times, t_start=None):
        """Integrate to each of ``times``; returns one snapshot dict per time plus the
        front history (t, x_f, u_f, p_f, p_b) of every step."""
        times = np.sort(np.asarray(times, dtype=float))
        # early enough that the impulse missed before it (~ t^(1+c3)) is negligible
        t = t_start if t_start is not None else 1e-6 * times[-1]
        r = self.gamma - 1.0
        x = self.initial_position(self.M)
        u = np.zeros_like(x)
        v = np.diff(x) / self.dm
        e = np.full(self.dm.size, 1e6)                              # ~cold: p << p_f
        p = r * e / v
        k = self._front_node(t)
        massive = self.front == "massive"

        def node_mass(k):
            mn = np.empty_like(x)
            mn[1:-1] = 0.5 * (self.dm[:-1] + self.dm[1:])
            mn[0], mn[-1] = 0.5 * self.dm[0], 0.5 * self.dm[-1]
            mn[k] = 0.5 * self.dm[k] + (self.m_front(t) if massive else 0.0)
            return mn

        hist = {key: [] for key in ("t", "x_f", "u_f", "p_f", "m_f")}
        snaps, dt, i_out = [], None, 0
        while i_out < times.size:
            # removing the cells the front has swept.  "massive": the front's inertia m_f(t)
            # already grows continuously, which adds the swept matter at the front velocity
            # (d(m_f u_f)/dt = F + m_f' u_f); the front takes the swept nodes' mean velocity.
            k_new = self._front_node(t)
            if k_new > k:
                if massive:
                    mn = node_mass(k)[k:k_new + 1]
                    u[k_new] = np.sum(mn * u[k:k_new + 1]) / np.sum(mn)
                k = k_new
            # viscosity from the current velocities
            du = np.diff(u)
            q = np.where(du < 0.0, (self.q_quad * du * du + self.q_lin * np.sqrt(self.gamma * p * v) * -du) / v, 0.0)
            P = p + q
            # momentum: nodes k+1..N-1 interior, k the front, N a rigid wall
            mn = node_mass(k)
            a = np.zeros_like(x)
            a[k + 1:-1] = -(P[k + 1:] - P[k:-1]) / mn[k + 1:-1]
            pf = self.p_front(t)
            a[k] = (pf - P[k]) / mn[k]
            # time step: acoustic CFL over the active cells, and no node moving more than a
            # fraction of its cell from rest in one step (the cold target has ~no sound speed);
            # and the front sweeping at most half a cell, so it tracks m_f(t) and a drive that
            # switches on later (a front pressure of 0 leaves no other limit) is not stepped over
            dx = np.diff(x)[k:]
            c = np.sqrt(self.gamma * p[k:] * v[k:]) + np.abs(du[k:])
            dt_cfl = self.cfl * np.min(dx / np.maximum(c, 1e-30))
            dt_acc = self.cfl * np.sqrt(np.min(dx / np.maximum(np.abs(a[k:-1]), 1e-30)))
            m_dot = max((self.m_front(1.001 * t) - self.m_front(t)) / (0.001 * t), 1e-30)
            dt_front = min(0.5 * self.dm[k] / m_dot, 1e-2 * t)
            dt = min(dt_cfl, dt_acc, dt_front) if dt is None else min(dt_cfl, dt_acc, dt_front, 1.2 * dt)
            dt = min(dt, times[i_out] - t)
            u_new = u + dt * a
            u_new[:k] = u_new[k]
            u_new[-1] = 0.0
            x = x + dt * u_new
            v_new = np.diff(x) / self.dm
            dv = v_new - v
            # ideal gas: e1 = e0 - (0.5 (p0 + p1) + q) dv with p1 = r e1 / v1
            e = np.maximum((e - (0.5 * p + q) * dv) / (1.0 + 0.5 * r * dv / v_new), 1e-30)
            v, u = v_new, u_new
            p = r * e / v
            t += dt
            hist["t"].append(t); hist["x_f"].append(x[k]); hist["u_f"].append(u[k])
            hist["p_f"].append(pf); hist["m_f"].append(self.M[k])
            if t >= times[i_out] * (1.0 - 1e-12):
                snaps.append(dict(t=t, k=k, M=self.M.copy(), x=x.copy(), u=u.copy(), v=v.copy(),
                                  e=e.copy(), p=p.copy()))
                i_out += 1
        return snaps, {key: np.asarray(val) for key, val in hist.items()}


def smooth_derivative(t, y, t_eval, width=0.1):
    """dy/dt at t_eval from a local linear fit in log t over +/- width (relative)."""
    out = np.empty(np.size(t_eval))
    for i, te in enumerate(np.atleast_1d(t_eval)):
        sel = np.abs(np.log(t / te)) < width
        out[i] = np.polyfit(t[sel] - te, y[sel], 1)[0]
    return out


def shock_mass(M, v, rho_init_cells):
    """Shock front from the cold side: where rho/rho_init - 1 falls through half its peak."""
    mc = 0.5 * (M[1:] + M[:-1])
    excess = (1.0 / v) / rho_init_cells - 1.0
    half = 0.5 * excess.max()
    j = int(np.flatnonzero(excess >= half)[-1])
    return float(np.interp(half, [excess[j + 1], excess[j]], [mc[j + 1], mc[j]]))


class PatchedAblationSolver:
    """The heat solution of an ``AblationSolver`` patched to a ``MovingFrontShock``.

    ``solve(mass, time)`` returns the same keys as ``AblationSolver.solve`` (profiles on
    the given mass coordinates) for the times given to the constructor.
    """

    def __init__(self, ablation_solver, *, times, front="massive", m_max=None, n_cells=3000):
        self.base = ablation_solver
        hs, ps = ablation_solver.heat_solver, ablation_solver.shock_solver
        self.hs, self.front = hs, front
        p0s = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3
        self.p_f = lambda t: p0s * t**hs.c3
        self.m_f = lambda t: float(hs.ablated_mass(time=t))
        times = np.sort(np.atleast_1d(np.asarray(times, dtype=float)))
        if m_max is None:
            m_max = 1.4 * float(np.ravel(ps.shocked_mass(time=times[-1]))[0])
        self.shock = MovingFrontShock(rho0=ps.rho0, omega=ps.omega, gamma=ps.gamma, m_front=self.m_f,
                                      p_front=self.p_f, m_max=m_max, n_cells=n_cells, front=front)
        snaps, self.history = self.shock.run(times)
        self.snaps = {s["t"]: s for s in snaps}
        h = self.history
        self.a_f = dict(zip(self.snaps, smooth_derivative(h["t"], h["u_f"], list(self.snaps))))

    def _snap(self, time):
        key = min(self.snaps, key=lambda s: abs(s - time))
        assert abs(key - time) <= 1e-9 * time, f"time {time:g} was not integrated"
        return key, self.snaps[key]

    def front_values(self, time):
        """(m_f, shock mass, pressure on the target p_b, front position, front velocity)."""
        key, s = self._snap(time)
        k = s["k"]
        w = self.shock.omega
        M = s["M"]
        rho_init = np.diff(M) / np.diff(self.shock.initial_position(M))
        m_f = self.m_f(time)
        p_b = self.p_f(time) - (m_f * self.a_f[key] if self.front == "massive" else 0.0)
        return dict(m_heat=m_f, m_shock=shock_mass(M, s["v"], rho_init), p_abl=p_b,
                    x_front=s["x"][k], u_front=s["u"][k], a_front=self.a_f[key])

    def solve(self, *, mass, time):
        key, s = self._snap(time)
        k, M = s["k"], s["M"]
        fv = self.front_values(time)
        m_f = fv["m_heat"]
        mass_heat, mass_shock = mass[mass <= m_f], mass[mass > m_f]
        heat = self.hs.solve(mass=mass_heat, time=time)
        heat["position"] = heat["position"] + fv["x_front"]
        heat["velocity"] = heat["velocity"] + fv["u_front"]
        if self.front == "massive":
            heat["pressure"] = heat["pressure"] - mass_heat * fv["a_front"]
            heat["sie"] = heat["pressure"] / (heat["density"] * self.hs.r)
        # target: cell values at cell-centre mass, node values (x, u) at nodes
        mc = 0.5 * (M[1:] + M[:-1])
        cells = slice(k, None)
        rho = np.interp(mass_shock, mc[cells], 1.0 / s["v"][cells])
        p = np.interp(mass_shock, mc[cells], s["p"][cells])
        e = np.interp(mass_shock, mc[cells], s["e"][cells])
        x = np.interp(mass_shock, M[k:], s["x"][k:])
        u = np.interp(mass_shock, M[k:], s["u"][k:])
        base = self.base
        T_shock = ((p * rho**(base.mu_shock - 1.0)) / (base.shock_solver.r * base.f_shock))**(1.0 / base.beta_shock)
        out = {}
        for key_, h, sh in (("density", heat["density"], rho), ("pressure", heat["pressure"], p),
                            ("velocity", heat["velocity"], u), ("sie", heat["sie"], e),
                            ("position", heat["position"], x), ("temperature", heat["temperature"], T_shock),
                            ("radiation_energy_flux", heat["radiation_energy_flux"], np.zeros_like(p))):
            out[key_] = np.concatenate((np.asarray(h, dtype=float), sh))
        out.update(heat_position=fv["x_front"], ablated_mass=m_f, boundary_position=heat["boundary_position"] + fv["x_front"],
                   shock_position=float(np.interp(fv["m_shock"], M, s["x"])), **fv)
        return out


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
    position *and velocity* at m_f (and, for "massive", its pressure by -m a_f), so p, u
    and x are all continuous at the front.  p_eff/p_f depends on m_f/m_s, which changes
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
