"""LEGACY (numerical): fix B of md/phase_c/why_the_solver_differs.md Sec. 3.3.

Not an analytic model.  ``MovingFrontShock.run`` integrates the Lagrangian Euler equations
of the target in (m, t) (a 1-D hydro simulation), so a curve built on it is a hybrid of the
self-similar heat wave and a numerical target, and cannot serve as an analytic benchmark
for IMC and diffusion.  Kept for the record of md/phase_c/legacy/patching_poc_fix_B.md:
it is what established the massless-front closure that the analytic patch in
``menahem_new.ablation_patching`` (fix C) is built on.

- ``MovingFrontShock``: the target rho0 x^-omega driven at the moving ablation front m_f(t);
  cells below m_f(t) are removed from the integration as the front sweeps them.
- ``PatchedAblationSolver``: the rigid heat solution patched to it, with a "massless" or
  "massive" front.
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
        # beyond the integrated domain the target is undisturbed
        out_ = mass_shock > M[-1]
        x[out_] = self.shock.initial_position(mass_shock[out_])
        w = self.shock.omega
        rho[out_] = self.shock.rho0 * x[out_]**(-w)
        p[out_], e[out_], u[out_] = 0.0, 0.0, 0.0
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
