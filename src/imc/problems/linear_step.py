#!/usr/bin/env python3
"""Milestone M1: one linear IMC step against the S_N twin (Theorem 2 in numbers).

With sigma, f and B = a T^4 frozen, a time step solves the linear equation (T)
of md/imc_mc_derivation.md:

    (1/c) dpsi/dt + mu dpsi/dx + sigma psi = (1-f) sigma phi + f sigma c B,

census in, census out, with a bath on the left and vacuum on the right.
The IMC mean must equal the exact solution; S_N (src/sn/sn.py, backward
Euler + diamond difference) approximates it, sub-stepped K times so that its
time error is small.  The S_N error is estimated by also running with
2K sub-steps and twice the ordinates.

The S_N module fixes the grid (100 cells, L = 0.1 cm) and its units
(cm, ns, MJ); this driver uses the same.

    uv run python -m imc.problems.linear_step
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imc.core import sources, transport
from imc.core.particles import Particles
from sn import sn

C = sn.C
N = sn.N_CELLS
DX = sn.DX
EDGES = np.linspace(0.0, N * DX, N + 1)
X = 0.5 * (EDGES[:-1] + EDGES[1:])


@dataclass(frozen=True)
class LinearProblem:
    sigma: np.ndarray            # absorption opacity per cell [1/cm]
    fleck: np.ndarray            # frozen Fleck factor per cell
    B: np.ndarray                # a T^4 per cell [MJ/cm^3]
    E0: float                    # initial isotropic radiation energy density [MJ/cm^3]
    F_inc: float                 # incident bath flux [MJ/(cm^2 ns)]
    dt: float                    # step [ns]


def make_problem(sigma_dx: float, dt: float = 1e-3) -> LinearProblem:
    """Smoothly varying f and B; uniform sigma with optical thickness sigma_dx per cell."""
    s = X / X[-1]
    return LinearProblem(
        sigma=np.full(N, sigma_dx / DX),
        fleck=0.3 + 0.6 * s,                      # 0.3 .. 0.9
        B=1e-3 * (1.0 + 0.8 * np.cos(np.pi * s)),
        E0=5e-4, F_inc=0.25 * C * 2e-3, dt=dt,
    )


def sn_solution(prob: LinearProblem, n_sub: int, n_mu: int) -> dict:
    """K backward-Euler S_N sub-steps of the frozen-coefficient equation."""
    mu, w = sn.make_quadrature(n_mu, "double-gauss")
    unit_current = 0.5 * np.sum(w[mu > 0] * mu[mu > 0])
    psi = np.full((n_mu, N), C * prob.E0)
    psi_bath = prob.F_inc / unit_current
    h = prob.dt / n_sub
    net_dep = np.zeros(N)
    leak_l = leak_r = 0.0
    for _ in range(n_sub):
        dep, out_l, out_r, _ = sn.transport_step(
            psi, prob.sigma, prob.fleck, prob.B, h, psi_bath, mu, w, 1e-12, 5000)
        net_dep += dep
        leak_l += out_l * h
        leak_r += out_r * h
    return {"E": sn.scalar_flux(psi, w) / C, "net_dep": net_dep, "leak_l": leak_l, "leak_r": leak_r}


def imc_solution(prob: LinearProblem, seed: int, n_census=200, n_emission=200_000,
                 n_bath=100_000) -> dict:
    """One IMC step (transport kernel only: the coefficients are frozen by hand)."""
    np.random.seed(seed)
    transport.seed(seed)
    census = sources.uniform_radiation_particles(EDGES, prob.E0, n_census, 0.0)
    emitted = prob.fleck * prob.sigma * C * prob.B * DX * prob.dt
    counts = sources.allocate_emission(emitted, n_emission)
    emission = sources.emission_particles(EDGES, emitted, counts, 0.0, prob.dt)
    E_b = prob.F_inc * prob.dt
    times = prob.dt * np.random.random(n_bath)
    bath = sources.bath_particles(EDGES, E_b, n_bath, times)
    p = Particles.concatenate([census, emission, bath])
    out = transport.transport_step(
        p.x, p.mu, p.w, p.t, p.cell, EDGES, prob.sigma, np.zeros(N), prob.fleck,
        C, prob.dt, transport.VACUUM, transport.VACUUM, 10**7)
    cw, ccell, deposited = out[2], out[3], out[4]
    E = np.bincount(ccell, weights=cw, minlength=N) / DX
    return {"E": E, "net_dep": (deposited - emitted) / DX, "leak_l": out[8], "leak_r": out[9]}


def compare(sigma_dx: float, replicas: int = 8, n_sub: int = 64, n_mu: int = 16) -> dict:
    prob = make_problem(sigma_dx)
    ref = sn_solution(prob, 2 * n_sub, 2 * n_mu)
    coarse = sn_solution(prob, n_sub, n_mu)
    reps = [imc_solution(prob, 100 + r) for r in range(replicas)]
    result = {"sigma_dx": sigma_dx}
    for key in ("E", "net_dep", "leak_l", "leak_r"):
        vals = np.array([r[key] for r in reps])
        mean = vals.mean(axis=0)
        se = vals.std(axis=0, ddof=1) / np.sqrt(replicas)
        sn_err = np.abs(ref[key] - coarse[key])
        result[key] = {"imc": mean, "se": se, "sn": ref[key], "sn_err": sn_err,
                       "z": (mean - ref[key]) / np.maximum(se + sn_err, 1e-300)}
    return result


def main():
    for sigma_dx in (0.1, 1.0, 10.0):
        r = compare(sigma_dx)
        print(f"\nsigma dx = {sigma_dx}")
        for key in ("E", "net_dep", "leak_l", "leak_r"):
            q = r[key]
            z = np.atleast_1d(q["z"])
            scale = np.max(np.abs(np.atleast_1d(q["sn"])))
            rel_se = np.median(np.atleast_1d(q["se"])) / scale
            rel_sn = np.median(np.atleast_1d(q["sn_err"])) / scale
            rel_diff = np.max(np.abs(np.atleast_1d(q["imc"] - q["sn"]))) / scale
            print(f"  {key:8s} max|IMC-SN|/scale={rel_diff:.2e}  median se/scale={rel_se:.2e}  "
                  f"median SN err/scale={rel_sn:.2e}  frac|z|<=3={np.mean(np.abs(z) <= 3):.2f}  "
                  f"max|z|={np.max(np.abs(z)):.2f}")


if __name__ == "__main__":
    main()
