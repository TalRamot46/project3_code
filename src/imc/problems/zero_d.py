#!/usr/bin/env python3
"""Milestones M2 and M2a: the 0-D gray problem against Wollaber (2016), Sec. 4.2.1.

An infinite homogeneous medium (one cell, reflecting walls) with constant heat
capacity and sigma ~ T^-p.  In Wollaber's scaled variables (T0 = 1, a = c = 1,
sigma(T0) = 1):

    q = a T0^3 / c_v,   tau = q sigma0 c t,   M = T / T0,   phi = E_r / (a T0^4),

energy conservation gives q phi + M = q + 1, and one IMC step from M_n is
solved exactly by his Eq. (120):

    M_{n+1} = q + 1 - q M_n^4 + (M_n - q - 1 + q M_n^4) exp(-f_n dtau / (q M_n^p)),
    f_n = 1 / (1 + 4 dtau M_n^(3-p))           (alpha = 1; f_n = 1 when explicit).

Linearizing about M = 1 gives the amplification factor, his Eq. (122):

    rho = (1 + 4q) exp(-f_0 dtau / q) - 4q,    f_0 = 1 / (1 + 4 dtau).

IMC is stable (|rho| < 1) for every dtau but oscillates when rho < 0.  The
explicit scheme (f = 1, M2a) has rho -> -4q for large dtau, unstable for q > 1/4.

    uv run python -m imc.problems.zero_d
"""
from __future__ import annotations

import numpy as np

from imc import sources, stepper
from imc.material import PowerLawMaterial
from imc.mesh import Mesh

MESH = Mesh.uniform(1000.0, 1)       # wide, so histories rarely reach the walls
RHO = 1.0


def material(p: float, q: float) -> PowerLawMaterial:
    """u_m = T / q (c_v = 1/q), sigma = T^-p, a = c = 1."""
    return PowerLawMaterial(f=1.0 / q, beta=1.0, mu=1.0, g=1.0, alpha=p, lam=0.0,
                            a=1.0, c=1.0, T_floor=1e-6)


def fleck_factor(M, p, dtau, fleck_mode="implicit"):
    return 1.0 if fleck_mode == "explicit" else 1.0 / (1.0 + 4.0 * dtau * M ** (3.0 - p))


def exact_step(M, p, q, dtau, fleck_mode="implicit"):
    """Wollaber Eq. (120): the exact IMC step in expectation."""
    f = fleck_factor(M, p, dtau, fleck_mode)
    return q + 1.0 - q * M**4 + (M - q - 1.0 + q * M**4) * np.exp(-f * dtau / (q * M**p))


def amplification(p, q, dtau, fleck_mode="implicit"):
    """Wollaber Eq. (122) (independent of p)."""
    f0 = fleck_factor(1.0, p, dtau, fleck_mode)
    return (1.0 + 4.0 * q) * np.exp(-f0 * dtau / q) - 4.0 * q


def run(p, q, dtau, M0, n_steps, seed, fleck_mode="implicit",
        n_emission=40_000, n_initial=40_000, max_census=40_000):
    """IMC trajectory M_0..M_n and the exact one-step predictions from each M_k."""
    mat = material(p, q)
    phi0 = (q + 1.0 - M0) / q
    stepper.seed_all(seed)
    census = sources.uniform_radiation_particles(MESH.edges, phi0, n_initial, 0.0)
    state = stepper.initial_state(np.array([M0]), MESH, RHO, mat, census=census)
    settings = stepper.IMCSettings(
        n_emission=n_emission, n_bath=0, max_census=max_census,
        left_bc="reflecting", right_bc="reflecting", fleck_mode=fleck_mode,
        max_events=10**6,
    )
    dt = dtau / q                                   # tau = q sigma0 c t
    M = [M0]
    predicted = [np.nan]
    energy_error = []
    for _ in range(n_steps):
        predicted.append(exact_step(M[-1], p, q, dtau, fleck_mode))
        state, _ = stepper.imc_step(state, MESH, RHO, mat, dt, None, settings)
        M.append(float(state.T[0]))
        energy_error.append(state.ledger.relative_error(stepper.total_energy(state, MESH, RHO, mat)))
    return np.array(M), np.array(predicted), np.array(energy_error)


def main():
    print("Wollaber Fig. 2 case: p = 5, q = 0.14, dtau = 10, M0 = 0.9")
    print(f"  linear amplification rho = {amplification(5, 0.14, 10.0):+.4f}  (Wollaber: about -0.2868)")
    M, pred, err = run(5, 0.14, 10.0, 0.9, 5, seed=1)
    for n, (m, pr) in enumerate(zip(M, pred)):
        print(f"  n={n}  IMC M={m:.5f}  exact step from IMC M_(n-1): {pr:.5f}")
    print(f"  max energy error {np.max(np.abs(err)):.1e}")
    for mode, q in (("implicit", 0.5), ("explicit", 0.5)):
        rho = amplification(5, q, 10.0, mode)
        M, pred, err = run(5, q, 10.0, 0.99, 6, seed=2, fleck_mode=mode)
        print(f"\n{mode}, q = {q}, dtau = 10, M0 = 0.99: rho = {rho:+.3f}")
        print("  M - 1 per step:", np.array2string(M - 1.0, precision=4))


if __name__ == "__main__":
    main()
