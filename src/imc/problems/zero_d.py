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

from imc.core import sources, stepper
from imc.core.material import PowerLawMaterial
from imc.core.mesh import Mesh

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


def exact_trajectory(p, q, dtau, M0, n_steps, fleck_mode="implicit"):
    """Iterate Eq. (120): the noise-free IMC trajectory."""
    M = [M0]
    for _ in range(n_steps):
        M.append(exact_step(M[-1], p, q, dtau, fleck_mode))
    return np.array(M)


def figure_1(path):
    """Wollaber Fig. 1: contours of rho(dtau, q) from Eq. (122)."""
    import matplotlib.pyplot as plt

    log_tau = np.linspace(-2.0, 6.0, 500)
    log_q = np.linspace(-2.0, 3.0, 400)
    TT, QQ = np.meshgrid(10**log_tau, 10**log_q)
    rho = amplification(5, QQ, TT)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    levels = [-0.29, -0.2, -0.1, 0.0, 0.2, 0.4, 0.6]
    cs = ax.contour(log_tau, log_q, rho, levels=levels, cmap="coolwarm")
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    ax.set_xlabel(r"$\log_{10}\Delta_\tau$")
    ax.set_ylabel(r"$\log_{10} q$")
    ax.set_title(r"Amplification factor $\rho$ (Wollaber Fig. 1, Eq. 122)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_2(path, replicas=16):
    """Wollaber Fig. 2: p = 5, q = 0.14, dtau = 10, M0 = 0.9, five steps.

    IMC mean +- 2 s.e. over replicas, the noise-free trajectory of Eq. (120),
    and the linear prediction 1 + (M0 - 1) rho^n of Eq. (122).
    """
    import matplotlib.pyplot as plt

    p, q, dtau, M0, n = 5, 0.14, 10.0, 0.9, 5
    runs = np.array([run(p, q, dtau, M0, n, seed=100 + r)[0] for r in range(replicas)])
    mean = runs.mean(axis=0)
    se = runs.std(axis=0, ddof=1) / np.sqrt(replicas)
    tau = dtau * np.arange(n + 1)
    rho = amplification(p, q, dtau)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(tau, mean, yerr=2 * se, fmt="o-", color="tab:blue", capsize=3,
                label=f"IMC (this code, {replicas} seeds, mean $\\pm$2 s.e.)")
    ax.plot(tau, exact_trajectory(p, q, dtau, M0, n), "s--", color="black", mfc="none",
            label="exact IMC map, Eq. (120)")
    ax.plot(tau, 1.0 + (M0 - 1.0) * rho ** np.arange(n + 1), "x:", color="tab:red",
            label=fr"linear prediction, $\rho$ = {rho:.4f}")
    ax.set_xlabel(r"dimensionless time $\tau$")
    ax.set_ylabel(r"dimensionless temperature $M = T/T_0$")
    ax.set_title(r"Wollaber Fig. 2: $\sigma\propto T^{-5}$, $q=0.14$, $\Delta_\tau=10$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return mean, se


def figure_explicit(path, replicas=12, explicit_replicas=64):
    """M2a: explicit vs implicit at q = 0.5, dtau = 10 (rho = -2 vs -0.158).

    Explicit steps are about 11x noisier (each emits 20x the material energy),
    so they get more replicas.  The implicit curve flattens at its noise floor.
    """
    import matplotlib.pyplot as plt

    p, q, dtau, M0 = 5, 0.5, 10.0, 0.99
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for mode, n, color in (("implicit", 8, "tab:blue"), ("explicit", 4, "tab:red")):
        reps = explicit_replicas if mode == "explicit" else replicas
        runs = np.array([run(p, q, dtau, M0, n, seed=200 + r, fleck_mode=mode)[0]
                         for r in range(reps)])
        rho = amplification(p, q, dtau, mode)
        ax.plot(np.arange(n + 1), np.abs(runs.mean(axis=0) - 1.0), "o-", color=color,
                label=fr"{mode} IMC, $\rho$ = {rho:+.3f}")
        ax.plot(np.arange(n + 1), np.abs(M0 - 1.0) * np.abs(rho) ** np.arange(n + 1), ":",
                color=color, label=r"$|M_0-1|\,|\rho|^n$")
    ax.set_yscale("log")
    ax.set_xlabel("step n")
    ax.set_ylabel(r"$|M_n - 1|$")
    ax.set_title(r"M2a: explicit ($f=1$) vs implicit, $q=0.5$, $\Delta_\tau=10$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="0-D IMC vs Wollaber (2016)")
    parser.add_argument("--figures", type=Path, default=None,
                        help="directory to write the reproduced figures into")
    args = parser.parse_args()
    main()
    if args.figures is not None:
        args.figures.mkdir(parents=True, exist_ok=True)
        figure_1(args.figures / "wollaber_fig1_amplification.png")
        figure_2(args.figures / "wollaber_fig2_damped_oscillation.png")
        figure_explicit(args.figures / "m2a_explicit_vs_implicit.png")
        print(f"Figures written to {args.figures}")
