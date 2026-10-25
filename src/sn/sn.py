#!/usr/bin/env python3
"""Deterministic gray 1D S_N check for the Back SiO2 problem.

This is the discrete-ordinates twin of src/imc/archive/sio2_original/sio2_1mm_gray_imc_check_og.py:
the same problem, material model, drive and Fleck-Cummings linearization.
Only the solution of each step's linear transport equation differs: here it
is deterministic (ordinates, diamond difference, source iteration) instead of
a random walk, so S_N minus IMC is IMC noise plus S_N discretization error.
Derivation and proofs: md/theory/sn_derivation.md.

Each step freezes sigma, the Fleck factor f and B = a T^4 at T^n and solves

    (1/c) dpsi/dt + mu dpsi/dx + sigma psi = (1-f) sigma phi + f sigma c B,
    phi = 0.5 * sum_m w_m psi_m = c E_r     (an isotropic psi equals phi),

then updates the material with what the radiation left behind:

    U^{n+1} = U^n + dt * f sigma (phi^{n+1} - c B^n).

Geometry and material model (as in the IMC script):
  * 1D slab, L = 1 mm, 100 cells, unit transverse area
  * rho = 0.05 g/cm^3 SiO2, gray pure absorption, Hammer-Rosen opacity and EOS
  * initial T_m = 300 K and no radiation
  * left blackbody bath from back_pulse.txt (synthetic), right vacuum

Discretization:
  * time: backward Euler, i.e. extra absorption 1/(c dt) and source psi^n/(c dt)
  * angle: Gauss-Legendre on [-1,1], or double-Gauss (exact half-range moments)
  * space: diamond difference, outgoing fluxes < 0 set to 0 (balance kept)
  * bath: psi_b on mu > 0, scaled so the discrete incoming current equals the
    step-averaged sigma_SB T_D^4 (or the naive psi_b = 4 F_inc, for comparison)

The result is deterministic; refine --n-mu, --dt and N_CELLS to estimate its
discretization error.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# Units: cm, ns, MJ, K.  Same constants and material as the IMC script.
C = 29.9792458                        # speed of light [cm/ns]
EV_TO_K = 11604.518121550082          # 1 eV expressed in kelvin
K_TO_EV = 1.0 / EV_TO_K
A_RAD = 7.56573325003e-28             # radiation constant a [MJ/(cm^3 K^4)]
SIGMA_SB = A_RAD * C / 4.0            # Stefan-Boltzmann sigma = ac/4 [MJ/(cm^2 ns K^4)]
E100 = A_RAD * (100.0 * EV_TO_K) ** 4  # a*T^4 at T = 100 eV: the radiation energy scale

L_CM = 0.1                            # slab thickness = 1 mm
N_CELLS = 100
DX = L_CM / N_CELLS                   # uniform cell width [cm]
RHO = 0.05                            # SiO2 foam density [g/cm^3]
ALPHA = 3.53                          # opacity temperature exponent, sigma ~ theta^-alpha
BETA = 1.10                           # EOS temperature exponent, u_m ~ theta^beta
EOS_MU = 0.09                         # EOS density exponent, u_m ~ rho^(1-mu)
LAMBDA = 0.75                         # opacity density exponent, sigma ~ rho^(1+lambda)
F_EOS = 8.78                          # EOS prefactor (Hammer-Rosen units)
G_OP = 1.0 / 9175.0                   # opacity prefactor (Hammer-Rosen units)
T0_K = 300.0                          # initial (cold) material temperature
THETA0 = T0_K * K_TO_EV / 100.0       # everything below uses theta = T / (100 eV)
THETA_FLOOR = 1.0e-12                 # keeps theta^-alpha and theta^(1/beta) finite

# Material energy density: U = E100 * H * theta^beta  [MJ/cm^3]
H = F_EOS * RHO ** (1.0 - EOS_MU) / E100
# Gray absorption opacity: sigma = SIGMA0 * theta^-alpha  [1/cm]
SIGMA0 = RHO ** (1.0 + LAMBDA) / G_OP

PHI_FLOOR = 1.0e-3                    # relative floor in the convergence test


class Drive:
    """Tabulated left-boundary blackbody drive temperature T_D(t)."""

    def __init__(self, path: Path):
        data = np.loadtxt(path)               # two columns: t [ns], T_D [eV]
        order = np.argsort(data[:, 0])        # tolerate unsorted input
        self.t = np.asarray(data[order, 0], dtype=float)
        self.T_eV = np.asarray(data[order, 1], dtype=float)
        if self.t[0] > 0.0:
            # Pad back to t = 0 with the cold initial temperature.
            self.t = np.insert(self.t, 0, 0.0)
            self.T_eV = np.insert(self.T_eV, 0, T0_K * K_TO_EV)

    def temperature_eV(self, t):
        return np.interp(t, self.t, self.T_eV)   # linear in t, flat outside the table

    def flux_average(self, t0, t1):
        # Eight-point Gauss-Legendre quadrature of sigma_SB*T_D(t)^4 over the step.
        nodes, weights = np.polynomial.legendre.leggauss(8)
        ts = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * nodes
        vals = SIGMA_SB * (self.temperature_eV(ts) * EV_TO_K) ** 4
        return 0.5 * np.sum(weights * vals)      # the *average* incident flux


def make_quadrature(n_mu, kind):
    """Ordinates mu (increasing, none at 0) and weights w with sum(w) = 2."""
    if n_mu < 2 or n_mu % 2 != 0:
        raise ValueError("n_mu must be an even integer >= 2.")
    if kind == "gauss":
        mu, w = np.polynomial.legendre.leggauss(n_mu)
    elif kind == "double-gauss":
        # Gauss-Legendre on [0,1], mirrored to [-1,0]: half-range moments exact.
        x, wx = np.polynomial.legendre.leggauss(n_mu // 2)
        half_mu, half_w = 0.5 * (x + 1.0), 0.5 * wx
        mu = np.concatenate((-half_mu[::-1], half_mu))
        w = np.concatenate((half_w[::-1], half_w))
    else:
        raise ValueError("quadrature must be 'gauss' or 'double-gauss'.")
    return mu.astype(np.float64), w.astype(np.float64)


@njit(cache=True)
def material_coefficients(theta, dt):
    """Per-cell opacity, Fleck factor, Planck energy density a*T^4 and material energy."""
    n = theta.size
    sigma = np.empty(n)
    fleck = np.empty(n)
    B = np.empty(n)
    U = np.empty(n)
    for i in range(n):
        th = max(theta[i], THETA_FLOOR)
        sigma[i] = SIGMA0 * th ** (-ALPHA)      # absorption opacity [1/cm]
        B[i] = E100 * th ** 4                   # a*T^4, equilibrium radiation energy density
        U[i] = E100 * H * th ** BETA            # material internal energy density
        # gamma = (dB/dT)/(dU/dT), the "beta" of Fleck-Cummings.
        gamma = 4.0 * th ** (4.0 - BETA) / (H * BETA)
        fleck[i] = 1.0 / (1.0 + C * sigma[i] * dt * gamma)   # Fleck factor f in (0,1]
    return sigma, fleck, B, U


@njit(cache=True)
def invert_material_energy(U):
    """Invert U = E100*H*theta^beta to recover theta from the updated energy."""
    theta = np.empty(U.size)
    for i in range(U.size):
        theta[i] = max(THETA_FLOOR, (max(U[i], 0.0) / (E100 * H)) ** (1.0 / BETA))
    return theta


@njit(cache=True)
def scalar_flux(psi, w):
    """phi_i = 0.5 * sum_m w_m psi[m, i]  (= c E_r)."""
    phi = np.zeros(psi.shape[1])
    for m in range(psi.shape[0]):
        for i in range(psi.shape[1]):
            phi[i] += 0.5 * w[m] * psi[m, i]
    return phi


@njit(cache=True)
def sweep(psi, rhs, mu, w, sig_eff, psi_bath):
    """Diamond-difference sweep of every ordinate: bath at x = 0, vacuum at x = L.

    Cell balance mu (psi_out - psi_in)/DX + sig_eff psi_cell = rhs with
    psi_cell = (psi_in + psi_out)/2.  If psi_out < 0, set it to 0 and keep
    the balance.  Returns the outgoing partial currents (left, right).
    """
    out_left = 0.0
    out_right = 0.0
    for m in range(mu.size):
        backward = mu[m] < 0.0                  # mu < 0 sweeps right to left
        mu_abs = abs(mu[m])
        two_mu_dx = 2.0 * mu_abs / DX
        psi_face = 0.0 if backward else psi_bath
        for j in range(N_CELLS):
            k = N_CELLS - 1 - j if backward else j
            psi_cell = (rhs[m, k] + two_mu_dx * psi_face) / (sig_eff[k] + two_mu_dx)
            psi_out = 2.0 * psi_cell - psi_face
            if psi_out < 0.0:
                # Set-to-zero fixup: thick cells would otherwise overshoot below 0.
                psi_cell = (rhs[m, k] + mu_abs * psi_face / DX) / sig_eff[k]
                psi_out = 0.0
            psi[m, k] = psi_cell
            psi_face = psi_out                  # outflow of k is inflow of the next cell
        if backward:
            out_left += 0.5 * w[m] * mu_abs * psi_face    # leaves through the drive face
        else:
            out_right += 0.5 * w[m] * mu_abs * psi_face   # leaves through the vacuum face
    return out_left, out_right


@njit(cache=True)
def transport_step(psi, sigma, fleck, B, dt, psi_bath, mu, w, tol, max_iters):
    """Backward-Euler S_N step for the Fleck-Cummings equation, by source iteration.

    Each iteration sweeps with the scattering source (1-f) sigma phi of the
    previous iterate; it converges at rate <= (1-f) sigma / (sigma + 1/(c dt)).
    Returns psi^{n+1} (in place), the energy deposited per cell [MJ/cm^3],
    the outgoing currents and the number of sweeps.
    """
    inv_c_dt = 1.0 / (C * dt)
    sig_eff = sigma + inv_c_dt                  # time absorption, backward Euler
    scatter = (1.0 - fleck) * sigma             # effective scattering
    emission = fleck * sigma * C * B            # effective emission f sigma c B
    psi_old = psi.copy()                        # psi^n, the angular census
    rhs = np.empty_like(psi)
    phi = scalar_flux(psi, w)
    phi_prev = phi
    out_left = 0.0
    out_right = 0.0
    sweeps = 0
    for it in range(max_iters):
        phi_prev = phi
        for m in range(mu.size):
            for i in range(N_CELLS):
                rhs[m, i] = scatter[i] * phi_prev[i] + emission[i] + psi_old[m, i] * inv_c_dt
        out_left, out_right = sweep(psi, rhs, mu, w, sig_eff, psi_bath)
        phi = scalar_flux(psi, w)
        sweeps = it + 1
        floor = PHI_FLOOR * phi.max() + 1.0e-300
        if np.max(np.abs(phi - phi_prev) / (phi + floor)) < tol:
            break
    # Absorbed minus re-emitted in the last sweep.  At convergence this equals
    # dt f sigma (phi - c B); taking it from the sweep itself makes the energy
    # balance exact for any tolerance.
    deposited = dt * (sigma * phi - scatter * phi_prev - emission)
    return deposited, out_left, out_right, sweeps


def run_one(drive, dt, end_time, n_mu, quadrature, bath_norm, tol, max_iters, progress=False):
    """Run the S_N calculation to end_time and return the diagnostics."""
    mu, w = make_quadrature(n_mu, quadrature)
    # Discrete incoming current of a unit isotropic bath: 1/4 if exact.
    unit_current = 0.5 * np.sum(w[mu > 0.0] * mu[mu > 0.0])

    theta = np.full(N_CELLS, THETA0)            # cold slab at t = 0
    psi = np.zeros((n_mu, N_CELLS))             # no radiation initially

    steps = int(np.ceil(end_time / dt))
    time_mid = np.empty(steps)                  # step-centred time [ns]
    outgoing = np.empty(steps)                  # right-face flux [MJ/(cm^2 ns)]
    material_T_max = np.empty(steps)            # max material temperature [eV]
    sweep_count = np.empty(steps, dtype=np.int64)
    energy_error = np.empty(steps)              # relative global balance error

    _, _, _, U0 = material_coefficients(theta, dt)
    initial_energy = np.sum(U0) * DX            # baseline for the energy balance
    cumulative_boundary = 0.0                   # energy injected by the drive
    cumulative_left = 0.0                       # energy leaked back out the left
    cumulative_right = 0.0                      # energy leaked out the right

    t0 = 0.0
    for step in range(steps):
        t1 = min(end_time, t0 + dt)             # last step may be shorter
        local_dt = t1 - t0
        sigma, fleck, B, U = material_coefficients(theta, local_dt)

        flux_avg = drive.flux_average(t0, t1)   # time-averaged incident flux
        # "current": the discrete inflow is exactly flux_avg; "naive": psi_b = 4 F_inc.
        psi_bath = flux_avg / unit_current if bath_norm == "current" else 4.0 * flux_avg
        cumulative_boundary += unit_current * psi_bath * local_dt

        deposited, escaped_left, escaped_right, sweeps = transport_step(
            psi, sigma, fleck, B, local_dt, psi_bath, mu, w, tol, max_iters,
        )

        # Material update: absorbed minus emitted, as energy density.
        U_new = U + deposited
        if np.any(U_new <= 0.0) or np.any(~np.isfinite(U_new)):
            raise RuntimeError(f"Non-positive material energy at step {step}, t={t1:g} ns.")
        theta = invert_material_energy(U_new)

        cumulative_left += escaped_left * local_dt
        cumulative_right += escaped_right * local_dt
        # Global check: matter + radiation should equal initial + in - out.
        current_total = (np.sum(U_new) + np.sum(scalar_flux(psi, w)) / C) * DX
        expected_total = initial_energy + cumulative_boundary - cumulative_left - cumulative_right

        time_mid[step] = 0.5 * (t0 + t1)
        outgoing[step] = escaped_right          # end-of-step flux, as backward Euler leaks it
        material_T_max[step] = 100.0 * np.max(theta)   # theta -> eV
        sweep_count[step] = sweeps
        energy_error[step] = (current_total - expected_total) / max(abs(expected_total), 1.0e-30)

        if progress and (step % max(1, steps // 20) == 0 or step == steps - 1):
            # About twenty progress lines per run, plus the final step.
            print(
                f"step={step+1:5d}/{steps}, t={t1:7.4f} ns, "
                f"Tmax={material_T_max[step]:7.2f} eV, sweeps={sweeps:4d}, "
                f"energy err={energy_error[step]:+.3e}",
                flush=True,
            )
        t0 = t1

    return {
        "time_ns": time_mid,
        "Fout": outgoing,
        "Tm_final_eV": 100.0 * theta,           # final material temperature profile [eV]
        "Tmax_eV": material_T_max,
        "sweeps": sweep_count,
        "energy_error": energy_error,
        "unit_current": unit_current,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path,
                        default=Path(__file__).resolve().parent.parent / "imc" / "data" / "back_pulse.txt")
    parser.add_argument("--dt", type=float, default=0.001, help="time step [ns]")
    parser.add_argument("--end", type=float, default=3.0, help="end time [ns]")
    parser.add_argument("--n-mu", type=int, default=16, help="number of ordinates (even)")
    parser.add_argument("--quadrature", choices=["gauss", "double-gauss"], default="double-gauss")
    parser.add_argument("--bath-norm", choices=["current", "naive"], default="current")
    parser.add_argument("--tol", type=float, default=1.0e-10, help="source-iteration tolerance")
    parser.add_argument("--max-iters", type=int, default=1000)          # sweeps per step
    parser.add_argument("--imc", type=Path, default=Path("sio2_1mm_gray_imc_check.npz"),
                        help="IMC result to overlay, if it exists")
    parser.add_argument("--prefix", type=str, default="sio2_1mm_gray_sn_check")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    wall0 = time.time()
    run = run_one(
        Drive(args.drive), args.dt, args.end, args.n_mu, args.quadrature,
        args.bath_norm, args.tol, args.max_iters, progress=not args.no_progress,
    )
    wall = time.time() - wall0

    t = run["time_ns"]
    norm = SIGMA_SB * (165.0 * EV_TO_K) ** 4    # plots are normalized to 165 eV
    flux = run["Fout"] / norm

    output_npz = Path(args.prefix + ".npz")
    np.savez_compressed(
        output_npz,
        time_ns=t,
        Fout=run["Fout"],
        normalization_flux=norm,
        dt_ns=args.dt,
        n_mu=args.n_mu,
        quadrature=args.quadrature,
        bath_norm=args.bath_norm,
        final_material_temperature_eV=run["Tm_final_eV"],
        sweeps=run["sweeps"],
        energy_error=run["energy_error"],
        wall_seconds=wall,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    if args.imc.exists():
        # Overlay the IMC mean and its 2 s.e. band from the IMC script's output.
        d = np.load(args.imc)
        ax.plot(d["smooth_time_ns"], d["smooth_normalized_mean"], color="black",
                linewidth=2.0, label="IMC mean")
        ax.fill_between(
            d["smooth_time_ns"],
            d["smooth_normalized_mean"] - 2.0 * d["smooth_normalized_stderr"],
            d["smooth_normalized_mean"] + 2.0 * d["smooth_normalized_stderr"],
            color="black", alpha=0.15, label=r"IMC $\pm2$ s.e.",
        )
    ax.plot(t, flux, color="tab:red", linewidth=1.6,
            label=f"$S_{{{args.n_mu}}}$ ({args.quadrature}, bath {args.bath_norm})")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"$F_{\rm out}/[\sigma_{\rm SB}(165\,{\rm eV})^4]$")
    ax.set_title(f"Gray $S_N$: L=1 mm, {N_CELLS} cells, dt={args.dt:g} ns")
    ax.set_xlim(0.8, args.end)    # nothing interesting before breakout
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = Path(args.prefix + "_Fout.png")
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    summary_path = Path(args.prefix + "_summary.txt")
    crossing = np.flatnonzero(flux >= 0.01)     # 1% of the 165 eV flux
    breakout = t[crossing[0]] if crossing.size else np.nan
    peak_index = int(np.argmax(flux))
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Deterministic gray S_N check for L=1 mm, {N_CELLS} cells\n")
        f.write(f"dt_ns {args.dt}\n")
        f.write(f"n_mu {args.n_mu}\n")
        f.write(f"quadrature {args.quadrature}\n")
        f.write(f"bath_norm {args.bath_norm}\n")
        f.write(f"discrete_unit_bath_current {run['unit_current']:.12g}\n")
        f.write(f"breakout_normalized_0p01_ns {breakout:.12g}\n")
        f.write(f"peak_time_ns {t[peak_index]:.12g}\n")
        f.write(f"peak_normalized {flux[peak_index]:.12g}\n")
        f.write(f"maximum_sweeps_per_step {int(run['sweeps'].max())}\n")
        f.write(f"maximum_absolute_energy_error {np.max(np.abs(run['energy_error'])):.12g}\n")
        f.write(f"wall_seconds {wall:.6f}\n")

    print(f"Completed S_N run in {wall:.1f} s")
    print(f"Breakout (normalized flux >=0.01): {breakout:.6f} ns")
    print(f"Peak: {flux[peak_index]:.6f} at {t[peak_index]:.6f} ns")
    print(f"Saved: {output_npz}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
