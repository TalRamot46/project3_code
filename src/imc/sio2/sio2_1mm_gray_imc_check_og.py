#!/usr/bin/env python3
"""Independent gray 1D IMC check for the Back SiO2 problem.

The code uses a standard Fleck-Cummings linearization with analog
effective-absorption and effective-scattering events (each collision absorbs
with probability f, otherwise scatters isotropically).  It is intentionally independent
of the moment solvers used for diffusion, P1 and DisAP1.

Geometry and material model:
  * 1D slab, L = 1 mm, 100 cells, unit transverse area
  * rho = 0.05 g/cm^3 SiO2
  * gray pure absorption, sigma_P = sigma_R
  * Hammer-Rosen power-law opacity and EOS used in the comparison paper
  * initial T_m = 300 K
  * synthetic Back-type drive from back_pulse.txt (not measured data)
  * left blackbody incident bath, right vacuum

The result is stochastic.  Use --seeds with several comma-separated seeds to
average independent histories.  The default is a moderate convergence check,
not a production-quality reference calculation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# Units: cm, ns, MJ, K.
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
        # Eight-point Gauss-Legendre quadrature of sigma_SB*T_D(t)^4.
        nodes, weights = np.polynomial.legendre.leggauss(8)
        tm = 0.5 * (t0 + t1)                     # midpoint of the step
        th = 0.5 * (t1 - t0)                     # half-width, maps [-1,1] -> [t0,t1]
        ts = tm + th * nodes
        TeV = self.temperature_eV(ts)
        vals = SIGMA_SB * (TeV * EV_TO_K) ** 4   # incident one-sided flux [MJ/(cm^2 ns)]
        # 0.5*sum(w*f) is the *average* over the step (the th Jacobian cancels).
        return 0.5 * np.sum(weights * vals)


@njit(cache=True)
def material_coefficients(theta, dt):
    """Per-cell opacity, Fleck factor, Planck function, energy, and emitted energy."""
    n = theta.size
    sigma = np.empty(n)
    fleck = np.empty(n)
    B = np.empty(n)
    U = np.empty(n)
    emission = np.empty(n)
    for i in range(n):
        th = max(theta[i], THETA_FLOOR)
        sigma[i] = SIGMA0 * th ** (-ALPHA)      # absorption opacity [1/cm]
        B[i] = E100 * th ** 4                   # a*T^4, equilibrium radiation energy density
        U[i] = E100 * H * th ** BETA            # material internal energy density
        # gamma = (dB/dT)/(dU/dT), the "beta" of Fleck-Cummings.
        gamma = 4.0 * th ** (4.0 - BETA) / (H * BETA)
        fleck[i] = 1.0 / (1.0 + C * sigma[i] * dt * gamma)   # Fleck factor f in (0,1]
        # Energy actually emitted by cell i this step, per unit area [MJ/cm^2].
        emission[i] = C * fleck[i] * sigma[i] * B[i] * DX * dt
    return sigma, fleck, B, U, emission


@njit(cache=True)
def invert_material_energy(U):
    """Invert U = E100*H*theta^beta to recover theta from the updated energy."""
    theta = np.empty(U.size)
    for i in range(U.size):
        theta[i] = max(THETA_FLOOR, (max(U[i], 0.0) / (E100 * H)) ** (1.0 / BETA))
    return theta


@njit(cache=True)
def allocate_material_packets(emission, target):
    """At least one packet per emitting cell, with extras proportional to energy."""
    n = emission.size
    counts = np.zeros(n, dtype=np.int64)
    total = 0.0
    active = 0
    for i in range(n):
        if emission[i] > 0.0:
            total += emission[i]      # total emitted energy this step
            active += 1               # number of cells that emit at all
    if total <= 0.0:
        return counts, 0

    for i in range(n):
        if emission[i] > 0.0:
            counts[i] = 1             # guarantee every emitting cell is sampled
    remaining = max(0, target - active)   # budget left after the one-per-cell floor
    assigned = active
    if remaining > 0:
        fractional = np.empty(n)
        for i in range(n):
            exact = remaining * emission[i] / total   # ideal (real-valued) share
            add = int(exact)                          # integer part now...
            counts[i] += add
            assigned += add
            fractional[i] = exact - add               # ...remainder queued below
        # Largest-remainder method: hand out the leftover packets one at a time.
        while assigned < target:
            best = 0
            best_value = -1.0
            for i in range(n):
                if fractional[i] > best_value:
                    best_value = fractional[i]
                    best = i
            counts[best] += 1
            fractional[best] = -1.0    # each cell wins at most one leftover
            assigned += 1
    return counts, assigned            # note: assigned > target if active > target,
                                       # but always assigned == counts.sum()


@njit(cache=True)
def create_sources(
    old_x, old_mu, old_w,              # census packets surviving from the last step
    emission, material_counts,         # per-cell emitted energy and packet counts
    boundary_energy, n_boundary,       # energy entering from the left drive
    t0, dt,
):
    """Build the full packet list for one step: census + emission + boundary source."""
    old_n = old_x.size
    n_material = 0
    for i in range(material_counts.size):
        n_material += material_counts[i]
    total_n = old_n + n_material + n_boundary

    x = np.empty(total_n)          # position [cm]
    mu = np.empty(total_n)         # direction cosine in [-1,1]
    weight = np.empty(total_n)     # packet energy [MJ/cm^2]
    start = np.empty(total_n)      # birth time within the step [ns]

    for p in range(old_n):
        # Census packets already exist at t0 and keep their state.
        x[p] = old_x[p]
        mu[p] = old_mu[p]
        weight[p] = old_w[p]
        start[p] = t0

    cursor = old_n
    for i in range(material_counts.size):
        count = material_counts[i]
        if count <= 0:
            continue
        packet_weight = emission[i] / count     # split the cell's emission evenly
        left = i * DX
        for _ in range(count):
            x[cursor] = left + DX * np.random.random()      # uniform inside the cell
            mu[cursor] = 2.0 * np.random.random() - 1.0     # isotropic emission
            weight[cursor] = packet_weight
            start[cursor] = t0 + dt * np.random.random()    # uniform birth time
            cursor += 1

    if n_boundary > 0 and boundary_energy > 0.0:
        packet_weight = boundary_energy / n_boundary
        for _ in range(n_boundary):
            x[cursor] = np.nextafter(0.0, 1.0)              # just inside the left face
            # Flux-weighted cosine law p(mu)=2mu, 0<mu<1.
            mu[cursor] = np.sqrt(np.random.random())        # inward-going only
            weight[cursor] = packet_weight
            start[cursor] = t0 + dt * np.random.random()
            cursor += 1

    return x, mu, weight, start


@njit(cache=True)
def transport_step(x, mu, weight, start, sigma, fleck, t1, cutoff):
    """Analog Fleck-Cummings IMC transport for one time step.

    A collision with the physical opacity is treated as effective absorption
    with probability f and as isotropic effective scattering with probability
    1-f.  This is much faster than continuous absorption in the initially
    ultrahigh-opacity cold foam and is the classic IMC interpretation.
    """
    n_packets = x.size
    census_x = np.empty(n_packets)      # packets still alive at t1
    census_mu = np.empty(n_packets)
    census_w = np.empty(n_packets)
    n_census = 0

    deposited = np.zeros(N_CELLS)       # energy absorbed per cell [MJ/cm^2]
    escaped_left = 0.0                  # leaked back out through the drive face
    escaped_right = 0.0                 # leaked out the vacuum face (the observable)
    collision_count = 0

    for p in range(n_packets):
        xp = x[p]
        mup = mu[p]
        wp = weight[p]
        tp = start[p]

        if wp <= 0.0 or tp >= t1:
            # Nothing to transport; keep it for the next step if it carries energy.
            if wp > 0.0:
                census_x[n_census] = min(max(xp, 0.0), L_CM)
                census_mu[n_census] = mup
                census_w[n_census] = wp
                n_census += 1
            continue

        events = 0
        while True:
            events += 1
            if events > 20000:
                # Runaway history (very optically thick cell): park it in census.
                census_x[n_census] = min(max(xp, 0.0), L_CM)
                census_mu[n_census] = mup
                census_w[n_census] = wp
                n_census += 1
                break

            cell = int(xp / DX)
            if cell < 0:
                escaped_left += wp
                break
            if cell >= N_CELLS:
                escaped_right += wp
                break

            sig = sigma[cell]
            if sig > 0.0:
                # Sample the exponential free path; clamp the log argument.
                distance_collision = -np.log(max(np.random.random(), 1.0e-300)) / sig
            else:
                distance_collision = 1.0e300

            abs_mu = abs(mup)
            if abs_mu < 1.0e-15:
                distance_face = 1.0e300          # travelling essentially transversely
            elif mup > 0.0:
                distance_face = ((cell + 1) * DX - xp) / mup   # to the right face
            else:   
                distance_face = (cell * DX - xp) / mup         # to the left face
            if distance_face < 0.0:
                distance_face = 0.0              # guard against round-off overshoot

            distance_census = C * max(0.0, t1 - tp)   # distance left in this time step
            distance = min(distance_collision, distance_face, distance_census)
            xp += mup * distance                      # advance in space...
            tp += distance / C                        # ...and in time

            if distance == distance_census or tp >= t1 - 1.0e-15:
                # Step ended first: freeze the packet into the census.
                census_x[n_census] = min(max(xp, 0.0), L_CM)
                census_mu[n_census] = mup
                census_w[n_census] = wp
                n_census += 1
                break

            if distance == distance_face:
                # bookkeeping: the packet is now in a new cell, but it may have escaped.
                if mup > 0.0:
                    if cell == N_CELLS - 1:
                        escaped_right += wp          # out the vacuum boundary
                        break
                    xp = (cell + 1) * DX + 1.0e-10 * DX   # nudge across the face
                else:
                    if cell == 0:
                        escaped_left += wp           # back out through the drive face
                        break
                    xp = cell * DX - 1.0e-10 * DX
                continue                             ### keep streaming in the new cell!

            collision_count += 1
            if np.random.random() < fleck[cell]:
                deposited[cell] += wp                # effective absorption, history ends
                break
            mup = 2.0 * np.random.random() - 1.0     # effective scatter, isotropic

    return (
        census_x[:n_census], census_mu[:n_census], census_w[:n_census],
        deposited, escaped_left, escaped_right, collision_count,
    )


@njit(cache=True)
def merge_census(x, mu, weight, max_packets):
    """Unbiased systematic resampling when the census becomes too large."""
    n = x.size
    if n <= max_packets:
        return x, mu, weight                 # nothing to do
    total = np.sum(weight)
    if total <= 0.0:
        return np.empty(0), np.empty(0), np.empty(0)

    cumulative = np.cumsum(weight)           # CDF in packet-energy space
    new_x = np.empty(max_packets)
    new_mu = np.empty(max_packets)
    new_w = np.full(max_packets, total / max_packets)   # equal weights, energy conserved
    offset = np.random.random() * total / max_packets   # random start of the comb
    j = 0
    for k in range(max_packets):
        target = offset + k * total / max_packets       # evenly spaced comb teeth
        while j < n - 1 and cumulative[j] < target:
            j += 1                                      # single sweep, O(n)
        new_x[k] = x[j]
        new_mu[k] = mu[j]
    return new_x, new_mu, new_w


def run_one(seed, drive, dt, end_time, n_boundary, n_material, max_census, progress=False):
    """Run one complete stochastic history set and return the diagnostics."""
    np.random.seed(seed)
    # Seed Numba's RNG in compiled code as well.
    seed_numba(seed)

    theta = np.full(N_CELLS, THETA0)     # cold slab at t = 0
    census_x = np.empty(0)               # no radiation in flight initially
    census_mu = np.empty(0)
    census_w = np.empty(0)

    steps = int(np.ceil(end_time / dt))
    time_mid = np.empty(steps)                    # step-centred time [ns]
    outgoing = np.empty(steps)                    # right-face flux [MJ/(cm^2 ns)]
    outgoing_raw_energy = np.empty(steps)         # right-face energy per step
    material_T_max = np.empty(steps)                # max material temperature [eV]
    census_count = np.empty(steps, dtype=np.int64)
    energy_error = np.empty(steps)                # relative global balance error

    _, _, _, U0, _ = material_coefficients(theta, dt)
    initial_energy = np.sum(U0) * DX              # baseline for the energy balance
    cumulative_boundary = 0.0                     # energy injected by the drive
    cumulative_left = 0.0                         # energy leaked back out the left
    cumulative_right = 0.0                        # energy leaked out the right

    t0 = 0.0
    for step in range(steps):
        t1 = min(end_time, t0 + dt)               # last step may be shorter
        local_dt = t1 - t0
        sigma, fleck, B, U, emission = material_coefficients(theta, local_dt)
        packet_counts, total_packet_counts = allocate_material_packets(emission, n_material)

        flux_avg = drive.flux_average(t0, t1)     # time-averaged incident flux
        boundary_energy = flux_avg * local_dt     # energy entering this step
        cumulative_boundary += boundary_energy

        x, mu, w, start = create_sources(
            census_x, census_mu, census_w,
            emission, packet_counts,
            boundary_energy, n_boundary,
            t0, local_dt,
        )

        # Representative packet energy, used to size the (currently unused) cutoff.
        boundary_weight = boundary_energy / max(n_boundary, 1) # n_boundary may be zero if the drive is off
        minimal_emission_weight = np.sum(emission) / max(total_packet_counts, 1)  
        if np.sum(emission) > 0.0:
            boundary_weight = max(boundary_weight, minimal_emission_weight)
        cutoff = max(1.0e-30, 1.0e-10 * boundary_weight)

        (
            census_x, census_mu, census_w,
            deposited, escaped_left, escaped_right, collisions,
        ) = transport_step(x, mu, w, start, sigma, fleck, t1, cutoff)

        census_x, census_mu, census_w = merge_census(
            census_x, census_mu, census_w, max_census
        )

        # Material update: absorbed minus emitted, converted back to energy density.
        U_new = U + (deposited - emission) / DX
        if np.any(U_new <= 0.0) or np.any(~np.isfinite(U_new)):
            # Over-emission: the linearization broke down for this dt / packet count.
            raise RuntimeError(
                f"Non-positive material energy at step {step}, t={t1:g} ns. "
                "Reduce dt or increase particle counts."
            )
        theta = invert_material_energy(U_new)

        cumulative_left += escaped_left
        cumulative_right += escaped_right
        # Global check: matter + in-flight radiation should equal initial + in - out.
        current_total = np.sum(U_new) * DX + np.sum(census_w)
        expected_total = initial_energy + cumulative_boundary - cumulative_left - cumulative_right

        time_mid[step] = 0.5 * (t0 + t1)
        outgoing_raw_energy[step] = escaped_right
        outgoing[step] = escaped_right / local_dt    # unit area, so energy/time = flux
        material_T_max[step] = 100.0 * np.max(theta)   # theta -> eV
        census_count[step] = census_w.size
        energy_error[step] = (current_total - expected_total) / max(abs(expected_total), 1.0e-30)

        if progress and (step % max(1, steps // 20) == 0 or step == steps - 1):
            # About twenty progress lines per run, plus the final step.
            print(
                f"seed={seed:7d} step={step+1:5d}/{steps}, "
                f"t={t1:7.4f} ns, Tmax={material_T_max[step]:7.2f} eV, "
                f"census={census_w.size:6d}, collisions={collisions:8d}, "
                f"energy err={energy_error[step]:+.3e}",
                flush=True,
            )
        t0 = t1

    return {
        "time_ns": time_mid,
        "Fout": outgoing,
        "Fout_energy": outgoing_raw_energy,
        "Tm_final_eV": 100.0 * theta,      # final material temperature profile [eV]
        "Tmax_eV": material_T_max,
        "census_count": census_count,
        "energy_error": energy_error,
    }


@njit(cache=True)
def seed_numba(seed):
    # Numba keeps its own RNG state; seeding must happen inside compiled code.
    np.random.seed(seed)


def block_average(time_values, values, width_ns):
    """Bin a noisy time series into fixed-width blocks to tame MC noise."""
    if width_ns <= 0.0:
        return time_values.copy(), values.copy()     # smoothing disabled
    tmin = time_values[0]
    tmax = time_values[-1]
    # Start half a sample early so the first bin is centred on the first point.
    edges = np.arange(tmin - 0.5 * (time_values[1] - time_values[0]), tmax + width_ns, width_ns)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(centers.size, np.nan)
    for i in range(centers.size):
        mask = (time_values >= edges[i]) & (time_values < edges[i + 1])
        if np.any(mask):
            out[i] = np.mean(values[mask])
    good = np.isfinite(out)                          # drop empty bins
    return centers[good], out[good]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path, default=Path(__file__).with_name("back_pulse.txt"))
    parser.add_argument("--dt", type=float, default=0.001, help="IMC timestep [ns]")
    parser.add_argument("--end", type=float, default=3.0, help="end time [ns]")
    parser.add_argument("--boundary-packets", type=int, default=4000)   # per step
    parser.add_argument("--material-packets", type=int, default=2500)   # target, per step
    parser.add_argument("--max-census", type=int, default=120000)       # resampling trigger
    parser.add_argument("--seeds", type=str, default="1729,2718,3141")  # independent runs
    parser.add_argument("--smooth", type=float, default=0.02, help="plot averaging width [ns]")
    parser.add_argument("--prefix", type=str, default="sio2_1mm_gray_imc_check")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    drive = Drive(args.drive)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    runs = []
    wall0 = time.time()
    for seed in seeds:
        runs.append(
            run_one(
                seed, drive, args.dt, args.end,
                args.boundary_packets, args.material_packets,
                args.max_census, progress=not args.no_progress,
            )
        )
    wall = time.time() - wall0

    t = runs[0]["time_ns"]                          # common time grid for all seeds
    fluxes = np.vstack([r["Fout"] for r in runs])   # shape (n_seeds, n_steps)
    mean_flux = np.mean(fluxes, axis=0)
    # Standard error of the mean across seeds; zero when there is only one run.
    stderr_flux = np.std(fluxes, axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean_flux)

    norm = SIGMA_SB * (165.0 * EV_TO_K) ** 4        # plots are normalized to 165 eV
    t_smooth, mean_smooth = block_average(t, mean_flux / norm, args.smooth)
    _, stderr_smooth = block_average(t, stderr_flux / norm, args.smooth)

    output_npz = Path(args.prefix + ".npz")
    np.savez_compressed(
        output_npz,
        time_ns=t,
        Fout_runs=fluxes,
        Fout_mean=mean_flux,
        Fout_stderr=stderr_flux,
        normalization_flux=norm,
        smooth_time_ns=t_smooth,
        smooth_normalized_mean=mean_smooth,
        smooth_normalized_stderr=stderr_smooth,
        dt_ns=args.dt,
        seeds=np.asarray(seeds),
        boundary_packets=args.boundary_packets,
        material_packets=args.material_packets,
        final_material_temperature_eV=np.vstack([r["Tm_final_eV"] for r in runs]),
        energy_error=np.vstack([r["energy_error"] for r in runs]),
        wall_seconds=wall,
    )

    # Plot IMC alone.
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for k, seed in enumerate(seeds):
        # Faint per-seed curves show the spread of individual histories.
        raw_t, raw_y = block_average(t, fluxes[k] / norm, args.smooth)
        ax.plot(raw_t, raw_y, alpha=0.35, linewidth=0.9, label=f"seed {seed}" if len(seeds) <= 4 else None)
    ax.plot(t_smooth, mean_smooth, color="black", linewidth=2.0, label="IMC mean")
    if len(seeds) > 1:
        ax.fill_between(          # 95%-ish confidence band on the mean
            t_smooth,
            mean_smooth - 2.0 * stderr_smooth,
            mean_smooth + 2.0 * stderr_smooth,
            alpha=0.18,
            label=r"mean $\pm2$ s.e.",
        )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"$F_{\rm out}/[\sigma_{\rm SB}(165\,{\rm eV})^4]$")
    ax.set_title(f"Independent gray IMC: L=1 mm, 100 cells, dt={args.dt:g} ns")
    ax.set_xlim(0.8, args.end)    # nothing interesting before breakout
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    imc_plot = Path(args.prefix + "_Fout.png")
    fig.savefig(imc_plot, dpi=220)
    plt.close(fig)

    # Compare with the existing deterministic curves if present.
    fields_path = Path(__file__).with_name("sio2_1mm_four_models_case_mu_100cells_fields.npz")
    comparison_plot = Path(args.prefix + "_vs_four_models.png")
    if fields_path.exists():
        d = np.load(fields_path)
        fig, ax = plt.subplots(figsize=(9.0, 5.8))
        labels = {                # the four deterministic moment models
            "diffusion": "Diffusion",
            "p1": r"Classical $P_1$",
            "dap1": "DisAP1, classical BC",
            "dap1_mu_bc": r"DisAP1, $\mu(\omega)$ BC",
        }
        styles = {"diffusion": "-", "p1": "--", "dap1": ":", "dap1_mu_bc": "-."}
        for key in labels:
            ax.plot(
                d[f"{key}_time_ns"], d[f"{key}_F_out"] / norm,
                styles[key], linewidth=1.5, label=labels[key],
            )
        ax.plot(t_smooth, mean_smooth, color="black", linewidth=2.2, label="Independent IMC")
        if len(seeds) > 1:
            ax.fill_between(
                t_smooth,
                mean_smooth - 2.0 * stderr_smooth,
                mean_smooth + 2.0 * stderr_smooth,
                color="black", alpha=0.12,
            )
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel(r"$F_{\rm out}/[\sigma_{\rm SB}(165\,{\rm eV})^4]$")
        ax.set_title("Gray 1D Back problem: independent IMC versus four moment models")
        ax.set_xlim(0.8, args.end)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(comparison_plot, dpi=220)
        plt.close(fig)

    summary_path = Path(args.prefix + "_summary.txt")
    crossing = np.flatnonzero(mean_smooth >= 0.01)          # 1% of the 165 eV flux
    breakout = t_smooth[crossing[0]] if crossing.size else np.nan
    peak_index = int(np.argmax(mean_smooth))
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Independent gray IMC check for L=1 mm, 100 cells\n")
        f.write(f"dt_ns {args.dt}\n")
        f.write(f"seeds {seeds}\n")
        f.write(f"boundary_packets_per_step {args.boundary_packets}\n")
        f.write(f"target_material_packets_per_step {args.material_packets}\n")
        f.write(f"smoothing_width_ns {args.smooth}\n")
        f.write(f"breakout_normalized_0p01_ns {breakout:.12g}\n")
        f.write(f"peak_time_ns {t_smooth[peak_index]:.12g}\n")
        f.write(f"peak_normalized {mean_smooth[peak_index]:.12g}\n")
        # Worst-case energy conservation error over all steps and all seeds.
        f.write(f"maximum_absolute_energy_error {max(np.max(np.abs(r['energy_error'])) for r in runs):.12g}\n")
        f.write(f"wall_seconds {wall:.6f}\n")

    print(f"Completed {len(seeds)} IMC histories sets in {wall:.1f} s")
    print(f"Breakout (smoothed normalized flux >=0.01): {breakout:.6f} ns")
    print(f"Peak: {mean_smooth[peak_index]:.6f} at {t_smooth[peak_index]:.6f} ns")
    print(f"Saved: {output_npz}")
    print(f"Saved: {imc_plot}")
    if fields_path.exists():
        print(f"Saved: {comparison_plot}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
