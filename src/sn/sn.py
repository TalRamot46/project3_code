"""Time-dependent discrete-ordinates (S_N) transport in a 1D slab or sphere.

Solves, for the angular flux psi(x, mu, t),

    (1/v) d(psi)/dt + mu d(psi)/dx + sigma_t psi = Q_s(mu) + S(x),
    phi = 0.5 * sum_m w_m psi_m          (so an isotropic psi equals phi),

with constant sigma_t, sigma_s, v.  Note that S enters without a factor 1/2:
integrating over mu gives (1/v) d(phi)/dt + dJ/dx + sigma_a phi = S, so phi/v
is the particle density.  The derivation, and the map to gray radiation
(psi = 4 pi I, phi = c E_r, v = c, S = k_a a c T^4), is in md/sn_derivation.md.

Discretization:
  * angle: Gauss-Legendre ordinates, n_mu even, sum(w) = 2
  * scattering: Legendre expansion Q_s = sigma_s sum_l (2l+1) f_l P_l(mu) phi_l
    (isotropic, Henyey-Greenstein f_l = g^l, or user-given f_l)
  * time: backward Euler, i.e. sigma_eff = sigma_t + 1/(v dt) and
    rhs = Q + psi^n / (v dt)
  * slab: diamond difference, vacuum on both faces
  * sphere: Lewis-Miller diamond difference with angular redistribution,
    a mu = -1 starting direction and reflection at r = 0
  * both: set-to-zero fixup for negative outgoing fluxes
  * the scattering source is converged by source iteration every step

The negative-source fixups (only active for anisotropic scattering) and the
sphere sweep follow the Fortran code sn1_spherical.f90 line by line.
"""
import time

import numpy as np
from numba import njit

FOUR_PI_OVER_THREE = 4.0 * np.pi / 3.0
PHI_FLOOR = 1e-3            # relative floor in the convergence and dt tests


def make_quadrature(n_mu):
    """Gauss-Legendre nodes (increasing, mu_0 < 0 < mu_{N-1}) and weights."""
    if n_mu < 2 or n_mu % 2 != 0:
        raise ValueError("n_mu must be an even integer >= 2.")
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    return mu.astype(np.float64), w.astype(np.float64)


@njit(cache=True)
def legendre_table(mu, lmax):
    """P[l, m] = P_l(mu_m) from (l+1) P_{l+1} = (2l+1) mu P_l - l P_{l-1}."""
    n_mu = len(mu)
    P = np.empty((lmax + 1, n_mu))
    for m in range(n_mu):
        P[0, m] = 1.0
        if lmax >= 1:
            P[1, m] = mu[m]
    for l in range(1, lmax):
        for m in range(n_mu):
            P[l + 1, m] = ((2 * l + 1) * mu[m] * P[l, m] - l * P[l - 1, m]) / (l + 1)
    return P


@njit(cache=True)
def scalar_flux(psi, w, phi):
    """phi_i = 0.5 * sum_m w_m psi[m, i], written into phi."""
    for i in range(psi.shape[1]):
        s = 0.0
        for m in range(psi.shape[0]):
            s += w[m] * psi[m, i]
        phi[i] = 0.5 * s


@njit(cache=True)
def angular_moments(psi, w, P):
    """phi_l(i) = 0.5 * sum_m w_m P_l(mu_m) psi[m, i]."""
    moments = np.zeros((P.shape[0], psi.shape[1]))
    for l in range(P.shape[0]):
        for m in range(psi.shape[0]):
            wP = 0.5 * w[m] * P[l, m]
            for i in range(psi.shape[1]):
                moments[l, i] += wP * psi[m, i]
    return moments


@njit(cache=True)
def scattering_source(moments, sigma_s, f_l, P):
    """Q_s(mu_m, i) = sigma_s * sum_l (2l+1) f_l P_l(mu_m) phi_l(i)."""
    q = np.zeros((P.shape[1], moments.shape[1]))
    for l in range(P.shape[0]):
        coeff = sigma_s * (2 * l + 1) * f_l[l]
        for m in range(P.shape[1]):
            cP = coeff * P[l, m]
            for i in range(moments.shape[1]):
                q[m, i] += cP * moments[l, i]
    return q


@njit(cache=True)
def fixup_scale_clip(q, q_iso, w):
    """Fortran anisotropic_method=2: shrink the isotropic level, then clip at 0.

    fac = q_iso / (q_iso - S_neg) with S_neg = 0.5 sum w min(q, 0); no-op
    (fac = 1) when no ordinate is negative, e.g. for isotropic scattering.
    """
    for i in range(q.shape[1]):
        fac = 0.0
        if q_iso[i] > 1e-40:
            sum_neg = 0.0                       # weighted negative part, <= 0
            for m in range(q.shape[0]):
                if q[m, i] < 0.0:
                    sum_neg += 0.5 * w[m] * q[m, i]
            fac = q_iso[i] / (q_iso[i] - sum_neg)
            q_iso[i] *= fac
        for m in range(q.shape[0]):
            q[m, i] = max(0.0, q_iso[i] + (q[m, i] - q_iso[i] / (fac + 1e-80)) * fac)


@njit(cache=True)
def fixup_min_blend(q, q_iso):
    """Fortran anisotropic_method=1: blend q toward q_iso until min(q) = 0."""
    for i in range(q.shape[1]):
        fac = 0.0
        if q_iso[i] > 1e-40:
            src_neg = 0.0                       # most negative ordinate source
            for m in range(q.shape[0]):
                if q[m, i] < src_neg:
                    src_neg = q[m, i]
            if src_neg >= min(0.0, q_iso[i]):
                continue                        # nothing negative in this cell
            fac = -q_iso[i] / (src_neg - q_iso[i])
        for m in range(q.shape[0]):
            q[m, i] = q_iso[i] + (q[m, i] - q_iso[i]) * fac


@njit(cache=True)
def sweep_slab(psi, rhs, mu, dx, sig_eff, do_fixup):
    """Diamond-difference sweep of every ordinate, vacuum on both faces.

    Cell balance mu (psi_out - psi_in)/dx + sig_eff psi_cell = rhs with
    psi_cell = (psi_in + psi_out)/2.  If psi_out < 0, set it to 0 and keep
    the balance: psi_cell = (rhs + |mu| psi_in/dx) / sig_eff.
    """
    n_mu, n_cells = psi.shape
    for m in range(n_mu):
        mu_abs = abs(mu[m])
        backward = m < n_mu // 2                # mu < 0 sweeps right to left
        psi_face = 0.0                          # vacuum inflow
        for j in range(n_cells):
            k = n_cells - 1 - j if backward else j
            two_mu_dx = 2.0 * mu_abs / dx[k]
            psi_cell = (rhs[m, k] + two_mu_dx * psi_face) / (sig_eff[k] + two_mu_dx)
            psi_out = 2.0 * psi_cell - psi_face
            if psi_out < 0.0 and do_fixup:
                psi_cell = (rhs[m, k] + mu_abs * psi_face / dx[k]) / sig_eff[k]
                psi_out = 0.0
            psi[m, k] = psi_cell
            psi_face = psi_out                  # outflow of k is inflow of the next cell


@njit(cache=True)
def lewis_miller_alpha(mu, w):
    """alpha_{1/2} = 0, alpha_{m+1/2} = alpha_{m-1/2} - mu_m w_m, alpha_{N+1/2} = 0."""
    alpha = np.zeros(len(mu) + 1)
    for j in range(len(mu)):
        alpha[j + 1] = alpha[j] - mu[j] * w[j]
    alpha[len(mu)] = 0.0                        # exact in theory; removes round-off
    return alpha


@njit(cache=True)
def sphere_cell(psi_in, psi_ang_in, rhs, mu_abs, w, alpha_lo, alpha_hi,
                area_in, area_out, dA, vol, sig, do_fixup):
    """Lewis-Miller balance of one cell and ordinate, with diamond in r and mu.

    Returns (psi_cell, spatial outflow, angular outflow).  A negative outflow
    is set to 0 (flags iff / muff) and the cell is re-solved without it.
    """
    iff = 0
    muff = 0
    psi_out = psi_in
    psi_ang_out = psi_ang_in
    while True:
        denom = (2.0 * mu_abs * area_out * (1.0 - iff)
                 + 2.0 / w * dA * alpha_hi * (1.0 - muff)
                 + vol * sig)
        numer = (mu_abs * (area_in + area_out * (1.0 - iff)) * psi_in
                 + 1.0 / w * dA * (alpha_hi * (1.0 - muff) + alpha_lo) * psi_ang_in
                 + vol * rhs)
        psi_cell = numer / denom
        if iff == 0:
            psi_out = 2.0 * psi_cell - psi_in
        if muff == 0:
            psi_ang_out = 2.0 * psi_cell - psi_ang_in
        if psi_out < 0.0 and iff == 0 and do_fixup:
            iff = 1
            psi_out = 0.0
            continue
        if psi_ang_out < 0.0 and muff == 0 and do_fixup:
            muff = 1
            psi_ang_out = 0.0
            continue
        return psi_cell, psi_out, psi_ang_out


@njit(cache=True)
def sweep_sphere(psi, rhs, q_total, mu, w, alpha, r_edges, vol, surf, sig_eff,
                 do_fixup, psi_start_old, psi_start_new, inv_v_dt):
    """Lewis-Miller sweep: mu = -1 start, mu < 0 inward, mu > 0 outward."""
    n_mu, n_cells = psi.shape
    psi_ang = np.empty(n_cells)                 # flux on the angular edge m - 1/2
    psi_center = np.zeros(n_mu)                 # inward flux reaching r = 0

    # Starting direction mu = -1: a slab-like inward sweep, no redistribution.
    # It uses the most negative ordinate's source and its own time census.
    psi_face = 0.0
    for k in range(n_cells - 1, -1, -1):
        dr = r_edges[k + 1] - r_edges[k]
        rhs_start = q_total[0, k] + psi_start_old[k] * inv_v_dt
        psi_cell = (2.0 * psi_face + dr * rhs_start) / (2.0 + dr * sig_eff[k])
        psi_out = 2.0 * psi_cell - psi_face
        if psi_out < 0.0 and do_fixup:
            psi_cell = (psi_face + dr * rhs_start) / (dr * sig_eff[k])
            psi_out = 0.0
        psi_start_new[k] = psi_cell
        psi_ang[k] = psi_cell
        psi_face = psi_out

    # mu < 0: inward, inflow through the outer face surf[k+1].
    for j in range(n_mu // 2):
        psi_face = 0.0                          # vacuum at the outer radius
        for k in range(n_cells - 1, -1, -1):
            psi[j, k], psi_face, psi_ang[k] = sphere_cell(
                psi_face, psi_ang[k], rhs[j, k], abs(mu[j]), w[j],
                alpha[j], alpha[j + 1], surf[k + 1], surf[k],
                surf[k + 1] - surf[k], vol[k], sig_eff[k], do_fixup)
        psi_center[j] = psi_face

    # mu > 0: outward, starting from the reflected inward flux at r = 0.
    for j in range(n_mu // 2, n_mu):
        psi_face = psi_center[n_mu - 1 - j]
        for k in range(n_cells):
            psi[j, k], psi_face, psi_ang[k] = sphere_cell(
                psi_face, psi_ang[k], rhs[j, k], abs(mu[j]), w[j],
                alpha[j], alpha[j + 1], surf[k], surf[k + 1],
                surf[k + 1] - surf[k], vol[k], sig_eff[k], do_fixup)


@njit(cache=True)
def time_loop(sphere, mu, w, alpha, P, f_l, sigma_t, sigma_s, v, dt, sig_eff,
              dx, r_edges, vol, surf, psi, source, do_fixup, neg_fixup,
              max_iters, conv_tol, snapshot_times, tracer_cells,
              logging_interval, adaptive, dtfac, dtmax, t_final, max_steps):
    """Backward-Euler time steps, each converged by source iteration.

    neg_fixup: 0 none, 1 min_blend, 2 scale_clip.  With adaptive=True,
    dt <- min(dtfac*dt/delta, 1.1*dt, dtmax), where delta is the largest
    relative change of phi over the step.
    """
    n_mu, n_cells = psi.shape
    psi_old = np.empty((n_mu, n_cells))         # psi^n for backward Euler
    rhs = np.empty((n_mu, n_cells))
    q_iso = np.empty(n_cells)
    phi = np.empty(n_cells)
    phi_prev = np.empty(n_cells)                # previous SI iterate
    phi_step = np.empty(n_cells)                # phi^n, for the dt controller
    psi_start_old = psi[0].copy()               # sphere mu = -1 census
    psi_start_new = psi[0].copy()
    inv_v_dt = 1.0 / (v * dt)

    n_snap = len(snapshot_times)
    phi_snap = np.zeros((n_snap, n_cells))
    avg_mu_snap = np.zeros((n_snap, n_cells))   # <mu>   = J / phi
    avg_mu2_snap = np.zeros((n_snap, n_cells))  # <mu^2> = 0.5 sum w mu^2 psi / phi
    time_snap = np.zeros(n_snap)
    size = max_steps + 1                        # history buffers, doubled if needed
    step_times = np.zeros(size)
    step_iters = np.zeros(size, dtype=np.int64)
    step_particles = np.zeros(size)
    n_tracers = len(tracer_cells)
    tracer_density = np.zeros((size if n_tracers > 0 else 0, n_tracers))

    snap = 0
    t = 0.0
    step = 0
    while True:
        if step >= size:                        # adaptive runs may outgrow the guess
            step_times = np.concatenate((step_times, np.zeros_like(step_times)))
            step_iters = np.concatenate((step_iters, np.zeros_like(step_iters)))
            step_particles = np.concatenate((step_particles, np.zeros_like(step_particles)))
            if n_tracers > 0:
                tracer_density = np.concatenate((tracer_density, np.zeros_like(tracer_density)))
            size *= 2

        # Diagnostics of the state at time t.
        scalar_flux(psi, w, phi)
        total = 0.0
        for i in range(n_cells):
            total += phi[i] / v * vol[i]        # particle count, phi/v = density
        step_times[step] = t
        step_particles[step] = total
        if step % logging_interval == 0:
            print("step=", step, " t=", round(t, 6), " dt=", round(dt, 8),
                  " particles=", round(total, 6))
        while snap < n_snap and t >= snapshot_times[snap] - 1e-10:
            for i in range(n_cells):
                phi_snap[snap, i] = phi[i]
                m1 = 0.0
                m2 = 0.0
                for m in range(n_mu):
                    m1 += w[m] * mu[m] * psi[m, i]
                    m2 += w[m] * mu[m] * mu[m] * psi[m, i]
                if phi[i] > 0.0:
                    avg_mu_snap[snap, i] = 0.5 * m1 / phi[i]
                    avg_mu2_snap[snap, i] = 0.5 * m2 / phi[i]
            time_snap[snap] = t
            snap += 1
        for tr in range(n_tracers):
            tracer_density[step, tr] = phi[tracer_cells[tr]] / v

        if adaptive:
            if t >= t_final - 1e-10 * (abs(t_final) + 1e-300):
                break
        elif step >= max_steps:
            break

        phi_step[:] = phi
        psi_old[:, :] = psi

        # Source iteration: build Q from the last psi, sweep, repeat.
        n_iters = 0
        for it in range(max_iters):
            n_iters = it + 1
            scalar_flux(psi, w, phi_prev)
            moments = angular_moments(psi, w, P)
            q_total = scattering_source(moments, sigma_s, f_l, P)
            for i in range(n_cells):
                q_total[:, i] += source[i]
                q_iso[i] = sigma_s * moments[0, i] + source[i]
            if neg_fixup == 2:
                fixup_scale_clip(q_total, q_iso, w)
            elif neg_fixup == 1:
                fixup_min_blend(q_total, q_iso)
            for m in range(n_mu):
                for i in range(n_cells):
                    rhs[m, i] = q_total[m, i] + psi_old[m, i] * inv_v_dt

            if sphere:
                sweep_sphere(psi, rhs, q_total, mu, w, alpha, r_edges, vol, surf,
                             sig_eff, do_fixup, psi_start_old, psi_start_new, inv_v_dt)
            else:
                sweep_slab(psi, rhs, mu, dx, sig_eff, do_fixup)

            for m in range(n_mu):
                for i in range(n_cells):
                    if psi[m, i] < 0.0:
                        print("NEGATIVE psi: step=", step, " iter=", n_iters,
                              " m=", m, " i=", i, " psi=", psi[m, i])
                        assert False, "Negative angular flux after sweep"

            # Converged when max_i |dphi| / (phi + 1e-3 phi_max) < conv_tol.
            scalar_flux(psi, w, phi)
            phi_min = PHI_FLOOR * phi.max() + 1e-300
            err = 0.0
            for i in range(n_cells):
                err = max(err, abs(phi[i] - phi_prev[i]) / (phi[i] + phi_min))
            if err < conv_tol:
                break
            if n_iters == 100:
                print("WARNING: SI not converged after 100 iters, step=",
                      step, " err=", err, " tol=", conv_tol)

        step_iters[step] = n_iters
        if sphere:
            psi_start_old[:] = psi_start_new    # advance the mu = -1 census
        t += dt
        step += 1

        if adaptive:
            phi_min = PHI_FLOOR * phi.max() + 1e-300
            delta = 1e-80
            for i in range(n_cells):
                delta = max(delta, abs(phi[i] - phi_step[i]) / (phi[i] + phi_min))
            dt = min(dt / delta * dtfac, 1.1 * dt, dtmax)
            inv_v_dt = 1.0 / (v * dt)
            sig_eff[:] = sigma_t + inv_v_dt

    n_out = step + 1
    return (phi_snap, avg_mu_snap, avg_mu2_snap, time_snap,
            step_times[:n_out], step_iters[:n_out], step_particles[:n_out],
            tracer_density[:n_out] if n_tracers > 0 else tracer_density)


def scattering_coefficients(scatter_model, g, lmax, scatter_fl):
    """Legendre coefficients f_l of the scattering kernel, and lmax."""
    if scatter_fl is not None:
        if scatter_model == "hg":
            raise ValueError("Use either scatter_model='hg' or scatter_fl, not both.")
        if lmax is None:
            raise ValueError("lmax is required when scatter_fl is given.")
        f_l = np.asarray(scatter_fl, dtype=np.float64)
        if not np.isclose(f_l[0], 1.0):
            raise ValueError("scatter_fl[0] must be 1.")
        if len(f_l) != lmax + 1:
            raise ValueError(f"len(scatter_fl)={len(f_l)} must equal lmax+1={lmax + 1}.")
        return f_l, lmax
    if scatter_model == "isotropic":
        return np.array([1.0]), 0
    if lmax is None:
        raise ValueError("lmax is required for scatter_model='hg'.")
    return np.array([g ** l for l in range(lmax + 1)]), lmax


def run_sn(geometry, x_edges, sigma_t, sigma_s, v, dt, t_final, n0, n_mu, *,
           snapshot_times=None, source_density=None, logging_interval=100,
           scatter_model="isotropic", g=0.0, lmax=None, scatter_fl=None,
           max_iters=5000, conv_tol=1e-10, fixup=True,
           negative_source_fixup="scale_clip", adaptive_dt=True, dtfac=0.1,
           dtmax=None, tracer_positions=None):
    """Run S_N from an isotropic initial state psi = v*n0 to t_final.

    geometry "slab" or "sphere"; x_edges uniform cell edges (radii for the
    sphere); n0 initial density per cell; source_density S per cell.
    Scattering: "isotropic", "hg" with g and lmax, or scatter_fl = [1, f_1, ...]
    with lmax.  negative_source_fixup: "scale_clip", "min_blend" or "none".
    adaptive_dt needs dtmax; dt is then the first step.

    Returns (phi_snap, avg_mu_snap, avg_mu2_snap, time_snap, step_times,
    step_n_iters, step_total_particles, tracer_density).
    """
    x_edges = np.asarray(x_edges, dtype=np.float64)
    n0 = np.asarray(n0, dtype=np.float64)
    n_cells = len(x_edges) - 1
    dx = np.diff(x_edges)

    if geometry not in ("slab", "sphere"):
        raise ValueError(f"Unknown geometry '{geometry}'. Use 'slab' or 'sphere'.")
    if np.any(dx <= 0):
        raise ValueError("x_edges must be strictly increasing.")
    if not np.allclose(dx, dx[0]):
        raise ValueError("Non-uniform grids are not supported.")
    if sigma_t < 0 or sigma_s < 0 or sigma_s > sigma_t:
        raise ValueError("Need 0 <= sigma_s <= sigma_t.")
    if v <= 0 or dt <= 0 or t_final < 0:
        raise ValueError("Need v > 0, dt > 0 and t_final >= 0.")
    if len(n0) != n_cells:
        raise ValueError(f"len(n0)={len(n0)} != n_cells={n_cells}.")
    if geometry == "sphere" and x_edges[0] < 0:
        raise ValueError("Sphere geometry requires x_edges[0] >= 0.")
    if scatter_model not in ("isotropic", "hg"):
        raise ValueError("scatter_model must be 'isotropic' or 'hg'.")
    if not -1.0 < g < 1.0:
        raise ValueError("HG parameter g must satisfy -1 < g < 1.")
    if adaptive_dt and (dtmax is None or dtmax <= 0 or dtfac <= 0):
        raise ValueError("adaptive_dt needs dtmax > 0 and dtfac > 0.")
    if negative_source_fixup not in ("scale_clip", "min_blend", "none"):
        raise ValueError("negative_source_fixup must be 'scale_clip', 'min_blend' or 'none'.")

    f_l, lmax = scattering_coefficients(scatter_model, g, lmax, scatter_fl)
    if lmax >= n_mu:
        raise ValueError(f"lmax={lmax} must be < n_mu={n_mu}.")
    mu, w = make_quadrature(n_mu)
    P = legendre_table(mu, lmax)
    alpha = lewis_miller_alpha(mu, w)

    if geometry == "slab":
        vol = dx.copy()                         # unit transverse area
        surf = np.zeros(n_cells + 1)
    else:
        vol = FOUR_PI_OVER_THREE * (x_edges[1:] ** 3 - x_edges[:-1] ** 3)
        surf = 4.0 * np.pi * x_edges ** 2

    source = np.zeros(n_cells) if source_density is None else np.asarray(source_density, dtype=np.float64)
    if len(source) != n_cells:
        raise ValueError(f"len(source_density)={len(source)} != n_cells={n_cells}.")
    snapshot_times = np.sort(np.array([t_final] if snapshot_times is None else snapshot_times, dtype=np.float64))

    tracer_cells = np.zeros(0, dtype=np.int64)
    if tracer_positions is not None:
        tracer_positions = np.asarray(tracer_positions, dtype=np.float64)
        if np.any(tracer_positions < x_edges[0]) or np.any(tracer_positions >= x_edges[-1]):
            raise ValueError(f"tracer_positions must lie in [{x_edges[0]}, {x_edges[-1]}).")
        tracer_cells = np.minimum(((tracer_positions - x_edges[0]) / dx[0]).astype(np.int64), n_cells - 1)

    # Fixed dt: exactly round(t_final/dt) steps.  Adaptive: a buffer-size guess.
    max_steps = int(np.ceil(t_final / dtmax)) * 3 + 1000 if adaptive_dt else int(round(t_final / dt))

    print(f"S_N {geometry}: cells={n_cells} n_mu={n_mu} sigma_t={sigma_t:.6g} "
          f"sigma_s={sigma_s:.6g} v={v:.6g} dt={dt:.6g} t_final={t_final:.6g} "
          f"scatter={scatter_model} lmax={lmax} adaptive={adaptive_dt}")
    wall0 = time.time()
    result = time_loop(
        geometry == "sphere", mu, w, alpha, P, f_l, sigma_t, sigma_s, v, dt,
        np.full(n_cells, sigma_t + 1.0 / (v * dt)),   # sigma_eff, backward Euler
        dx, x_edges.copy(), vol, surf,
        np.tile(v * n0, (n_mu, 1)),                   # isotropic psi = v*n0
        source, fixup, {"none": 0, "min_blend": 1, "scale_clip": 2}[negative_source_fixup],
        max_iters, conv_tol, snapshot_times, tracer_cells, logging_interval,
        adaptive_dt, dtfac, dtmax if dtmax is not None else dt, t_final, max_steps)
    print(f"S_N runtime (including JIT compilation on first call): {time.time() - wall0:.2f} s")
    return result
