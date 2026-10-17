"""
Discrete-ordinates Sn transport solver for 1-D slab and spherical geometries.

Provides a single entry point ``solve(geometry, ...)`` that runs a
time-dependent Sn simulation with backward Euler time discretization
and diamond-difference spatial discretization.

Angular discretization
----------------------
Gauss-Legendre quadrature with n_mu ordinates (must be even).

    μₘ, wₘ = Gauss-Legendre nodes/weights on [-1, 1],  Σwₘ = 2.

Scattering source via Legendre expansion:

    Qₛ(μₘ) = σₛ · Σₗ (2l+1)·fₗ·Pₗ(μₘ)·φₗ
    where φₗ = ½ Σₘ wₘ·Pₗ(μₘ)·ψₘ   (angular moments)

Scattering models
-----------------
scatter_model="isotropic": fₗ = δ_{l,0}  (only l=0 term).
scatter_model="hg":        fₗ = gˡ       (Henyey-Greenstein).
scatter_fl=[f₀, f₁, ...]:  direct Legendre coefficients (requires lmax).

Spatial discretization
----------------------
Slab:  diamond-difference with vacuum BCs.
Sphere:  Lewis-Miller diamond-difference with angular redistribution
         (α coefficients), three-stage sweep (μ=-1 start, μ<0 inward,
         μ>0 outward with center reflection), and iff/muff fixup.

Time discretization
-------------------
Backward Euler:
    (ψⁿ⁺¹ − ψⁿ)/(v·Δt) + μ·∂ψ/∂x + Σₜ·ψ = Q
    → Σ_eff = Σₜ + 1/(v·Δt),   RHS = Q + ψⁿ/(v·Δt)

Source iteration solves the scattering source lagged in angle.

Input fields
------------
n0[j] : initial particle density (isotropic)
    slab:   [particles / length]
    sphere: [particles / volume]
    Initial scalar flux: φ(xⱼ, 0) = v · n0[j].

source_density[j] : volumetric source rate S(xⱼ)
    slab:   [particles / (length · time)]
    sphere: [particles / (volume · time)]
"""

import numpy as np
from numba import njit
import time as _time

_FOUR_PI_OVER_THREE = 4.0 * np.pi / 3.0
_FAC_MIN = 1e-3


# ── Quadrature helper (Python) ──────────────────────────────────────

def _make_quadrature(n_mu):
    """Gauss-Legendre quadrature on [-1, 1].

    Returns (mu, w) with mu sorted increasing: μ₀ < 0 < μ_{N-1}.
    Weights satisfy Σwₘ = 2.
    """
    if n_mu < 2 or n_mu % 2 != 0:
        raise ValueError("n_mu must be an even integer >= 2.")
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    return mu.astype(np.float64), w.astype(np.float64)


# ── Legendre table (JIT) ────────────────────────────────────────────

@njit(cache=False)
def _legendre_table(mu, lmax):
    """Pₗ(μₘ) via three-term recurrence.

    Returns P[l, m] of shape (lmax+1, n_mu).
    P₀(μ) = 1,  P₁(μ) = μ,  (l+1)Pₗ₊₁ = (2l+1)μPₗ − lPₗ₋₁
    """
    n_mu = len(mu)
    P = np.empty((lmax + 1, n_mu))
    for m in range(n_mu):
        P[0, m] = 1.0
    if lmax >= 1:
        for m in range(n_mu):
            P[1, m] = mu[m]
    for l in range(1, lmax):
        for m in range(n_mu):
            P[l + 1, m] = ((2 * l + 1) * mu[m] * P[l, m]
                           - l * P[l - 1, m]) / (l + 1)
    return P


# ── Shared: angular moments & scattering source ─────────────────────

@njit(cache=False)
def _compute_moments(psi, w, P, n_mu, n_cells, lmax):
    """Angular flux moments via quadrature.

    φₗ(i) = ½ Σₘ wₘ · Pₗ(μₘ) · ψ(μₘ, i)

    Returns moments[l, i] of shape (lmax+1, n_cells).
    """
    moments = np.zeros((lmax + 1, n_cells))
    for l in range(lmax + 1):
        for m in range(n_mu):
            wP = 0.5 * w[m] * P[l, m]
            for i in range(n_cells):
                moments[l, i] += wP * psi[m, i]
    return moments


@njit(cache=False)
def _build_scattering_source(moments, sigma_s, coeffs, P,
                             n_mu, n_cells, lmax):
    """Scattering source from Legendre expansion (geometry-independent).

    Qₛ(μₘ, i) = σₛ · Σₗ (2l+1) · fₗ · Pₗ(μₘ) · φₗ(i)

    Returns q_s[m, i] of shape (n_mu, n_cells).
    """
    q_s = np.zeros((n_mu, n_cells))
    for l in range(lmax + 1):
        coeff = sigma_s * (2 * l + 1) * coeffs[l]
        for m in range(n_mu):
            cP = coeff * P[l, m]
            for i in range(n_cells):
                q_s[m, i] += cP * moments[l, i]
    return q_s


# ── Negative source fixup (JIT, geometry-independent) ────────────────

@njit(cache=False)
def _negative_source_fixup(q, q_iso, w, n_mu, n_cells):
    """Negative source fixup matching Fortran anisotropic_method=2.

    Prevents negative angular source terms from anisotropic scattering.
    Mutates q and q_iso IN PLACE.

    Algorithm (Fortran lines 557-591):
      S_neg = ½ Σₘ wₘ·min(q(μₘ), 0)    (weighted negative sum, ≤ 0)
      fac = q_iso / (q_iso − S_neg)      (0 < fac ≤ 1)
      q_iso ← q_iso · fac                (reduce isotropic level)
      q(μₘ) = max(0, q_iso + (q(μₘ) − q_iso/(fac+ε)) · fac)
    """
    for i in range(n_cells):
        fac = 0.0
        if q_iso[i] > 1e-40:
            sum_neg = 0.0
            for m in range(n_mu):
                if q[m, i] < 0.0:
                    sum_neg += 0.5 * w[m] * q[m, i]
            # 0 < fac ≤ 1;  fac = 1 when sum_neg = 0 (no-op)
            fac = q_iso[i] / (q_iso[i] - sum_neg)
            q_iso[i] *= fac
        # Apply unconditionally (Fortran lines 586-589)
        for m in range(n_mu):
            q[m, i] = max(0.0,
                          q_iso[i] + (q[m, i] - q_iso[i] / (fac + 1e-80)) * fac)


@njit(cache=False)
def _negative_source_fixup_min_blend(q, q_iso, w, n_mu, n_cells):
    """Negative source fixup matching Fortran anisotropic_method=1.

    Linear blend toward isotropic level.  Mutates q IN PLACE (q_iso unchanged).

    Algorithm (Fortran sn1_spherical.f90, lines 567-584):
      src_neg = minₘ q(μₘ)                  (most negative angular source)
      sanis   = src_neg − q_iso
      skip cell if src_neg ≥ min(0, q_iso)   (no negatives to fix)
      fac     = −q_iso / sanis              (0 < fac ≤ 1)
      q(μₘ)  = q_iso + (q(μₘ) − q_iso)·fac  (blend toward isotropic)
    """
    for i in range(n_cells):
        fac = 0.0
        if q_iso[i] > 1e-40:
            src_neg = 0.0
            for m in range(n_mu):
                if q[m, i] < src_neg:
                    src_neg = q[m, i]
            sanis = src_neg - q_iso[i]
            if src_neg >= min(0.0, q_iso[i]):
                continue
            fac = -q_iso[i] / sanis
        for m in range(n_mu):
            q[m, i] = q_iso[i] + (q[m, i] - q_iso[i]) * fac


# ── Geometry-specific: slab diamond-difference sweep ─────────────────

@njit(cache=False)
def _sweep_slab(psi, rhs, mu, dx_arr, sig_eff, n_mu, n_cells, do_fixup):
    """Slab diamond-difference sweep with vacuum BCs.

    Diamond-difference cell balance:
      μ·(ψ_out − ψ_in)/Δx + Σ_eff·ψ_cell = Q
      ψ_cell = ½(ψ_in + ψ_out)
      → ψ_cell = (Q + 2|μ|ψ_in/Δx) / (Σ_eff + 2|μ|/Δx)

    Fixup (Fortran set-to-zero scheme, lines 344-347):
      Denominator: Σ_eff only (NO |μ|/Δx).
      Numerator: |μ|·ψ_face/Δx (factor 1, not 2).
      Outgoing face → 0.
    """
    n_half = n_mu // 2
    for m in range(n_mu):
        mu_abs = abs(mu[m])
        psi_face = 0.0
        if m < n_half:
            # μ < 0: sweep right to left
            for k in range(n_cells - 1, -1, -1):
                dx = dx_arr[k]
                two_mu_dx = 2.0 * mu_abs / dx
                psi_cell = (rhs[m, k] + two_mu_dx * psi_face) / (sig_eff[k] + two_mu_dx)
                psi_out = 2.0 * psi_cell - psi_face
                if psi_out < 0.0 and do_fixup:
                    # Fixup: denominator Σ_eff only, numerator |μ|·ψ_face/Δx
                    psi_cell = (rhs[m, k] + mu_abs * psi_face / dx) / sig_eff[k]
                    psi_out = 0.0
                psi[m, k] = psi_cell
                psi_face = psi_out
        else:
            # μ > 0: sweep left to right
            for k in range(n_cells):
                dx = dx_arr[k]
                two_mu_dx = 2.0 * mu_abs / dx
                psi_cell = (rhs[m, k] + two_mu_dx * psi_face) / (sig_eff[k] + two_mu_dx)
                psi_out = 2.0 * psi_cell - psi_face
                if psi_out < 0.0 and do_fixup:
                    psi_cell = (rhs[m, k] + mu_abs * psi_face / dx) / sig_eff[k]
                    psi_out = 0.0
                psi[m, k] = psi_cell
                psi_face = psi_out


# ── Geometry-specific: spherical Lewis-Miller sweep ──────────────────

@njit(cache=False)
def _compute_alpha(mu, w, n_mu):
    """Lewis-Miller angular redistribution coefficients.

    α₁ = 0,  αⱼ₊₁ = αⱼ − μⱼ·wⱼ,  α_{N+1} = 0.
    Returns alpha of size (n_mu + 1,).
    """
    alpha = np.zeros(n_mu + 1)
    for j in range(n_mu):
        alpha[j + 1] = alpha[j] - mu[j] * w[j]
    alpha[n_mu] = 0.0
    return alpha


@njit(cache=False)
def _sweep_sphere(psi, rhs, q_total, mu, w, alpha, r_edges,
                  vol, surf, sig_eff, n_mu, n_cells, do_fixup,
                  psi_start_old, psi_start_current, psi_mu_half,
                  inv_v_dt):
    """Spherical Lewis-Miller 3-stage sweep.

    Starting direction (μ = −1): inward, uses q_total[0,k] (most negative
        ordinate source), matching Fortran q(1,k).
    μ < 0 ordinates: inward with angular redistribution.
    μ > 0 ordinates: outward with center reflection.

    psi_start_old: starting-direction flux from previous TIME STEP (read-only).
    psi_start_current: updated by starting direction each SI iteration (written).
    psi_mu_half: angular-interface flux array of size (n_cells,) (work array).
    """
    n_half = n_mu // 2
    psi12mu_safa = np.zeros(n_mu)

    # ── Starting direction (μ = −1), inward sweep ──
    # Source from most negative ordinate (Fortran: q(1,k)), time lag from
    # dedicated starting-direction ψ (Fortran: psi_old(0,k)).
    psi_face = 0.0
    for k in range(n_cells - 1, -1, -1):
        dr = r_edges[k + 1] - r_edges[k]
        psi_face_old = psi_face
        rhs_start = q_total[0, k] + psi_start_old[k] * inv_v_dt
        psi_cell = (2.0 * psi_face + dr * rhs_start) / (2.0 + dr * sig_eff[k])
        psi_out = 2.0 * psi_cell - psi_face_old
        if psi_out < 0.0 and do_fixup:
            # Fixup: drop 2 → Δr·Σ_eff, incoming coeff 2 → 1
            psi_cell = (psi_face_old + dr * rhs_start) / (dr * sig_eff[k])
            psi_out = 0.0
        psi_start_current[k] = psi_cell
        psi_mu_half[k] = psi_cell
        psi_face = psi_out

    # ── Stage B: μ < 0 ordinates, inward sweep ──
    for j in range(n_half):
        mu_abs = abs(mu[j])
        psi_face = 0.0
        for k in range(n_cells - 1, -1, -1):
            iff = 0
            muff = 0
            psil2n_old = psi_face
            psi_nm12i_old = psi_mu_half[k]
            dA = surf[k + 1] - surf[k]
            # Initialize outgoing values; updated by diamond only when flag==0
            psil2n = psil2n_old
            psi_nm12i_new = psi_nm12i_old

            while True:
                denom = (2.0 * mu_abs * surf[k] * (1.0 - iff)
                         + 2.0 / w[j] * dA * alpha[j + 1] * (1.0 - muff)
                         + vol[k] * sig_eff[k])
                numer = (mu_abs * (surf[k + 1] + surf[k] * (1.0 - iff)) * psil2n_old
                         + 1.0 / w[j] * dA * (alpha[j + 1] * (1.0 - muff) + alpha[j]) * psi_nm12i_old
                         + vol[k] * rhs[j, k])
                psi_cell = numer / denom

                if iff == 0:
                    psil2n = 2.0 * psi_cell - psil2n_old
                if muff == 0:
                    psi_nm12i_new = 2.0 * psi_cell - psi_nm12i_old

                if psil2n < 0.0 and iff == 0 and do_fixup:
                    iff = 1
                    psil2n = 0.0
                    continue
                if psi_nm12i_new < 0.0 and muff == 0 and do_fixup:
                    muff = 1
                    psi_nm12i_new = 0.0
                    continue
                break

            psi[j, k] = psi_cell
            psi_face = psil2n
            psi_mu_half[k] = psi_nm12i_new

        psi12mu_safa[j] = psi_face

    # ── Stage C: μ > 0 ordinates, outward sweep with center reflection ──
    for j in range(n_half, n_mu):
        mirror = n_mu - 1 - j
        psi_face = psi12mu_safa[mirror]
        for k in range(n_cells):
            iff = 0
            muff = 0
            psil2n_old = psi_face
            psi_nm12i_old = psi_mu_half[k]
            dA = surf[k + 1] - surf[k]
            psil2n = psil2n_old
            psi_nm12i_new = psi_nm12i_old

            while True:
                denom = (2.0 * abs(mu[j]) * surf[k + 1] * (1.0 - iff)
                         + 2.0 / w[j] * dA * alpha[j + 1] * (1.0 - muff)
                         + vol[k] * sig_eff[k])
                numer = (abs(mu[j]) * (surf[k + 1] * (1.0 - iff) + surf[k]) * psil2n_old
                         + 1.0 / w[j] * dA * (alpha[j + 1] * (1.0 - muff) + alpha[j]) * psi_nm12i_old
                         + vol[k] * rhs[j, k])
                psi_cell = numer / denom

                if iff == 0:
                    psil2n = 2.0 * psi_cell - psil2n_old
                if muff == 0:
                    psi_nm12i_new = 2.0 * psi_cell - psi_nm12i_old

                if psil2n < 0.0 and iff == 0 and do_fixup:
                    iff = 1
                    psil2n = 0.0
                    continue
                if psi_nm12i_new < 0.0 and muff == 0 and do_fixup:
                    muff = 1
                    psi_nm12i_new = 0.0
                    continue
                break

            psi[j, k] = psi_cell
            psi_face = psil2n
            psi_mu_half[k] = psi_nm12i_new


# ── Unified time/SI driver (JIT) ────────────────────────────────────

@njit(cache=False)
def _run_sn(geometry_mode, n_mu, n_cells, max_steps, mu, w, alpha,
            P, coeffs, lmax,
            sigma_t, sigma_s, v, dt_init, sig_eff,
            dx_arr, r_edges, vol, surf,
            psi_init,
            source_density_arr,
            do_fixup, neg_fixup_mode, max_iters, conv_tol,
            snapshot_times_arr, tracer_cells,
            logging_interval,
            adaptive, dtfac, dtmax, t_final):
    """JIT driver: time-stepping loop with source iteration.

    geometry_mode  : 0 = slab, 1 = sphere.
    neg_fixup_mode : 0 = none, 1 = min_blend (method 1), 2 = scale_clip (method 2).
    adaptive       : if True, dt is adjusted after each completed step
                    using the Fortran error-based formula:
                        δ = max_i |φⁿ⁺¹ − φⁿ| / (φⁿ⁺¹ + φ_min)
                        dt_new = min(dt/δ · dtfac,  1.1·dt,  dtmax)
    """
    # ── Work arrays (preallocated outside time/SI loops) ──
    psi = psi_init.copy()
    psi_time_old = np.empty((n_mu, n_cells))
    q_total = np.empty((n_mu, n_cells))
    q_iso_arr = np.empty(n_cells)
    rhs = np.empty((n_mu, n_cells))
    phi = np.empty(n_cells)
    phi_prev = np.empty(n_cells)
    phi_old_time = np.empty(n_cells)

    # Spherical-only work arrays
    psi_start_old = np.empty(n_cells)
    psi_start_current = np.empty(n_cells)
    psi_mu_half = np.empty(n_cells)

    if geometry_mode == 1:
        for i in range(n_cells):
            psi_start_old[i] = psi[0, i]
            psi_start_current[i] = psi[0, i]

    dt = dt_init
    inv_v_dt = 1.0 / (v * dt)

    # ── Output arrays (grown dynamically if needed) ──
    n_snap = len(snapshot_times_arr)
    phi_snap = np.zeros((n_snap, n_cells))
    avg_mu_snap = np.zeros((n_snap, n_cells))
    avg_mu2_snap = np.zeros((n_snap, n_cells))
    time_snap = np.zeros(n_snap)
    buf_size = max_steps + 1
    step_times = np.zeros(buf_size)
    step_n_iters = np.zeros(buf_size, dtype=np.int64)
    step_total_particles = np.zeros(buf_size)

    n_tracers = len(tracer_cells)
    if n_tracers > 0:
        tracer_density = np.zeros((buf_size, n_tracers))
    else:
        tracer_density = np.zeros((0, 0))

    snap_id = 0
    t = 0.0
    step = 0
    actual_steps = 0

    while True:
        # ── Grow output buffers if needed (adaptive mode) ──
        if step >= buf_size:
            new_size = buf_size * 2
            new_st = np.zeros(new_size)
            new_ni = np.zeros(new_size, dtype=np.int64)
            new_tp = np.zeros(new_size)
            for j in range(buf_size):
                new_st[j] = step_times[j]
                new_ni[j] = step_n_iters[j]
                new_tp[j] = step_total_particles[j]
            step_times = new_st
            step_n_iters = new_ni
            step_total_particles = new_tp
            if n_tracers > 0:
                new_td = np.zeros((new_size, n_tracers))
                for j in range(buf_size):
                    for tr in range(n_tracers):
                        new_td[j, tr] = tracer_density[j, tr]
                tracer_density = new_td
            buf_size = new_size

        # ── Compute φ from current ψ ──
        for i in range(n_cells):
            s = 0.0
            for m in range(n_mu):
                s += w[m] * psi[m, i]
            phi[i] = 0.5 * s

        # ── Per-step diagnostics ──
        step_times[step] = t
        total_p = 0.0
        for i in range(n_cells):
            total_p += phi[i] / v * vol[i]
        step_total_particles[step] = total_p

        # ── Logging ──
        if step % logging_interval == 0:
            if adaptive:
                print("step=", step, " t=", round(t, 6),
                      " dt=", round(dt, 8),
                      " particles=", round(total_p, 6))
            else:
                print("step=", step, "/", max_steps,
                      " particles=", round(total_p, 6))

        # ── Snapshots (time-based) ──
        while snap_id < n_snap and t >= snapshot_times_arr[snap_id] - 1e-10:
            for i in range(n_cells):
                phi_snap[snap_id, i] = phi[i]
                m1 = 0.0
                mu2_raw = 0.0
                for m in range(n_mu):
                    m1 += w[m] * mu[m] * psi[m, i]
                    mu2_raw += w[m] * mu[m] * mu[m] * psi[m, i]
                m1 *= 0.5
                mu2_raw *= 0.5
                if phi[i] > 0.0:
                    avg_mu_snap[snap_id, i] = m1 / phi[i]
                    avg_mu2_snap[snap_id, i] = mu2_raw / phi[i]
            time_snap[snap_id] = t
            snap_id += 1

        # ── Tracers ──
        if n_tracers > 0:
            for tr in range(n_tracers):
                tracer_density[step, tr] = phi[tracer_cells[tr]] / v

        actual_steps = step

        # ── Termination ──
        if adaptive:
            if t >= t_final - 1e-10 * (abs(t_final) + 1e-300):
                break
        else:
            if step >= max_steps:
                break

        # ── Save φ for adaptive dt comparison (pre-SI state) ──
        if adaptive:
            for i in range(n_cells):
                phi_old_time[i] = phi[i]

        # ── Save ψⁿ for backward Euler ──
        for m in range(n_mu):
            for i in range(n_cells):
                psi_time_old[m, i] = psi[m, i]

        # ── Source iteration ──
        n_iters = 0
        for it in range(max_iters):
            n_iters = it + 1

            # φ_prev for convergence (reuse preallocated array)
            for i in range(n_cells):
                s = 0.0
                for m in range(n_mu):
                    s += w[m] * psi[m, i]
                phi_prev[i] = 0.5 * s

            # 1. Compute angular moments
            moments = _compute_moments(psi, w, P, n_mu, n_cells, lmax)

            # 2. Build scattering source
            q_scatter = _build_scattering_source(moments, sigma_s, coeffs,
                                                 P, n_mu, n_cells, lmax)

            # 3. Add external source and build q_iso
            for m in range(n_mu):
                for i in range(n_cells):
                    q_total[m, i] = q_scatter[m, i] + source_density_arr[i]
            for i in range(n_cells):
                q_iso_arr[i] = sigma_s * moments[0, i] + source_density_arr[i]

            # 4. Negative source fixup
            if neg_fixup_mode == 2:
                _negative_source_fixup(q_total, q_iso_arr, w, n_mu, n_cells)
            elif neg_fixup_mode == 1:
                _negative_source_fixup_min_blend(q_total, q_iso_arr, w, n_mu, n_cells)

            # 5. Build RHS = q_total + ψⁿ/(v·Δt)
            for m in range(n_mu):
                for i in range(n_cells):
                    rhs[m, i] = q_total[m, i] + psi_time_old[m, i] * inv_v_dt

            # 6. Sweep
            if geometry_mode == 0:
                _sweep_slab(psi, rhs, mu, dx_arr, sig_eff,
                            n_mu, n_cells, do_fixup)
            else:
                _sweep_sphere(psi, rhs, q_total, mu, w, alpha,
                              r_edges, vol, surf, sig_eff,
                              n_mu, n_cells, do_fixup,
                              psi_start_old, psi_start_current, psi_mu_half,
                              inv_v_dt)

            # 7. Post-sweep positivity check
            for m in range(n_mu):
                for i in range(n_cells):
                    if psi[m, i] < 0.0:
                        print("NEGATIVE psi: step=", step, " iter=", n_iters,
                              " m=", m, " i=", i, " psi=", psi[m, i])
                        assert False, "Negative angular flux after sweep"

            # 8. Convergence check
            phi_max = 0.0
            for i in range(n_cells):
                s = 0.0
                for m in range(n_mu):
                    s += w[m] * psi[m, i]
                phi[i] = 0.5 * s
                if phi[i] > phi_max:
                    phi_max = phi[i]

            phi_min = _FAC_MIN * phi_max + 1e-300
            err = 0.0
            for i in range(n_cells):
                rel = abs(phi[i] - phi_prev[i]) / (phi[i] + phi_min)
                if rel > err:
                    err = rel

            if err < conv_tol:
                break

            if n_iters == 100:
                print("WARNING: SI not converged after 100 iters, step=",
                      step, " err=", err, " tol=", conv_tol)

        step_n_iters[step] = n_iters

        # After SI convergence: update spherical starting direction
        if geometry_mode == 1:
            for i in range(n_cells):
                psi_start_old[i] = psi_start_current[i]

        # ── Advance time ──
        t += dt
        step += 1

        # ── Adaptive dt update (Fortran error-based formula) ──
        if adaptive:
            phi_max_a = 0.0
            for i in range(n_cells):
                if phi[i] > phi_max_a:
                    phi_max_a = phi[i]
            phi_min_a = _FAC_MIN * phi_max_a + 1e-300

            deltt = 1e-80
            for i in range(n_cells):
                change = abs(phi[i] - phi_old_time[i]) / (phi[i] + phi_min_a)
                if change > deltt:
                    deltt = change

            dt_proposed = dt / deltt * dtfac
            dt = min(dt_proposed, 1.1 * dt)
            dt = min(dt, dtmax)

            inv_v_dt = 1.0 / (v * dt)
            for i in range(n_cells):
                sig_eff[i] = sigma_t + inv_v_dt

    # ── Trim output arrays to actual size ──
    n_out = actual_steps + 1
    return (phi_snap, avg_mu_snap, avg_mu2_snap, time_snap,
            step_times[:n_out], step_n_iters[:n_out],
            step_total_particles[:n_out],
            tracer_density[:n_out] if n_tracers > 0 else tracer_density)


# ── JIT warmup tracking ─────────────────────────────────────────────

_warmup_done = {"sn": False}


# ── Public API ───────────────────────────────────────────────────────

def run_sn(geometry, x_edges, sigma_t, sigma_s, v, dt,
           t_final, n0, n_mu, *,
           snapshot_times=None,
           source_density=None,
           logging_interval=100,
           scatter_model="isotropic",
           g=0.0,
           lmax=None,
           scatter_fl=None,
           max_iters=5000,
           conv_tol=1e-10,
           fixup=True,
           negative_source_fixup="scale_clip",
           adaptive_dt=True,
           dtfac=0.1,
           dtmax=None,
           tracer_positions=None):
    """Solve time-dependent Sn transport in slab or spherical geometry.

    Parameters
    ----------
    geometry : str
        "slab" or "sphere".
    x_edges : array (n_cells+1,)
        Spatial cell edges [length].
        Slab: x-coordinates.  Sphere: radial shell boundaries (r ≥ 0).
    sigma_t, sigma_s : float
        Total / scattering cross sections Σₜ, Σₛ [1/length].
    v : float
        Particle speed [length/time].
    dt : float
        Time step [time].  Fixed step when adaptive_dt=False.
        Initial step when adaptive_dt=True.
    t_final : float
        Total simulation time [time].
    n0 : array (n_cells,)
        Initial particle density (isotropic).
        Initial scalar flux: φ(xⱼ, 0) = v · n0[j].
    n_mu : int
        Number of angular ordinates (even, ≥ 2).

    Keyword-only
    ------------
    snapshot_times : array or None
        Physical times at which to record flux snapshots.
        Default: [t_final].
    source_density : array (n_cells,) or None
        Volumetric source rate S(x).  Default: None (no source).
    logging_interval : int
        Screen-output interval (steps).  Default: 100.
    scatter_model : str
        "isotropic" or "hg".  Default: "isotropic".
    g : float
        HG asymmetry parameter, −1 < g < 1.  Default: 0.0.
    lmax : int or None
        Max Legendre order for anisotropic scattering.
        Required when scatter_model="hg" or scatter_fl is given.
        Default: None (isotropic → lmax=0 automatically).
    scatter_fl : array (lmax+1,) or None
        Legendre coefficients fₗ of the scattering kernel f(μ).
        Must have length lmax+1;  f₀ must be 1.0.
        Both scatter_fl and lmax are required together.
        Cannot be combined with scatter_model="hg".
    max_iters : int
        Max source iterations per time step.  Default: 5000.
        Warning printed if SI exceeds 100 iterations without converging.
    conv_tol : float
        SI convergence tolerance.  Default: 1e-10.
    fixup : bool
        Diamond-difference edge-flux fixup.  Default: True.
    negative_source_fixup : str
        Negative angular-source repair scheme.  Default: "scale_clip".
        "scale_clip" : Fortran anisotropic_method=2 (sn1_spherical.f90:576-589).
            fac = q_iso/(q_iso − S_neg),  q_iso *= fac,
            q(μ) = max(0, q_iso + (q(μ) − q_iso/(fac+ε))·fac).
        "min_blend"  : Fortran anisotropic_method=1 (sn1_spherical.f90:567-584).
            fac = −q_iso/(src_neg − q_iso),
            q(μ) = q_iso + (q(μ) − q_iso)·fac.  Linear blend, no clipping.
        "none"       : No negative source repair.
    adaptive_dt : bool
        Enable error-based adaptive timestep control.  Default: True.
        When True, dt is adjusted after each step using:
            δ = max_i |φⁿ⁺¹ − φⁿ| / (φⁿ⁺¹ + φ_min)
            dt_new = min(dt/δ · dtfac,  1.1·dt,  dtmax)
    dtfac : float
        Adaptive scaling factor.  Default: 0.1 (Fortran default).
    dtmax : float or None
        Maximum allowed dt (hard ceiling).  Required when adaptive_dt=True.
    tracer_positions : array or None
        Positions at which to record density at every step.

    Returns
    -------
    phi_snap            : (n_snap, n_cells) scalar flux snapshots
    avg_mu_snap         : (n_snap, n_cells) <mu> = M1/phi snapshots
    avg_mu2_snap        : (n_snap, n_cells) <mu^2> = M2_raw/phi snapshots
    time_snap           : (n_snap,) snapshot times
    step_times          : (n_out,) time at each step
    step_n_iters        : (n_out,) SI iterations per step
    step_total_particles: (n_out,) total particle count
    tracer_density      : (n_out, n_tracers) density at tracers
    """
    # ── Validate geometry ──
    if geometry not in ("slab", "sphere"):
        raise ValueError(f"Unknown geometry '{geometry}'. Use 'slab' or 'sphere'.")

    # ── Coerce arrays ──
    x_edges = np.asarray(x_edges, dtype=np.float64)
    n0 = np.asarray(n0, dtype=np.float64)
    n_cells = len(x_edges) - 1

    # ── Validate inputs ──
    if np.any(np.diff(x_edges) <= 0):
        raise ValueError("x_edges must be strictly increasing.")
    if not np.allclose(np.diff(x_edges), x_edges[1] - x_edges[0]):
        raise ValueError("Non-uniform grids are not supported in v1.")
    if sigma_t < 0 or sigma_s < 0:
        raise ValueError("Cross sections must be non-negative.")
    if sigma_s > sigma_t:
        raise ValueError("sigma_s must be <= sigma_t.")
    if v <= 0:
        raise ValueError("v must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_final < 0:
        raise ValueError("t_final must be non-negative.")
    if len(n0) != n_cells:
        raise ValueError(f"len(n0)={len(n0)} != n_cells={n_cells}.")
    if geometry == "sphere" and x_edges[0] < 0:
        raise ValueError("Sphere geometry requires x_edges[0] >= 0.")

    # ── Validate scatter model ──
    if scatter_model not in ("isotropic", "hg"):
        raise ValueError("scatter_model must be 'isotropic' or 'hg'.")
    if not (-1.0 < g < 1.0):
        raise ValueError("HG parameter g must satisfy -1 < g < 1.")
    # ── Validate adaptive dt parameters ──
    if adaptive_dt:
        if dtmax is None:
            raise ValueError("dtmax is required when adaptive_dt=True.")
        if dtmax <= 0:
            raise ValueError("dtmax must be positive.")
        if dtfac <= 0:
            raise ValueError("dtfac must be positive.")

    if negative_source_fixup not in ("scale_clip", "min_blend", "none"):
        raise ValueError(
            "negative_source_fixup must be 'scale_clip', 'min_blend', or 'none'.")

    # ── Defaults ──
    if snapshot_times is None:
        snapshot_times = np.array([t_final])
    else:
        snapshot_times = np.asarray(snapshot_times, dtype=np.float64)

    if source_density is None:
        source_density_arr = np.zeros(n_cells, dtype=np.float64)
    else:
        source_density_arr = np.asarray(source_density, dtype=np.float64)
        if len(source_density_arr) != n_cells:
            raise ValueError(
                f"len(source_density)={len(source_density_arr)} != n_cells={n_cells}.")

    # ── Resolve scattering coefficients ──
    if scatter_fl is not None:
        if scatter_model == "hg":
            raise ValueError(
                "Cannot use both scatter_model='hg' and scatter_fl. "
                "Use scatter_model='isotropic' (default) when providing scatter_fl.")
        coeffs = np.asarray(scatter_fl, dtype=np.float64)
        if not np.isclose(coeffs[0], 1.0):
            raise ValueError("scatter_fl[0] must be ≈ 1.0 (f_0 = 1).")
        if lmax is None:
            raise ValueError("lmax is required when scatter_fl is given.")
        if len(coeffs) != lmax + 1:
            raise ValueError(
                f"len(scatter_fl)={len(coeffs)} must equal lmax+1={lmax + 1}.")
        coeff_source = "scatter_fl"
    elif scatter_model == "isotropic":
        coeffs = np.array([1.0])
        lmax = 0
        coeff_source = "isotropic"
    else:
        if lmax is None:
            raise ValueError(
                "lmax is required for anisotropic scattering "
                "(scatter_model='hg').")
        coeffs = np.array([g ** l for l in range(lmax + 1)])
        coeff_source = "hg"

    if lmax >= n_mu:
        raise ValueError(f"lmax={lmax} must be < n_mu={n_mu}.")

    # ── Quadrature ──
    mu, w = _make_quadrature(n_mu)

    # ── Legendre table ──
    P = _legendre_table(mu, lmax)

    # ── Geometry mode ──
    geometry_mode = 0 if geometry == "slab" else 1

    # ── Cell widths, volumes, surfaces ──
    dx_arr = np.diff(x_edges)
    if geometry == "slab":
        vol = dx_arr.copy()
        surf = np.zeros(n_cells + 1)
        r_edges = x_edges.copy()
    else:
        vol = _FOUR_PI_OVER_THREE * (x_edges[1:] ** 3 - x_edges[:-1] ** 3)
        surf = 4.0 * np.pi * x_edges ** 2
        r_edges = x_edges.copy()

    # ── Effective cross section ──
    sig_eff = np.full(n_cells, sigma_t + 1.0 / (v * dt))

    # ── Alpha coefficients (spherical) ──
    alpha = _compute_alpha(mu, w, n_mu)

    # ── Time stepping ──
    n_steps = int(round(t_final / dt))
    if adaptive_dt:
        effective_dtmax = dtmax
        max_steps = int(np.ceil(t_final / effective_dtmax)) * 3 + 1000
    else:
        max_steps = n_steps

    snapshot_times_sorted = np.sort(snapshot_times)

    # ── Tracer preprocessing ──
    if tracer_positions is not None:
        tracer_positions = np.asarray(tracer_positions, dtype=np.float64)
        dx = x_edges[1] - x_edges[0]
        tracer_cell_indices = np.empty(len(tracer_positions), dtype=np.int64)
        for k, xp in enumerate(tracer_positions):
            if xp < x_edges[0] or xp >= x_edges[-1]:
                raise ValueError(
                    f"tracer_positions[{k}]={xp} is outside the grid "
                    f"[{x_edges[0]}, {x_edges[-1]}).")
            tracer_cell_indices[k] = int((xp - x_edges[0]) / dx)
            if tracer_cell_indices[k] >= n_cells:
                tracer_cell_indices[k] = n_cells - 1
    else:
        tracer_cell_indices = np.zeros(0, dtype=np.int64)

    # ── Initial angular flux: isotropic ψ(μ, x) = v·n0(x) ──
    psi_init = np.empty((n_mu, n_cells))
    for m in range(n_mu):
        psi_init[m, :] = v * n0

    # ── Fixup flags ──
    do_fixup = fixup
    neg_fixup_mode = {"none": 0, "min_blend": 1, "scale_clip": 2}[negative_source_fixup]

    # ── JIT warmup ──
    if not _warmup_done["sn"]:
        print("Compiling Sn JIT kernels...")
        _e = np.linspace(0.0, 1.0, 3)
        _dx = np.diff(_e)
        _v = np.array([0.5, 0.5])
        _snap_t = np.array([0.2])
        _tc = np.zeros(0, dtype=np.int64)
        _se = np.full(2, 2.0)
        _coeffs_w = np.array([1.0])
        _mu_w, _w_w = _make_quadrature(4)
        _P_w = _legendre_table(_mu_w, 0)
        _alpha_w = _compute_alpha(_mu_w, _w_w, 4)
        _psi_w = np.ones((4, 2))
        _src_w = np.zeros(2)
        _surf_w = np.zeros(3)
        if geometry_mode == 1:
            _surf_w = 4.0 * np.pi * _e ** 2
        t_compile = _time.time()
        _run_sn(geometry_mode, 4, 2, 2, _mu_w, _w_w, _alpha_w,
                _P_w, _coeffs_w, 0,
                1.0, 0.5, 1.0, 0.1, _se,
                _dx, _e, _v, _surf_w,
                _psi_w, _src_w,
                True, 0, 5, 1e-8,
                _snap_t, _tc, 100,
                False, 0.1, 1.0, 0.2)
        print(f"Compilation time: {_time.time() - t_compile:.1f} s")
        _warmup_done["sn"] = True

    # ── Log solver parameters ──
    print(f"--- sn_solver [{geometry}] ---")
    print(f"  sigma_t={sigma_t:.6g}")
    print(f"  sigma_s={sigma_s:.6g}")
    if sigma_t > 0:
        print(f"  c={sigma_s / sigma_t:.6g}")
    print(f"  v={v:.6g}")
    print(f"  scatter_model={scatter_model} (coeffs from {coeff_source})")
    print(f"  g={g:.6g},  lmax={lmax}")
    print(f"  n_mu={n_mu}")
    print(f"  cells={n_cells}")
    if adaptive_dt:
        print(f"  dt_init={dt:.6g}  (adaptive: dtfac={dtfac}, dtmax={dtmax})")
    else:
        print(f"  dt={dt:.6g}")
    print(f"  t_final={t_final:.6g}")
    if not adaptive_dt:
        print(f"  time steps={n_steps}")
    print(f"  snapshots={len(snapshot_times_sorted)}")
    print(f"  max_iters={max_iters},  conv_tol={conv_tol:.2e}")
    print(f"  fixup={fixup},  neg_fixup={negative_source_fixup}")
    N_total = np.sum(n0 * vol)
    print(f"  initial particles (N0)={N_total:.6g}")
    if len(tracer_cell_indices) > 0:
        print(f"  tracers={len(tracer_cell_indices)}")

    # ── Run ──
    _dtmax_val = dtmax if dtmax is not None else dt
    t0 = _time.time()
    result = _run_sn(
        geometry_mode, n_mu, n_cells, max_steps, mu, w, alpha,
        P, coeffs, lmax,
        sigma_t, sigma_s, v, dt, sig_eff,
        dx_arr, r_edges, vol, surf,
        psi_init,
        source_density_arr,
        do_fixup, neg_fixup_mode, max_iters, conv_tol,
        snapshot_times_sorted, tracer_cell_indices,
        logging_interval,
        adaptive_dt, dtfac, _dtmax_val, t_final)
    print(f"Simulation runtime: {_time.time() - t0:.2f} s")
    return result
