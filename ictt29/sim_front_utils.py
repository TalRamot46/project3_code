# ictt29/sim_front_utils.py
"""
Shared utility functions for detecting simulation fronts and computing
analytic-fit front trajectories.

Imported by both ``plot_xt.py`` and ``full_fitting_eulerian.py`` so that
the same detection logic is used everywhere.

Public API
----------
find_shock_front(rho, m_coordinate, *, rho_unshocked, gamma,
                 Hugoniot_threshold=0.9)
    Locate the rightmost strongly-compressed cell (shock front).

detect_sim_ablation_boundary(x_sim)
    Return the time-series of the ablation boundary position (left edge of
    the first cell) in cm.

detect_sim_ablation_front(history_rho, history_m, x_sim, *, rho_unshocked,
                          gamma, smooth_window=5)
    Detect the ablation / heat-wave front as the left edge of the shocked
    region by finding the leftmost cell whose density is above the Hugoniot
    compression threshold.


detect_sim_shock_front_trajectory(history_rho, history_m, x_sim, *,
                                  rho_unshocked, gamma,
                                  smooth_window=5, extrap_t_ns=0.002,
                                  extrap_times=None)
    Return a 1-D array of shock-front x-positions (cm) for every row in
    history_rho / x_sim.  Early-time frames where the shock is not yet
    detectable are extrapolated from a log-log power-law fit to the
    later-time detections.

compute_fit_front_trajectories(times_model, ablation_solver, sub_params,
                               shock_params)
    Evaluate the four analytic-fit front positions
    (boundary, ablation front, piston, shock) over a 1-D time array,
    returning a dict of numpy arrays in cm.

_rolling_mean(a, window)
    Causal convolution smoothing (internal helper, exported for reuse).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "find_shock_front",
    "detect_sim_ablation_boundary",
    "detect_sim_ablation_front",
    "detect_sim_shock_front_trajectory",
    "compute_fit_front_trajectories",
    "_rolling_mean",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Symmetric (``mode='same'``) boxcar average over *window* samples."""
    if window <= 1:
        return np.asarray(a, dtype=float)
    w = int(max(1, window))
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(np.asarray(a, dtype=float), kernel, mode="same")


# ---------------------------------------------------------------------------
# Shock-front detection
# ---------------------------------------------------------------------------

def find_shock_front(
    rho: np.ndarray,
    m_coordinate: np.ndarray,
    *,
    rho_unshocked: float,
    gamma: float,
    Hugoniot_threshold: float = 0.9,
) -> tuple[int, float]:
    """Detect the shock front as the right edge of the compressed region.

    The Rankine-Hugoniot strong-shock limit gives a maximum compression ratio
    of ``(gamma+1)/(gamma-1)``.  We search for the **rightmost** cell whose
    density exceeds ``Hugoniot_threshold`` times that maximum.  Using 0.9
    (90 %) gives a tight bracket that reliably identifies the shock without
    triggering on lightly-compressed upstream cells.

    Parameters
    ----------
    rho:
        Cell-centred density array (already smoothed if desired).
    m_coordinate:
        Lagrangian mass-coordinate array (same length as *rho*).
    rho_unshocked:
        Ambient (pre-shock) density in g/cm³.
    gamma:
        Adiabatic index of the gas.
    Hugoniot_threshold:
        Fraction of the maximum Hugoniot compression to use as the detection
        threshold.  Default is 0.9.

    Returns
    -------
    (i, rho_i) where *i* is the cell index of the shock front and *rho_i*
    is the density there.  Returns ``(-1, nan)`` if no shock is found.
    """
    rho_arr = np.asarray(rho, dtype=float)
    m_arr   = np.asarray(m_coordinate, dtype=float)
    if rho_arr.size < 3:
        return -1, float("nan")

    # Maximum strong-shock compression ratio scaled by the threshold fraction.
    rho_thresh = float(rho_unshocked) * Hugoniot_threshold * (gamma + 1.0) / (gamma - 1.0)
    compressed = rho_arr > rho_thresh
    compressed_idx = np.flatnonzero(compressed)
    if compressed_idx.size > 0:
        # Rightmost cell above threshold = right edge of the compressed region.
        i = int(compressed_idx[-1])
        return i, float(rho_arr[i])

    # Fallback: steepest negative density gradient in Lagrangian mass space.
    drho_dm = np.gradient(rho_arr, m_arr)
    i_steep = int(np.argmin(drho_dm))
    if np.isfinite(drho_dm[i_steep]):
        return i_steep, float(rho_arr[i_steep])
    return -1, float("nan")


# ---------------------------------------------------------------------------
# Ablation boundary (left domain edge)
# ---------------------------------------------------------------------------

def detect_sim_ablation_boundary(x_sim: np.ndarray) -> np.ndarray:
    """Return the time-series of the ablation boundary position in cm.

    The ablation boundary is the left face of the first Lagrangian cell.
    Since ``history.x`` stores cell *centres*, the left edge of cell 0 is
    estimated as::

        x_boundary[k] = x_sim[k, 0] - 0.5 * (x_sim[k, 1] - x_sim[k, 0])

    Parameters
    ----------
    x_sim:
        Cell-centre positions, shape ``(K, Ncells)``.

    Returns
    -------
    1-D array of shape ``(K,)`` in cm.
    """
    x_sim = np.asarray(x_sim, dtype=float)
    dx0 = x_sim[:, 1] - x_sim[:, 0]          # width of the leftmost cell
    return x_sim[:, 0] - 0.5 * dx0


# ---------------------------------------------------------------------------
# Ablation / heat-wave front detection
# ---------------------------------------------------------------------------

def detect_sim_ablation_front(
    history_rho: np.ndarray,
    history_m: np.ndarray,
    x_sim: np.ndarray,
    *,
    rho_unshocked: float,
    gamma: float,
    smooth_window: int = 5,
) -> np.ndarray:
    """Detect the ablation / heat-wave front from the simulation density.

    The ablation front (heat-wave front) is the *left* boundary of the
    shocked / compressed region.  We find the leftmost cell that is still
    strongly compressed (above the Hugoniot threshold), which marks where
    the heat wave has pushed mass into the shock.

    Parameters
    ----------
    history_rho, history_m, x_sim:
        Arrays of shape ``(K, Ncells)`` containing the cell-centred density,
        cumulative mass coordinate, and cell-centre positions for each
        snapshot.
    rho_unshocked:
        Ambient density in g/cm³.
    gamma:
        Adiabatic index.
    smooth_window:
        Boxcar half-width used before detection.

    Returns
    -------
    1-D array ``x_ablation_front`` of shape ``(K,)`` in cm.
    ``nan`` for frames where no compressed region is found.
    """
    history_rho = np.asarray(history_rho, dtype=float)
    history_m   = np.asarray(history_m,   dtype=float)
    x_sim       = np.asarray(x_sim,       dtype=float)
    K = history_rho.shape[0]

    rho_thresh = (
        float(rho_unshocked)
        * 0.5   # same Hugoniot threshold fraction as find_shock_front
        * (gamma + 1.0) / (gamma - 1.0)
    )

    x_ablation_front = np.full(K, np.nan, dtype=float)
    # for k in range(1, K):
    #     rhok = _rolling_mean(history_rho[k], smooth_window)
    #     compressed_idx = np.flatnonzero(rhok > rho_thresh)
    #     if compressed_idx.size > 0:
    #         # Leftmost strongly-compressed cell = ablation / heat-wave front.
    #         i_left = int(compressed_idx[0])
    #         x_ablation_front[k] = float(x_sim[k, i_left])

    # Find the leftmost drop in density gradient
    rho_grad = np.gradient(history_rho, axis=1)
    Ncells = history_rho.shape[1]
    # return the leftmost local maximum that is above its 3 closest neighbors.
    for k in range(3, K - 3):
        for i in range(3, Ncells - 3):
            if rho_grad[k, i] > rho_grad[k, i - 1] and \
               rho_grad[k, i] > rho_grad[k, i + 1] and \
               rho_grad[k, i] > rho_grad[k, i - 2] and \
               rho_grad[k, i] > rho_grad[k, i + 2] and \
               rho_grad[k, i] > rho_grad[k, i - 3] and \
               rho_grad[k, i] > rho_grad[k, i + 3]:
                x_ablation_front[k] = float(x_sim[k, i])
                break
    return x_ablation_front

# ---------------------------------------------------------------------------
# Shock front trajectory (vectorised over all timesteps)
# ---------------------------------------------------------------------------

def detect_sim_shock_front_trajectory(
    history_rho: np.ndarray,
    history_m: np.ndarray,
    x_sim: np.ndarray,
    *,
    rho_unshocked: float,
    gamma: float,
    smooth_window: int = 5,
    extrap_t_ns: float = 0.1,
    extrap_times: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the shock-front x-position for every snapshot in the history.

    Parameters
    ----------
    history_rho, history_m, x_sim:
        Arrays of shape ``(K, Ncells)``.  These should already be the
        *downsampled* arrays (same indexing as the time axis you will plot).
    rho_unshocked, gamma:
        EOS parameters passed to ``find_shock_front``.
    smooth_window:
        Boxcar width for density smoothing before detection.
    extrap_t_ns:
        Early-time cutoff in nanoseconds.  Frames at ``t < extrap_t_ns`` [ns]
        that return no shock detection are filled by a log-log power-law
        extrapolation fitted to the detected frames at ``t >= extrap_t_ns``.
    extrap_times:
        Physical times in **seconds** corresponding to the rows of
        ``history_rho``.  Required when ``extrap_t_ns > 0``.  If ``None``,
        extrapolation is skipped.

    Returns
    -------
    1-D array ``x_shock`` of shape ``(K,)`` in cm.  Index 0 is always
    ``nan`` (the t=0 frame has no shock).
    """
    history_rho = np.asarray(history_rho, dtype=float)
    history_m   = np.asarray(history_m,   dtype=float)
    x_sim       = np.asarray(x_sim,       dtype=float)
    K = history_rho.shape[0]

    x_shock = np.full(K, np.nan, dtype=float)
    for k in range(1, K):
        rhok = _rolling_mean(history_rho[k], smooth_window)
        ishock, _ = find_shock_front(
            rhok,
            history_m[k],
            rho_unshocked=rho_unshocked,
            gamma=gamma,
        )
        if ishock >= 1:
            x_shock[k] = float(x_sim[k, ishock])

    # Log-log power-law extrapolation for early times where shock is too weak.
    if extrap_times is not None and extrap_t_ns > 0.0:
        t = np.asarray(extrap_times, dtype=float)
        later_mask  = (t * 1e9 >= extrap_t_ns) & np.isfinite(x_shock)
        early_mask  = (t * 1e9 <  extrap_t_ns)
        if later_mask.sum() >= 2:
            log_t = np.log(t[later_mask] + 1e-20)
            log_x = np.log(x_shock[later_mask])
            slope, intercept = np.polyfit(log_t, log_x, 1)
            for k in np.where(early_mask)[0]:
                x_shock[k] = np.exp(slope * np.log(t[k] + 1e-20) + intercept)

    return x_shock


# ---------------------------------------------------------------------------
# Analytic-fit front trajectories
# ---------------------------------------------------------------------------

def compute_fit_front_trajectories(
    times_model: np.ndarray,
    ablation_solver,
    sub_params,
    shock_params,
) -> dict[str, np.ndarray]:
    """Evaluate the four analytic-fit front positions over a time array.

    Extracted from ``full_fitting_eulerian.plot_front_trajectories_eulerian``
    (lines 744–764) so the same formula can be reused in ``plot_xt.py``.

    The four fronts are:

    * ``boundary``       — ablation boundary (left edge, where T = T_drive)
    * ``ablation_front`` — heat-wave / ablation front (interface between
                           subsonic and shocked regions)
    * ``piston``         — shock piston (inner edge of shocked region)
    * ``shock``          — leading shock front

    Parameters
    ----------
    times_model:
        1-D array of physical times in **seconds**.
    ablation_solver:
        A fully initialised ``AblationSolver`` instance.
    sub_params:
        Subsonic-fit parameter object returned by ``perform_subsonic_fitting``.
    shock_params:
        Shock-fit parameter object returned by ``perform_shock_fitting``.

    Returns
    -------
    dict with keys ``"boundary"``, ``"ablation_front"``, ``"piston"``,
    ``"shock"``, each a 1-D ``np.ndarray`` of positions in **cm** with the
    same length as *times_model*.
    """
    # Lazy import to avoid hard dependency when the module is used without
    # the fitting pipeline (e.g., in plot_xt.py which does not always fit).
    try:
        from shock_fitting import fit_by_params as shock_fit_by_params  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "compute_fit_front_trajectories requires 'shock_fitting' to be "
            "importable (add the ictt29 directory to sys.path)."
        ) from exc

    hs = ablation_solver.heat_solver
    ss = ablation_solver.shock_solver
    q1 = 1.0 - ss.omega
    q2 = (2.0 - ss.omega) / (ss.tau + 2.0)

    times_model = np.asarray(times_model, dtype=float)
    n = times_model.size

    x_boundary       = np.full(n, np.nan, dtype=float)
    x_ablation_front = np.full(n, np.nan, dtype=float)
    x_piston         = np.full(n, np.nan, dtype=float)
    x_shock          = np.full(n, np.nan, dtype=float)

    for i, t in enumerate(times_model):
        t_val = max(float(t), 1e-18)

        # Ablated mass at this time
        m_f    = hs.ablated_mass(time=t_val)
        # Mass-to-xi conversion factor for the shock solver
        xsi_mf = m_f * ss.xsi_over_m(time=t_val)
        # Normalised coordinate of the ablation front inside the shock profile
        y_mf   = xsi_mf / ss.xsi_s

        # Evaluate the shock fit at the ablation-front location
        _, _, U_fit_mf, rho_fit_mf = shock_fit_by_params(
            np.array([y_mf]), shock_params
        )
        V_fit_mf = 1.0 / rho_fit_mf[0]

        # Position scale factor (dimensional time factor from shock similarity)
        pos_scale = ss._position_temporal_factor(time=t_val)

        # Ablation-front (heat-wave interface) position in the fit
        x_af = pos_scale * (q1 * xsi_mf * V_fit_mf + q2 * U_fit_mf[0])
        x_ablation_front[i] = x_af

        # Piston position (q2 * U0 in the shock similarity variable)
        x_piston[i] = pos_scale * q2 * ss.U0

        # Ablation boundary: heat-solver boundary shifted to lab frame
        x_boundary[i] = hs.boundary_position(time=t_val) + x_af

        # Shock-front position from the shock solver analytic formula
        x_shock[i] = ss.shock_position(time=t_val)

    return {
        "boundary":       x_boundary,
        "ablation_front": x_ablation_front,
        "piston":         x_piston,
        "shock":          x_shock,
    }
