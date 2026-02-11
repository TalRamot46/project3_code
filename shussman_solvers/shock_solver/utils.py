from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

def integrate_ode(
    fun, t_span, y0, *,
    rtol=1e-9, atol=1e-9,
    max_step=None,
    dense_output=False,
    # --- new safety knobs ---
    stop_on_nonfinite=True,
    positivity_idx=(),        # e.g. (0,1) to enforce y[0]>0 and y[1]>0
    positivity_eps=-0.01,       # use >0 or >eps
    fail_fast_on_nonfinite_rhs=True,
):
    """
    Wrapper around solve_ivp that can terminate early (MATLAB-style) when
    the integration becomes invalid.

    Parameters
    ----------
    stop_on_nonfinite : if True, add a terminal event that stops when t or y is non-finite
    positivity_idx    : indices in y that must remain > positivity_eps (e.g. V and P)
    fail_fast_on_nonfinite_rhs : if True, convert non-finite RHS evaluations into a hard stop
                                 by raising and returning success=False.
    """

    y0 = np.asarray(y0, dtype=float)
    t0, tf = float(t_span[0]), float(t_span[1])

    # --- Optional fail-fast wrapper around RHS ---
    def wrapped_fun(t, y):
        y = np.asarray(y, dtype=float)

        # If state is already non-finite, don't keep evaluating
        if stop_on_nonfinite and (not np.isfinite(t) or not np.all(np.isfinite(y))):
            # returning NaNs can make the integrator struggle; raising is clearer
            if fail_fast_on_nonfinite_rhs:
                raise FloatingPointError(f"Non-finite state: t={t}, y={y}")
            return np.full_like(y, np.nan)

        # Physical positivity checks
        if positivity_idx:
            for k in positivity_idx:
                if not (y[k] > positivity_eps):
                    if fail_fast_on_nonfinite_rhs:
                        raise FloatingPointError(f"State violated positivity at index {k}: y[{k}]={y[k]}")
                    return np.full_like(y, np.nan)

        f = fun(t, y)
        f = np.asarray(f, dtype=float)

        # If RHS returns NaNs/Infs, stop early
        if stop_on_nonfinite and not np.all(np.isfinite(f)):
            if fail_fast_on_nonfinite_rhs:
                raise FloatingPointError(f"Non-finite RHS: t={t}, y={y}, f={f}")
            return np.full_like(y, np.nan)

        return f

    events = []

    # --- Terminal event: stop on non-finite or positivity violation ---
    if stop_on_nonfinite or positivity_idx:
        def invalid_event(t, y):
            # event is triggered when it crosses zero; return +1 when valid, -1 when invalid
            if stop_on_nonfinite:
                if (not np.isfinite(t)) or (not np.all(np.isfinite(y))):
                    return -1.0

            if positivity_idx:
                for k in positivity_idx:
                    if not (y[k] > positivity_eps):
                        return -1.0

            return 1.0  # valid

        invalid_event.terminal = True
        invalid_event.direction = 0
        events.append(invalid_event)

    kwargs = dict(rtol=rtol, atol=atol, dense_output=dense_output)
    if max_step is not None:
        kwargs["max_step"] = max_step
    if events:
        kwargs["events"] = events

    # --- Run solve_ivp with exception -> convert to "failed solution" ---
    try:
        sol = solve_ivp(wrapped_fun, (t0, tf), y0, **kwargs)
    except FloatingPointError as e:
        # mimic solve_ivp object shape
        class _Sol:
            success = False
            status = -1
            message = f"Aborted: {e}"
            t = np.array([t0], dtype=float)
            y = y0.reshape(-1, 1)
            t_events = None
            y_events = None
            nfev = 0
            njev = 0
            nlu = 0
        return _Sol()

    # If we stopped due to invalid_event, mark as failure (MATLAB-like warning)
    if events and getattr(sol, "t_events", None) is not None and len(sol.t_events) > 0:
        # If any terminal event fired, integration terminated early
        fired = any(te.size > 0 for te in sol.t_events)
        if fired:
            sol.success = False
            sol.message = (sol.message or "") + " (terminated early: invalid state/event)"

    # Also downgrade success if final y contains non-finite values
    if stop_on_nonfinite and (not np.all(np.isfinite(sol.y)) or not np.all(np.isfinite(sol.t))):
        sol.success = False
        sol.message = (sol.message or "") + " (non-finite values in solution arrays)"

    return sol

def trapz(y, x):
    return float(np.trapezoid(y, x))

def ensure_finite(arr: np.ndarray, name: str = "array"):
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"Non-finite values encountered in {name}.")
