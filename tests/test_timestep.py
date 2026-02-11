"""Smoke tests for CFL timestep computation."""

import numpy as np
import pytest

from project_3.hydro_sim.core.timestep import (
    compute_dt_acoustic,
    compute_dt_cfl,
    compute_dt_crossing,
    compute_dt_volchange,
    update_dt_relchange,
)
from project_3.hydro_sim.core.state import HydroState


def _make_state(ncells=100, p0=1.0, rho0=1.0, u0=0.0):
    """Helper: create a minimal HydroState for timestep tests."""
    x = np.linspace(0.0, 1.0, ncells + 1)
    u = np.full(ncells + 1, u0)
    rho = np.full(ncells, rho0)
    p = np.full(ncells, p0)
    return HydroState(
        t=0.0, x=x, u=u, a=None,
        V=None, rho=rho, e=None, p=p, q=None, m_cells=None,
    )


# ---------------------------------------------------------------------------
# compute_dt_acoustic
# ---------------------------------------------------------------------------

def test_dt_acoustic_positive_finite():
    """Acoustic timestep should be positive and finite."""
    state = _make_state()
    dt = compute_dt_acoustic(state, gamma=1.4, CFL=0.3)
    assert dt > 0
    assert np.isfinite(dt)


def test_dt_acoustic_scales_with_cfl():
    """Doubling CFL should roughly double the timestep."""
    state = _make_state()
    dt1 = compute_dt_acoustic(state, gamma=1.4, CFL=0.3)
    dt2 = compute_dt_acoustic(state, gamma=1.4, CFL=0.6)
    np.testing.assert_allclose(dt2 / dt1, 2.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# compute_dt_crossing
# ---------------------------------------------------------------------------

def test_dt_crossing_no_collapse():
    """When no cells are collapsing, dt_crossing should be inf."""
    x = np.linspace(0.0, 1.0, 11)
    u = np.zeros(11)  # no motion
    dt = compute_dt_crossing(x, u, CFL=0.3)
    assert dt == np.inf


def test_dt_crossing_collapsing():
    """When cells are collapsing, dt_crossing should be finite and positive."""
    x = np.linspace(0.0, 1.0, 11)
    u = np.linspace(1.0, -1.0, 11)  # converging flow
    dt = compute_dt_crossing(x, u, CFL=0.3)
    assert dt > 0
    assert np.isfinite(dt)


# ---------------------------------------------------------------------------
# compute_dt_volchange
# ---------------------------------------------------------------------------

def test_dt_volchange_positive():
    """Volume-change timestep should be positive and finite."""
    x = np.linspace(0.0, 1.0, 51)
    u = np.linspace(0.0, 0.1, 51)
    dt = compute_dt_volchange(x, u)
    assert dt > 0
    assert np.isfinite(dt)


# ---------------------------------------------------------------------------
# compute_dt_cfl (combined)
# ---------------------------------------------------------------------------

def test_dt_cfl_positive_finite():
    """Combined CFL timestep should be positive and finite."""
    N = 100
    x = np.linspace(0.0, 1.0, N + 1)
    u = np.zeros(N + 1)
    rho = np.ones(N)
    p = np.ones(N)
    dt = compute_dt_cfl(x, u, rho, p, gamma=1.4, CFL=0.3)
    assert dt > 0
    assert np.isfinite(dt)


# ---------------------------------------------------------------------------
# update_dt_relchange (radiation adaptive dt)
# ---------------------------------------------------------------------------

def test_relchange_returns_positive():
    """Adaptive dt should always be positive."""
    N = 50
    E = np.ones(N)
    UR = np.ones(N)
    new_E = E * 1.01
    new_UR = UR * 0.99
    dt_new = update_dt_relchange(1e-10, new_E, E, new_UR, UR)
    assert dt_new > 0


def test_relchange_respects_growth_cap():
    """dt should not grow faster than growth_cap per step."""
    N = 50
    E = np.ones(N)
    UR = np.ones(N)
    # Tiny change => dt wants to grow a lot, but growth_cap limits it
    new_E = E * (1 + 1e-12)
    new_UR = UR * (1 + 1e-12)
    dt_old = 1e-10
    dt_new = update_dt_relchange(dt_old, new_E, E, new_UR, UR, growth_cap=1.1)
    assert dt_new <= 1.1 * dt_old + 1e-30
