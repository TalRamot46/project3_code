"""Smoke tests for boundary condition application."""

import numpy as np
import pytest

from project_3.hydro_sim.core.boundary import apply_velocity_bc, apply_pressure_bc


# ---------------------------------------------------------------------------
# apply_velocity_bc
# ---------------------------------------------------------------------------

def test_velocity_outflow_left():
    """Outflow BC copies u[1] into u[0]."""
    u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u_bc = apply_velocity_bc(u, "outflow", "none", t=0.0)
    assert u_bc[0] == u[1]
    # Interior unchanged
    np.testing.assert_array_equal(u_bc[1:], u[1:])


def test_velocity_outflow_right():
    """Outflow BC copies u[-2] into u[-1]."""
    u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u_bc = apply_velocity_bc(u, "none", "outflow", t=0.0)
    assert u_bc[-1] == u[-2]


def test_velocity_none_leaves_unchanged():
    """'none' BC should not modify the array."""
    u = np.array([10.0, 20.0, 30.0])
    u_bc = apply_velocity_bc(u, "none", "none", t=0.0)
    np.testing.assert_array_equal(u_bc, u)


def test_velocity_does_not_mutate_input():
    """apply_velocity_bc should return a copy, not modify in-place."""
    u = np.array([1.0, 2.0, 3.0])
    u_orig = u.copy()
    _ = apply_velocity_bc(u, "outflow", "outflow", t=0.0)
    np.testing.assert_array_equal(u, u_orig)


def test_velocity_bad_left_raises():
    """Unknown left BC should raise ValueError."""
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        apply_velocity_bc(u, "invalid_bc", "none", t=0.0)


def test_velocity_bad_right_raises():
    """Unknown right BC should raise ValueError."""
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        apply_velocity_bc(u, "none", "invalid_bc", t=0.0)


# ---------------------------------------------------------------------------
# apply_pressure_bc
# ---------------------------------------------------------------------------

def test_pressure_outflow_left():
    """Outflow BC copies p[1] into p[0]."""
    p = np.array([0.0, 5.0, 10.0, 15.0])
    p_bc = apply_pressure_bc(p, "outflow", "none", t=0.0)
    assert p_bc[0] == p[1]


def test_pressure_outflow_right():
    """Outflow BC copies p[-2] into p[-1]."""
    p = np.array([0.0, 5.0, 10.0, 15.0])
    p_bc = apply_pressure_bc(p, "none", "outflow", t=0.0)
    assert p_bc[-1] == p[-2]


def test_pressure_dict_left():
    """Dict-style pressure drive sets p[0] to prescribed value."""
    p = np.array([0.0, 5.0, 10.0])
    bc_left = {"type": "pressure", "p": 99.0}
    p_bc = apply_pressure_bc(p, bc_left, "none", t=0.0)
    assert p_bc[0] == 99.0


def test_pressure_does_not_mutate_input():
    """apply_pressure_bc should return a copy, not modify in-place."""
    p = np.array([1.0, 2.0, 3.0])
    p_orig = p.copy()
    _ = apply_pressure_bc(p, "outflow", "outflow", t=0.0)
    np.testing.assert_array_equal(p, p_orig)
