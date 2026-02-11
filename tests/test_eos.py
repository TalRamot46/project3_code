"""Smoke tests for the equation of state module."""

import numpy as np
import pytest

from project_3.hydro_sim.core.eos import (
    internal_energy_from_prho,
    pressure_ideal_gas,
    sound_speed,
)


# ---------------------------------------------------------------------------
# pressure_ideal_gas
# ---------------------------------------------------------------------------

def test_pressure_ideal_gas_scalar():
    """P = (gamma - 1) * rho * e for known values."""
    rho = np.array([1.0])
    e = np.array([2.5])
    gamma = 1.4
    p = pressure_ideal_gas(rho, e, gamma)
    expected = 0.4 * 1.0 * 2.5  # 1.0
    np.testing.assert_allclose(p, expected)


def test_pressure_ideal_gas_array():
    """Vectorised call returns same-shape array with correct values."""
    rho = np.array([1.0, 2.0, 3.0])
    e = np.array([2.5, 1.0, 0.5])
    gamma = 5.0 / 3.0
    p = pressure_ideal_gas(rho, e, gamma)
    expected = (gamma - 1.0) * rho * e
    np.testing.assert_allclose(p, expected)
    assert p.shape == rho.shape


def test_pressure_zero_energy():
    """Zero internal energy gives zero pressure."""
    rho = np.array([1.0, 10.0])
    e = np.zeros(2)
    p = pressure_ideal_gas(rho, e, 1.4)
    np.testing.assert_allclose(p, 0.0)


# ---------------------------------------------------------------------------
# internal_energy_from_prho  (inverse of pressure_ideal_gas)
# ---------------------------------------------------------------------------

def test_energy_pressure_roundtrip():
    """e -> P -> e should give back the original energy."""
    rho = np.array([1.0, 5.0, 0.1])
    e_orig = np.array([2.5, 1.0, 10.0])
    gamma = 1.4
    p = pressure_ideal_gas(rho, e_orig, gamma)
    e_back = internal_energy_from_prho(p, rho, gamma)
    np.testing.assert_allclose(e_back, e_orig)


# ---------------------------------------------------------------------------
# sound_speed
# ---------------------------------------------------------------------------

def test_sound_speed_positive():
    """Sound speed should be positive and finite for physical inputs."""
    rho = np.array([1.0, 2.0])
    p = np.array([1.0, 4.0])
    gamma = 1.4
    cs = sound_speed(rho, p, gamma)
    assert np.all(cs > 0)
    assert np.all(np.isfinite(cs))


def test_sound_speed_known_value():
    """c = sqrt(gamma * P / rho) for a single known case."""
    rho = np.array([1.0])
    p = np.array([1.0])
    gamma = 1.4
    cs = sound_speed(rho, p, gamma)
    np.testing.assert_allclose(cs, np.sqrt(1.4))
