"""
Unit test for SupersonicInstantaneousAnalytic solver.
"""

import numpy as np
import scipy.integrate
import pytest
import matplotlib.pyplot as plt
from rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

from menahem_new.supersonic_instantaneous_analytic import (
    SupersonicInstantaneousAnalytic,
    Units,
)


def test_gold_supersonic_instantaneous_solver():
    """Test solver initialization and consistency for Gold parameters."""
    # Au parameters typical for rad-hydro problems
    solver = SupersonicInstantaneousAnalytic(
        g=1.0, 
        alpha=2,
        lambdap=1,
        f=1.0,
        beta=1.6,
        mu=0,
        rho0=19.32,
        omega=0.2,
        T0_HeV=1.0,
    )
    t_ns = 0.5
    t_sec = t_ns * 1e-9  # 0.5 ns
    r_h = solver.heat_front_radius(t_sec)
    assert r_h > 0, "Heat front radius should be positive"

    # Test grid evaluation
    r_grid = np.linspace(1e-12, r_h * 1.5, 1000)
    T = solver.temperature_profile(r_grid, t_sec)
    u = solver.energy_density(r_grid, t_sec)

    # Check edge condition: T = 0 for r >= r_h
    outside_mask = r_grid > r_h
    assert np.all(T[outside_mask] == 0.0), "Temperature beyond r_h should be 0"

    # Check origin drive consistency: T(r->0, t) ~ T0 * t_ns^tau in Kelvin
    T_origin = solver.temperature_profile(1e-6, t_sec)
    expected_T_origin = Units.hev_kelvin * (t_ns ** solver.tau)
    assert np.isclose(T_origin, expected_T_origin, rtol=1e-3), (
        f"T at origin {T_origin} K expected ~ {expected_T_origin} K"
    )

    # plt.plot(r_grid, T, label="T")
    plt.plot(r_grid, T**1.6, label="T^1.6")
    plt.plot(r_grid[1:], u[1:], label="u")
    plt.legend()
    plt.show()

    # Check numerical integral of total energy Q
    Q_calc = scipy.integrate.simpson(u[1:], r_grid[1:])
    assert np.isclose(Q_calc, solver.Q, rtol=1e-2), (
        f"Integrated Q = {Q_calc} erg/cm^2 does not match theoretical Q = {solver.Q} erg/cm^2"
    )
    print(f"integrated Q = {Q_calc}, theoretical Q = {solver.Q}, percent error = {np.abs(Q_calc - solver.Q) / solver.Q * 100}%")


if __name__ == "__main__":
    test_gold_supersonic_instantaneous_solver()
    print("All tests passed cleanly!")
