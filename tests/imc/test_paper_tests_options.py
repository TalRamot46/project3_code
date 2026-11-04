"""Phase C 1/2 options for the paper's Tests 1-6 (md/phase_c/phase_c_tests3to6_results.md).

The uniform-mass mesh, the bath calibration switch (bath_drive) and the front detector of the
comparison driver.  Every option is opt-in: the defaults are the Phase C code.
"""
import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("TQDM_DISABLE", "1")

from hydro_sim.problems.simulation_config import SimulationConfig  # noqa: E402
from imc.coupling.rad_hydro import power_law_bath  # noqa: E402
from rad_hydro_sim.problems import presets_config as P  # noqa: E402
from rad_hydro_sim.simulation.iterator import (  # noqa: E402
    bath_correction, bath_correction_for, bath_correction_path, initialize_problem)
from rad_hydro_sim.simulation.radiation_step import a_Kelvin, c  # noqa: E402
from rad_hydro_sim.verification.imc_vs_diffusion import fronts_of_profile  # noqa: E402

TEST4 = P.PRESET_TEST_CASES[P.CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL]


@pytest.mark.parametrize("key", [P.CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL, P.PRESET_PAPER_TEST1])
def test_uniform_mass_mesh_gives_equal_cell_masses(key):
    case = P.PRESET_TEST_CASES[key]
    state = initialize_problem(case, SimulationConfig(N=200, mesh="uniform_mass"))
    np.testing.assert_allclose(state.m_cells, state.m_cells.mean(), rtol=1e-9)
    assert state.x[0] == 0.0 and state.x[-1] == case.x_max


def test_default_mesh_is_unchanged():
    default = initialize_problem(TEST4, SimulationConfig(N=50))
    explicit = initialize_problem(TEST4, SimulationConfig(N=50, mesh="default"))
    np.testing.assert_array_equal(default.x, explicit.x)
    with pytest.raises(ValueError, match="mesh"):
        initialize_problem(TEST4, SimulationConfig(N=50, mesh="nonsense"))


def test_bath_drive_switch(tmp_path):
    table = tmp_path / "corr.txt"
    np.savetxt(table, np.array([[0.1, 1.10], [1.0, 1.20], [2.0, 1.20]]))
    case = replace(TEST4, bath_correction=str(table))
    assert bath_correction_for(case, SimulationConfig()) is None                       # option (c), default
    factor = bath_correction_for(case, SimulationConfig(bath_drive="calibrated"))      # option (a)
    assert factor(0.05e-9) == pytest.approx(1.10)                                      # held before the table
    assert factor(np.sqrt(0.1 * 1.0) * 1e-9) == pytest.approx(1.15)                   # linear in log t
    assert bath_correction_path(case, "calibrated_flux") == str(tmp_path / "corr_flux.txt")
    with pytest.raises(ValueError, match="no table"):
        bath_correction_for(case, SimulationConfig(bath_drive="calibrated_flux"))
    with pytest.raises(ValueError, match="bath_drive"):
        bath_correction_for(case, SimulationConfig(bath_drive="nonsense"))


def test_bath_table_is_reread_after_a_rewrite(tmp_path):
    """The calibration rewrites one file per iteration; the cache must not keep the old one."""
    table = tmp_path / "corr.txt"
    np.savetxt(table, np.array([[0.1, 1.0], [2.0, 1.0]]))
    assert bath_correction(str(table))(1e-9) == pytest.approx(1.0)
    os.utime(table, (1, 1))
    np.savetxt(table, np.array([[0.1, 1.3], [2.0, 1.3]]))
    assert bath_correction(str(table))(1e-9) == pytest.approx(1.3)


class _Law:
    Tb, tau, bath_B, bath_power = 1.16e6, 0.0, 3.35e-5, -79.0 / 192.0

    def Tbath(self, *, time):
        return self.Tb * time**self.tau * (1.0 + self.bath_B * time**self.bath_power) ** 0.25


def test_imc_bath_takes_the_calibration_factor():
    plain = power_law_bath(_Law(), a_Kelvin, c)
    scaled = power_law_bath(_Law(), a_Kelvin, c, factor=lambda t: np.full_like(np.asarray(t, float), 1.1))
    assert scaled.step_energy(1e-9, 1.1e-9) == pytest.approx(1.1**4 * plain.step_energy(1e-9, 1.1e-9), rel=1e-12)
    assert scaled.flux(1e-9) == pytest.approx(1.1**4 * plain.flux(1e-9), rel=1e-12)


def test_fronts_of_profile_on_a_synthetic_profile():
    m = np.linspace(0.01, 1.0, 100)                          # cell right-edge masses
    mc = m - 0.005
    T = np.where(mc < 0.3, 1.0, 0.05)                        # heat front between the centres 0.295 and 0.305
    rho0 = np.full_like(m, 10.0)
    rho = np.where(mc < 0.3, 0.1, np.where(mc < 0.7, 40.0, 10.0))   # compressed up to 0.7
    p = np.where(mc < 0.7, 2.0, 0.0)
    m_heat, m_shock, p_abl = fronts_of_profile(m, T, rho, p, rho0)
    assert 0.295 < m_heat < 0.305 and 0.695 <= m_shock <= 0.705 and p_abl == pytest.approx(2.0)
