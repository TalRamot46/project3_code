"""initial_condition="analytic_ablation": a run seeded from the semi-analytic AblationSolver at t_sec_start.

Test 3 (gold, omega = +0.5) on its default mesh, seeded at 0.1 ns.  Building the solver takes ~10 s.
"""
from dataclasses import replace

import numpy as np
import pytest

from hydro_sim.problems.simulation_config import SimulationConfig
from rad_hydro_sim.problems import presets_config as P
from rad_hydro_sim.simulation.hydro_steps import kinetic_energy
from rad_hydro_sim.simulation.iterator import _ablation_cache_key, _get_ablation_solver, initialize_problem

T0 = 1e-10


@pytest.fixture(scope="module")
def seeded():
    cold_case = P.PRESET_TEST_CASES[P.CONSTANT_TEMPERATURE_OMEGA_0_5_FULL]
    case = replace(cold_case, initial_condition="analytic_ablation", t_sec_start=T0)
    cfg = SimulationConfig(N=25, show_plot=False)
    solver = _get_ablation_solver(_ablation_cache_key(case))
    m_shock = float(np.ravel(solver.shock_solver.shocked_mass(time=T0))[0])
    return case, initialize_problem(cold_case, cfg), initialize_problem(case, cfg), solver, m_shock


def test_mesh_keeps_its_masses_and_the_far_field(seeded):
    case, cold, state, _, m_shock = seeded
    assert state.t == T0
    np.testing.assert_allclose(state.m_cells, cold.m_cells, rtol=1e-9)
    assert np.all(np.diff(state.x) > 0)
    m_nodes = np.concatenate(([0.0], np.cumsum(cold.m_cells)))
    ahead = m_nodes > m_shock
    np.testing.assert_allclose(state.x[ahead], cold.x[ahead], rtol=1e-12)
    ahead_cells = m_nodes[:-1] > m_shock
    np.testing.assert_array_equal(state.rho[ahead_cells], cold.rho[ahead_cells])
    np.testing.assert_array_equal(state.e_material[ahead_cells], cold.e_material[ahead_cells])
    assert np.all(state.u[ahead] == 0.0)


def test_shocked_matter_carries_the_solution(seeded):
    """Behind the shock and past the heat front the cells hold the solver's density and pressure,
    so their entropy p / rho^gamma, which the cold start gets wrong by up to 4x, is the solver's."""
    case, _, state, solver, m_shock = seeded
    mc = np.cumsum(state.m_cells) - 0.5 * state.m_cells
    inside = (mc > 3e-3) & (mc < 0.9 * m_shock)
    assert inside.sum() > 50
    exact = solver.solve(mass=mc[inside], time=T0)
    np.testing.assert_allclose(state.rho[inside], exact["density"], rtol=2e-3)
    np.testing.assert_allclose(state.p[inside], exact["pressure"], rtol=2e-3)


def test_energy_is_the_solutions(seeded):
    """Internal plus kinetic energy above the cold target: the solver's, within 1%."""
    case, cold, state, solver, m_shock = seeded
    E_seeded = (np.sum(state.m_cells * state.e_material) - np.sum(cold.m_cells * cold.e_material)
                + kinetic_energy(state.u, state.m_cells))
    m = np.geomspace(1e-14, 0.9999 * m_shock, 50000)
    exact = solver.solve(mass=m, time=T0)
    rho, p, u = (np.asarray(exact[k]) for k in ("density", "pressure", "velocity"))
    rho_cold = 2.0 * case.rho0**2 / m                         # rho0 x^-0.5 at m = 2 rho0 x^0.5
    e_cold = case.f_Kelvin * case.T_initial_Kelvin**case.beta_Rosen * rho_cold**(-case.mu)
    E_exact = np.trapz(p / (case.r * rho) - e_cold + 0.5 * u**2, m)
    assert E_seeded == pytest.approx(E_exact, rel=1e-2)
