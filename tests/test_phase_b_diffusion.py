"""Phase B (M3 + A1) changes to the diffusion rad-hydro code.

See md/phase_b_diffusion_changes.md.  Every option defaults to the original
behaviour; the golden file holds final states produced by the code *before*
the Phase B edits, so the first test proves the defaults are bit-identical.
"""
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TQDM_DISABLE", "1")

from hydro_sim.problems.simulation_config import SimulationConfig  # noqa: E402
from rad_hydro_sim.problems import presets_config as P  # noqa: E402
import importlib  # noqa: E402

# rad_hydro_sim.simulation re-exports the *function* radiation_step, which
# shadows the module of the same name, so import the module explicitly.
rs = importlib.import_module("rad_hydro_sim.simulation.radiation_step")
from rad_hydro_sim.simulation.iterator import (  # noqa: E402
    compute_boundary_temperatures, initialize_problem, simulate_rad_hydro,
)

GOLDEN = Path(__file__).parent / "data" / "phase_b_original_code_golden.npz"


def _run(key, t_end, N=120, **opts):
    case = replace(P.PRESET_TEST_CASES[key], t_sec_end=t_end)
    cfg = SimulationConfig(N=N, store_every=10**9, show_plot=False, **opts)
    return simulate_rad_hydro(case, cfg)


def _max_budget_error(key, t_end, N=60, **opts):
    _, _, meta, _ = _run(key, t_end, N=N, energy_diagnostics=True, **opts)
    return float(np.max(np.abs(meta["energy_budget"]["relative_error"])))


# --- defaults unchanged ------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("name, key", [
    ("rad_only_marshak", P.CONSTANT_TEMPERATURE_OMEGA_0_5_RADIATION_ONLY),
    ("rad_only_dirichlet", P.PRESET_CONSTANT_T_RADIATION_ONLY),
    ("full_fig8_marshak", P.PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK),
    ("hydro_only", P.CONSTANT_TEMPERATURE_OMEGA_0_5_HYDRO_ONLY),
])
def test_defaults_are_bit_identical_to_the_original_code(name, key):
    golden = np.load(GOLDEN)
    _, state, _, _ = _run(key, 0.05e-9)
    for field in ("x", "u", "e_material", "T_material", "E_rad", "t"):
        np.testing.assert_array_equal(np.asarray(getattr(state, field)), golden[f"{name}__{field}"],
                                      err_msg=f"{name}: {field}")


def test_with_output_paths_keeps_phase_b_fields():
    cfg = SimulationConfig(material_update="conservative", hydro_energy="compatible", dt_ramp_max=1e-12)
    out = cfg.with_output_paths("some case")
    assert out.material_update == "conservative"
    assert out.hydro_energy == "compatible"
    assert out.dt_ramp_max == 1e-12
    assert out.save_path is not None


# --- M3(a) conservative material update ---------------------------------------

def test_conservative_update_closes_the_radiation_only_budget():
    key = P.CONSTANT_TEMPERATURE_OMEGA_0_5_RADIATION_ONLY
    original = _max_budget_error(key, 0.05e-9)
    conservative = _max_budget_error(key, 0.05e-9, material_update="conservative")
    assert original > 1e-4            # the u_r-form update leaks energy
    assert conservative < 1e-10       # measured ~1e-12


def test_conservative_update_with_dirichlet_drive():
    assert _max_budget_error(P.PRESET_CONSTANT_T_RADIATION_ONLY, 0.05e-9,
                             material_update="conservative") < 1e-10


# --- M3(b) manuscript material energy u_m = u - a T^4 ------------------------

def _set_material(case):
    (rs.alpha, rs.beta_Rosen, rs.mu, rs.f_Kelvin, rs.chi, rs.lambda_, rs.g_Kelvin) = case._get_params()


@pytest.mark.parametrize("subtract", [False, True])
def test_material_energy_inversion(subtract):
    case = P.PRESET_TEST_CASES[P.PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK]
    _set_material(case)
    T = np.geomspace(300.0, 3e6, 40)
    rho = np.geomspace(1e-3, 20.0, 40)
    u = rs.material_energy_density(T, rho, subtract)
    np.testing.assert_allclose(rs.material_temperature(u, rho, subtract), T, rtol=1e-11)
    if not subtract:
        np.testing.assert_allclose(rs.fleck_beta(T, rho, False),
                                   rs.calculate_beta_from_temperature_and_density(T, rho), rtol=1e-14)


def test_subtracted_initial_energy_and_budget():
    key = P.CONSTANT_TEMPERATURE_OMEGA_0_5_RADIATION_ONLY
    case = replace(P.PRESET_TEST_CASES[key], t_sec_end=0.05e-9)
    cfg = SimulationConfig(N=60, subtract_radiation_energy=True)
    state = initialize_problem(case, cfg)
    T, rho = state.T_material, state.rho
    expected = case.f_Kelvin * T**case.beta_Rosen * rho**-case.mu - rs.a_Kelvin * T**4 / rho
    np.testing.assert_allclose(state.e_material, expected, rtol=1e-14)
    assert _max_budget_error(key, 0.05e-9, material_update="conservative",
                             subtract_radiation_energy=True) < 1e-10


def test_subtract_is_refused_with_hydro():
    with pytest.raises(NotImplementedError):
        _run(P.PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK, 1e-12, N=20, subtract_radiation_energy=True)


def test_phase_b_options_need_the_time_dependent_gray_mode():
    case = replace(P.PRESET_TEST_CASES[P.PRESET_SUPERSONIC_INSTANTANEOUS_ANALYTIC])
    state = initialize_problem(case, SimulationConfig(N=20))
    with pytest.raises(ValueError, match="force_black=None"):
        rs.radiation_step(state, 1e-27, case, None, material_update="conservative")


# --- M3(d) right Marshak vacuum ----------------------------------------------

def test_right_marshak_vacuum_replaces_the_ghost_leak():
    case = P.PRESET_TEST_CASES[P.CONSTANT_TEMPERATURE_OMEGA_0_5_RADIATION_ONLY]
    _set_material(case)
    rng = np.random.default_rng(0)
    n = 12
    rho = rng.uniform(1, 20, n)
    m = rng.uniform(1e-5, 1e-4, n)
    T = rng.uniform(1e5, 1e6, n)
    sigma = rs.calculate_sigma_from_temperature_and_density(T, rho)
    D = rs.calculate_D_from_sigma(sigma)
    A = rs.calculate_A(rs.calculate_beta_from_temperature_and_density(T, rho), sigma, 1e-12)
    E = rs.a_Kelvin * T**4
    args = (sigma, D, A, m, rho, E, T, 1e-12)
    ghost = rs.calculate_abcd(*args, bc_type="Marshak", T_left=1e6)
    vac = rs.calculate_abcd(*args, bc_type="Marshak", T_left=1e6, right_rad_bc="marshak_vacuum")
    dx = m / rho
    fc = rs.face_flux_coefficients(D, m, rho)
    np.testing.assert_allclose(vac[1][-1] - ghost[1][-1], -fc[-1] / dx[-1] + rs.c / (2 * dx[-1]), rtol=1e-12)
    for k in (0, 2, 3):
        np.testing.assert_array_equal(vac[k], ghost[k])
    np.testing.assert_array_equal(vac[1][:-1], ghost[1][:-1])


# --- A1 compatible hydro, M3(c) per-mass radiation, M3(e) the budget ----------

def test_compatible_hydro_conserves_energy_to_round_off():
    key = P.CONSTANT_TEMPERATURE_OMEGA_0_5_HYDRO_ONLY
    standard = _max_budget_error(key, 0.01e-9)
    compatible = _max_budget_error(key, 0.01e-9, hydro_energy="compatible")
    assert standard > 1e-8            # measured 3e-4 at 0.05 ns, N=120
    assert compatible < 1e-12


def test_full_rad_hydro_budget_needs_all_three_flags():
    key = P.PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK
    without_per_mass = _max_budget_error(key, 0.02e-9, material_update="conservative",
                                         hydro_energy="compatible")
    all_flags = _max_budget_error(key, 0.02e-9, material_update="conservative",
                                  hydro_energy="compatible", rad_energy_per_mass=True)
    assert all_flags < 1e-12
    assert without_per_mass > 100 * all_flags   # E_rad V changes with the cell volume


# --- tabulated bath ------------------------------------------------------------

def test_tabulated_bath_temperature():
    case = P.PRESET_TEST_CASES[P.PRESET_SIO2_FOAM_RADIATION_ONLY]
    state = initialize_problem(case, SimulationConfig(N=10))
    state.t = 0.075e-9                                      # halfway 25 eV -> 55 eV
    T_left, _, T_bath, _ = compute_boundary_temperatures(state, case, solver=None)
    K_per_eV = rs.eV_to_erg / rs.k_B
    assert T_left == T_bath
    np.testing.assert_allclose(T_bath, 40.0 * K_per_eV, rtol=1e-12)
