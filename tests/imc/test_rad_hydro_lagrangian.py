"""Milestones M5 and M5b: IMC in Lagrangian rad-hydro (md/phase_c/phase_c_changes.md)."""
import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("TQDM_DISABLE", "1")

from hydro_sim.problems.simulation_config import SimulationConfig  # noqa: E402
from imc.core.mesh import Mesh  # noqa: E402
from rad_hydro_sim.problems import presets_config as P  # noqa: E402
from rad_hydro_sim.simulation.iterator import get_dimensionless_boundary_flux, simulate_rad_hydro  # noqa: E402
from rad_hydro_sim.verification.menahem_comparison import run_menahem_front_masses  # noqa: E402

TEST1 = P.PRESET_TEST_CASES[P.PRESET_PAPER_TEST1]
TEST2 = P.PRESET_TEST_CASES[P.PRESET_PAPER_TEST2]

# The configuration every Phase C comparison uses: the Phase B conservation
# flags, the dt ramp, and (for IMC) modest packet counts.
MATCHED = dict(material_update="conservative", right_rad_bc="marshak_vacuum",
               rad_energy_per_mass=True, hydro_energy="compatible", dt_ramp_max=2e-12)
IMC = dict(radiation_solver="imc", imc_n_emission=2000, imc_n_bath=2000, imc_max_census=50000)


def _run(case, t_end=0.02e-9, N=40, **opts):
    cfg = SimulationConfig(N=N, store_every=10**9, show_plot=False, energy_diagnostics=True, **opts)
    return simulate_rad_hydro(replace(case, t_sec_end=t_end), cfg)


def _max_error(meta):
    return float(np.max(np.abs(meta["energy_budget"]["relative_error"])))


def test_carry_moves_packets_with_their_cells():
    old = Mesh(np.array([0.0, 1.0, 3.0, 4.0]))
    new = Mesh(np.array([-2.0, 0.5, 1.0, 5.0]))
    cell = np.array([0, 1, 1, 2, 2])
    x = np.array([0.25, 1.0, 2.5, 3.0, 4.0])          # including both faces of a cell
    xi = (x - old.edges[cell]) / old.widths[cell]
    moved = old.carry(x, cell, new)
    np.testing.assert_allclose(moved, new.edges[cell] + xi * new.widths[cell], rtol=1e-15)
    assert np.all((moved >= new.edges[cell]) & (moved <= new.edges[cell + 1]))


def test_hydro_only_is_bit_identical_with_imc_selected():
    case = P.PRESET_TEST_CASES[P.CONSTANT_TEMPERATURE_OMEGA_0_5_HYDRO_ONLY]
    _, ref, _, _ = _run(case, 0.01e-9)
    _, imc, _, _ = _run(case, 0.01e-9, **IMC)
    for field in ("x", "u", "e_material", "p", "t"):
        np.testing.assert_array_equal(np.asarray(getattr(imc, field)), np.asarray(getattr(ref, field)))


@pytest.mark.parametrize("coupling", ["matched", "physical"])
def test_imc_full_rad_hydro_closes_the_energy_budget(coupling):
    _, state, meta, _ = _run(TEST1, **MATCHED, **IMC, rad_coupling=coupling)
    assert _max_error(meta) < 1e-11                      # measured ~2e-13
    assert meta["imc"]["max_ledger_error"] < 1e-11
    assert meta["imc"]["max_relative_cell_residual"] < 1e-10
    assert meta["imc"]["capped_histories"] == 0
    assert state.x[0] < 0.0                              # the surface ablated
    if coupling == "physical":
        assert state.P_rad is not None and np.all(state.P_rad >= 0.0)
        assert meta["imc"]["radiation_work_total"] > 0.0
    else:
        assert meta["imc"]["radiation_work_total"] == 0.0


def test_diffusion_physical_coupling_closes_the_energy_budget():
    _, matched, meta_m, _ = _run(TEST1, **MATCHED)
    _, physical, meta_p, _ = _run(TEST1, **MATCHED, rad_coupling="physical")
    assert _max_error(meta_m) < 1e-12
    assert _max_error(meta_p) < 1e-12
    assert not np.array_equal(physical.u, matched.u)     # P_rad does push the matter


def test_physical_coupling_needs_the_conservative_hydro():
    with pytest.raises(ValueError, match="compatible"):
        _run(TEST1, 1e-12, N=20, **{**MATCHED, "hydro_energy": "standard"}, rad_coupling="physical")
    with pytest.raises(ValueError, match="rad_energy_per_mass"):
        _run(TEST1, 1e-12, N=20, **{**MATCHED, "rad_energy_per_mass": False}, **IMC)


@pytest.mark.parametrize("solver", ["diffusion", "imc"])
def test_subtracted_radiation_energy_in_full_rad_hydro(solver):
    opts = dict(MATCHED, subtract_radiation_energy=True)
    if solver == "imc":
        opts.update(IMC)
    _, _, meta, _ = _run(TEST1, **opts)
    assert _max_error(meta) < 1e-11


def test_relativistic_terms_are_negligible_early():
    """Plan item A9: v/c and tau v/c stay far below the dynamic-diffusion regime."""
    _, _, meta, _ = _run(TEST1, **MATCHED)
    b = meta["energy_budget"]
    assert np.max(b["max_v_over_c"]) < 1e-3
    assert np.max(b["tau_v_over_c"]) < 0.1


@pytest.mark.parametrize("case, bath_coeff, bath_power, m_coeff, m_power", [
    (TEST1, 0.169141, -79 / 192, 1.01889e-3, 33 / 64),
    (TEST2, 0.204154, -632 / 1285, 8.34319e-4, 193 / 257),
])
def test_paper_presets_match_the_manuscript(case, bath_coeff, bath_power, m_coeff, m_power):
    """Manuscript Sec. "Test cases": the bath law and m_front(t) of Tests 1 and 2."""
    solver, _ = get_dimensionless_boundary_flux(case)
    assert solver.bath_power == pytest.approx(bath_power, rel=1e-10)
    assert solver.bath_B * 1e-9**solver.bath_power == pytest.approx(bath_coeff, rel=2e-5)
    t = np.array([0.5e-9, 2e-9])
    fronts = run_menahem_front_masses(case, t)
    np.testing.assert_allclose(fronts["heat"], m_coeff * (t / 1e-9) ** m_power, rtol=2e-5)
    assert fronts["shock"][-1] < 0.9 * case.x_max * case.rho0    # the shock stays in the domain
