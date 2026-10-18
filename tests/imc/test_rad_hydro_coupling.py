"""Milestone M4: IMC as a drop-in radiation solver for rad_hydro_sim (src/imc/coupling)."""
import os
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

os.environ.setdefault("TQDM_DISABLE", "1")

from hydro_sim.problems.simulation_config import SimulationConfig  # noqa: E402
from imc.coupling.rad_hydro import power_law_bath, tabulated_bath  # noqa: E402
from imc.sio2.problem import DRIVE_PATH, block_average  # noqa: E402
from rad_hydro_sim.problems import presets_config as P  # noqa: E402
from rad_hydro_sim.simulation.iterator import simulate_rad_hydro, tabulated_bath_temperature  # noqa: E402
from rad_hydro_sim.simulation.radiation_step import a_Kelvin, c, eV_to_erg, k_B  # noqa: E402
from sn import sn  # noqa: E402

SIO2 = P.PRESET_TEST_CASES[P.PRESET_SIO2_FOAM_RADIATION_ONLY]


class _FakeBathSolver:
    """The SubsonicHeatWave bath law T = Tb t^tau (1 + B t^p)^(1/4) (Test 1 exponents)."""
    Tb, tau, bath_B, bath_power = 1.16e6, 0.0, 3.35e-5, -79.0 / 192.0

    def Tbath(self, *, time):
        return self.Tb * time**self.tau * (1.0 + self.bath_B * time**self.bath_power) ** 0.25


def test_power_law_bath_energy_is_exact_across_the_singularity():
    bath = power_law_bath(_FakeBathSolver(), a_Kelvin, c)
    for t0, t1 in [(0.0, 1e-12), (1e-12, 3e-12), (1e-9, 1.001e-9)]:
        numeric, _ = quad(bath.flux, t0, t1, limit=200, epsabs=0.0, epsrel=1e-12)
        np.testing.assert_allclose(bath.step_energy(t0, t1), numeric, rtol=1e-9)


def test_tabulated_bath_matches_rad_hydro():
    bath = tabulated_bath(DRIVE_PATH, a_Kelvin, c)
    t = np.linspace(0.0, 3.4e-9, 50)
    T_rad_hydro = tabulated_bath_temperature(SIO2.bath_table, t)
    np.testing.assert_allclose(bath.flux(t), 0.25 * a_Kelvin * c * T_rad_hydro**4, rtol=1e-12)


def _imc_config(**opts):
    base = dict(N=50, store_every=10**9, show_plot=False, energy_diagnostics=True, dt_ramp_max=1e-12,
                radiation_solver="imc", imc_n_emission=2000, imc_n_bath=2000, imc_max_census=50000)
    base.update(opts)
    return SimulationConfig(**base)


def test_adapter_closes_both_energy_ledgers():
    _, state, meta, _ = simulate_rad_hydro(replace(SIO2, t_sec_end=0.3e-9), _imc_config())
    assert np.max(np.abs(meta["energy_budget"]["relative_error"])) < 1e-10
    assert meta["imc"]["max_ledger_error"] < 1e-11
    assert meta["imc"]["max_relative_cell_residual"] < 1e-10
    assert meta["imc"]["capped_histories"] == 0
    assert state.E_rad.shape == state.rho.shape


def test_imc_refuses_what_it_does_not_model():
    with pytest.raises(NotImplementedError):
        simulate_rad_hydro(replace(P.PRESET_TEST_CASES[P.PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK],
                                   t_sec_end=1e-12), _imc_config(N=20))
    with pytest.raises(NotImplementedError, match="incident bath"):
        simulate_rad_hydro(replace(P.PRESET_TEST_CASES[P.PRESET_CONSTANT_T_RADIATION_ONLY],
                                   t_sec_end=1e-12), _imc_config(N=20))


@pytest.mark.slow
def test_adapter_reproduces_the_standalone_sio2_problem():
    """T1 through rad_hydro_sim vs the deterministic S_N twin (which the standalone IMC matches)."""
    norm = 0.25 * a_Kelvin * c * (165.0 * eV_to_erg / k_B) ** 4
    width = 0.05
    blocks = []
    for seed in (1, 2, 3):
        cfg = _imc_config(N=100, imc_seed=seed, imc_n_emission=2500, imc_n_bath=4000, imc_max_census=120000)
        _, _, meta, _ = simulate_rad_hydro(SIO2, cfg)
        b = meta["energy_budget"]
        t = np.asarray(b["t"])
        flux = np.diff(b["rad_out_right"]) / np.diff(t) / norm
        tc, fb = block_average(0.5 * (t[1:] + t[:-1]) * 1e9, flux, width)
        blocks.append(np.interp(np.arange(width / 2, 3.0, width), tc, fb))
        assert np.max(np.abs(b["relative_error"])) < 1e-10
    mean = np.mean(blocks, axis=0)
    ref = sn.run_one(sn.Drive(DRIVE_PATH), 1e-3, 3.0, 16, "double-gauss", "current", 1e-10, 1000)
    t_sn, f_sn = block_average(ref["time_ns"], ref["Fout"] / ref_norm(ref), width)
    tgrid = np.arange(width / 2, 3.0, width)
    f_sn = np.interp(tgrid, t_sn, f_sn)
    late = tgrid > 1.5
    rel = (mean - f_sn)[late] / f_sn[late]
    assert abs(rel.mean()) < 0.005            # measured +0.0004
    assert np.sqrt(np.mean(rel**2)) < 0.01    # measured 0.0035


def ref_norm(ref):
    """sigma_SB (165 eV)^4 in the S_N module's units (cm, ns, MJ)."""
    return sn.SIGMA_SB * (165.0 * sn.EV_TO_K) ** 4
