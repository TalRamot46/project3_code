"""Milestone M0 unit tests for the imc package.

Exact checks against the frozen script (material, allocation, bath energy),
analytic checks of the random walk (attenuation, free streaming, reflection),
and the two energy ledgers.
"""
import numpy as np
import pytest

from imc.core import population, sources, stepper, transport
from imc.archive.sio2_original import sio2_1mm_gray_imc_check_og as frozen
from imc.core.material import PowerLawMaterial
from imc.core.mesh import Mesh
from imc.core.particles import Particles
from imc.problems import sio2


def _run_kernel(p, edges, k_a, k_s, fleck, c, t1, left="vacuum", right="vacuum", seed=7):
    transport.seed(seed)
    return transport.transport_step(
        p.x, p.mu, p.w, p.t, p.cell, edges,
        np.asarray(k_a, float), np.asarray(k_s, float), np.asarray(fleck, float),
        c, t1, stepper.BC[left], stepper.BC[right], 10**6,
    )


# --- material and sources: identical to the frozen script --------------------

def test_sio2_material_matches_frozen_script():
    rng = np.random.default_rng(0)
    theta = 10 ** rng.uniform(-3.5, 0.5, 200)
    dt = 1e-3
    sigma, fleck, B, U, emission = frozen.material_coefficients(theta, dt)
    m = sio2.MATERIAL
    k_a, k_s = m.opacities(theta, sio2.RHO)
    np.testing.assert_allclose(k_a, sigma, rtol=1e-12)
    assert np.all(k_s == 0.0)
    np.testing.assert_allclose(m.fleck(theta, sio2.RHO, dt), fleck, rtol=1e-12)
    np.testing.assert_allclose(m.u_r(theta), B, rtol=1e-12)
    np.testing.assert_allclose(m.u_m(theta, sio2.RHO), U, rtol=1e-12)
    dx = sio2.L_CM / sio2.N_CELLS
    mine = m.fleck(theta, sio2.RHO, dt) * k_a * m.c * m.u_r(theta) * dx * dt
    np.testing.assert_allclose(mine, emission, rtol=1e-12)
    np.testing.assert_allclose(m.temperature(U, sio2.RHO), theta, rtol=1e-12)


@pytest.mark.parametrize("subtract", [False, True])
def test_temperature_inversion_roundtrip(subtract):
    m = PowerLawMaterial(f=2.0, beta=1.6, mu=0.14, g=1.0, alpha=1.5, lam=0.2,
                         a=1e-3, c=1.0, subtract_radiation=subtract)
    T = np.linspace(0.1, 3.0, 50)
    rho = np.linspace(0.5, 20.0, 50)
    np.testing.assert_allclose(m.temperature(m.u_m(T, rho), rho), T, rtol=1e-12)


def test_chi_above_one_has_no_transport_equivalent():
    with pytest.raises(ValueError):
        PowerLawMaterial(f=1, beta=1, mu=0, g=1, alpha=0, lam=0, a=1, c=1, chi=1000.0)


def test_from_case_uses_rad_hydro_formulas():
    from rad_hydro_sim.problems.presets_config import (
        PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK, PRESET_TEST_CASES,
    )
    case = PRESET_TEST_CASES[PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK]
    m = PowerLawMaterial.from_case(case)
    T, rho = 1.16e6, 19.32
    sigma_rad_hydro = 1.0 / (case.g_Kelvin * T**case.alpha * rho ** (-case.lambda_ - 1))
    np.testing.assert_allclose(m.opacity_total(T, rho), sigma_rad_hydro, rtol=1e-14)
    np.testing.assert_allclose(m.u_m(T, rho) / rho, case.f_Kelvin * T**case.beta_Rosen * rho**-case.mu,
                               rtol=1e-14)


def test_allocation_matches_frozen_script():
    rng = np.random.default_rng(1)
    for target in (5, 50, 2500):
        emission = rng.exponential(1.0, 100) * (rng.random(100) > 0.2)
        mine = sources.allocate_emission(emission, target)
        theirs, total = frozen.allocate_material_packets(emission, target)
        np.testing.assert_array_equal(mine, theirs)
        assert mine.sum() == total


def test_bath_energy_matches_frozen_script():
    bath = sio2.make_bath()
    drive = frozen.Drive(sio2.DRIVE_PATH)
    for t0 in (0.0, 0.4, 1.2, 2.7):
        np.testing.assert_allclose(bath.step_energy(t0, t0 + 1e-3),
                                   drive.flux_average(t0, t0 + 1e-3) * 1e-3, rtol=1e-12)


def test_bath_times_follow_the_flux():
    np.random.seed(3)
    bath = sources.BathSource(lambda t: t, n_sub=64)          # flux ~ t on [0, 1]
    times = bath.sample_times(0.0, 1.0, 200_000)
    # density 2t: mean 2/3; the sub-interval approximation is O(1/n_sub^2) here.
    assert abs(times.mean() - 2.0 / 3.0) < 5e-3


def test_mesh_from_masses():
    mesh = Mesh.from_masses([1.0, 2.0, 3.0], [1.0, 2.0, 0.5])
    np.testing.assert_allclose(mesh.edges, [0.0, 1.0, 2.0, 8.0])


# --- random walk -------------------------------------------------------------

def test_attenuation_in_pure_absorber():
    n, sigma, L = 200_000, 2.0, 1.0
    edges = np.linspace(0.0, L, 11)
    p = Particles(np.zeros(n), np.ones(n), np.ones(n), np.zeros(n), np.zeros(n, dtype=np.int64))
    out = _run_kernel(p, edges, [sigma] * 10, [0.0] * 10, [1.0] * 10, c=1.0, t1=10.0)
    deposited, esc_r = out[4], out[9]
    expected = np.exp(-sigma * L)
    assert abs(esc_r / n - expected) < 4 * np.sqrt(expected * (1 - expected) / n)
    cells = np.exp(-sigma * edges[:-1]) - np.exp(-sigma * edges[1:])
    np.testing.assert_allclose(deposited / n, cells, atol=4 * np.sqrt(cells.max() / n))


def test_free_streaming_is_exact():
    rng = np.random.default_rng(2)
    n = 5000
    edges = np.sort(np.concatenate(([0.0, 1.0], rng.uniform(0, 1, 20))))
    x0 = rng.uniform(0, 1, n)
    mu0 = rng.uniform(-1, 1, n)
    cell0 = np.searchsorted(edges, x0, side="right") - 1
    p = Particles(x0, mu0, np.ones(n), np.zeros(n), cell0.astype(np.int64))
    t1 = 0.3
    zeros = np.zeros(edges.size - 1)
    cx, cmu, cw, ccell = _run_kernel(p, edges, zeros, zeros, zeros + 1, c=1.0, t1=t1)[:4]
    x_final = x0 + mu0 * t1
    inside = (x_final > 0) & (x_final < 1)
    assert cx.size == inside.sum()
    np.testing.assert_allclose(np.sort(cx), np.sort(x_final[inside]), atol=1e-12)
    assert np.all((cx >= edges[ccell]) & (cx <= edges[ccell + 1]))


def test_reflecting_box_keeps_every_packet():
    rng = np.random.default_rng(4)
    n = 2000
    edges = np.linspace(0.0, 1.0, 6)
    p = Particles(rng.uniform(0, 1, n), rng.uniform(-1, 1, n), rng.uniform(0.5, 1.5, n),
                  np.zeros(n), np.zeros(n, dtype=np.int64))
    p.cell[:] = np.minimum((p.x * 5).astype(np.int64), 4)
    out = _run_kernel(p, edges, [0.0] * 5, [3.0] * 5, [1.0] * 5, c=1.0, t1=5.0,
                      left="reflecting", right="reflecting")
    cw, esc_l, esc_r = out[2], out[8], out[9]
    assert esc_l == 0.0 and esc_r == 0.0
    np.testing.assert_allclose(cw.sum(), p.w.sum(), rtol=1e-13)


def test_energy_ledgers_close_to_round_off():
    """Global and per-cell ledgers on a nonuniform mesh with scattering and a bath."""
    rng = np.random.default_rng(5)
    mat = PowerLawMaterial(f=1.0, beta=1.3, mu=0.2, g=0.05, alpha=1.0, lam=0.1,
                           a=1.0, c=1.0, chi=0.6)
    mesh = Mesh(np.concatenate(([0.0], np.cumsum(rng.uniform(0.2, 2.0, 25)))))
    rho = rng.uniform(0.5, 2.0, 25)
    bath = sources.BathSource(lambda t: 0.3 * (1.0 + np.sin(t)))
    census = sources.uniform_radiation_particles(mesh.edges, 0.1, 40, 0.0)
    state = stepper.initial_state(np.full(25, 0.7), mesh, rho, mat, census=census)
    settings = stepper.IMCSettings(n_emission=2000, n_bath=1000, max_census=3000)
    stepper.seed_all(11)
    for _ in range(15):
        state, tl = stepper.imc_step(state, mesh, rho, mat, 0.4, bath, settings)
        scale = max(tl.census_after.max(), tl.emitted.max(), tl.deposited.max())
        assert np.abs(tl.cell_balance_residual()).max() < 1e-12 * scale
        assert abs(tl.radiation_balance_residual()) < 1e-12 * scale * mesh.n_cells
        err = state.ledger.relative_error(stepper.total_energy(state, mesh, rho, mat))
        assert abs(err) < 1e-12


def test_comb_conserves_energy_and_is_unbiased():
    rng = np.random.default_rng(6)
    n, cells = 3000, 10
    w = rng.exponential(1.0, n)
    cell = rng.integers(0, cells, n)
    census = Particles(rng.random(n), rng.uniform(-1, 1, n), w, np.zeros(n), cell)
    target = census.energy_per_cell(cells)
    np.random.seed(0)
    acc = np.zeros(cells)
    reps = 400
    for _ in range(reps):
        out = population.comb(census, 500)
        assert out.size == 500
        np.testing.assert_allclose(out.energy(), census.energy(), rtol=1e-12)
        acc += out.energy_per_cell(cells)
    np.testing.assert_allclose(acc / reps, target, rtol=0.02)
