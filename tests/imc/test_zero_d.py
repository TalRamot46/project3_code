"""Milestones M2 / M2a: the 0-D IMC problem against Wollaber (2016), Eqs. (120) and (122)."""
import numpy as np
import pytest

from imc.problems import zero_d

REPLICAS = 12


def _replicas(p, q, dtau, M0, n_steps, mode="implicit"):
    runs = [zero_d.run(p, q, dtau, M0, n_steps, seed=10 + r, fleck_mode=mode,
                       n_emission=20_000, n_initial=20_000, max_census=20_000)
            for r in range(REPLICAS)]
    M = np.array([r[0] for r in runs])
    pred = np.array([r[1] for r in runs])
    err = max(np.max(np.abs(r[2])) for r in runs)
    return M, pred, err


@pytest.mark.parametrize("p, q, dtau, mode", [
    (5, 0.14, 10.0, "implicit"),     # Wollaber Fig. 2
    (3, 0.14, 10.0, "implicit"),
    (5, 0.5, 1.0, "implicit"),
    (5, 0.5, 10.0, "explicit"),      # M2a: unstable scheme, still solved exactly per step
])
def test_each_step_matches_wollaber_eq_120(p, q, dtau, mode):
    """Given the IMC's own M_n, the mean of M_(n+1) is Eq. (120) (Theorem 2 in 0-D)."""
    M0 = 0.9 if mode == "implicit" else 0.99
    n_steps = 5 if mode == "implicit" else 4      # explicit: stop before the blow-up
    M, pred, err = _replicas(p, q, dtau, M0, n_steps, mode)
    # Round-off scales with the energy cycled per step: explicit emission is
    # c sigma dt = dtau/q = 20 times the material energy here, implicit f times that.
    assert err < (1e-12 if mode == "implicit" else 1e-10)
    residual = M[:, 1:] - pred[:, 1:]                       # per replica and step
    z = residual.mean(axis=0) / (residual.std(axis=0, ddof=1) / np.sqrt(REPLICAS))
    assert np.all(np.abs(z) <= 3.5), z


def test_fig2_damped_oscillation():
    """Wollaber Fig. 2: overshoot above 1, then alternating, shrinking deviations."""
    M, _, _ = _replicas(5, 0.14, 10.0, 0.9, 4)
    dev = M.mean(axis=0)[1:] - 1.0
    assert 0.03 < dev[0] < 0.04                             # Fig. 2 peaks near 1.035
    assert np.all(np.sign(dev[:-1]) != np.sign(dev[1:]))    # rho < 0: alternating
    assert np.all(np.abs(dev[1:]) < np.abs(dev[:-1]))       # |rho| < 1: damped


def test_amplification_factor():
    # Wollaber quotes about -0.2868 for the Fig. 2 parameters.
    assert abs(zero_d.amplification(5, 0.14, 10.0) - (-0.2868)) < 5e-4
    # Eq. (122) is the derivative of the exact map (120) at equilibrium, for any p.
    eps = 1e-6
    for p in (3, 5):
        slope = (zero_d.exact_step(1 + eps, p, 0.14, 10.0) - zero_d.exact_step(1 - eps, p, 0.14, 10.0)) / (2 * eps)
        assert abs(slope - zero_d.amplification(p, 0.14, 10.0)) < 1e-6
    # IMC stays stable for every dtau; |rho| is largest as dtau -> infinity (-0.2984 for q*).
    taus = np.logspace(-3, 6, 200)
    for q in (0.01, 0.14, 1.0, 10.0):
        assert np.all(np.abs(zero_d.amplification(5, q, taus)) < 1.0)
    # The explicit scheme tends to -4q: unstable for q > 1/4.
    assert abs(zero_d.amplification(5, 0.5, 100.0, "explicit") + 2.0) < 1e-6


def test_explicit_diverges_where_implicit_is_stable():
    """M2a: q = 0.5, dtau = 10.  Explicit rho = -2, implicit rho = -0.158."""
    Mi, _, _ = _replicas(5, 0.5, 10.0, 0.99, 5, "implicit")
    Me, _, _ = _replicas(5, 0.5, 10.0, 0.99, 4, "explicit")
    dev_i = np.abs(Mi.mean(axis=0) - 1.0)
    dev_e = np.abs(Me.mean(axis=0) - 1.0)
    assert dev_i[-1] < 0.5 * dev_i[0]                       # decays (to noise)
    assert dev_e[-1] > 6.0 * dev_e[0]                       # grows by ~2 per step (16 nominal)


def test_explicit_ends_with_negative_material_energy():
    """Without the Fleck factor, Proposition 1 fails: emission eventually exceeds u_m."""
    with pytest.raises(RuntimeError, match="Non-positive material energy"):
        zero_d.run(5, 0.5, 10.0, 0.99, 30, seed=3, fleck_mode="explicit",
                   n_emission=20_000, n_initial=20_000, max_census=20_000)


def test_implicit_never_loses_positivity():
    M, _, _ = zero_d.run(5, 0.5, 10.0, 0.99, 30, seed=3,
                         n_emission=20_000, n_initial=20_000, max_census=20_000)
    assert np.all(M > 0.9)
