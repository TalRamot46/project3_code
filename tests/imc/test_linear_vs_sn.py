"""Milestone M1: one frozen-coefficient IMC step agrees with S_N (Theorem 2 in numbers).

z = (IMC mean - S_N) / (IMC standard error + S_N error estimate), where the
S_N error estimate is |S_N(2K, 2 n_mu) - S_N(K, n_mu)| (time and angle only:
the S_N grid is fixed by src/sn/sn.py).

What the exploration (python -m imc.problems.linear_step) showed:
  * Census energy agrees cell by cell at every optical thickness.
  * S_N's net deposition oscillates from cell to cell (the odd-even mode of
    diamond difference in cells with sigma dx >~ 1), so deposition is compared
    on sums over adjacent cell pairs, which cancel the oscillation.
  * At sigma dx = 10 the boundary layers at both faces are 0.1 cell thick and
    diamond difference with the set-to-zero fixup cannot represent them; there
    S_N's own spatial error (not estimated here) dominates, so the comparison
    is restricted to the interior cells 5..89.
"""
import numpy as np
import pytest

from imc.problems import linear_step as ls

REPLICAS = 32


def _compare(sigma_dx):
    prob = ls.make_problem(sigma_dx)
    ref = ls.sn_solution(prob, 128, 32)
    coarse = ls.sn_solution(prob, 64, 16)
    reps = [ls.imc_solution(prob, 1000 + r) for r in range(REPLICAS)]
    out = {}
    for key in ("E", "net_dep", "leak_l", "leak_r"):
        vals = np.array([r[key] for r in reps])
        out[key] = (vals, ref[key], np.abs(ref[key] - coarse[key]))
    return out


def _z(vals, ref, sn_err):
    se = vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
    return (vals.mean(axis=0) - ref) / (se + sn_err + 1e-300)


@pytest.mark.parametrize("sigma_dx", [0.1, 1.0, 10.0])
def test_linear_step_matches_sn(sigma_dx):
    out = _compare(sigma_dx)
    cells = slice(None) if sigma_dx < 5 else slice(5, 90)
    pairs = slice(None) if sigma_dx < 5 else slice(3, 45)

    vals, ref, err = out["E"]
    z = _z(vals, ref, err)[cells]
    assert np.mean(np.abs(z) <= 3.0) >= 0.97, z

    vals, ref, err = out["net_dep"]
    pv = vals.reshape(vals.shape[0], -1, 2).sum(axis=2)
    z = _z(pv, ref.reshape(-1, 2).sum(axis=1), err.reshape(-1, 2).sum(axis=1))[pairs]
    assert np.mean(np.abs(z) <= 3.0) >= 0.95, z

    for key in ("leak_l", "leak_r"):
        vals, ref, err = out[key]
        assert abs(_z(vals, ref, err)) <= 3.0, key
