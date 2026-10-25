"""Milestone M0 gate: the packaged SiO2 run reproduces the frozen script and S_N.

The anchor is the deterministic S_N twin (src/sn/sn.py), which the frozen
script already matches (breakout 1.4675 ns, peak 0.9506).  Reference numbers
of the frozen script at default settings: breakout 1.47 ns, peak 0.9538,
energy error 5.3e-14.
"""
import numpy as np
import pytest

from imc.core import stepper
from imc.problems import sio2
from sn import sn


@pytest.mark.slow
def test_sio2_package_matches_sn_and_frozen_script():
    settings = stepper.IMCSettings()                    # frozen-script defaults
    bath = sio2.make_bath()
    runs = [sio2.run_one(seed, 1e-3, 3.0, settings, bath) for seed in (1729, 2718, 3141)]

    t = runs[0]["time_ns"]
    fluxes = np.vstack([r["Fout"] for r in runs])
    s = sio2.summarize(t, fluxes, 0.02)
    assert abs(s["breakout"] - 1.47) <= 0.011
    assert abs(s["peak"] - 0.9538) <= 2.5 * s["peak_stderr"]
    assert max(np.max(np.abs(r["energy_error"])) for r in runs) < 1e-12
    assert max(np.max(r["cell_residual"]) for r in runs) < 1e-10
    assert sum(r["n_capped"] for r in runs) == 0

    ref = sn.run_one(sn.Drive(sio2.DRIVE_PATH), 1e-3, 3.0, 16, "double-gauss", "current", 1e-10, 1000)
    norm = s["norm"]
    width = 0.05
    blocks = np.array([sio2.block_average(t, f / norm, width)[1] for f in fluxes])
    tc = sio2.block_average(t, fluxes[0] / norm, width)[0]
    t_sn, f_sn = sio2.block_average(ref["time_ns"], ref["Fout"] / norm, width)
    late = tc > 1.5
    rel = (blocks.mean(axis=0) - np.interp(tc, t_sn, f_sn))[late] / np.interp(tc, t_sn, f_sn)[late]
    assert abs(rel.mean()) < 0.005                      # no bias beyond noise
    assert np.sqrt(np.mean(rel**2)) < 0.01              # measured: 0.003
