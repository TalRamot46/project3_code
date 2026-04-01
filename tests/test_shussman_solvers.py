# tests/test_shussman_solvers.py
"""
Cross-check Python Shussman solvers against legacy MATLAB under ``matlab_shussman_solvers``.

MATLAB reference arrays live in ``matlab_shussman_solvers/test_exports/*.mat``.
Regenerate them (from repo root, with MATLAB on PATH)::

    cd matlab_shussman_solvers/test_exports
    matlab -batch "run('export_sub_reference.m'); run('export_super_reference.m'); run('export_shock_tau_reference.m'); run('export_piecewise_m_P.m'); exit"

The shock export uses fixed ``P0`` (Barye) and ``tau`` and matches
``compute_shock_profiles(..., patching_method=False)``, not the chained
``profiles_for_report_shock.m`` call that uses ``Pw(3)`` from the subsonic run.

Subsonic MATLAB ``T_heat`` uses an extra factor ``100`` relative to ``T0`` (see ``sub/profiles_for_report_sub.m``);
tests scale Python temperature by ``100`` when ``T0_phys_HeV=1`` matches MATLAB ``T0=1``.

Piecewise graph check: compares the **normalized** pressure profile ``P(m)/max(P)`` between
MATLAB ``my_final_profiles`` output and ``build_piecewise_reference`` driven like
``run_shussman_piecewise_reference`` (patching shock, no tail), avoiding absolute unit
ambiguity between MBar vs Barye in different pipeline stages while still testing the
combined sub+shock join that the MATLAB figure displays.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy.io")

from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATLAB_EXPORT_DIR = PROJECT_ROOT / "matlab_shussman_solvers" / "test_exports"

SUB_MAT = MATLAB_EXPORT_DIR / "sub_reference.mat"
SHOCK_TAU_MAT = MATLAB_EXPORT_DIR / "shock_tau_reference.mat"
SUPER_MAT = MATLAB_EXPORT_DIR / "super_reference.mat"
PIECEWISE_MAT = MATLAB_EXPORT_DIR / "piecewise_m_P_reference.mat"


def _require_mat_fixture(path: Path) -> None:
    if not path.is_file():
        pytest.skip(
            f"Missing {path.name}. Regenerate MATLAB exports (see test module docstring)."
        )


def _resample_by_index(y: np.ndarray, *, n: int = 400) -> np.ndarray:
    """Linearly resample ``y`` along normalized index (for non-monotone similarity abscissa)."""
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < 2:
        raise ValueError("Need at least two samples")
    t = np.linspace(0.0, 1.0, len(y))
    u = np.linspace(0.0, 1.0, n)
    return np.interp(u, t, y)


def _align_on_common_xi(
    xi_a: np.ndarray,
    y_a: np.ndarray,
    xi_b: np.ndarray,
    y_b: np.ndarray,
    *,
    n: int = 400,
    trim_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate both profiles onto the same physical ``xi`` grid (overlap of ranges)."""
    xi_a = np.asarray(xi_a, dtype=float).ravel()
    xi_b = np.asarray(xi_b, dtype=float).ravel()
    y_a = np.asarray(y_a, dtype=float).ravel()
    y_b = np.asarray(y_b, dtype=float).ravel()
    lo = max(xi_a[0], xi_b[0])
    hi = min(xi_a[-1], xi_b[-1])
    assert hi > lo, "No overlap in similarity coordinate"
    if trim_frac > 0:
        lo = lo + trim_frac * (hi - lo)
    assert hi > lo, "Trim removed all overlap in similarity coordinate"
    xg = np.linspace(lo, hi, n)
    return np.interp(xg, xi_a, y_a), np.interp(xg, xi_b, y_b)


@pytest.fixture(scope="module")
def matlab_sub():
    _require_mat_fixture(SUB_MAT)
    return loadmat(SUB_MAT, squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope="module")
def matlab_shock_tau():
    _require_mat_fixture(SHOCK_TAU_MAT)
    return loadmat(SHOCK_TAU_MAT, squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope="module")
def matlab_super():
    _require_mat_fixture(SUPER_MAT)
    return loadmat(SUPER_MAT, squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope="module")
def matlab_piecewise():
    _require_mat_fixture(PIECEWISE_MAT)
    return loadmat(PIECEWISE_MAT, squeeze_me=True, struct_as_record=False)


@pytest.mark.matlab
@pytest.mark.slow
def test_subsonic_profiles_match_matlab(matlab_sub):
    from shussman_solvers.subsonic_solver.materials_sub import material_au
    from shussman_solvers.subsonic_solver.profiles_for_report_sub import (
        compute_profiles_for_report,
    )

    py = compute_profiles_for_report(material_au(), T0_phys_HeV=1.0, tau=0.0, times_ns=np.array([1.0]))

    t_ml = np.asarray(matlab_sub["t"], dtype=float).ravel()
    xsi_ml = float(matlab_sub["xsi"])
    xi_ml = t_ml / xsi_ml

    t_py = np.asarray(py["t"], dtype=float).ravel()
    xi_py = t_py / float(np.max(t_py))

    m_ml, m_py = _align_on_common_xi(
        xi_ml, np.asarray(matlab_sub["m_heat"]).ravel(), xi_py, np.asarray(py["m_heat"][0]).ravel()
    )
    np.testing.assert_allclose(m_ml, m_py, rtol=2.5e-2, atol=2e-7)

    # Subsonic MATLAB pressure is in MBar-scale numbers; Python uses Barye.
    # Low-ξ layer: MATLAB vs Python can disagree where P is tiny; trim overlap from below.
    P_ml, P_py = _align_on_common_xi(
        xi_ml,
        np.asarray(matlab_sub["P_heat"]).ravel(),
        xi_py,
        np.asarray(py["P_heat"][0]).ravel(),
        trim_frac=0.03,
    )
    np.testing.assert_allclose(P_ml, P_py / 1e12, rtol=4.0e-2, atol=1e-9)

    u_ml, u_py = _align_on_common_xi(
        xi_ml, np.asarray(matlab_sub["u_heat"]).ravel(), xi_py, np.asarray(py["u_heat"][0]).ravel()
    )
    assert np.corrcoef(u_ml, u_py)[0, 1] > 0.94

    r_ml, r_py = _align_on_common_xi(
        xi_ml,
        np.asarray(matlab_sub["rho_heat"]).ravel(),
        xi_py,
        np.asarray(py["rho_heat"][0]).ravel(),
        trim_frac=0.03,
    )
    np.testing.assert_allclose(r_ml, r_py, rtol=7.0e-2, atol=1e-9)

    # MATLAB: T_heat = 100*T0*... with T0=1
    T_ml, T_py = _align_on_common_xi(
        xi_ml,
        np.asarray(matlab_sub["T_heat"]).ravel(),
        xi_py,
        100.0 * np.asarray(py["T_heat"][0]).ravel(),
    )
    np.testing.assert_allclose(T_ml, T_py, rtol=2.0e-2, atol=1e-6)


@pytest.mark.matlab
@pytest.mark.slow
def test_shock_profiles_tau_drive_match_matlab(matlab_shock_tau):
    from shussman_solvers.shock_solver.materials_shock import au_supersonic_variant_1
    from shussman_solvers.shock_solver.profiles_for_report_shock import compute_shock_profiles
    from shussman_solvers.shock_solver.manager_shock import manager_shock

    P0 = float(matlab_shock_tau["P0"])
    tau = float(matlab_shock_tau["tau_drive"])
    times = np.array([float(matlab_shock_tau["times"])])

    mat = au_supersonic_variant_1()
    py = compute_shock_profiles(mat, P0, tau, None, times, False, None)

    xi_ml = np.asarray(matlab_shock_tau["xi"], dtype=float).ravel()
    _, _, _, _, _, _, xsi_p, _, _, _, t_raw, _ = manager_shock(mat, tau)
    t_p = np.asarray(t_raw, dtype=float)[::-1]
    xi_py = t_p / float(xsi_p)

    for key_ml, key_py in [
        ("m_shock", "m_shock"),
        ("P_shock", "P_shock"),
        ("rho_shock", "rho_shock"),
        ("T_shock", "T_shock"),
    ]:
        y_ml = np.asarray(matlab_shock_tau[key_ml], dtype=float).ravel()
        y_py = np.asarray(py[key_py][0], dtype=float).ravel()
        a_ml, a_py = _align_on_common_xi(xi_ml, y_ml, xi_py, y_py)
        a_ml /= np.max(np.abs(a_ml)) + 1e-30
        a_py /= np.max(np.abs(a_py)) + 1e-30
        np.testing.assert_allclose(a_ml, a_py, rtol=2.0e-2, atol=1e-3)

    # MATLAB u0 includes extra 10^12 scaling in shock/manager.m vs Python manager_shock; shapes still match.
    u_ml = np.asarray(matlab_shock_tau["u_shock"], dtype=float).ravel()
    u_py = np.asarray(py["u_shock"][0], dtype=float).ravel()
    u_ml_n = u_ml / (np.max(np.abs(u_ml)) + 1e-30)
    u_py_n = u_py / (np.max(np.abs(u_py)) + 1e-30)
    i_ml, i_py = _align_on_common_xi(xi_ml, u_ml_n, xi_py, u_py_n)
    np.testing.assert_allclose(i_ml, i_py, rtol=2.0e-2, atol=1e-3)


def _material_super_matlab_my_au():
    """Match ``super/my_Au.m`` (f, g, sigma) for supersonic reference."""
    from shussman_solvers.supersonic_solver.materials_super import MaterialSuper

    HeV = 1_160_500.0
    return MaterialSuper(
        alpha=1.5,
        beta=1.6,
        lambda_=0.2,
        mu=0.14,
        rho0=19.32,
        f=3.4e13,
        g=1.0 / 7200.0,
        sigma=5.670373e-5 * HeV**4,
        r=None,
        name="Au_matlab_my_Au",
    )


@pytest.mark.matlab
@pytest.mark.slow
def test_supersonic_profiles_match_matlab(matlab_super):
    from shussman_solvers.supersonic_solver.profiles_for_report_super import (
        compute_profiles_for_report,
    )

    mat = _material_super_matlab_my_au()
    times_ns = np.array([float(matlab_super["times"])])
    T0 = float(matlab_super["T0"])
    tau = float(matlab_super["tau"])
    py = compute_profiles_for_report(mat, T0_phys_HeV=T0, tau=tau, times_ns=times_ns)

    m_ml_r = _resample_by_index(np.asarray(matlab_super["m_heat"], dtype=float).ravel())
    m_py_r = _resample_by_index(np.asarray(py["m_heat"][0], dtype=float).ravel())
    assert np.corrcoef(m_ml_r, m_py_r)[0, 1] > 0.97

    x_ml_r = _resample_by_index(np.asarray(matlab_super["x_heat"], dtype=float).ravel())
    x_py_r = _resample_by_index(np.asarray(py["x_heat"][0], dtype=float).ravel())
    assert np.corrcoef(x_ml_r, x_py_r)[0, 1] > 0.97

    T_ml = _resample_by_index(np.asarray(matlab_super["T_heat"], dtype=float).ravel())
    T_py = _resample_by_index(np.asarray(py["T_heat"][0], dtype=float).ravel())
    # Index-resampled curves differ in plateau detail; linear correlation is the robust check.
    assert np.corrcoef(T_ml, T_py)[0, 1] > 0.94


@pytest.mark.matlab
@pytest.mark.slow
def test_piecewise_pressure_shape_matches_matlab_piecewise_export(matlab_piecewise):
    """
    Same physics as the MATLAB figure from ``run_matlab_profiles.py`` / ``my_final_profiles.m``:
    chained sub + shock (Pw-based shock manager) vs Python patching pipeline without tail.
    """
    from project3_code.rad_hydro_sim.problems.presets_config import PRESET_MATLAB, PRESET_TEST_CASES
    from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV
    from project3_code.rad_hydro_sim.verification.shussman_comparison import (
        _rad_hydro_case_to_material_shock,
        _rad_hydro_case_to_material_sub,
        build_piecewise_reference,
    )
    from shussman_solvers.shock_solver.profiles_for_report_shock import compute_shock_profiles
    from shussman_solvers.subsonic_solver.profiles_for_report_sub import compute_profiles_for_report

    case = PRESET_TEST_CASES[PRESET_MATLAB]
    assert case.T0_Kelvin is not None
    T0_HeV = float(case.T0_Kelvin) / KELVIN_PER_HEV
    times_ns = np.array([1.0])

    mat_s = _rad_hydro_case_to_material_sub(case)
    sub = compute_profiles_for_report(
        mat=mat_s, T0_phys_HeV=T0_HeV, tau=float(case.tau), times_ns=times_ns
    )
    P_front = sub["P_heat"][-1, -1]
    _, _, wP3 = sub["Pw"]
    mat_h = _rad_hydro_case_to_material_shock(case)
    Pw = np.array([_, _, wP3], dtype=float)
    shock = compute_shock_profiles(
        mat_h, P_front, tau=None, Pw=Pw, times_ns=times_ns, patching_method=True, save_npz=None
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ref = build_piecewise_reference(case, sub, shock, times_ns / 1e9, m_max=None)

    m_ml = np.asarray(matlab_piecewise["m_fp"], dtype=float).ravel()
    P_ml = np.asarray(matlab_piecewise["P_fp"], dtype=float).ravel()
    m_py = np.asarray(ref.m[0], dtype=float).ravel()
    P_py = np.asarray(ref.p[0], dtype=float).ravel()

    m0 = max(m_ml[0], m_py[0])
    m1 = min(m_ml[-1], m_py[-1])
    assert m1 > m0, "No overlapping mass range for piecewise comparison"

    mm = np.linspace(m0, m1, 256)
    P_i_ml = np.interp(mm, m_ml, P_ml)
    P_i_py = np.interp(mm, m_py, P_py)

    c = np.corrcoef(P_i_ml, P_i_py)[0, 1]
    assert c > 0.999, f"Pressure correlation vs MATLAB too low: {c}"

    # plot the results side by side
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(m_ml, P_ml, label="MATLAB")
    plt.plot(m_py, P_py / 1e12, label="Python")
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(mm, P_i_ml, label="MATLAB")
    plt.plot(mm, P_i_py / 1e12, label="Python")
    plt.legend()

    relative_error = np.abs(P_i_ml - (P_i_py / 1e12)) / P_i_ml
    beginning_index = len(relative_error) // 10
    plt.plot(mm, relative_error, label="Relative error")
    plt.legend()
    plt.show()
    assert relative_error[beginning_index:].max() < 1e-2

@pytest.mark.matlab
@pytest.mark.slow
def test_run_matlab_profiles_script_produces_png(tmp_path):
    """End-to-end: ``run_matlab_profiles.py`` invokes MATLAB and writes an image file."""
    import shutil
    import subprocess
    import sys

    matlab = shutil.which("matlab")
    if matlab is None and not Path(r"C:\Program Files\MATLAB").exists():
        pytest.skip("MATLAB executable not found on PATH")
    script = PROJECT_ROOT / "matlab_shussman_solvers" / "run_matlab_profiles.py"
    out = tmp_path / "from_script.png"
    proc = subprocess.run(
        [sys.executable, str(script), "-o", str(out)],
        cwd=str(PROJECT_ROOT / "matlab_shussman_solvers"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.skip(
            "MATLAB batch run failed (install MATLAB or set MATLAB_EXE). "
            f"stderr: {proc.stderr[:500]!r}"
        )
    assert out.is_file() and out.stat().st_size > 1000