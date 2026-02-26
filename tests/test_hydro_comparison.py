"""Unit consistency tests for hydro-only verification (rad_hydro, hydro_sim, shock solver).

Verifies that P0*t^tau boundary conditions and units are consistent across:
- rad_hydro integrator (P0_Barye, tau, t in seconds -> p_drive in Barye)
- hydro_sim driven shock (P0, tau, t in seconds -> p_drive in Barye)
- shock solver (P0_phys_Barye, tau, times_ns -> P_prof in Barye)
"""
import pytest
import numpy as np

from project_3.rad_hydro_sim.verification.verification_config import (
    VerificationMode,
    get_preset_for_mode,
)
from project_3.rad_hydro_sim.problems.presets_utils import get_preset
from project_3.rad_hydro_sim.verification.run_comparison import (
    run_shock_solver_hydro_reference,
    _rad_hydro_case_to_shock_material,
)


def test_hydro_only_preset_has_pressure_drive():
    """Hydro-only preset must have P0_Barye and tau for shock solver."""
    preset_name = get_preset_for_mode(VerificationMode.HYDRO_ONLY)
    case, _ = get_preset(preset_name)
    assert case.P0_Barye is not None
    assert case.tau is not None
    assert case.P0_Barye > 0


def test_shock_material_from_rad_hydro_case():
    """_rad_hydro_case_to_shock_material builds Material from f_Kelvin, g_Kelvin."""
    preset_name = get_preset_for_mode(VerificationMode.HYDRO_ONLY)
    case, _ = get_preset(preset_name)
    mat = _rad_hydro_case_to_shock_material(case)
    assert mat is not None
    assert mat.alpha == case.alpha
    assert mat.beta == case.gamma
    assert mat.lambda_ == case.lambda_
    assert mat.mu == case.mu
    assert 1.0 / mat.V0 == case.rho0


def test_shock_solver_p0_tau_order():
    """Shock solver output: P should be ~ O(P0*t_ns^tau) [Barye] at drive boundary."""
    preset_name = get_preset_for_mode(VerificationMode.HYDRO_ONLY)
    case, _ = get_preset(preset_name)
    P0 = float(case.P0_Barye)
    tau = float(case.tau)

    t_sec = 0.5e-9
    t_ns = t_sec * 1e9
    times_sec = np.array([t_sec])

    shock_data = run_shock_solver_hydro_reference(case, times_sec)
    if shock_data is None:
        pytest.skip("Shock solver not available")

    # P = P0 * t^tau * P_tilde; P_tilde is O(1). Check scale.
    p_scale = P0 * (t_ns**tau)
    p_max = np.max(shock_data.p[0])
    # p_max should be within 0.01x to 100x of P0*t^tau (P_tilde typically 0.1..10)
    assert p_scale * 0.01 <= p_max <= p_scale * 100, (
        f"P_max={p_max:.2e} out of expected range [{p_scale*0.01:.2e}, {p_scale*100:.2e}]"
    )


def test_shock_solver_times_in_seconds():
    """Shock solver output times should be in seconds."""
    preset_name = get_preset_for_mode(VerificationMode.HYDRO_ONLY)
    case, _ = get_preset(preset_name)
    times_sec = np.array([0.1e-9, 0.5e-9, 1.0e-9])

    shock_data = run_shock_solver_hydro_reference(case, times_sec)
    if shock_data is None:
        pytest.skip("Shock solver not available")

    # Output times should match input (within numerical tolerance)
    np.testing.assert_allclose(shock_data.times, times_sec, rtol=1e-10)
    assert len(shock_data.p) == len(times_sec)
    assert len(shock_data.m) == len(times_sec)
