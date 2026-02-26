# verification/verification_config.py
"""
Configuration for rad_hydro_sim verification comparisons.

Verification modes:
  1. Radiation-only: run_rad_hydro (radiation_only_constant_temperature_drive) vs
     1D Diffusion self similar in gold (constant temperature drive) vs
     Supersonic solver (radiation self-similar, same physics).
  2. Hydro-only: run_rad_hydro (hydro_only_power_law_pressure_drive) vs
     hydro_sim run_hydro (matching driven shock case).
  3. Full rad_hydro: run_rad_hydro (constant temperature drive) vs piecewise Shussman
     reference (subsonic solver up to shock front, then shock solver driven by
     pressure at front from subsonic; shock front diagnosed from rad_hydro).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum
from datetime import datetime

from project_3.rad_hydro_sim.problems.presets_config import (
    PRESET_FIRST_ATTEMPT,
    PRESET_CONSTANT_T_RADIATION,
    PRESET_CONSTANT_PRESSURE,
    PRESET_POWER_LAW,
    PRESET_FIG_8,
    PRESET_FIG_9,
    PRESET_FIG_10,
)


class VerificationMode(str, Enum):
    """Which verification comparison to run."""
    RADIATION_ONLY = "radiation_only"   # vs 1D Diffusion
    HYDRO_ONLY = "hydro_only"            # vs hydro_sim (and later Shussman)
    FULL_RAD_HYDRO = "full_rad_hydro"   # vs Shussman subsonic + shock (constant T drive)


# ============================================================================
# Preset names (physical case keys) and mode → preset mapping
# ============================================================================

RADIATION_ONLY_PRESET = PRESET_CONSTANT_T_RADIATION   # constant_temperature_drive
HYDRO_ONLY_PRESET = PRESET_POWER_LAW                  # constant_pressure_drive
FULL_RAD_HYDRO_PRESET = PRESET_FIG_8                 # fig_8_comparison

# Map each verification mode to its preset name (SIMPLE_TEST_CASES key)
MODE_TO_PRESET: dict[VerificationMode, str] = {
    VerificationMode.RADIATION_ONLY: RADIATION_ONLY_PRESET,
    VerificationMode.HYDRO_ONLY: HYDRO_ONLY_PRESET,
    VerificationMode.FULL_RAD_HYDRO: FULL_RAD_HYDRO_PRESET,
}


def get_preset_for_mode(mode: VerificationMode) -> str:
    """Return the preset name for the given verification mode."""
    return MODE_TO_PRESET[mode]


def get_output_prefix_for_mode(mode: VerificationMode) -> str:
    """Return the output path prefix for the given mode (e.g. 'radiation_only', 'hydro_only', 'full_rad_hydro')."""
    return mode.value


# ============================================================================
# Output paths (under results/)
# ============================================================================

def get_verification_output_dir() -> Path:
    """Base output directory for rad_hydro_sim verification figures."""
    from project_3.hydro_sim.problems.simulation_config import get_results_dir
    base = get_results_dir() / "rad_hydro_sim_verification"
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_verification_output_paths(case_name: str) -> Tuple[Path, Path]:
    """(png_path, gif_path) for a given case name.

    Includes a timestamp in the filename so repeated runs of the same
    preset do not overwrite previous results.
    """
    base = get_verification_output_dir()
    png_dir = base / "png"
    gif_dir = base / "gif"
    png_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    safe = case_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe}_{timestamp}"
    return png_dir / f"{filename}.png", gif_dir / f"{filename}.gif"


