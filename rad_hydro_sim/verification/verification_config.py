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


class VerificationMode(str, Enum):
    """Which verification comparison to run."""
    RADIATION_ONLY = "radiation_only"   # vs 1D Diffusion
    HYDRO_ONLY = "hydro_only"            # vs hydro_sim (and later Shussman)
    FULL_RAD_HYDRO = "full_rad_hydro"   # vs Shussman subsonic + shock (constant T drive)


# ============================================================================
# Output paths (under results/)
# ============================================================================

def get_verification_output_dir() -> Path:
    """Base output directory for verification figures."""
    from project_3.hydro_sim.problems.simulation_config import get_results_dir
    base = get_results_dir() / "verification"
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_verification_output_paths(case_name: str) -> Tuple[Path, Path]:
    """(png_path, gif_path) for a given case name."""
    base = get_verification_output_dir()
    png_dir = base / "png"
    gif_dir = base / "gif"
    png_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    safe = case_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
    return png_dir / f"{safe}.png", gif_dir / f"{safe}.gif"


# ============================================================================
# Preset names used for verification
# ============================================================================

RADIATION_ONLY_PRESET = "radiation_only_constant_temperature_drive"
HYDRO_ONLY_PRESET = "hydro_only_power_law_pressure_drive"
FULL_RAD_HYDRO_PRESET = "rad_hydro_constant_temperature_drive"
