# output_paths.py
"""
Central paths for rad_hydro_sim outputs.

- NPZ and other simulation data: rad_hydro_sim/data/
- Figures (PNG, GIF): results/rad_hydro_sim/figures/ (png and gif subdirs)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple


def get_rad_hydro_root() -> Path:
    """Root of the rad_hydro_sim package."""
    return Path(__file__).resolve().parent


def get_rad_hydro_data_dir() -> Path:
    """Directory for NPZ and other simulation data (inside rad_hydro_sim)."""
    data_dir = get_rad_hydro_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_rad_hydro_results_dir() -> Path:
    """Base directory for rad_hydro_sim figures under project results/."""
    from project_3.hydro_sim.problems.simulation_config import get_results_dir
    base = get_results_dir() / "rad_hydro_sim"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_rad_hydro_figures_dir() -> Path:
    """Figures directory: results/rad_hydro_sim/figures (with png/ and gif/ inside)."""
    base = get_rad_hydro_results_dir() / "figures"
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_rad_hydro_output_paths(case_name: str) -> Tuple[Path, Path]:
    """
    (png_path, gif_path) for a given case name.
    Uses results/rad_hydro_sim/figures/png and .../gif with a safe filename.
    """
    base = get_rad_hydro_figures_dir()
    png_dir = base / "png"
    gif_dir = base / "gif"
    png_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    safe = (
        (case_name or "rad_hydro_run")
        .replace(" ", "_")
        .replace("=", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(":", "")
    )
    return png_dir / f"{safe}.png", gif_dir / f"{safe}.gif"


def get_rad_hydro_npz_path(base_name: str, prefix: str = "sim_data") -> Path:
    """Full path for an NPZ file in rad_hydro_sim/data. E.g. prefix='sim_data', base_name='fig_10' -> data/sim_data_fig_10.npz."""
    safe = (base_name or "run").replace(" ", "_").replace("=", "").replace("(", "").replace(")", "").replace(",", "").replace(":", "")
    return get_rad_hydro_data_dir() / f"{prefix}_{safe}.npz"
