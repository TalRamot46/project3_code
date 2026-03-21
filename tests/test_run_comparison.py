"""
Automated end-to-end verification tests that run each VerificationMode
and assert that PNG and GIF artefacts are produced.

Each test invokes ``run_comparison`` with a single mode (RADIATION_ONLY,
HYDRO_ONLY, FULL_RAD_HYDRO), saves outputs to *results/*, and checks
that at least one new .png and .gif file was created.

Output naming convention (produced by the existing helpers):
    results/rad_hydro_sim_verification/png/<mode>_<case_title>_<timestamp>.png
    results/rad_hydro_sim_verification/gif/<mode>_<case_title>_<timestamp>.gif

Skip slow tests project-wide:
    pytest -m "not slow"

Run only these tests (use -s so tqdm bars and prints are visible):
    pytest tests/test_run_comparison.py -v -s
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Set

import matplotlib
import pytest

matplotlib.use("Agg")

from project3_code.rad_hydro_sim.verification.verification_config import (
    VerificationMode,
    get_verification_output_dir,
)
from project3_code.rad_hydro_sim.verification.run_comparison import run_comparison


def _collect_files(directory: Path, suffix: str) -> Set[str]:
    """Return the set of file paths (as strings) under *directory* with the given suffix."""
    if not directory.exists():
        return set()
    return {str(p) for p in directory.rglob(f"*{suffix}")}


def _print_summary(mode_name: str, elapsed: float, new_pngs: Set[str], new_gifs: Set[str]) -> None:
    """Print a human-readable summary after a test run."""
    print(f"\n{'=' * 60}")
    print(f"  {mode_name}  —  finished in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    for p in sorted(new_pngs):
        print(f"  PNG  {p}")
    for g in sorted(new_gifs):
        print(f"  GIF  {g}")
    print()


@pytest.fixture()
def output_dirs():
    """Yield (png_dir, gif_dir) and ensure they exist before the test."""
    base = get_verification_output_dir()
    png_dir = base / "png"
    gif_dir = base / "gif"
    png_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    return png_dir, gif_dir


# ---------------------------------------------------------------------------
# One test per VerificationMode
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_radiation_only_produces_outputs(output_dirs):
    """RADIATION_ONLY: full pipeline produces at least one PNG and one GIF."""
    png_dir, gif_dir = output_dirs
    pngs_before = _collect_files(png_dir, ".png")
    gifs_before = _collect_files(gif_dir, ".gif")

    t0 = time.perf_counter()
    run_comparison(
        VerificationMode.RADIATION_ONLY,
        show_plot=False,
        save_png=True,
        save_gif=True,
    )
    elapsed = time.perf_counter() - t0

    new_pngs = _collect_files(png_dir, ".png") - pngs_before
    new_gifs = _collect_files(gif_dir, ".gif") - gifs_before

    _print_summary("RADIATION_ONLY", elapsed, new_pngs, new_gifs)

    assert new_pngs, "No new PNG file was created for RADIATION_ONLY"
    assert new_gifs, "No new GIF file was created for RADIATION_ONLY"
    for p in new_pngs:
        assert os.path.getsize(p) > 0, f"PNG file is empty: {p}"
    for g in new_gifs:
        assert os.path.getsize(g) > 0, f"GIF file is empty: {g}"


@pytest.mark.slow
def test_hydro_only_produces_outputs(output_dirs):
    """HYDRO_ONLY: full pipeline produces at least one PNG and one GIF."""
    png_dir, gif_dir = output_dirs
    pngs_before = _collect_files(png_dir, ".png")
    gifs_before = _collect_files(gif_dir, ".gif")

    t0 = time.perf_counter()
    run_comparison(
        VerificationMode.HYDRO_ONLY,
        show_plot=False,
        save_png=True,
        save_gif=True,
    )
    elapsed = time.perf_counter() - t0

    new_pngs = _collect_files(png_dir, ".png") - pngs_before
    new_gifs = _collect_files(gif_dir, ".gif") - gifs_before

    _print_summary("HYDRO_ONLY", elapsed, new_pngs, new_gifs)

    assert new_pngs, "No new PNG file was created for HYDRO_ONLY"
    assert new_gifs, "No new GIF file was created for HYDRO_ONLY"
    for p in new_pngs:
        assert os.path.getsize(p) > 0, f"PNG file is empty: {p}"
    for g in new_gifs:
        assert os.path.getsize(g) > 0, f"GIF file is empty: {g}"


@pytest.mark.slow
def test_full_rad_hydro_produces_outputs(output_dirs):
    """FULL_RAD_HYDRO: full pipeline produces at least one PNG and one GIF."""
    png_dir, gif_dir = output_dirs
    pngs_before = _collect_files(png_dir, ".png")
    gifs_before = _collect_files(gif_dir, ".gif")

    t0 = time.perf_counter()
    run_comparison(
        VerificationMode.FULL_RAD_HYDRO,
        show_plot=False,
        save_png=True,
        save_gif=True,
    )
    elapsed = time.perf_counter() - t0

    new_pngs = _collect_files(png_dir, ".png") - pngs_before
    new_gifs = _collect_files(gif_dir, ".gif") - gifs_before

    _print_summary("FULL_RAD_HYDRO", elapsed, new_pngs, new_gifs)

    assert new_pngs, "No new PNG file was created for FULL_RAD_HYDRO"
    assert new_gifs, "No new GIF file was created for FULL_RAD_HYDRO"
    for p in new_pngs:
        assert os.path.getsize(p) > 0, f"PNG file is empty: {p}"
    for g in new_gifs:
        assert os.path.getsize(g) > 0, f"GIF file is empty: {g}"
