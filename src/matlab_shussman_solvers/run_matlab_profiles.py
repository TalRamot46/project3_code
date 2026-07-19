#!/usr/bin/env python3
"""
Run the legacy MATLAB Shussman solvers (subsonic report -> shock report -> my_final_profiles)
and save the resulting figure for comparison with rad_hydro_sim.

The MATLAB chain must run with correct working directories so each folder's Au.m and
manager.m are used (see run_shussman_final_profiles.m).

Requirements:
  - MATLAB installed; executable on PATH as ``matlab``, or set environment variable
    MATLAB_EXE to the full path of matlab.exe (Windows) or the ``matlab`` binary (Unix).

Usage:
  python run_matlab_profiles.py
  python run_matlab_profiles.py -o comparison.png
  python run_matlab_profiles.py --show-matlab-log
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_matlab_root() -> Path:
    """Directory containing run_shussman_final_profiles.m (this file's folder)."""
    return Path(__file__).resolve().parent


def find_matlab_executable() -> str | None:
    env = os.environ.get("MATLAB_EXE", "").strip()
    if env and Path(env).is_file():
        return env
    which = shutil.which("matlab")
    if which:
        return which
    if sys.platform == "win32":
        prog = os.environ.get("ProgramFiles", r"C:\Program Files")
        matlab_dir = Path(prog) / "MATLAB"
        if matlab_dir.is_dir():
            for release in sorted(matlab_dir.iterdir(), reverse=True):
                cand = release / "bin" / "matlab.exe"
                if cand.is_file():
                    return str(cand)
    return None


def matlab_path_string(p: Path) -> str:
    """Single-quoted path safe for MATLAB -batch string (forward slashes)."""
    s = p.resolve().as_posix()
    return s.replace("'", "''")


def run_profiles(
    output_png: Path,
    *,
    matlab_exe: str | None = None,
    capture_log: bool = False,
) -> None:
    root = _repo_matlab_root()
    driver = root / "run_shussman_final_profiles.m"
    if not driver.is_file():
        raise FileNotFoundError(f"Missing MATLAB driver: {driver}")

    out = output_png.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    matlab = matlab_exe or find_matlab_executable()
    if not matlab:
        raise RuntimeError(
            "MATLAB not found. Add MATLAB to PATH, or set MATLAB_EXE to the full path "
            "to the MATLAB executable."
        )

    out_quoted = matlab_path_string(out)
    root_quoted = matlab_path_string(root)

    # addpath then call driver so mfilename('fullpath') resolves inside the package root
    stmt = (
        f"addpath('{root_quoted}'); "
        f"run_shussman_final_profiles('{out_quoted}'); "
        "exit;"
    )

    cmd = [matlab, "-batch", stmt]
    kwargs: dict = {}
    if capture_log:
        kwargs["capture_output"] = True
        kwargs["text"] = True

    proc = subprocess.run(cmd, **kwargs)
    if capture_log and proc.stdout:
        print(proc.stdout, end="")
    if capture_log and proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MATLAB exited with code {proc.returncode}. "
            f"Command: {matlab} -batch \"...\""
        )
    if not out.is_file():
        raise RuntimeError(f"Expected output image was not created: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MATLAB Shussman profile scripts and save the pressure comparison figure."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: shussman_pressure_profiles.png next to this script)",
    )
    parser.add_argument(
        "--matlab",
        type=str,
        default=None,
        help="Override MATLAB executable path (same as MATLAB_EXE)",
    )
    parser.add_argument(
        "--show-matlab-log",
        action="store_true",
        help="Print MATLAB stdout/stderr from -batch run",
    )
    args = parser.parse_args()

    root = _repo_matlab_root()
    out = args.output if args.output is not None else root / "shussman_pressure_profiles.png"

    try:
        run_profiles(
            out,
            matlab_exe=args.matlab,
            capture_log=args.show_matlab_log,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
