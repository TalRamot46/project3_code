#!/usr/bin/env python3
"""
Simple test runner. Run all or individual tests.

Usage:
  python run_tests.py                    # Run all tests
  python run_tests.py --fast             # Skip slow tests (verification simulations)
  python run_tests.py -k diffusion       # Run tests matching "diffusion"
  python run_tests.py tests/             # All tests in tests/
  python run_tests.py tests/test_configs.py
  python run_tests.py tests/test_verification_fit.py::test_radiation_diffusion_runs

Or after pip install -e .:  run-tests, run-tests --fast
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    args = list(sys.argv[1:])
    if "--fast" in args:
        args.remove("--fast")
        args = ["-m", "not slow", *args]
    cmd = [sys.executable, "-m", "pytest", "-v", *args]
    return subprocess.run(cmd, cwd=project_root).returncode


if __name__ == "__main__":
    sys.exit(main())
