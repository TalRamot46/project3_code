# project3_code — Directory layout and usage

## Directory layout

- **hydro_sim/** — Lagrangian hydrodynamics (run_hydro, driven shock, Sedov, Riemann).
  - **core/** — EOS, grid, geometry, integrator, boundary, timestep, state.
  - **problems/** — Case definitions and simulation config (output paths under `results/`).
  - **simulations/** — Lagrangian sim loop, Riemann exact.
  - **plotting/** — hydro_plots (PNG, GIF).
- **rad_hydro_sim/** — Radiation-hydrodynamics (run_rad_hydro).
  - **simulation/** — iterator, integrator, hydro_steps, radiation_step.
  - **problems/** — RadHydroCase, presets, presets_config.
  - **plotting/** — slider, RadHydroHistory, gif.
  - **verification/** — Compares rad_hydro to references:
    - Radiation-only: 1D Diffusion + Supersonic solver.
    - Hydro-only: hydro_sim + Shussman shock solver (via **hydro_shock/**).
  - **verification/hydro_shock/** — Hydro-only comparison (compare_shock_plots, presets); uses `shussman_solvers.shock_solver`.
- **shussman_solvers/** — Self-similar (semi-analytic) solvers:
  - **shock_solver/** — Hydro/shock solver (used by rad_hydro verification hydro-only).
  - **supersonic_solver/** — Radiation-diffusion solver (used by rad_hydro verification radiation-only).
  - **subsonic_solver/** — Subsonic self-similar solver (Python port of MATLAB).
- **tests/** — Pytest tests (hydro_sim core: eos, grid, boundary, timestep, configs). Run with `pytest` from repo root (see below).
- **results/** — Simulation outputs (figures, data). All runs and verification write here. Git-ignored except README and `.gitkeep`. Layout: `results/hydro_sim/`, `results/rad_hydro_sim_verification/`, `results/hydro_sim_verification/`.
- **docs/** — Documentation. Contains a **Git submodule** pointing to the **"Numerical problems"** LaTeX repo (shared with other projects). The submodule keeps its own history; project3_code only stores a reference. See **docs/README.md** for one-time setup (`git submodule add ...`), cloning with `--recurse-submodules`, and how to commit LaTeX changes. No separate branch in project3_code is needed for the .tex files.

All verification is triggered from **rad_hydro_sim/verification/run_comparison.py** (set `MODE = RADIATION_ONLY` or `HYDRO_ONLY` in `main()`).

---

## Repo root and package name

- The **repository root** is the directory that **contains** the `project3_code` folder. So you have `repo_root/project3_code/hydro_sim`, `project3_code/rad_hydro_sim`, etc.
- The top-level Python package is **project3_code**. Imports use `from project3_code.hydro_sim...`, `from project3_code.rad_hydro_sim.simulation...`, etc.
- All commands below assume you are in the **repo root** (parent of `project3_code`), so that `project3_code` is on `sys.path` when you run `python -m project3_code....`
- If your workspace is only the `project3_code` folder: then “repo root” for running is the parent of that folder; run `make` and `pytest` from that parent, or set `PYTHONPATH` / install the package (e.g. `pip install -e .` from repo root).

---

## How to run

| Goal | Command (from repo root) |
|------|---------------------------|
| Run hydro | `python -m project3_code.hydro_sim.run_hydro` |
| Run rad-hydro | `python -m project3_code.rad_hydro_sim.run_rad_hydro` |
| Run verification | `python -m project3_code.rad_hydro_sim.verification.run_comparison` |
| Run tests | `python -m pytest project3_code/tests -v` (ensure `project3_code` is on path, e.g. `pythonpath = ["."]` in pytest or run from repo root) |
| Format / lint | `ruff format project3_code/` and `ruff check project3_code/` |

### Makefile

Run `make` from **inside** `project3_code`. The Makefile sets `REPO_ROOT := $(abspath ..)` (parent of `project3_code`) and runs the Python commands from there so that `project3_code` is importable.

- `make run-hydro` — run hydro_sim.
- `make run-rad-hydro` — run rad_hydro_sim.
- `make run-verification` — run verification comparison.
- `make test` — run pytest on `project3_code/tests`.
- `make fmt` — format and auto-fix with Ruff.
- `make lint` — Ruff check only.

---

## Outputs (results/)

- **hydro_sim** and **rad_hydro_sim** (when using the shared `SimulationConfig` and `with_output_paths`) write figures to **results/hydro_sim/** (png/, gif/).
- **Verification** (radiation-only and hydro-only) writes to **results/rad_hydro_sim_verification/** and **results/hydro_sim_verification/**.
- `get_results_dir()` is defined in `hydro_sim.problems.simulation_config` and used by verification configs so all outputs stay under `results/`.

---

## Tests

- **Location:** `project3_code/tests/` (e.g. `test_eos.py`, `test_grid.py`, `test_boundary.py`, `test_timestep.py`, `test_configs.py`).
- **Run:** From repo root: `python -m pytest project3_code/tests -v`, or `make test` from inside `project3_code`.
- **Purpose:** Unit/regression tests for hydro_sim core. Add new tests here for new features; tests should not depend on writing plots (use temp dirs or mocks if needed).
