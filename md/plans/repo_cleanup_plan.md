# Repository layout: what moved on 2026-11-19, and the plan for the legacy folders

## Done
| from | to |
|---|---|
| `results/test3_*/`, `results/light_analytic_start/` (run data) and `results/phase_c/figures/test3_*/` (figures) | `results/misalignment/<study>/{runs,figures}/` |
| `md/phase_c/test3_*.md`, `md/phase_c/why_the_solver_differs.md` | `md/misalignment/`; on 2026-11-20 rewritten by topic and numbered in order of first appearance (README + 01_surface_equilibrium … 06_runs; figures in context); the original notebook is in `md/misalignment/archive/` |
| `src/ictt29/` | `src/fitting/{test_docs,core,legacy_ictt29}/` (see `src/fitting/README.md`; generator re-run on Test 1: same numbers to ~1e-6) |
| `src/results/` | `results/legacy/src_results/`; cause fixed (`hydro_sim/problems/simulation_config.get_results_dir` pointed at `src/`) |
| `Claude outputs/` | `results/legacy/claude_session_exports/` |
| `results/overnight/`, `results/phase_c_logs/` | `results/logs/{overnight,phase_c}/` |
| `src/rad_hydro_sim/data/{sim,ref}_data_*.npz` (2026-11-20) | `results/rad_hydro_sim/runs/`; `output_paths.get_rad_hydro_data_dir` now points there. `src/rad_hydro_sim/data/` keeps only inputs (bath-correction tables) |
| `src/fitting/test_docs/cache/` (2026-11-20) | `results/fitting/cache/` (new default of `paths.CACHE`) |

The script constants (`OUT`, `FIGURES`) and `.gitignore` follow the new paths. An empty `src/ictt29/` folder is left behind; delete it by hand.

## Plan for the legacy code (not done; each step is one commit, run `pytest` after each)

Guiding rule: `src/` keeps what the two current lines of work import, the misalignment study (rad_hydro_sim, imc, hydro_sim, menahem_new, sn) and the paper (fitting/test_docs). Everything else goes to `src/legacy/`, with a one-line README saying what it was for.

1. **`src/comparison/`** (June: Marshak, Fig. 8 χ, Menahem vs Shussman, rt vs rad-hydro). Imported by nothing outside itself. → `src/legacy/comparison/`. (The `from comparison.presets import …` in `hydro_sim/verification/presets.py` is only in its docstring, and that module does not exist.)
2. **`src/malka_article/`** (April, one script) → `src/legacy/malka_article/`.
3. **`src/matlab_shussman_solvers/`** (MATLAB originals + exports): reference material for `shussman_solvers/`, which `tests/test_shussman_solvers.py` and `hydro_sim/verification` still import. Move the MATLAB folder to `src/legacy/`, or out of `src/` to `reference/`. Keep `shussman_solvers/`.
4. **`src/menahem_new/`**: it is the solver both studies use, so it stays, but it holds four generations of the same files. `subsonic_heat_wave_mine.py` and `instantaneous_d3_to_d1.py` have no importers → `menahem_new/legacy/`. `_og` (19 importers, used by `fitting/`) and the unsuffixed versions are both live: decide which one is the paper's solver, then rename (e.g. `menahem_new` → `solver`) in one pass. That rename touches ~50 imports, so do it after the paper's figures are frozen.
5. **`src/scratch/`** (38 one-off scripts) → triage once: anything a doc cites moves next to that doc's code, the rest to `src/legacy/scratch/`.
6. **`src/fitting/legacy_ictt29/`**: keep until the paper's fit section is final (it holds the energy and Eulerian checks); then archive with `core/`.
7. **Results**: `results/ictt/` (3 GB, the ICTT-29 runs and caches) and `results/imc_vs_diffusion_n1000/`: move to `results/legacy/` once nobody runs `fitting/legacy_ictt29` (its `data_loader` reads `results/ictt/cache` relative to the working directory). `results/rad_hydro_sim_verification/` and `results/legacy/src_results/` are timestamped GIF/PNG dumps of the same runs: keep the newest per case, drop the rest.
8. **`md/phase_c/legacy/` and `results/phase_c/figures/legacy/`**: already well separated; leave.

Suggested top-level shape when done:
```
results/  phase_b/  phase_c/  misalignment/  (paper/)  logs/  legacy/  + untracked run-data folders
md/       theory/ imc_code/ phase_b/ phase_c/ misalignment/ plans/
src/      imc/ sn/ hydro_sim/ rad_hydro_sim/ menahem_new/ shussman_solvers/ fitting/ steinberg_comparison/ legacy/
```
A `results/paper/` (figures exactly as `subsonic_self_similar_paper/figs/` uses them, with the script that makes each) would close the loop between the two lines of work.
