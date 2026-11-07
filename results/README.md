# Results

Every test result lives here; `md/` holds only the text that discusses them. Git tracks the figures and tables the docs show. The run data (`.npz`, solver pickles, logs, caches) stays untracked (`.gitignore`). A copy of the run data that the current figures are drawn from is on the GitHub branch `data/runs-backup`; to restore it:

```
git fetch project3_code data/runs-backup
git archive project3_code/data/runs-backup results | tar -x
```

## Figures and tables (tracked)

| folder | what | written by | discussed in |
|---|---|---|---|
| `phase_b/figures/` | Phase B: diffusion conventions, IMC on SiO₂ and on gold | `rad_hydro_sim.verification.imc_vs_diffusion --figures-dir results/phase_b/figures` | `md/phase_b/phase_b_results.md` |
| `imc_code/figures/` | checks of the IMC code: the 0-D problem, one linear step against S_N | `imc.problems.zero_d`, `imc.problems.linear_step` | `md/imc_code/` |
| `phase_c/figures/` | Phase C: full profiles of Tests 1–6 per bath, overlays, every view of Tests 3–4, and legacy figures. Layout: `phase_c/figures/README.md` | `rad_hydro_sim.verification.bath_calibration --profiles / --overlay / --views` (legacy: `verification/legacy/`) | `md/phase_c/` (gallery: `md/phase_c/profile_figures.md`) |
| `phase_c/omega_sweep/` | the gold ω sweep: fronts against time, deviations against ω, mesh convergence, full profiles | `rad_hydro_sim.verification.omega_sweep --figures` | `md/phase_c/omega_sweep/README.md` |
| `hydro_sim/png/` | the hydro test problems (Sod, Sedov, pressure drives, …) | `hydro_sim` runs (`simulation_config.make_output_paths`) | |
| `hydro_sim_verification/png/` | hydro shock verification against the semi-analytic shock | `hydro_sim.verification` (`comparison_config`) | |
| `rad_hydro_sim/figures/png/` | `rad_hydro_sim` runs | `rad_hydro_sim.run_rad_hydro` (`output_paths`) | |
| `steinberg_comparison/` | the Steinberg comparison figures and the profile tables they are drawn from | `src/steinberg_comparison/steinberg_comparison.py` | |

## Run data (untracked)

| folder | what | written by |
|---|---|---|
| `imc_vs_diffusion/`, `imc_vs_diffusion_calibrated/`, `imc_vs_diffusion_calibrated_flux/` | the rad-hydro runs of Tests 1–6 under the paper, T-calibrated and flux-calibrated baths, on the default and the fine meshes (`test*_full`, `test*_fine`, `test3_smooth`); `logs/` | `imc_vs_diffusion.run_hydro`, `bath_calibration --jobs` |
| `omega_sweep/`, `omega_sweep_calibrated_flux/` | the ω-sweep runs, default and fine meshes, and `solvers/` (one pickled solver per ω); `logs/` | `omega_sweep --run` (`--jobs`, `--fine`) |
| `patching_poc/` | pickled solvers of Tests 1–6 and the bath tables | `bath_calibration.solver_for`, `bath_calibration` |
| `bath_calibration/` | the per-case bath tables | `bath_calibration` |
| `imc_vs_diffusion_n1000/` | the earlier N = 1000 runs | `imc_vs_diffusion` |
| `ictt/` | the ICTT-29 presentation runs (~3 GB) | `src/ictt29/` |
| `rad_hydro_sim/`, `rad_hydro_sim_verification/` | `rad_hydro_sim` run outputs | `rad_hydro_sim` |
| `hydro_sim/gif/`, `hydro_sim_verification/gif/`, `hydro_sim_verification/shock_profiles.npz` | animations and the shock profiles of the hydro checks | `hydro_sim` |
| `menahem_new/position_verification/` | the front-position check of the analytic solver | `src/menahem_new/position_evolution_verification.py` |
| `overnight/`, `phase_c_logs/` | logs of long runs | |

Inputs are not results and stay with the code: the bath-correction tables (`src/rad_hydro_sim/data/bath_correction/`), the SiO₂ drive (`src/imc/data/`), and the test fixtures (`tests/data/`).
