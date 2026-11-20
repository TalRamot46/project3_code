# Runs, meshes and how to reproduce them

Reference for the topic docs. Branch `feat/patching-poc`. Run data (`.npz`, untracked) is in `results/misalignment/<study>/runs/`, figures and tables in `.../figures/` ([results/misalignment/README.md](../../results/misalignment/README.md)). Phase C runs (Tests 1–6 under three baths) are in `results/imc_vs_diffusion{,_calibrated,_calibrated_flux}/` ([../phase_c/bath_calibration.md](../phase_c/bath_calibration.md)).

## Meshes (Test 3)

| name | where used | layout |
|---|---|---|
| old (`test3_smooth`, 1237 cells) | Phase C, fix A | 139 surface cells of 0.0216 mg/cm² to ~3 mg/cm²; 5% growth to ~10; 0.3 mg/cm² beyond |
| old default (`test3_full`, 347 cells) | Phase C | same surface cells, coarse shocked region |
| graded (`mesh="log_mass"`, 1760 cells) | cold starts from 14 Nov | first cell 4×10⁻⁴ mg/cm², 5% growth to 0.0216, equal to 1.5 mg/cm²; 0.3% growth to 0.3 mg/cm² |
| graded fine (1847 cells) | `*_fine` | first cell 1×10⁻⁴ mg/cm², 3% growth |
| seeded (1698 cells) | analytic starts | equal 0.0216 mg/cm² surface cells (the solver's table does not resolve smaller), then as graded |

IMC everywhere: 2000 emission + 2000 bath packets per step, census capped at 20 000 (×4 where marked); 3 seeds. Al runs: 5000 + 5000.

## Studies

| study (module in `src/test_misalignment/`; run as `python -m test_misalignment.<study>`) | runs | cost | doc |
|---|---|---|---|
| `rad_hydro_sim.verification.bath_calibration` (Phase C), `test3_smooth_analytic` | fix A: diffusion, IMC seeds 1–3 | 56 min; IMC 16.7 h per seed | [04_analytic_start.md § 2](04_analytic_start.md) |
| `test3_hydro_only` | 9 hydro-only runs: `{cold,seeded}_{p_solver,p_diffusion,p_diffusion_raw,p_imc,piston}`, `seeded_p_solver_30K` | 12–19 min each | [05_shocked_region.md § 1](05_shocked_region.md) |
| `test3_on_the_solver` | `cold_1T_{30K,300K}`, `cold_2T_30K[_chi10,_chi100,_chi1000,_chi100_fine]`, `cold_IMC_30K_seed1-3`, `seeded_2T_{30K,300K}` (sharp, on a face), `seeded_2T_30K_smeared[_chi100]`, `seeded_IMC_30K_smeared_seed1-3`, `seeded_IMC_30K_smeared_x4_seed1` | 16–105 min; 1T 88 min; IMC ×4 4 h | [03_startup.md](03_startup.md), [04_analytic_start.md](04_analytic_start.md) |
| `test3_omega_scan` | ω = 0.4…0.05, IMC ×3 and 2T, to 1 ns (ω = 0.5 from `test3_on_the_solver` at 1 ns) | ~1 h on 2 cores | [03_startup.md § 5](03_startup.md) |
| `test3_transparency` | `dense_cold_2T_30K[_chi100]`, `dense_cold_IMC_30K_seed1` (profiles every 1000 steps); `--pdrive` twins `{cold,seeded}_{rad,hydro}_p_{solver,2T}` | 22 / 26 / 103 min; twins 14–87 min | [01_surface_equilibrium.md § 2](01_surface_equilibrium.md), [03_startup.md § 3](03_startup.md), [05_shocked_region.md § 2](05_shocked_region.md) |
| `light_analytic_start` | `{al,be}_{cold,analytic}_paper[_n2500]`, `al_analytic_calibrated[_flux]`, `be_analytic_calibrated_flux`; calibration loops | ~1 min each | [01_surface_equilibrium.md § 3](01_surface_equilibrium.md) |
| `al_long` | `{cold,seeded}_{2T,chi1000,imc1-3}`, Al to 50 ns | 2 min (2T); 18–23 min (IMC) | [01_surface_equilibrium.md § 3](01_surface_equilibrium.md), [02_patching.md § 4](02_patching.md) |

Energy is conserved to ~10⁻¹³ (2T), ~10⁻⁹ (IMC, Test 3) and ~10⁻¹¹ (Al). It is not measured for the 1T mode.

## Commands

```
# fix A (Phase C machinery)
uv run python -m rad_hydro_sim.verification.bath_calibration --run-one test3_smooth_analytic paper 0     # 0 = diffusion, 1-3 = IMC seed
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test3_smooth_analytic --baths paper --views
uv run pytest tests/test_analytic_ablation_start.py

# hydro only
uv run python -m test_misalignment.test3_hydro_only --run-all --jobs 4
uv run python -m test_misalignment.test3_hydro_only --figures --table

# the six causes, chi scan, IMC
uv run python -m test_misalignment.test3_on_the_solver --run-all --jobs 4
uv run python -m test_misalignment.test3_on_the_solver --figures --table
uv run python -m test_misalignment.test3_on_the_solver --slider <run>      # keys: f fit y, x/y log-lin, arrows step

# omega
uv run python -m test_misalignment.test3_omega_scan --run-all --jobs 2
uv run python -m test_misalignment.test3_omega_scan --figures

# transparency, pressure-driven twins
uv run python -m test_misalignment.test3_transparency --dense cold_2T_30K cold_2T_30K_chi100 cold_IMC_30K_seed1
uv run python -m test_misalignment.test3_transparency --pdrive cold_rad_p_solver cold_hydro_p_solver
uv run python -m test_misalignment.test3_transparency --pdrive seeded_rad_p_2T seeded_hydro_p_2T cold_rad_p_2T cold_hydro_p_2T --t-end 2e-9
uv run python -m test_misalignment.test3_transparency_figures

# Be and Al seeded at 0.1 ns
uv run python -m test_misalignment.light_analytic_start --run al_cold_paper al_analytic_paper be_cold_paper be_analytic_paper al_analytic_calibrated_flux
uv run python -m test_misalignment.light_analytic_start --calibrate al_analytic --t-min 0.4
uv run python -m test_misalignment.light_analytic_start --calibrate-flux be_analytic
uv run python -m test_misalignment.light_analytic_start --run be_analytic_calibrated_flux
uv run python -m test_misalignment.light_analytic_start_figures

# Al to 50 ns
uv run python -m test_misalignment.al_long --run cold_2T seeded_2T cold_chi1000 seeded_chi1000
uv run python -m test_misalignment.al_long --run cold_imc1 cold_imc2 cold_imc3 seeded_imc1 seeded_imc2 seeded_imc3
uv run python -m test_misalignment.al_long_figures
cd results/misalignment/al_long/figures/slider && python -m http.server
```

## The slider

`uv run python -m test_misalignment.test3_on_the_solver --slider <run>` shows density, pressure, velocity, temperature and entropy over the whole domain against the solver, 220 frames from 0.01 ns (0.1 ns for seeded runs) to 2 ns, on fixed axes. `<run>`: any run of `test3_on_the_solver` above. Zoom with the toolbar (kept while sliding); `f` fits the y ranges to the window, `x`/`y` switch log/linear, arrows step one frame. Al's slider is an HTML page: `results/misalignment/al_long/figures/slider/index.html` (serve the folder).

## Not in the repo

 the piston check of Test 4 (scratch `piston_check.py`, [02_patching.md § 3](02_patching.md)) and the short exploratory runs behind [03_startup.md § 2](03_startup.md) (graded meshes at 50 ps) and [05_shocked_region.md § 5](05_shocked_region.md) (growth factors). Their settings are given in those sections.
