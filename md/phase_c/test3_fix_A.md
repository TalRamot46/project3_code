# Test 3, fix A: start from the analytic solution

Branch `feat/patching-poc`, 2026-11-10. Summary of the fix; the full analysis is in [test3_analytic_start.md](test3_analytic_start.md).

## The problem

In Test 3 (gold, ω = +0.5, paper bath) the density of diffusion and of IMC falls below the solver's between the heat front and ~30 mg/cm², by different amounts in the two codes (ρ/ρ_solver = 0.40 in diffusion and 0.59 in IMC at 2 mg/cm²), while pressure, velocity and temperature agree. The temperature of the same matter is too high by the matching factor. It is a startup imprint: until ~0.6 ps the analytic ablated layer is thinner than one surface cell, the first cells are heated as blocks, and the matter the early shock crosses keeps 1.3–4× the solver's entropy for the rest of the run.

## The fix

`initial_condition="analytic_ablation"` with `t_sec_start = 0.1 ns` starts the run from the `AblationSolver` profile instead of a cold target at t = 0. At 0.1 ns the ablated layer spans ~14 surface cells.

- **Code:** `_seed_ablation_profile` in `src/rad_hydro_sim/simulation/iterator.py`. The mesh keeps its cell masses. Nodes behind the solver's shock take the solution's positions and velocities, so each cell has the solution's density. The specific energy is the solution's mass average over the cell.
- **Cases:** `test3_smooth_analytic` (fine mesh, 1237 cells) and `test3_analytic` (default mesh) in `imc_vs_diffusion.HYDRO_CASES`, through the new `HydroCase.case_items`.
- **Tests:** `tests/test_analytic_ablation_start.py`.
- **Figures:** `results/phase_c/figures/paper_bath/test3_analytic_start/<region>_<scale>.png` and `paper_bath/test3_analytic_start_profiles.png`, diffusion and IMC (mean of 3 seeds).

## Result (fine mesh, 1–2 ns)

ρ / ρ_solver at 2 ns:

| m [mg/cm²] | 2 | 3 | 5 | 8 | 12 | 20 | 30 | 50 | 70 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusion, cold start | 0.40 | 0.49 | 0.59 | 0.70 | 0.82 | 0.90 | 0.94 | 0.95 | 0.94 | 0.94 | 0.94 | 0.94 |
| IMC, cold start | 0.59 | 0.68 | 0.80 | 0.86 | 0.98 | 1.01 | 1.04 | 1.05 | 1.00 | 0.97 | 0.97 | 0.96 |
| diffusion, fix A | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.93 | 0.94 | 0.94 |
| IMC, fix A | 1.01 | 0.99 | 0.97 | 0.99 | 1.00 | 1.01 | 1.01 | 0.99 | 0.93 | 0.93 | 0.96 | 0.95 |

- **The startup band is gone in both codes.** Inside the seeded region (m < 61.6 mg/cm²) the entropy p/ρ^γ is the solver's within 0.1% in diffusion and within 0.1–2.2% in IMC (largest at 2–8 mg/cm², next to the heat front; the three seeds agree to 0.5%).
- **The fronts are unchanged.** At 2 ns against the solver: heat front −0.8% (diffusion) and −1.1% (IMC), shock +0.1% and −0.4%, ablation pressure −4.8% and −4.5%.
- **Energy error:** 2×10⁻¹⁴ (diffusion, 56 min), 1.2×10⁻⁹ (IMC, 16.7 h per seed).

## What fix A does not remove

- **The 300 K offset.** Matter the run shocks itself (m > 62 mg/cm²) is 6–7% less dense than the solver's, entropy × 1.05–1.09. The 300 K initial temperature lowers the shock compression (`test3_analytic_start.md` § 4). A colder target would remove it.
- **One cell at the seam.** The cell that held the solver's shock at 0.1 ns (61.75 mg/cm²) starts half shocked and keeps ρ −31%. Seeding the shock on a cell face would remove it.
- **The flicker in the shocked region.** The analytic curves are smooth; the two simulations are not. Three effects (analysis of 2026-11-08):
  1. *Sound waves trailing the shock, both codes.* About ±3% in density in the 50 cells behind the shock, 0.1–0.4% deeper in, at the isentropic ratios (δρ/ρ = δp/γp, entropy flat). The captured shock changes shape once per cell it crosses and emits a small wave each time; the wavelength is 3.6–3.8 cells. It is a property of the artificial-viscosity hydro and does not shift the profiles.
  2. *Monte Carlo noise, IMC only.* It matters where radiation still heats the shocked matter (Test 4). In Test 3 the shocked layer is cold, and IMC's extra jitter there is sound arriving from the noisy hot layer.
  3. *Isolated steps frozen into the matter, both codes.* A density dip with a matching temperature peak at fixed mass, pressure smooth across it. Cold start: at ~3 and ~10 mg/cm², where the cell mass starts to grow and where it reaches its cap. Fix A: at 62 mg/cm² (the seam) and at 85 and 110 mg/cm², which the shock crossed at 0.19 and 0.31 ns; the timing fits pressure waves from the first ~20 ps after the seeding catching the shock.

The hydro-only test of the shocked region (pressure or piston boundary, no radiation) is in [test3_hydro_only.md](test3_hydro_only.md).

## Reproduce

```
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test3_smooth_analytic --baths paper --jobs 4 --seeds 1 2 3
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test3_smooth_analytic --baths paper --views
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test3_smooth_analytic --baths paper --profiles
uv run pytest tests/test_analytic_ablation_start.py
```
