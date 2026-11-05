# Phase C figures

Every figure here is a full-profile figure (ρ, p, u, T against Lagrangian mass): diffusion and IMC against the semi-analytic `AblationSolver`. The folder says which bath drove the simulations (`../bath_calibration.md`).

| folder | what it is |
|---|---|
| `paper_bath/` | simulations driven by the manuscript's bath relation: Tests 1–6 and the earlier ω scan (ω = −0.75, −0.5 at N = 400, −0.25, +0.25 at N = 2000) |
| `calibrated_T_bath/` | bath calibrated so that the matter's surface temperature follows T_s(t): Tests 1, 2, 4, 5, 6 |
| `calibrated_flux_bath/` | bath calibrated so that the net energy flux equals the solver's s(0, t): Tests 1, 4, 5, 6 |
| `overlay/` | one figure per case with more than one bath, at 2 ns: diffusion under every bath against the solver, the closest (smallest profile L1) drawn bold, the baths ranked in a table with their front errors |
| `legacy/fix_B/` | figures that involve fix B, a numerical moving-front target. **Not analytic** (`../legacy/patching_poc_fix_B.md`) |
| `legacy/impulse_patch/` | the impulse-matched patch, per bath and overlays (`../legacy/analytic_patching_impulse.md`) |
| `legacy/non_profile/` | the earlier non-profile figures: fronts against time, `opacity_summary.png`, `bath_comparison.png` |

Test 3's file is `test3_full_profiles.png` (its fine-mesh run, `test3_smooth`). The gold ω sweep of the fronts against time is a separate set: `../omega_sweep/`.

Regenerate (seconds per case once the solvers are cached in `results/patching_poc/`):

```
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full test2_full test3_smooth test4_full test5_full test6_full --profiles
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full test2_full test4_full test5_full test6_full --overlay
uv run python -m rad_hydro_sim.verification.bath_calibration --cases --omega-scan=-0.75 --profiles   # likewise -0.5, -0.25; 0.25 with --N 2000
```
