# Phase C figures

Every figure here is a full-profile figure (ρ, p, u, T against Lagrangian mass): diffusion and IMC against the semi-analytic `AblationSolver`. The folder says which bath drove the simulations (`../bath_calibration.md`).

| folder | what it is |
|---|---|
| `paper_bath/` | simulations driven by the manuscript's bath relation: Tests 1–6 and the earlier ω scan (ω = −0.75, −0.5 at N = 400, −0.25, +0.25 at N = 2000) |
| `calibrated_T_bath/` | bath calibrated so that the matter's surface temperature follows T_s(t): Tests 1, 2, 4, 5, 6 |
| `calibrated_flux_bath/` | bath calibrated so that the net energy flux equals the solver's s(0, t): Tests 1–6 (Test 3 on its fine mesh) |
| `overlay/` | one figure per case with more than one bath, at 2 ns: diffusion under every bath against the solver, the closest (smallest profile L1) drawn bold, the baths ranked in a table with their front errors |
| `legacy/fix_B/` | figures that involve fix B, a numerical moving-front target. **Not analytic** (`../legacy/patching_poc_fix_B.md`) |
| `legacy/impulse_patch/` | the impulse-matched patch, per bath and overlays (`../legacy/analytic_patching_impulse.md`) |
| `legacy/non_profile/` | the earlier non-profile figures: fronts against time, `opacity_summary.png`, `bath_comparison.png` |

Test 3's file is `test3_full_profiles.png` (its fine-mesh run, `test3_smooth`).

**Tests 3 and 4, every view.** Each folder that has the case (Test 3: `paper_bath/`, `calibrated_flux_bath/` and `overlay/`; Test 4: the three bath folders and `overlay/`) also has a subfolder `test3_full/` or `test4_full/` with six figures, `<region>_<scale>.png`:

| region | mass window | | scale | axes |
|---|---|---|---|---|
| `full` | the whole domain | | `linear` | linear mass, linear ρ, p, u, T |
| `heat` | the ablated layer, up to 1.3 m_heat | | `loglog` | log mass, log ρ, p, T; u (which changes sign) on a symmetric-log axis, linear within ±1 km/s |
| `shock` | from 0.85 m_heat to 1.1 m_shock | | | |

The y limits come from the simulations inside the window. In the overlays the ranking table is always over the full domain. The top-level `test3_full_profiles.png` (log mass axis, linear values) and `test4_full_profiles.png` (linear) are the figures the docs link to. The gold ω sweep of the fronts against time is a separate set: `../omega_sweep/`.

Regenerate (seconds per case once the solvers are cached in `results/patching_poc/`):

```
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full test2_full test3_smooth test4_full test5_full test6_full --profiles
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full test2_full test4_full test5_full test6_full --overlay
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test3_smooth test4_full --views
uv run python -m rad_hydro_sim.verification.bath_calibration --cases --omega-scan=-0.75 --profiles   # likewise -0.5, -0.25; 0.25 with --N 2000
```
