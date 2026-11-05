# Phase C figures

Every figure here is a full-profile figure (ρ, p, u, T against Lagrangian mass): diffusion and IMC against an **analytic** solution. The folders say which analytic solution, and which bath drove the simulations.

```
<bath>/<analytic method>/<case>_profiles.png
overlay/<case>_profiles.png
legacy/
```

| folder | what it is |
|---|---|
| `paper_bath/` | simulations driven by the manuscript's bath relation (option (c)) |
| `calibrated_T_bath/` | bath calibrated so that the matter's surface temperature follows the solver's T_s(t) (option (a)) |
| `calibrated_flux_bath/` | bath calibrated so that the net energy flux equals the solver's s(0, t) (`why_the_solver_differs.md` §2.3(i)) |
| `…/original_patch/` | analytic: Menahem's `AblationSolver`, the original patch (PistonShock with the piston at m = 0, cut at m_f) |
| `…/impulse_patch/` | analytic: the impulse-matched patch, `MatchedPistonPatch(match="impulse")` (`../analytic_patching.md`) |
| `overlay/` | one figure per case at 2 ns: every bath × patch option the case has, the best one (smallest profile L1) drawn bold, the options ranked in a table with their front errors |
| `legacy/fix_B/` | figures that involve fix B, the numerical moving-front target. **Not analytic**; kept as the record of `../legacy/patching_poc_fix_B.md` |
| `legacy/non_profile/` | the earlier non-profile figures: fronts against time, `opacity_summary.png`, `bath_comparison.png` |

Which cases have which bath: the paper bath has all of them (Tests 1–6 and the ω scan, ω = −0.75, −0.5 at N = 400, −0.25, +0.25 at N = 2000). The calibrated baths were run for Tests 1, 2, 4, 5 and 6 (T-calibrated) and 1, 4, 5 and 6 (flux-calibrated). Test 3's file is `test3_full_profiles.png` (its fine-mesh run, `test3_smooth`).

Regenerate (seconds per case once the solvers are cached in `results/patching_poc/`):

```
uv run python -m rad_hydro_sim.verification.patching_poc --cases test1_full test2_full test3_smooth test4_full test5_full test6_full --profiles
uv run python -m rad_hydro_sim.verification.patching_poc --cases test1_full ... --overlay
uv run python -m rad_hydro_sim.verification.patching_poc --cases --omega-scan=-0.75 --profiles   # likewise -0.5, -0.25; 0.25 with --N 2000
```
