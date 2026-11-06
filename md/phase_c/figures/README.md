# Phase C figures

Every figure here is a full-profile figure (ρ, p, u, T against Lagrangian mass): diffusion and IMC against the semi-analytic `AblationSolver`. The folder says which bath drove the simulations (`../bath_calibration.md`).

Each per-bath figure is titled with the test, the region, the bath and the scale. It shows diffusion at 1, 1.5 and 2 ns (light to dark blue), IMC (the mean of 3 seeds, orange) and the analytic solution (black). The overlays use one colour per bath; the closest bath is drawn wide underneath and listed first in the table.

The analytic curve leaves out the heat solution's last sliver before its front, where it is denser or colder than the shocked matter just past the front. That sliver is where the two self-similar solutions meet: the heat solution's rigid front (gold: ρ → 313 g/cm³, T → 0.02 HeV) against a lighter, hotter shocked state. Drawn, it is a spike in ρ and T at the ablation front, an artifact of the patch. It holds at most 0.5% of m_f (Test 4; 0.001% in Test 1), 3.5% for ω = −0.8 at 1 ns, and nothing for Test 3, Cu, Al and ω ≥ +0.2. The front values, the tables and the overlay ranking do not use it (`imc_vs_diffusion.front_spike_mask`).

| folder | what it is |
|---|---|
| `paper_bath/` | simulations driven by the manuscript's bath relation: Tests 1–6 |
| `calibrated_T_bath/` | bath calibrated so that the matter's surface temperature follows T_s(t): Tests 1, 2, 4, 5, 6 |
| `calibrated_flux_bath/` | bath calibrated so that the net energy flux equals the solver's s(0, t): Tests 1–6 (Test 3 on its fine mesh) |
| `overlay/` | one figure per case with more than one bath, at 2 ns: diffusion under every bath against the solver, the closest (smallest profile L1) drawn bold, the baths ranked in a table with their front errors |
| `legacy/fix_B/` | figures that involve fix B, a numerical moving-front target. **Not analytic** (`../legacy/patching_poc_fix_B.md`) |
| `legacy/impulse_patch/` | the impulse-matched patch, per bath and overlays (`../legacy/analytic_patching_impulse.md`) |
| `legacy/omega_scan/` | the earlier gold ω scan (ω = −0.75, −0.5, −0.25 at N = 400, +0.25 at N = 2000; paper bath, 1 IMC seed), superseded by `../omega_sweep/` |
| `legacy/non_profile/` | the earlier non-profile figures: fronts against time, `opacity_summary.png`, `bath_comparison.png` |

Test 3's file is `test3_full_profiles.png` (its fine-mesh run, `test3_smooth`).

**Tests 3 and 4, every view.** Each folder that has the case (Test 3: `paper_bath/`, `calibrated_flux_bath/` and `overlay/`; Test 4: the three bath folders and `overlay/`) also has a subfolder `test3_full/` or `test4_full/` with twelve figures, `<region>_<scale>.png`:

| region | mass window | | scale | axes |
|---|---|---|---|---|
| `full` | the whole domain | | `linear` | linear mass, linear ρ, p, u, T |
| `heat` | the ablated layer, up to 1.3 m_heat | | `semilogx` | log mass, linear ρ, p, u, T |
| `shock` | from 0.85 m_heat to 1.1 m_shock (simulated or analytic, whichever is further); from 1.05 m_heat when the ablated layer is thin against the shocked one (Test 3) | | `semilogy` | linear mass, log ρ, p, T; u on a symmetric-log axis |
| | | | `loglog` | log mass, log ρ, p, T; u on a symmetric-log axis |

On a log field axis u, which changes sign, goes on a symmetric-log axis, linear within ±1 km/s.

The y limits come from the simulations inside the window. In the shocked region they come from the shocked matter alone, simulations and solver together, between each one's heat front and shock, so the plateaus and their differences fill the panels. In the overlays the ranking table is always over the full domain. The top-level `test3_full_profiles.png` (log mass axis, linear values) and `test4_full_profiles.png` (linear) are the figures the docs link to. The gold ω sweep of the fronts against time is a separate set: `../omega_sweep/`.

Regenerate (seconds per case once the solvers are cached in `results/patching_poc/`):

```
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full test2_full test3_smooth test4_full test5_full test6_full --profiles
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full test2_full test4_full test5_full test6_full --overlay
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test3_smooth test4_full --views
```
