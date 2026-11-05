# Phase C½ results: IMC vs. diffusion vs. the semi-analytic solution on the paper's six tests

This is the step between Phase C and Phase D that you asked for (plan Addendum B). It runs the full rad-hydro comparison of `md/phase_c/phase_c_results.md` on all six tests of the manuscript. **Why the semi-analytic curve differs from the simulations** in Tests 4–6 is explained in its own document, `md/phase_c/why_the_solver_differs.md`; this one gives the results.

| Test | Material | Initial density | Drive |
|---|---|---|---|
| 1 | Au | uniform (ω = 0) | T_s = 1 HeV (τ = 0) |
| 2 | Au | uniform (ω = 0) | τ = 158/1285 |
| 3 | Au | ρ₀x^(−0.5) (ω = +0.5) | τ = 0 |
| 4 | Au | ρ₀x^(+0.5) (ω = −0.5) | τ = 0 |
| 5 | Cu | uniform | τ = 0 |
| 6 | Al | uniform | τ = 0 |

**Setup.** Both solvers ran in rad_hydro_sim with the Phase C configuration (the Phase B conservation flags plus `rad_energy_per_mass` and `compatible` hydro), the same mesh and the same dt rule. IMC ran with 3 seeds (1 at N = 1000 for Tests 1–2). The black curve is Menahem's `AblationSolver`, evaluated directly. Fronts are defined in `md/phase_c/phase_c_changes.md` §3.5, with the heat front now temperature-based (§5 below).

## 1. Summary

**IMC and diffusion agree in every test.** The agreement is best in the opaque gold tests and loosens as the hot ablated plasma becomes thinner, exactly in the order of the optical depth 𝒯 of that plasma:

| | Au (Tests 1–4) | Cu (Test 5) | Al (Test 6) |
|---|---|---|---|
| 𝒯 of the hot plasma (T > T_s/2) at 2 ns | 8–10 | 3.2 | 2.3 |
| IMC vs. diffusion, worst quantity at 2 ns | 2.3% | 4.2% | 5.3% |

**The semi-analytic solution** agrees with both within ~3% in Tests 1–3. In Tests 5–6 it agrees within ~4% once the bath delivers the solution's energy flux (the flux-calibrated bath, §3). In Test 4 its own patching approximation costs ~6–14% (§4).

### At 2 ns, the manuscript's bath (option (c))

| Test | cells | IMC seeds | IMC vs. diffusion: heat / shock / abl. p / energy | diffusion vs. solver: heat / shock / abl. p / energy | 𝒯 (T > T_s/2) | surface T [HeV] |
|---|---|---|---|---|---|---|
| 1 (Au, τ = 0) | 1000 | 1 | −0.5% / −1.5% / +0.6% / −2.3% | −0.5% / −1.4% / −2.7% / −2.7% | 9.0 | 0.995 |
| 2 (Au, τ = 158/1285) | 1000 | 1 | −0.5% / −1.4% / −0.5% / −2.1% | −0.7% / −0.1% / −4.7% / −1.5% | 8.0 | 1.082 (solver 1.089) |
| 3 (Au, ω = +0.5) | 347 | 3 | −0.6% / −1.8% / +0.3% / −2.3% | −1.0% / −2.1% / −4.5% / −1.8% | 8.8 | 0.995 |
| 4 (Au, ω = −0.5) | 1000 | 3 | −0.7% / −0.9% / −1.3% / −1.2% | +6.0% / −12.5% / +10.2% / −10.7% | 10.1 | 0.994 |
| 5 (Cu) | 1000 | 3 | −2.5% / −2.5% / −1.6% / −4.2% | −1.5% / −4.7% / −5.1% / −10.0% | 3.2 | 0.942 |
| 6 (Al) | 400 | 3 | −4.6% / −3.3% / −1.2% / −5.3% | −4.1% / −7.8% / −8.5% / −11.2% | 2.3 | 0.959 |

Energy is conserved to round-off in every run: diffusion 10⁻¹⁴–10⁻¹³; IMC 10⁻¹³–10⁻¹¹ (Test 3: 7×10⁻¹⁰, over 2×10⁶ steps).

### At 2 ns, the flux-calibrated bath (option (a), energy flux matched)

| Test | cells | IMC seeds | IMC vs. diffusion: heat / shock / abl. p / energy | diffusion vs. solver: heat / shock / abl. p / energy | 𝒯 (T > T_s/2) | surface T [HeV] |
|---|---|---|---|---|---|---|
| 1 (Au, τ = 0) | 400 | 3 | +1.5% / −2.0% / +1.1% / −3.2% | +2.0% / +1.4% / −2.2% / −0.5% | 8.7 | 0.999 |
| 4 (Au, ω = −0.5) | 1000 | 3 | −0.6% / −0.8% / +0.2% / −1.1% | +13.8% / −6.4% / +15.1% / −1.1% | 10.6 | 1.011 |
| 5 (Cu) | 1000 | 3 | −2.3% / −2.5% / −1.8% / −4.1% | +3.8% / −0.2% / −1.8% / −2.2% | 3.3 | 0.956 |
| 6 (Al) | 400 | 3 | −4.5% / −3.3% / −4.0% / −5.2% | +2.3% / −2.9% / −2.8% / −2.3% | 2.6 | 0.974 |

With the flux-matched bath, copper and aluminum lie within ~4% of the solver on every quantity, and IMC still agrees with diffusion as it did with the paper's bath. The IMC-vs-diffusion difference is a property of the material, not of the bath. Profiles: `md/phase_c/figures/calibrated_flux_bath/original_patch/`.

## 2. Figures

In `md/phase_c/figures/` (layout: `figures/README.md`); the profiles of this page are the original patch with the paper bath, `paper_bath/original_patch/`, the others are under `legacy/non_profile/`:
- `testN_full_profiles.png`: ρ, p, u, T at 1, 1.5 and 2 ns in the style of `results/ictt/const_T/dimensional_verification/const_T_fully_patched_comparison.png`. Diffusion is drawn per time (blue, orange, red), the solver in black, IMC dashed green, with zoom insets on the ablation layer and the shocked region. Test 3 uses a log mass axis: its ablated layer, 1.5 mg/cm², sits on a 300 mg/cm² domain. Each figure's subtitle gives the hot plasma's 𝒯, the share of its mass in cells with τ < 1, and the surface temperature against the solver's.
- `testN_full_fronts.png`: the heat and shock fronts, ablation pressure and energy delivered vs. time, against the self-similar laws.
- `opacity_summary.png`: 𝒯 per test, next to the IMC-vs-diffusion and diffusion-vs-solver differences.
- `bath_comparison.png`: diffusion vs. the solver for the three baths (paper, surface temperature matched, energy flux matched) in Tests 1, 4, 5 and 6.

## 3. Which materials can lose consistency between diffusion and IMC

Transport and diffusion part where the hot ablated plasma is optically thin. Gray diffusion is exact only in the thick limit, and the drive's radiation crosses the hot plasma.

| | Au | Cu | Al |
|---|---|---|---|
| κ_R at 1 HeV, 0.01 g/cm³ | 2866 cm²/g | 588 cm²/g | 272 cm²/g |
| opacity temperature exponent α (κ ∝ T^−α) | 1.5 | 2.21 | 3.1 |
| 𝒯 of the hot plasma | 8–10 | 3.2 | 2.3 |
| T_rad / T_mat at the surface | 1.003–1.007 | 1.065 | 1.047 |
| IMC vs. diffusion (worst) | ≤ 2.3% | ≤ 4.2% | ≤ 5.3% |

**Aluminum is the least opaque and the most likely to lose consistency**, copper is intermediate, and gold is safe. The steep α makes it worse in aluminum: its hottest plasma is its most transparent.

The same thinness drives the main difference from the semi-analytic solution in Tests 5–6. The solution assumes radiation and matter are in equilibrium at the surface, while the simulations show the radiation 5–6.5% hotter than the matter there. The manuscript's bath then delivers ~10% too little energy, and a bath tuned to the matter's surface temperature delivers 15–24% too much. A bath tuned to the solution's energy flux brings the copper and aluminum diffusion runs within 3–4% of it. That calibration is `bath_drive="calibrated_flux"`: `why_the_solver_differs.md` §2, `bath_comparison.png`.

## 4. Test 4: the patching

With ω = −0.5 the ablated mass is 28–37% of the shocked mass, and the shocked matter moves at ~14 km/s. The patched solution assumes a rigid ablation front, and these numbers violate that. IMC and diffusion agree with each other to ≤ 2.2%, and a pure-hydro run of the solver's own piston problem reproduces its shock to 0.1%. Both simulations therefore differ from the black curve for the same reason, the patch. The shock is 12.5% behind with the manuscript's bath, 6.4% with the flux-matched one, while the heat front and ablation pressure move further off. Details, the code walk-through of `AblationSolver.solve`, and three possible fixes are in `why_the_solver_differs.md` §3.

## 5. Changes to the analysis made along the way

- **The heat front is temperature-based.** It is the first point from the surface where T < 0.3 T_s, between cell centres. The density-based detector of Phase C failed for ω = +0.5: the surface cells start at ~10⁶ g/cm³ and never return to their initial density after ablating. Tables use the stored profiles at 1, 1.5 and 2 ns (within 1% of those times), so runs made before the change are analyzed with the same definition. The Phase C document's gold numbers used the old definition; the two agree within a cell.
- **Test 4 runs on a uniform-mass mesh** (`SimulationConfig(mesh="uniform_mass")`). The default graded mesh put 628 of its 1397 cells in the first 10⁻⁵ g/cm², and only 51 coarse cells across the shocked region. That mesh's 0.934 HeV surface and 10⁻¹¹ energy error went away on the uniform mesh (0.994 HeV, 10⁻¹⁴), and the step count dropped 20×. Its gap to the solver did not.
- **Test 3 runs with 1/10 of the packets.** ρ ∝ x^(−0.5) puts ~10⁶ g/cm³ in a 10⁻¹⁰ cm first cell; the acoustic CFL then forces dt ~ 10⁻¹⁵ s, ~2×10⁶ steps to 2 ns. **Test 4's census is combed at 3×10⁴ packets**: its near-zero-density surface cells are optically thin and held the census at its 2×10⁵ cap.
- **Test 3 on a fine mesh** (`test3_smooth`; `SimulationConfig(max_cell_mass=3e-4)`). The default graded mesh resolves the ablated layer well: 0.0216 mg/cm² cells, ~70 across it. Its shocked region, though, grows to 1–12 mg/cm² cells, with the shock at 200–290 mg/cm².
  - **How the cap works:** it keeps every cell lighter than 0.3 mg/cm², including the 3×10⁻¹³ cm surface cells that set dt, then grows the cell mass by ≤ 5% per cell up to the cap. The result is 1237 cells at the same step count. Each step costs more, so diffusion takes 2.4–7 h depending on machine load, against 22 min on the coarse mesh.
  - **What it changes:** diffusion's shock goes from −2.1% to 0.0% of the solver's, and the pressure plateau lies on it; the heat front (−1.0%) and the ablation pressure (−4.5%) are unchanged.
  - **A failed first version:** an earlier version of the cap split each heavy cell into whole pieces. That left neighbours at mass ratios down to 0.52, and the hydro made density spikes at exactly those jumps, so it was replaced.
  - **IMC vs. diffusion on this mesh** (3 seeds, 2.5–8.8 h each), at 2 ns: heat front −0.6%, shock −1.0%, ablation pressure +0.1%, energy −2.2%. Energy error 7×10⁻¹⁰ (IMC), 10⁻¹³ (diffusion).
  - **The density deficit:** behind the ablation front (1–10 mg/cm²) diffusion's density is 20–25% below the solver's at equal pressure, i.e. hotter material. **IMC's density there is clearly higher than diffusion's and closer to the solver**, with the same hydro, so this is not a hydro (wall-heating) error, as first suggested here. It points at radiation. Gray diffusion without a flux limiter carries too much flux into the cold, dense material just ahead of the ablation front, preheating it, while transport limits that flux. Both stay below the solver's density, which rises steeply toward the front for ω = +0.5. This is the one visible place in Test 3 where transport and diffusion differ. The mechanism is inferred from the figure, not yet confirmed: a flux-limited diffusion run (plan M11) would test it.
  - `paper_bath/original_patch/test3_full_profiles.png` and `legacy/non_profile/test3_full_fronts.png` show this fine mesh, with IMC.
- **Copper's domain is cut to 3×10⁻² g/cm²**, holding the 2.2×10⁻² g/cm² shock, so that 1000 cells resolve its ablated layer.

## 6. Reproducing

```bash
M="uv run python -m rad_hydro_sim.verification.imc_vs_diffusion"
$M --cases test1_full test2_full test3_full test4_full test5_full test6_full --figures-dir md/phase_c/figures
$M --calibrate-bath --calibrate-target flux --cases test5_full test6_full      # the energy-flux tables
$M --cases test5_full test6_full --bath calibrated_flux                         # runs with them
```

Seeds can run in parallel processes that share the cache (`--seeds 2,3 --imc-only`). Costs on this machine:

| Test | Diffusion | IMC per seed |
|---|---|---|
| 1, 2 (N = 400) | ~1 min | 4–8 min |
| 3 | 22 min | 3.1–4.7 h |
| 4 (uniform mesh) | 1.5 min | 6–11 min |
| 5 (Cu) | 1 min | ~23 min |
| 6 (Al) | 35 s | 10–17 min |
