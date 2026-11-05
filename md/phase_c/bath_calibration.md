# The bath drive: paper, temperature-calibrated and flux-calibrated

Branch `feat/patching-poc`, 2026-11-05. The reference is Menahem's semi-analytic `AblationSolver`, unchanged. The attempts to improve its patching are legacy (§4). This page compares the three ways of driving the simulations against it.

**In one paragraph.** The bath is the only thing that changes between the three sets of simulations; the analytic curve is the same for all of them. For gold with ω ≥ 0 the **paper bath** is already closest (Tests 1–2), and calibrating moves the simulations 1–2% further from the solver. For copper and aluminum the **flux-calibrated bath** is clearly closest: whole-profile error 2.6% vs. 9.4% (paper) for Cu, and 7.6% vs. 15.9% for Al. The **temperature-calibrated bath** is the worst option there (26% and 16%). For Test 4 (gold, ω = −0.5) the flux-calibrated bath is also closest (17.8% vs. 25.4%). It doesn't close the gap, though: it trades the shock error (−12.5% → −6.4%) for the heat front (+6% → +14%), and what remains comes from the solver's patching (`why_the_solver_differs.md` §3).

---

## 1. The three baths

The simulations are driven through a Marshak boundary at a bath temperature T_bath(t). The solver instead imposes "the surface is at T_s(t)" in local thermodynamic equilibrium (LTE). Translating one into the other is the whole question:

| bath | `--bath` | what it matches | where it is right |
|---|---|---|---|
| **paper** | `paper` | the manuscript's Marshak relation between T_bath and the solver's surface flux (option (c)) | where the surface is in LTE (gold): radiation and matter at the same temperature |
| **T-calibrated** | `calibrated` | the matter's surface temperature, iterated to follow T_s(t) (option (a), `calibrate_bath(target="temperature")`) | only in LTE; elsewhere it over-drives, because the radiation is hotter than the matter |
| **flux-calibrated** | `calibrated_flux` | the net energy flux into the slab, iterated to equal the solver's s(0, t), so that the energy delivered matches its closed form E(t) (`calibrate_bath(target="flux")`) | everywhere the energy budget is what matters; it leaves the surface temperature free |

Why they differ (from `why_the_solver_differs.md` §2): in copper and aluminum the hot ablated plasma is only marginally opaque (optical depth 3.2 and 2.3, against 8–10 in gold), so at the surface the radiation is 5–7% hotter than the matter. The phrase "the surface is at 1 HeV" then has two meanings, and they give different fluxes. The paper bath puts the *radiation* there at 1 HeV: the matter settles at 0.94–0.96 HeV and the energy in falls ~10% short. The T-calibrated bath puts the *matter* at 1 HeV: the radiation is then at 1.06 HeV and the energy overshoots by 15–24%. The flux-calibrated bath delivers the solver's energy and lets the surface settle in between.

The calibration tables live in `src/rad_hydro_sim/data/bath_correction/` (`<preset>.txt` for temperature, `<preset>_flux.txt` for flux). IMC runs with the same table as diffusion, so both solvers keep the same drive.

### 1.1 How the calibration iterates

`calibrate_bath(label, target)` in `verification/imc_vs_diffusion.py`. The table format and the fixed caching bug are also in `phase_c_changes.md` §6.1.

**How a new bath reaches the next run.** The case object never changes inside the loop. It carries only the *path* of the table: `case = replace(case, bath_correction=…/<key>.txt)`, and the run's config has `bath_drive="calibrated"` or `"calibrated_flux"`. The update goes through the file:
1. At the top of each iteration the current correction is written to the table, `np.savetxt(path, (t [ns], c))`, with `path = bath_correction_path(case, drive)` (`<key>_flux.txt` for the flux target).
2. `simulate_rad_hydro(case, cfg)` reads it through `bath_correction_for` → `bath_correction(path)` (`simulation/iterator.py`). It drives the boundary with T_bath(t) = c(t) · T_bath,law(t), with c interpolated linearly in log t and held constant outside the table. Before the grid's first time, 0.2 ns, the bath therefore uses c(0.2 ns).
3. At the bottom of the iteration c is updated, and the next iteration's `np.savetxt` overwrites the file before the next run.

The reader is cached (`lru_cache`) by path **and modification time**, so each rewrite is read again. An earlier version cached by path alone and silently reused the first table; that is the fixed bug, covered by `test_bath_table_is_reread_after_a_rewrite`.

**One iteration**, on the grid t_g = 60 points log-spaced from 0.2 ns to t_end, with every quantity in units of T⁴:
- **Measured Q.** For the temperature target, the matter temperature of the first cell, T(cell 0)⁴, interpolated onto t_g in log t. For the flux target, the net energy flux into the slab divided by 2σ_SB. It is taken from the cumulative energy that entered through the left boundary, as a finite difference over a ±5% window in t: (E_in(1.05 t) − E_in(t/1.05)) / (window length).
- **Target Q_target.** T_s(t)⁴ = (T₀ t_ns^τ)⁴ for temperature. For flux, s(0, t)/(2σ_SB) = T_b⁴ B t^(4τ+p), the derivative of the solver's closed-form energy E(t).
- **Error** = max over t_g of |(Q / Q_target)^(1/4) − 1|. The loop stops when this is below tol = 2×10⁻³, or after iteration 6 (at most 7 runs). It stops *before* updating, so the table left on disk is the c of the last run, and later `--bath calibrated_flux` runs read exactly that table.
- **Update**, pointwise on t_g: T_bath⁴ ← T_bath⁴ + g (Q_target − Q). The gain is g = 1 on the first iteration. Afterwards it is a secant estimate from the last two runs, g = 1 / clip(ΔQ / ΔT_bath⁴, 0.3, 3); points where the bath hardly changed (relative change < 10⁻⁶) keep g = 1.
- **Clip.** c_new = clip(T_bath,new⁴ / T_bath,law⁴, 0.7⁴, 1.5⁴)^(1/4), so the factor on T_bath stays within [0.7, 1.5].
- **Smoothing.** `c = np.convolve(np.pad(c_new, 1, mode="edge"), np.ones(3) / 3, mode="valid")` is a 3-point moving average of c_new over t_g. Since t_g is log-spaced, each point is averaged with its neighbours in log t ("light smoothing in log t" in the code). The edge padding repeats the first and last values once, so the result again has 60 points, with the endpoints weighted twice: c[0] = (2 c_new[0] + c_new[1]) / 3. It is there because the measured flux, a finite difference of the cumulative energy, is noisy from point to point, and without smoothing that noise would go straight into the next run's bath and could make the iteration oscillate. Its cost is that it is applied again every iteration and slightly flattens real structure in c(t), mostly near the ends of the grid. The converged table is a smoothed solution, not an exact pointwise match. That is a likely reason, not a checked one, why some cases stop above the tolerance.

**What it reached.** In the gold ω sweep (`omega_sweep/`), with the flux target: ω = −0.8 stopped at 1.0% after 7 runs, −0.6 at 0.65%, −0.4 at 0.31%, −0.2 converged (0.16%) at the 7th run, ω = 0 converged after 4 runs and +0.2 after 3. The error falls every iteration (ω = −0.8: 6.2% → 1.0%), which also confirms that each rewritten table is picked up.

**Caveat.** Each iteration overwrites the tracked table in `src/rad_hydro_sim/data/bath_correction/`. A calibrated run of the same case started while its calibration is still going would read a half-finished table, so run them in sequence.

## 2. Results at 2 ns

Profile L1 is the mass-weighted |diffusion − analytic| over the domain, relative to the analytic profile, averaged over ρ, p, u and T (`bath_calibration.profile_score`); the fronts are diffusion vs. the solver (+ = diffusion ahead / higher). Closest bath per case in bold.

| case | bath | profile L1 | heat front | shock | front pressure |
|---|---|---|---|---|---|
| Test 1 (Au, ω = 0) | **paper** | **3.4%** | +1.2% | +0.6% | −2.1% |
| | flux-calibrated | 4.4% | +2.0% | +1.4% | −2.2% |
| | T-calibrated | 4.9% | +2.2% | +1.8% | −2.1% |
| Test 2 (Au, τ > 0) | **paper** | **4.8%** | +1.0% | +2.0% | −4.5% |
| | T-calibrated | 6.6% | +2.1% | +3.5% | −4.0% |
| Test 4 (Au, ω = −0.5) | **flux-calibrated** | **17.8%** | +13.8% | −6.4% | +15.1% |
| | T-calibrated | 23.4% | +8.1% | −10.9% | +11.9% |
| | paper | 25.4% | +6.0% | −12.5% | +10.2% |
| Test 5 (Cu) | **flux-calibrated** | **2.6%** | +3.8% | −0.2% | −1.8% |
| | paper | 9.4% | −1.5% | −4.7% | −5.1% |
| | T-calibrated | 25.8% | +20.3% | +13.2% | +8.6% |
| Test 6 (Al) | **flux-calibrated** | **7.6%** | +2.3% | −2.9% | −2.8% |
| | T-calibrated | 15.7% | +14.3% | +6.6% | +2.8% |
| | paper | 15.9% | −4.1% | −7.8% | −8.5% |

With the paper bath only: Test 3 (Au, ω = +0.5) 3.4%, with a heat front −1.0%, shock 0.0% and front pressure −4.5%. The gold ω sweep, −0.8 to +0.6, is in [`omega_sweep/`](omega_sweep/README.md).

IMC follows diffusion under every bath (`phase_c_tests3to6_results.md` §1): within ≤ 2.3% for gold and ≤ 5.3% for Cu and Al, independent of the bath.

## 3. Conclusions

1. **Gold with ω ≥ 0: keep the paper bath.** The surface is in LTE, so the paper relation is already right: the matter surface is at 0.997–1.000 HeV. Both calibrations move the simulations 1–2% further from the solver, which is within resolution.
2. **Cu and Al: use the flux-calibrated bath.** It brings every front within ~4% of the solver, against 5–9% with the paper bath. The T-calibrated bath is the wrong target where the surface is out of LTE: it overshoots by 13–20% on the fronts.
3. **Test 4: the bath is not the problem.** The flux-calibrated bath has the smallest whole-profile error, but by trading the shock (−12.5% → −6.4%) for the heat front (+6% → +14%) and the front pressure (+10% → +15%). No bath reconciles the simulations with the solver there. That is the signature of the solver's rigid-front patching (`why_the_solver_differs.md` §3), not of the drive.
4. **For the paper:** report Tests 1–3 with the paper bath, Tests 5–6 with the flux-calibrated bath (and state why: the surface optical depth), and Test 4 as the limit of the patched solution, with m_f/m_s next to it.

## 4. What was tried and is legacy

Two attempts to improve the solver's patching itself, for Test 4 and ω < 0:
- **Fix B** (`legacy/patching_poc_fix_B.md`) replaced PistonShock by a numerical hydro of the target driven at the moving ablation front. It is not an analytic model, so it can't benchmark IMC and diffusion. It did establish that the ablated layer does not decelerate with the front.
- **The impulse-matched patch** (`legacy/analytic_patching_impulse.md`) kept everything self-similar and set the PistonShock amplitude by the momentum a massless front delivers. It is analytic, but not a consistent improvement: it flips the shock error, worsens the front pressure and leaves a pressure step at the front.

## 5. Code and figures

```
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full test4_full          # tables
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full ... --profiles      # figures/<bath>/
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full ... --overlay       # figures/overlay/
uv run python -m rad_hydro_sim.verification.imc_vs_diffusion --cases test5_full --calibrate-bath --calibrate-target flux
uv run python -m rad_hydro_sim.verification.imc_vs_diffusion --cases test5_full --bath calibrated_flux
```

Figures: `figures/<bath>/` (one full-profile figure per case and bath) and `figures/overlay/` (one per case with more than one bath: every bath against the solver, the closest drawn bold). Layout: `figures/README.md`. Tables: `results/bath_calibration/<case>_table.md`.
