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
