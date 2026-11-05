# LEGACY: analytic patching of the semi-analytic solution, the impulse-matched patch

> **Legacy, 2026-11-05.** The impulse match is not a consistent improvement. With the paper bath it lowers the whole-profile error for ω < 0 and Cu/Al, but it flips the shock error (+7 … +13% for ω ≤ −0.5), makes the front pressure worse (+23% in Test 4) and leaves a pressure step at the front. With the flux-calibrated bath it is the best option only for Al. The current reference is the original `AblationSolver` under each bath: [`../bath_calibration.md`](../bath_calibration.md). Code: `src/menahem_new/legacy/matched_piston_patch.py`, `src/rad_hydro_sim/verification/legacy/patching_poc.py`. Figures: `md/phase_c/figures/legacy/impulse_patch/`.

Follow-up to `../why_the_solver_differs.md` §3.3 (fix C). Branch `feat/patching-poc`, 2026-11-05. Supersedes `patching_poc_fix_B.md` as the current result. That page used a numerical target (fix B) and is kept only as the record of how the closure below was found.

**In one paragraph.** Only analytic curves are used here: the self-similar heat wave and the self-similar PistonShock, joined in different ways. The **impulse-matched patch** keeps both pieces self-similar. It changes the PistonShock amplitude so that the momentum the target carries beyond the cut equals the momentum a *massless* ablation front delivers to it. With the paper bath, it lies closer to the simulations than the original patch over the whole profile in every ω < 0 case: profile L1 33 vs. 40% at ω = −0.75, 19 vs. 25% in Test 4, 9.5 vs. 11.9% at ω = −0.25. It also does better for copper and aluminum. It ties with the original for ω > 0, and is worse in Test 1 (5.8 vs. 3.4%). It fixes the original's main defect, the shock running 12–16% ahead for ω ≤ −0.5, but overshoots the other way (+7% in Test 4, +13% at ω = −0.75). It also leaves a pressure step at the front, about 10% below p_f in Test 4. Neither patch removes the heat-side error for ω < 0, where the simulated heat front runs 6–16% ahead of the self-similar one.

---

## 1. The patch

`MatchedPistonPatch` in `src/menahem_new/legacy/matched_piston_patch.py`. The heat region is the rigid self-similar solution, unchanged. The shocked region is the self-similar PistonShock (piston at m = 0), unchanged in shape, with an effective amplitude p_eff(t) in place of p_f(t). At the cut m_f, the heat layer is shifted to the PistonShock position and velocity, so x and u are continuous there.

**Why change the amplitude.** In the original patch, the slab 0 < m < m_f is shocked matter between the piston and the cut. It decelerates with the piston, which raises the pressure the target feels by ≈ m_f|du/dt|: +20% at the cut in Test 4. In effect, the original treats the ablated layer as decelerating with the front, a "massive" front. The simulations show the opposite: inside the ablated layer, the simulated pressure follows the rigid heat profile, with no inertial term. The ablated matter keeps the velocity it had when it crossed the front.

**The impulse condition.** For a massless front, the momentum of the target beyond the cut obeys

  dΠ_target/dt = p_f − ṁ_f u_f,

where u_f is the velocity of the matter crossing the front. The patch chooses p_eff(t) so that PistonShock's momentum beyond the cut,

  P_eff t^(1+c₃)/(1+c₃) − ∫₀^m_f u_PS dm,

equals ∫₀^t (p_f − ṁ_f u_f) dt', with u_f taken from the same PistonShock at the cut. The integral is a scalar marched on a 60-point log grid in time, with one root-find per point. No field is integrated: every profile still comes from the two self-similar ODE solutions, so the result is analytic in the same sense as `AblationSolver`.

At early times the heat front runs ahead of any shock it could drive (m_f > m_s). There the matter swept into the layer takes nearly all the impulse: the loss term is capped at the push, and p_eff is bounded below by the amplitude that puts the shock on the cut.

**What it does not do.** The target starts at p_PS(m_f; p_eff). That is not p_f, so the pressure is not continuous at the front (Test 4: 1.80 vs. 2.00 Mbar). Being continuous in p and having the right momentum are two different conditions. The second is the physical one, but the pressure step is a visible defect.

`MatchedPistonPatch(match="pressure")` (C_pressure) is the quasi-static alternative: it imposes p_PS(m_f) = p_f at every time. It lands within 1% on the shock for ω = −0.5 and −0.25 at late times, but by coincidence. It is +9 … +32% off at ω = −0.75, because it applies the current m_f/m_s to the whole history. It is in the tables (`results/patching_poc/*_table.md`), not in the figures.

## 2. Results

### 2.1 Every option, by whole-profile agreement (2 ns)

Profile L1 is the mass-weighted |diffusion − analytic| over the domain, relative to the analytic profile, averaged over ρ, p, u and T (`patching_poc.profile_score`). It is dominated by where the shocks sit, which is the purpose here. Best per case in bold. Figures: `../figures/legacy/impulse_patch/overlay/`.

| case | ω | paper, original | paper, impulse | T-calib., original | T-calib., impulse | flux-calib., original | flux-calib., impulse |
|---|---|---|---|---|---|---|---|
| ω scan | −0.75 | 40.2% | **33.4%** | | | | |
| Test 4 | −0.5 | 25.4% | 19.0% | 23.4% | 21.9% | **17.8%** | 30.4% |
| ω scan (N = 400) | −0.5 | 25.8% | **19.5%** | | | | |
| ω scan | −0.25 | 11.9% | **9.5%** | | | | |
| Test 1 | 0 | **3.4%** | 5.8% | 4.9% | 7.8% | 4.4% | 7.3% |
| Test 2 (τ > 0) | 0 | 4.8% | **4.3%** | 6.6% | 6.4% | | |
| ω scan (N = 2000) | +0.25 | 2.0% (tie) | 2.1% (tie) | | | | |
| Test 3 | +0.5 | 3.4% (tie) | 3.4% (tie) | | | | |
| Test 5 (Cu) | 0 | 9.4% | 5.4% | 25.8% | 31.4% | **2.6%** | 4.9% |
| Test 6 (Al) | 0 | 15.9% | 8.8% | 15.7% | 22.8% | 7.6% | **5.0%** |

### 2.2 The fronts, paper bath (2 ns; diffusion vs. analytic, + = diffusion ahead / higher)

| case | heat front | shock: original | shock: impulse | front p: original | front p: impulse |
|---|---|---|---|---|---|
| ω = −0.75 | +15.6% | −16.2% | +12.9% | +16.0% | +20.7% |
| Test 4 (ω = −0.5) | +6.0% | −12.5% | +6.7% | +10.2% | +22.9% |
| ω = −0.25 | +1.6% | −5.5% | +3.4% | +1.8% | +9.3% |
| Test 1 | +1.2% | +0.6% | +2.8% | −2.1% | −0.6% |
| Test 2 | +1.0% | +2.0% | +2.4% | −4.5% | −4.4% |
| ω = +0.25 | −0.6% | +0.5% | +0.8% | −4.1% | −4.0% |
| Test 3 | −1.0% | 0.0% | 0.0% | −4.5% | −4.5% |
| Test 5 (Cu) | −1.5% | −4.7% | −2.5% | −5.1% | −3.6% |
| Test 6 (Al) | −4.1% | −7.8% | −3.9% | −8.5% | −6.3% |

The heat front is the same for both patches: the heat solution is not changed. With the other baths: `results/patching_poc/<case>_<bath>_table.md`. IMC agrees with diffusion on the shock to 0.9–1.4% in the new ω runs, as in the six tests.

## 3. Conclusions

1. **The impulse-matched patch is the better analytic reference for ω < 0, and for Cu and Al with the paper bath.** Over the whole profile it is 17–25% (relative) closer to the simulations than the original patch at ω = −0.75, −0.5 and −0.25. It nearly halves the gap for Cu (9.4 → 5.4%) and Al (15.9 → 8.8%).
2. **It is not a fix of everything.** For ω ≤ −0.5 it moves the shock error from −12 … −16% to +7 … +13%. Its front pressure is further from the simulations than the original's (+23% in Test 4), because the target starts below p_f.
3. **For ω ≥ 0 the original patch stays.** The two patches tie for ω > 0 (m_f/m_s ≤ 2.4%). In Test 1 the original is better (3.4 vs. 5.8%).
4. **The bath matters as much as the patch for Cu, Al and Test 4.** The flux-calibrated bath gives the best case overall for Test 4 (with the original patch, 17.8%), Cu (original, 2.6%) and Al (impulse, 5.0%). The temperature-calibrated bath is the worst option for Cu and Al (`../why_the_solver_differs.md` §2.2).
5. **What limits both patches for ω < 0 is the heat side.** The simulated heat front runs ahead of the rigid self-similar one: +6% in Test 4, +16% at ω = −0.75, and more at early times. No joining of the two pieces can fix that. The likely cause is the heat solution's front condition ṽ(ξ_f) = 0, a front against an infinitely dense wall. For ω < 0, the target near the surface is light (ρ₀x^(−ω) → 0). This is untested.

## 4. Next steps (theory, all analytic)

1. **A first-order theory of the massless closure.** The impulse march is a model, not a derivation. A perturbation of PistonShock in ε = m_f/m_s with a moving-mass boundary would give the corrected profile in closed form, at least where ε is small (ω ≥ −0.25). It might also remove the pressure step.
2. **The early, supersonic phase.** At ω = −0.75 most of the impulse history has m_f > m_s. The cap used here is crude. An analytic supersonic-to-subsonic transition of the front would make the patch usable there.
3. **The heat side.** A heat wave with a finite-density inflow at the front, as a perturbation in v_target/v_layer, is the candidate for conclusion 5.

## 5. Code and figures

| | |
|---|---|
| `src/menahem_new/legacy/matched_piston_patch.py` | `MatchedPistonPatch` (impulse and pressure matching) |
| `src/rad_hydro_sim/verification/legacy/patching_poc.py` | tables per bath, the ω-scan runs, `--profiles` (the partitioned figures) and `--overlay` |
| `md/phase_c/figures/` | layout in `../figures/README.md`: `<bath>/<original_patch or impulse_patch>/`, `overlay/` and `legacy/` |
| legacy (fix B, numerical) | `src/menahem_new/legacy/moving_front_shock.py`, `src/rad_hydro_sim/verification/legacy/patching_fix_b.py`, `patching_poc_fix_B.md`, `figures/legacy/fix_B/` |

```
uv run python -m rad_hydro_sim.verification.legacy.patching_poc --cases test1_full test4_full     # tables, every bath
uv run python -m rad_hydro_sim.verification.legacy.patching_poc --cases test4_full --profiles      # figures/<bath>/<method>/
uv run python -m rad_hydro_sim.verification.legacy.patching_poc --cases test4_full --overlay       # figures/overlay/
uv run python -m rad_hydro_sim.verification.legacy.patching_poc --cases --omega-scan=-0.75 --seeds 1   # an omega-scan run
```

The first `AblationSolver` construction per case takes ~20 min. Solvers are pickled to `results/patching_poc/solver_<label>.pkl`.
