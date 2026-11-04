# Why the semi-analytic curve does not lie on the simulations

The black curve in every profile figure is Menahem's `AblationSolver`: the subsonic heat-wave solution patched to the piston-shock solution, evaluated directly (no fits). On the paper's gold Tests 1–3 it lies on the diffusion and IMC curves within 1–2%. On copper, aluminum and gold with ω = −0.5 (Tests 4–6) it does not, by 5–15%, and calibrating the bath (option (a)) does not fix it.

The simulations are not the problem: IMC and diffusion agree with each other in every test, and the hydro code reproduces the solver's own shock exactly when given the same problem (§4.1). The black curve solves a slightly different, idealized problem. It makes **two approximations**, and each test breaks a different one:

1. **Local thermodynamic equilibrium (LTE) at the surface.** The ablated plasma has one temperature: radiation and matter are in equilibrium, E_r = aT⁴. This fails where the hot plasma is optically thin. It costs copper and aluminum ~10%.
2. **A rigid ablation front in the patching.** The heat-wave solution assumes the ablation front is at rest. The shock solution assumes a piston starting at m = 0, and the two are joined by pressure only. This fails where the shocked material moves fast, or where the ablated mass is not small next to the shocked mass. It costs Test 4 ~12%, and aluminum and copper a few %.

The table summarizes the evidence. All values are diffusion at 2 ns, finest mesh; IMC agrees with diffusion to ≤ 2% in Tests 1, 2 and 4, and ≤ 5% in Tests 5–6.

| Test | T_rad / T_mat at the surface | Shocked-matter velocity | m_f / m_s | Shock vs. solver: paper bath / calibrated | Energy in vs. solver: paper / calibrated | What breaks |
|---|---|---|---|---|---|---|
| 1 (Au, ω = 0) | 1.003 | 3.8 km/s | 7% | +0.6% / +1.8% | −2.0% / 0.0% | nothing beyond resolution |
| 2 (Au, ω = 0, τ > 0) | — | — | 8% | +2.0% / +3.5% | −0.9% / +1.5% | nothing beyond resolution |
| 3 (Au, ω = +0.5) | — | — | 0.5% | −2.1% / — | −1.8% / — | nothing beyond resolution |
| 4 (Au, ω = −0.5) | 1.007 | 14.4 km/s | 28% | −12.5% / −10.9% | −10.7% / −8.1% | **2: the patching** |
| 5 (Cu) | **1.065** | 7.9 km/s | 8% | −4.7% / +13.2% | −10.0% / **+23.8%** | **1: surface LTE**, plus some of 2 |
| 6 (Al) | **1.047** | 15.2 km/s | 16% | −7.8% / +6.6% | −11.2% / **+15.4%** | **1 and 2** |

(— : not computed for that test, or no calibrated run.)

---

## 1. What the black curve is

`AblationSolver` builds the solution of the manuscript's §2 from two self-similar pieces:

- **The ablated region (0 < m < m_f).** The subsonic heat wave of Eqs. `cont/mom/energy_lagrange_heat`. The radiation enters only as a heat flux, s = −(c/3κ_R) ∂(aT⁴)/∂m: **equilibrium (one-temperature) diffusion**, with no separate radiation field. Its boundary conditions:
  - at the surface m = 0: T = T_s(t) (1 HeV, × t^τ in Test 2) and p = 0;
  - at the ablation front m = m_f(t): **ṽ = 0 and ũ = 0** (manuscript §"The energy in the system": "using the boundary conditions p̃(0) = 0, ṽ(ξ_f) = 0, and ũ(ξ_f) = 0"). The front is an infinitely dense wall at rest.
- **The shocked region (m > m_f).** The `PistonShock` solution: a piston at **m = 0**, pushing into ρ₀x^(−ω) with pressure p_f(t) = p₀ₛ t^(c₃), the ablated region's pressure at its front.
- **The patch.** The shock solution is evaluated at the true mass coordinates m > m_f. The heat-region positions are shifted to sit at the shock solution's position of mass m_f. The two regions are matched in **pressure only**; the velocity jumps at m_f (§3).

The simulations solve the full problem: two temperatures (matter T, radiation E_r), a Marshak bath, and one continuous Lagrangian flow with no patch. Every difference between the black curve and the simulations therefore comes from one of these idealizations, or from mesh resolution.

## 2. Approximation 1: equilibrium at the surface (Tests 5 and 6)

### 2.1 What happens

The solver imposes "the surface is at 1 HeV". In LTE that sentence has one meaning. Out of LTE it has two: the matter at 1 HeV, or the radiation at 1 HeV. They are different statements, and they give different energy fluxes.

At the copper and aluminum surfaces the radiation is measurably hotter than the matter. The hot ablated plasma is only marginally opaque: its optical depth is 𝒯 ≈ 3.2 (Cu) and 2.3 (Al), against 8–10 in gold (`opacity_summary.png`). The table's T_rad/T_mat column shows it: copper 1.065, aluminum 1.047, gold 1.003–1.007.

### 2.2 Why no bath fixes it, and why calibration overshot

- **The paper's bath** follows from the Marshak relation between T_bath and the solver's surface flux, a relation derived assuming LTE. In copper, it brings the *radiation* at the surface to 1.003 HeV, but the *matter* settles at 0.942 HeV. The matter's temperature sets the ablation pressure and the heat front, so the shock lags (−4.7%) and the energy in falls short (−10%).
- **The calibrated bath (option (a))** raises T_bath until the *matter* at the surface is at 1.000 HeV. The radiation there is then 1.063 HeV, and the flux into the slab follows the radiation (F ∝ T_bath⁴ − T_rad⁴ through the Marshak face). The energy in overshoots (+24%) and so does the shock (+13%).

The solver's single-temperature surface sits between these two. No bath reproduces both its surface temperature and its energy flux, because in copper and aluminum the true surface is not in equilibrium. In gold the two temperatures agree to 0.3–0.7%, so both baths give the same answer within 2%.

So for copper and aluminum the disagreement is real physics. The paper's LTE solution is the right reference only where the ablated plasma is thick, and these drives make it marginal.

### 2.3 What can be done

- **(i) Calibrate to the energy flux instead of the matter temperature: done, and it works.** The bath is chosen so that the net flux into the slab equals the solver's s(0, t) (`bath_drive="calibrated_flux"`, `--calibrate-bath --calibrate-target flux`). The total energy then matches the solver's closed form E(t) = −s(0,t)·t/(2c_u + c_m), and what remains is the non-equilibrium redistribution inside the slab. Diffusion vs. the solver at 2 ns:

  | | heat front | shock | ablation p | energy in | matter surface T |
  |---|---|---|---|---|---|
  | Cu, paper bath | −1.5% | −4.7% | −5.1% | −10.0% | 0.942 |
  | Cu, matter T matched | +20.3% | +13.2% | +8.6% | +23.8% | 1.000 |
  | **Cu, energy flux matched** | **+3.8%** | **−0.2%** | **−1.8%** | **−2.2%** | 0.956 |
  | Al, paper bath | −4.1% | −7.8% | −8.5% | −11.2% | 0.959 |
  | Al, matter T matched | +14.3% | +6.6% | +2.8% | +15.4% | 1.000 |
  | **Al, energy flux matched** | **+2.3%** | **−2.9%** | **−2.8%** | **−2.3%** | 0.974 |
  | Au (Test 1), all three | within 2.2% | within 1.8% | −2.1 … −2.2% | −2.0 … 0.0% | 0.997–1.000 |

  The matter at the surface settles between the two extremes, as expected where it is not in equilibrium with the radiation. The remaining 2–4% is not decomposed. Likely contributors are the energy delivered before 0.2 ns, where the calibration does not act because the first cell does not yet resolve the heated layer, and the patching's share in aluminum (§3). Figure: `figures/bath_comparison.png`.
- **(ii) Report Tests 5–6 as the limit of the LTE solution.** State 𝒯 of the hot plasma next to each test (the figures now show it). Say that for 𝒯 ≲ 3 the LTE semi-analytic solution has a ~10% systematic error in the delivered energy, which both transport (IMC) and non-equilibrium diffusion see. This is a physics result, and the agreement between IMC and diffusion shows it is not a code artifact.
- **(iii) Choose drives where 𝒯 is larger** (a higher density, a longer pulse, or a lower T_s for the light materials), if Tests 5–6 must agree with the solver to a few %.

## 3. Approximation 2: the rigid front of the patching (Test 4, partly Tests 5–6)

### 3.1 What happens

The heat-wave solution ends at a front with ũ = 0: in its own solution the ablation front does not move. The shock solution, at the same mass, has the piston moving into the target at u_p > 0. The patch keeps both, so the black velocity curve **jumps** at m_f. The jump is visible in the velocity panels: small in gold Test 1, large in Test 4 (black curve, m ≈ 1–1.5 mg/cm²). In the simulation the velocity is continuous. The ablation front is carried forward with the shocked matter, which changes the ablated flow and the pressure history the shock receives.

The same assumption shows up in mass: the shock solution counts the ablated mass m_f as part of a slab pushed from m = 0. That is harmless while m_f ≪ m_s.

Two numbers measure how wrong the rigid front is:
- the shocked-matter velocity u_p, against the ablated flow speeds (~100 km/s);
- the ratio m_f/m_s.

| | Test 1 | Test 5 (Cu) | Test 6 (Al) | Test 4 (ω = −0.5) |
|---|---|---|---|---|
| u_p | 3.8 km/s | 7.9 km/s | 15.2 km/s | 14.4 km/s |
| m_f/m_s at 2 ns | 7% | 8% | 16% | 28% (37% at 1 ns) |
| shock vs. solver | +0.6% | (mixed with §2) | (mixed with §2) | −12.5% |

In Test 4 the shocked region is ρ ∝ x^(+0.5): light and easily pushed, so the piston moves fast and the shock sweeps little mass. In Test 3 (ω = +0.5) the target is dense and m_f/m_s = 0.5%, so the patching is nearly exact there.

### 3.1a The patch in the code, line by line

`AblationSolver.solve` (`src/menahem_new/ablation_solver_og.py`; `ablation_solver.py` has the same `solve`):

1. **The shock region.** `mass_shock = [ablated_mass, *mass[mass > ablated_mass]]`, then `result_shock = self.shock_solver.solve(mass=mass_shock, time=time)`. `PistonShock` is the solution with the piston at **m = 0**: it started at x = 0 and is now at `piston_position(t)` > 0. It is evaluated only at m ≥ m_f.
2. **`heat_position = result_shock["position"][0]`** is one number: the Eulerian position that the *shock* solution gives the tracer of mass m_f. It is not the piston. In the shock solution the slab 0 < m < m_f is shocked material in front of the piston, so `heat_position` > `piston_position`. At 2 ns: Test 1 8.74 vs 8.01 µm; Test 4 49.4 vs 33.7 µm; Cu 18.5 vs 16.5 µm; Al 38.5 vs 33.9 µm.
3. **The heat region.** `SubsonicHeatWave.solve` returns positions measured from its own ablation front: `get_position_self_similar_factor` is 0 at ξ ≥ ξ_f, negative inside, and the surface is at `boundary_position(t)` = x_b < 0. `result_heat["position"] += heat_position` moves the whole ablated layer so that its front sits on the shock solution's tracer m_f. Positions are continuous at m_f.
4. **Velocities are not shifted.** The heat region keeps its own velocities, in which the front is at rest (ũ(ξ_f) = 0). The shock region at m_f moves at about the piston velocity. Hence the jump in the black velocity curve, at 2 ns:

| | u just behind m_f | u just ahead | jump | piston u |
|---|---|---|---|---|
| Test 1 (Au) | −0.9 km/s | +3.2 km/s | 4.1 | 3.1 |
| Test 4 (Au, ω = −0.5) | −1.1 km/s | +13.5 km/s | 14.6 | 10.5 |
| Test 5 (Cu) | −4.9 km/s | +6.8 km/s | 11.7 | 6.5 |
| Test 6 (Al) | −8.8 km/s | +14.2 km/s | 22.9 | 13.4 |

The mass 0 < m < m_f is described twice: as shocked material pushed by a piston at m = 0 (shock solution, then discarded), and as ablated plasma flowing back (heat solution, kept). The patch hides the overlap in position, but not in velocity.

**The jump is a symptom, not the cause of the shock error.** The patch is one-way: the shock region sees the heat region only through p_f(t). Adding u_piston to the heat-region velocities would remove the visible jump but leave the shock where it is. The bookkeeping fix of putting the piston at m_f (m_s → m_f + m_s(PistonShock)) moves Test 4's analytic shock *further ahead* (5.2 → 6.7 mg/cm²), while the simulations' shock is 12% *behind* it. The error is dynamical: the real piston, the ablation front, has its own velocity and deceleration, and, for ω = −0.5, denser material directly ahead of it than the piston-at-m = 0 solution assumes. None of that is in the patch. Removing it requires solving the two regions together: fix (B) numerically, or (C) in theory.

### 3.2 Evidence that the codes are right and the patching is the cause

1. **The piston check** (scratch script, recorded here). A pure-hydro run with the same code, ρ₀x^(−ω) and a piston at m = 0 driven by the solver's own p_f(t) = p₀ₛ t^(c₃), reproduces the solver's shock: −0.17% at 1 ns and −0.07% at 2 ns for ω = −0.5; −0.18% / −0.11% for ω = 0. Energy is conserved to 10⁻¹². So the hydro solver and `PistonShock` agree when they solve the same problem.
2. **Test 4 is not the drive and not the mesh.** On the uniform-mass mesh the surface is at 0.994 HeV (1.000 calibrated), and two different meshes give the same −12.5% / −13.3%. Calibration moves the shock only from −12.5% to −10.9%.
3. **Transport is not involved.** IMC and diffusion agree to 0.6–2.2% in Test 4.
4. **The size of the error follows m_f/m_s and u_p.** Among the opaque gold tests: 0.5% → ~2% (Test 3, at a coarser mesh); 7–8% → ~1% (Tests 1–2); 28% → 12.5% (Test 4).

What is not proven is the exact split between the velocity mismatch and the mass mismatch; both come from the same rigid-front assumption.

### 3.3 Suggested fixes

- **(A) For the paper now: state the validity condition.** The patched solution assumes a rigid ablation front, which holds when m_f/m_s ≪ 1 and u_p is small against the ablation flow. Report m_f/m_s next to each test. Test 4 then reads as "at m_f/m_s = 28% the patched solution's shock is 12% ahead of the full solution". Because m_f/m_s ∝ t^(−0.42) for ω = −0.5, the error shrinks with time; a longer t_end (or a drive with a slower heat wave) brings Test 4 into the regime where it agrees.
- **(B) A better reference for Test 4: a numerical shock region.** Keep the self-similar heat solution, but replace `PistonShock` with a 1-D Lagrangian hydro solution of the target driven at the *moving* ablation front. The piston boundary sits at the heat solution's m_f(t), with pressure p_f(t), and with the target ahead of it starting at x(m_f). This is a pure-hydro run like the piston check above, with the left boundary placed at m_f(t) instead of 0. It removes the mass mismatch exactly, and is cheap (seconds). It is no longer closed-form, so it is a reference curve, not a formula.
- **(C) A semi-analytic correction.** Patch both pressure and velocity: put the heat solution in the frame of the moving piston, where it feels the frame's deceleration, a body force −m du_p/dt on the ablated mass. Because u_p and m_f scale with different powers of t, the corrected problem is not self-similar, but the correction can be carried to first order in m_f/m_s. This is a real piece of theory work, and it belongs with Menahem.

My recommendation: (A) in the manuscript text, and (B) as the reference curve for Test 4 if a closer agreement is wanted in the figure. (C) only if the paper wants an improved closed form.

## 4. What calibration did and did not do

Option (a) changes only the drive: it makes the simulated matter surface follow the solver's T_s(t).
- **Gold, ω ≥ 0 (Tests 1–3):** the drive was already right (T_s ≈ 0.997), so calibration changes the curves by 1–2%, within resolution.
- **Test 4:** calibration removes the small drive deficit (−12.5% → −10.9% on the shock), but the patching error (§3) is untouched.
- **Copper and aluminum:** calibration over-corrects, because the target "matter at 1 HeV" is not what the LTE solver means at a non-equilibrium surface (§2.2).

So calibration to the matter temperature is the right tool only where the surface is in LTE, and there it was barely needed. Calibration to the energy flux (§2.3(i)) is the version that makes sense for copper and aluminum, and it brings both within ~4% of the solver. For Test 4 the flux-matched bath halves the shock gap (−12.5% → −6.4%) but moves the heat front (+6% → +14%) and the ablation pressure (+10% → +15%) further off. No drive reconciles the patched solution there, which is the signature of a structural approximation, not a drive mismatch.

## 5. Resolution, for completeness

At N = 400 the gold heat fronts sit 4–9% off the solver; at N = 1000 they are within 0.8–2.6% (`phase_c_results.md` §5). All numbers above are at the finest mesh of each test: Tests 1, 2, 4 and 5 at 1000 cells, Test 6 at 400 (at 1000 cells its differences from the solver move by ≤ 1.1 percentage points for diffusion and ≤ 0.15 for IMC), Test 3 at its graded 347-cell mesh.

## 6. Status of the checks behind this document

| Check | Where | Result |
|---|---|---|
| Piston at m = 0 vs. `PistonShock`, ω = −0.5 and 0 | scratch script `piston_check.py` | −0.07 … −0.18% |
| T_rad / T_mat at the surface | diffusion runs at 2 ns | Cu 1.065, Al 1.047, Au 1.003–1.007 |
| Paper vs. calibrated bath | `results/imc_vs_diffusion{,_calibrated}` | table above |
| Test 4 mesh independence | graded (1397 cells) vs. uniform-mass (1000) | shock −13.3% / −12.5% |
| IMC vs. diffusion | `phase_c_results.md`, tracker | ≤ 2.2% (Au), ≤ 5.5% (Cu, Al) |
