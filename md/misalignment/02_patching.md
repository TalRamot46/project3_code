# B. The rigid-front patch (Test 4; part of Al and Cu)

Elaborates cause B of the [README](README.md). It is the only cause that lies in the solver itself and that no initial condition, bath or coupling removes.

---

## 1. The patch in the code

`AblationSolver.solve` (`src/menahem_new/ablation_solver_og.py`; `ablation_solver.py` has the same `solve`):

1. **Shock region.** `result_shock = self.shock_solver.solve(mass=[m_f, *mass[mass > m_f]], time)`. `PistonShock` is the solution with the piston at **m = 0**, driven by the heat region's front pressure p_f(t). It is evaluated only at m ≥ m_f.
2. **`heat_position = result_shock["position"][0]`**: the position the *shock* solution gives the tracer m_f. This is not the piston: there the slab 0 < m < m_f is shocked material ahead of the piston (Test 4 at 2 ns: 49.4 against 33.7 µm).
3. **Heat region.** `SubsonicHeatWave.solve` measures positions from its own front (ũ(ξ_f) = ṽ(ξ_f) = 0: a rigid wall at rest). `result_heat["position"] += heat_position` puts that front on the shock solution's tracer, so positions are continuous at m_f.
4. **Velocities and pressures are not matched.** The heat region keeps its own velocities, with the front at rest, and the shock region starts with its own values at m_f.

The mass 0 < m < m_f is described twice: as shocked slab (shock solution, discarded) and as ablated plasma (heat solution, kept). The patch hides the overlap in position, not in velocity or pressure.

## 2. The size of the error: how deep the cut is

At the cut the shock solution has the pressure and velocity found a fraction m_f/m_s into its own shocked slab, where the pressure has already built up. At 2 ns (`PistonShock.solve`):

| test | m_f / m_s | p(m_f) / p_f | u(m_f) / u_piston | velocity jump at m_f [km/s] | u_p [km/s] | shock vs. simulations |
|---|---|---|---|---|---|---|
| 3 (Au, ω = +0.5) | 0.5% | 1.000 | 1.000 | | | on it |
| 1 (Au, ω = 0) | 7.0% | 1.027 | 1.043 | 4.1 | 3.8 | +0.6% |
| 5 (Cu) | 8.2% | 1.030 | 1.046 | 11.7 | 7.9 | mixed with [A](01_surface_equilibrium.md) |
| 6 (Al) | 16.1% | 1.059 | 1.058 | 22.9 | 15.2 | ~2% after A is removed (§ 4) |
| 4 (Au, ω = −0.5) | 28.0% | **1.201** | **1.291** | 14.6 | 14.4 | **−12.5%** |

A real flow cannot carry a pressure step across the front. In the simulations the shocked matter is pushed by a pressure between p_f and p(m_f) (Test 4: +9–10% of p_f), so the simulated shock is weaker than the patched one. In Test 4 the target is light (ρ ∝ x^0.5), so the piston moves fast and sweeps little mass. In Test 3 it is dense and the patch is nearly exact. Since m_f/m_s ∝ t^−0.42 for ω = −0.5, the error shrinks with time. The step is visible in the black pressure curve of [test4_full_profiles.png](../../results/phase_c/figures/paper_bath/test4_full_profiles.png).

**The jump is a symptom, not the cause.** The coupling is one-way: the shock region sees the heat region only through p_f(t). Two easy fixes fail. Adding u_piston to the heat velocities removes the visible jump but not the shock error. Moving the piston to m_f (m_s → m_f + m_s) moves Test 4's analytic shock *further ahead* (5.2 → 6.7 mg/cm²). The error is dynamical: the real piston is the ablation front, with its own velocity and deceleration and denser matter ahead of it for ω < 0.

## 3. Evidence that the patch, not the codes, is the cause

1. **Piston check.** Pure hydro with the solver's own p_f(t) at m = 0 reproduces `PistonShock` to −0.07…−0.18% (ω = −0.5 and 0), with energy conserved to 10⁻¹². This was a scratch script, `piston_check.py`, and is not in the repo. `test3_hydro_only.py` does the same for Test 3 ([05_shocked_region.md § 1](05_shocked_region.md)).
2. **Not the drive, not the mesh, not transport.** In Test 4 the surface is at 0.994 HeV (1.000 calibrated). Two meshes give −12.5% and −13.3%, and IMC and diffusion agree to ≤ 2.2%.
3. **The error follows m_f/m_s** among the opaque gold tests: 0.5% → ~1–2%, 7–8% → ~1%, 28% → 12.5%.
4. **No bath reconciles Test 4.** The flux-calibrated bath halves the shock gap (−12.5% → −6.4%) but moves the heat front (+6% → +14%) and the front pressure (+10% → +15%) further off. That is the signature of a structural approximation, not of a drive mismatch.

## 4. The patch in Al (from [01_surface_equilibrium.md § 3](01_surface_equilibrium.md))

Once the surface non-equilibrium is removed (seeded at 2 ns, or χ = 1000), Al still has its shock 2.0% behind and 5% less entropy in the matter shocked after 2 ns, from both starts and at every χ. The size matches p(m_f)/p_f = 1.059. As m_f/m_s falls (16% at 2 ns, 8% at 50 ns), newly shocked matter approaches the solver: the χ = 1000 entropy median goes from 0.93 to 0.95.

The seeded runs carry a second mark of the patch. The matter shocked just after 2 ns (20–30 mg/cm² at 50 ns) has entropy 0.83–0.85. The patched profile, with its step at m_f, is not a state the simulations can hold, and its relaxation sends a weak pulse into the shock. A seed made continuous at m_f was not tried.

## 5. Fixes tried (5–6 Nov), all dropped

Each fix is a different answer to one question: when the front swallows matter, how much momentum leaves the target with it? The original patch is, to first order, a **massive** front: the swallowed matter decelerates with the front. That gives p_b = p_f − m_f a_f > p_f.

- **Fix B: a numerical target driven at the moving front.** It is exact on the target side: driven by the simulation's own front history, it reproduces the simulated shock to 0.3–1.8% (Au, ω = −0.75…+0.5). Driven by the rigid heat solution, Test 4 goes from −12.5% to +5.9%. The remainder is on the heat side: the real heat front runs 6–16% ahead of the rigid one for ω < 0, converged in mesh. B is not analytic, so it cannot be a benchmark.
- **Fix C: impulse matching with a massless front** (each element keeps the velocity it had when swallowed). It is analytic, but it flips the shock error for ω ≤ −0.5, makes the front pressure worse and leaves a step.
- **Measured truth:** the momentum the simulated target carries beyond the cut lies a fraction θ of the way from massless (0) to massive (1): θ = 1.15, 0.49, 0.37, 0.29 at ω = 0, −0.25, −0.5, −0.75. No fixed closure is right across ω.

Equations and code: [../phase_c/legacy/patching_fixes_explained.md](../phase_c/legacy/patching_fixes_explained.md). Results: [patching_poc_fix_B.md](../phase_c/legacy/patching_poc_fix_B.md), [analytic_patching_impulse.md](../phase_c/legacy/analytic_patching_impulse.md). Code: `src/rad_hydro_sim/verification/legacy/`, `src/menahem_new/legacy/`. Figures: `results/phase_c/figures/legacy/{fix_B,impulse_patch}/`.

## 6. What follows

- **For the paper:** state the validity condition m_f/m_s ≪ 1 (u_p small against the ablation flow) and give m_f/m_s next to each test. Test 4 then reads "at m_f/m_s = 28% the patched shock is 12% ahead of the full solution". For Al, compare at ≥ 20 ns (m_f/m_s ≤ 10%).
- **A better closed form** needs both regions solved together, with the front decelerating. It is not self-similar (u_p and m_f scale differently) but could be carried to first order in m_f/m_s. This is theory work, for Menahem.

## 7. Open

- The Al residual is attributed to the patch by its size and its independence of χ. The direct test, hydro with the pressure applied at the moving front, was not run for Al.
- The heat side for ω < 0 (front 6–16% ahead of the rigid solution) is not understood. The suspect is the front condition ṽ(ξ_f) = 0.
