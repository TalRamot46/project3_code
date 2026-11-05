# Which solutions should agree, and why: IMC, diffusion and the semi-analytic solver

Three solutions of the manuscript's subsonic Marshak-wave problem are compared on its six tests:

- **IMC**: gray transport;
- **diffusion**: gray two-temperature diffusion;
- **the semi-analytic solver**: Menahem's patched self-similar solution, the black curve in every figure.

This document derives, from the assumptions of each, which of them should agree with which, and under what conditions. It then checks each prediction against the runs, and ends with the parameters that decide the agreement and with the limits of the solver, which assumes complete thermal equilibrium between matter and radiation.

| Test | Material | Initial density | Drive |
|---|---|---|---|
| 1 | Au | uniform (ω = 0) | τ = 0 |
| 2 | Au | uniform | τ = 158/1285 |
| 3 | Au | ρ₀x^(−1/2) (ω = +0.5) | τ = 0 |
| 4 | Au | ρ₀x^(+1/2) (ω = −0.5) | τ = 0 |
| 5 | Cu | uniform | τ = 0 |
| 6 | Al | uniform | τ = 0 |

All drives have T_s = 1 HeV at 1 ns (Test 2 rises as t^τ).

Results: `md/phase_c/phase_c_results.md` (Tests 1–2), `md/phase_c/phase_c_tests3to6_results.md` (all six), figures in `md/phase_c/figures/`. Every number here comes from the runs in `results/imc_vs_diffusion*/` or from the solver itself, at t = 2 ns unless stated. The scripts are in `rad_hydro_sim.verification.imc_vs_diffusion`; a few one-off checks are listed in §6. Statements that are inferences, not measurements, are marked as such.

## Contents

0. The answer in brief
1. The three solutions and what each one assumes
2. The hierarchy of approximations, and what each comparison isolates
3. Predictions for the six tests, from the assumptions alone
4. Numerical validation
5. Conclusions: the determining parameters, and the limits of the equilibrium solver
6. Data and reproduction

---

## 0. The answer in brief

- **IMC vs. diffusion** differ only in how radiation propagates: transport vs. the P1/Fick closure. Theory says they agree where the hot ablated plasma is optically thick, and the runs confirm it. The order follows the optical depth 𝒯 of that plasma: gold (𝒯 ≈ 8–10) ≤ 2.3%, copper (3.2) ≤ 4.2%, aluminum (2.3) ≤ 5.3%. Neither code assumes equilibrium between radiation and matter, so their agreement does **not** test the solver's equilibrium assumption.
- **The solver vs. both codes** adds two approximations of its own.
  1. *One temperature (complete equilibrium)*, imposed at the surface as T = T_s. This needs an opaque surface layer. It holds in gold (surface T_rad/T_mat = 1.005–1.007) and fails in copper and aluminum (1.047–1.065). There the manuscript's bath delivers ~10% too little energy, and no bath can reproduce both the solver's surface temperature and its energy flux.
  2. *The patching*, which cuts a piston-at-m = 0 shock solution at the ablated mass m_f. At the cut its pressure exceeds the ablation pressure by about m_f·|du/dt|, the inertia of the ablated slab, which the real flow does not have. The excess is ≤ 3% in every test except Test 4 (ω = −0.5), where it is 20% and the shock misses by 6–13%.
- **The deciding parameters** are:
  - for the equilibrium assumption, the solver's own lower-bound optical depth 𝒯_lb = m_f κ_R(T_s, ρ_s), computable without any simulation. It is ≥ 2.3 for gold, 0.56 for copper, 0.46 for aluminum;
  - for the patching, the pressure step at the cut, Δp/p_f;
  - then mesh resolution of the ablated layer and of the shocked region.

---

## 1. The three solutions and what each one assumes

### 1.1 IMC (`src/imc/`, coupled through `src/imc/coupling/rad_hydro.py`)

IMC solves the gray radiation transport equation of the manuscript's §"Transport setup" (Eq. `Treq`), coupled to the material energy equation (Eq. `tr_mat`), by Monte Carlo. Its assumptions:

1. **Gray transport, angle-resolved.** The intensity I(x, μ, t) is followed in angle; there is no closure. Absorption opacity k_a = χρκ_R, physical scattering (1 − χ)ρκ_R; χ = 1 in all six tests, so pure absorption.
2. **Two temperatures.** The radiation field and the matter exchange energy at the rate c k_a (E_r − aT⁴); they are not forced to equilibrium.
3. **Fleck–Cummings time linearization**, O(Δt): emission from Tⁿ with the Fleck factor, the "effective scattering". This is the standard IMC approximation, shared with diffusion (§1.2).
4. **Monte Carlo noise.** 3 seeds; the standard errors on fronts are ~0.01–0.1%, much smaller than any difference discussed here.
5. **Boundaries.** An isotropic blackbody bath at T_bath(t) enters at the surface, the manuscript's Eq. `Ibath_tr`, integrated exactly over each step. The right face is a vacuum.
6. **The same hydro as diffusion:**
   - Lagrangian, compatible (energy-conserving) scheme;
   - ideal gas p = rρe, with the material energy e = fT^βρ^−μ;
   - artificial viscosity;
   - radiation carried per unit mass through the hydro step, with no radiation pressure (the manuscript's equations);
   - same mesh and same time steps.

### 1.2 Diffusion (`src/rad_hydro_sim/simulation/radiation_step.py`)

Diffusion solves the manuscript's gray diffusion equations (Eqs. `gray_diff`, `gray_diff_mat`) with backward Euler and the same Fleck–Cummings linearization. Its assumptions differ from IMC's in exactly two places:

1. **The P1/Fick closure.** The flux is F = −(c/3ρκ_R) ∂E_r/∂x. This is exact only when the radiation is nearly isotropic, i.e. when the photon mean free path λ = 1/(ρκ_R) is short compared with the length L_E over which E_r changes. There is **no flux limiter**, so the flux is not bounded by cE_r where gradients are steep.
2. **A Marshak boundary with the cell-centre value**: F = (c/2)(E_bath − E₀). This is the P1 form of the incident half-range current; it uses the first cell's E₀, with no half-cell resistance (`md/phase_b/phase_b_diffusion_changes.md` §5.3).

Everything else (two temperatures, linearization, hydro, mesh, time steps, EOS) is the same as IMC's. **Diffusion does not assume equilibrium either:** it carries E_r separately from aT⁴.

### 1.3 The semi-analytic solver (`src/menahem_new/ablation_solver_og.py`, `subsonic_heat_wave_og.py`, `piston_shock_og.py`)

The solver builds the solution of the manuscript's §2 from two self-similar solutions.

**The ablated region, 0 < m < m_f(t): `SubsonicHeatWave`.** It solves Eqs. `cont/mom/energy_lagrange_heat`, where radiation enters only as the heat flux s = −(c/3κ_R) ∂(aT⁴)/∂m. Its assumptions:
1. **Complete thermal equilibrium.** E_r = aT⁴ at every point: one temperature, the radiation's angular structure and its departure from the matter both ignored (equilibrium diffusion, manuscript Eq. `lte_diff_m`).
2. **No separate radiation energy.** The radiation's energy is not a variable; Rosen's e is read as the total energy of matter plus radiation (Eq. `um_def`). The codes carry E_r as well, which is the same thing only if E_r ≪ ρe.
3. **A Dirichlet surface:** T(m = 0, t) = T_s(t) exactly, with p = 0 at the surface.
4. **A rigid front:** at m_f, ṽ = 0 and ũ = 0. The front is an infinitely dense wall at rest.
5. **Self-similarity:** power-law material and drive, a cold semi-infinite target, no length scale.

**The shocked region, m > m_f: `PistonShock`.** A strong shock driven by a piston at **m = 0**, with pressure p_f(t) = p₀ₛ t^(c₃) equal to the ablated region's front pressure, into a cold target ρ₀x^−ω.

**The patch, `AblationSolver.solve`:**
- `mass_shock = [ablated_mass, *mass[mass > ablated_mass]]`; the shock solution is evaluated only at m ≥ m_f.
- `heat_position = result_shock["position"][0]` is one number: where the *shock* solution places the tracer of mass m_f. That is not the piston: in the shock solution the slab 0 < m < m_f is shocked material in front of the piston.
- `result_heat["position"] += heat_position` moves the whole ablated layer so that its front sits there. Positions are continuous at m_f.
- Velocities and pressures are **not** adjusted. The heat region's front is at rest with pressure p_f. The shock solution at m_f has a pressure p(m_f) and a velocity u(m_f) that are not the piston's (§4.3).

So the solver adds five approximations to the codes' physics:
- three about radiation: equilibrium; radiation energy folded into e; a Dirichlet surface;
- one about its construction: the rigid-front patch;
- one about the domain: a semi-infinite target with no discretization.

---

## 2. The hierarchy of approximations, and what each comparison isolates

The three solutions are nested, each step adding approximations:

```
gray transport (IMC)
   |  P1/Fick closure: photon mean free path << radiation gradient length; Marshak face
   v
two-temperature gray diffusion (diffusion)
   |  complete equilibrium E_r = aT^4 (fast coupling, opaque surface layer);
   |  radiation energy << matter energy; surface held at T_s instead of a bath
   v
one-temperature equilibrium diffusion + hydro (the problem the solver solves)
   |  self-similar solution with the rigid-front patch (small step at the cut);
   |  continuous, semi-infinite
   v
the patched semi-analytic solver (the black curve)
```

All three share the ideal-gas EOS, the material model, the gray opacity and the planar geometry. The two codes also share the hydro scheme, the mesh, the time steps and the time linearization.

| Comparison | What differs | Controlling parameter | Measured by |
|---|---|---|---|
| IMC vs. diffusion | angular closure; boundary treatment | optical depth 𝒯 of the hot plasma, i.e. λ/L_E | front, pressure and energy differences (§4.1) |
| diffusion vs. the equilibrium problem | two temperatures vs. one; bath vs. Dirichlet surface; radiation energy | opacity of the surface layer; E_r/(ρe) | T_rad/T_mat; surface temperature deficit; energy delivered (§4.2) |
| equilibrium problem vs. the black curve | the patch | the pressure step Δp/p_f at the cut (∝ m_f/m_s) | the piston check; the shock gap once the drive is removed (§4.3) |
| any code vs. its limit | discretization | cells across the ablated layer and the shocked region | mesh refinement (§4.4) |

Two consequences shape everything below:
- **IMC agreeing with diffusion says nothing about equilibrium.** Both codes allow T_rad ≠ T_mat, so they can agree with each other while both differ from the equilibrium solver.
- **When IMC and diffusion agree with each other but both differ from the black curve, the difference lies in the solver's own approximations:** the equilibrium surface or the patch. Neither code can be the cause.

---

## 3. Predictions for the six tests, from the assumptions alone

The controlling parameters can be computed from the solver alone, with no simulation:

- **𝒯_lb = m_f κ_R(T_s, ρ_s)**, the manuscript's lower-bound optical depth of the ablated layer (§"Gray diffusion setup", Eq. `T_lower_bound`), with ρ_s the solver's surface density;
- **𝒯_hot**, the solver's optical depth of its hot plasma, T > T_s/2: the region the drive's radiation crosses;
- **m_f/m_s** and **Δp/p_f**, the shock solution's pressure step at the cut.

| Test | 𝒯_lb | 𝒯_hot (solver) | m_f/m_s | Δp/p_f |
|---|---|---|---|---|
| 1 Au, ω = 0 | 2.59 | 8.48 | 7.0% | 2.7% |
| 2 Au, τ > 0 | 2.28 | 7.78 | 8.1% | 0.6% |
| 3 Au, ω = +0.5 | 2.56 | 8.70 | 0.5% | 0.0% |
| 4 Au, ω = −0.5 | 2.61 | 8.74 | 28.0% | **20.1%** |
| 5 Cu | **0.56** | 2.84 | 8.2% | 3.0% |
| 6 Al | **0.46** | 2.16 | 16.1% | 5.9% |

What the assumptions then predict:

| Test | IMC ↔ diffusion | equilibrium (solver's surface) | patch | expected agreement with the black curve |
|---|---|---|---|---|
| 1, 2, 3 | good (𝒯 ≈ 8) | good (𝒯_lb > 2) | small | good, to resolution |
| 4 | good | good | **poor** (20% step) | poor, from the patch alone |
| 5 | fair (𝒯 ≈ 3) | **poor** (𝒯_lb ≈ 0.5) | small | poor, from the surface |
| 6 | fair (𝒯 ≈ 2) | **poor** | moderate (6% step) | poor, from the surface and partly the patch |

Gold's 𝒯_lb is almost the same in Tests 1–4, because the ablated region does not depend on ω. Copper and aluminum are 4–5× less opaque at the surface: κ_R at 1 HeV and 0.01 g/cm³ is 2866 (Au), 588 (Cu) and 272 cm²/g (Al), and their opacity falls faster with temperature (κ ∝ T^−α with α = 1.5, 2.21, 3.1).

---

## 4. Numerical validation

### 4.1 IMC vs. diffusion: is the P1/Fick closure good?

**Prediction:** agreement improving with 𝒯_hot; the largest differences in the quantity most sensitive to the surface (the energy delivered), because the two codes treat the bath face differently (exact incident current vs. cell-centre Marshak).

IMC vs. diffusion at 2 ns, with the manuscript's bath, at each test's finest mesh. IMC is the mean of 3 seeds (1 seed at N = 1000 for Tests 1–2).

| Test | 𝒯_hot (simulated) | heat front | shock | ablation pressure | energy delivered |
|---|---|---|---|---|---|
| 1 | 9.0 | −0.5% | −1.5% | +0.6% | −2.3% |
| 2 | 8.0 | −0.5% | −1.4% | −0.5% | −2.1% |
| 3 (coarse mesh) | 8.8 | −0.6% | −1.8% | +0.3% | −2.3% |
| 3 (fine mesh, branch `imc/test3-fine-mesh`) | 8.8 | −0.6% | −1.0% | +0.1% | −2.2% |
| 4 | 10.1 | −0.7% | −0.9% | −1.3% | −1.2% |
| 5 | 3.2 | −2.5% | −2.5% | −1.6% | −4.2% |
| 6 | 2.3 | −4.6% | −3.3% | −1.2% | −5.3% |

What this shows:
- **The predicted order holds:** the worst difference is ~2% in gold (𝒯 ≈ 8–10), 4.2% in copper (3.2), 5.3% in aluminum (2.3). Across the three materials it falls roughly like 𝒯^(−0.6). This is an empirical fit over three points, not a derived law.
- **The energy delivered is the worst quantity in five of the six tests.** IMC takes in 1–5% less, consistent with the surface boundary layer, where the two codes' boundary conditions differ most.
- **The fronts converge with the mesh.** In Test 1 the IMC-vs-diffusion heat-front difference goes from +0.8…+3.9% at 400 cells to −0.1…−0.9% at 1000 (`phase_c_results.md` §5, under the earlier density-based front definition). What remains at 1000 cells is transport, not resolution.
- **The one visible local difference is in Test 3.** On the fine mesh, behind the ablation front (1–10 mg/cm²), IMC's density is clearly higher than diffusion's at the same pressure, and closer to the solver's. Diffusion's material there is hotter. The likely cause is that diffusion without a flux limiter carries too much flux into the cold dense material just ahead of the front, which transport limits. This is an inference from the profiles; a flux-limited diffusion run (plan M11) would test it.

**Verdict:** diffusion reproduces transport to ~2% where 𝒯_hot ≳ 8, and to ~5% at 𝒯_hot ≈ 2–3. The P1 closure is never the dominant error against the black curve.

### 4.2 Diffusion vs. the equilibrium problem: does matter–radiation equilibrium hold?

**Prediction:** equilibrium holds in the interior of every test, about one mean free path inside the plasma. At the surface it holds only where the surface layer is opaque (𝒯_lb ≳ 2). Where it does not hold, the manuscript's bath, which follows from the Marshak relation derived under equilibrium, cannot produce the solver's surface.

**Equilibrium measured in the diffusion runs at 2 ns:**

| Test | T_rad/T_mat at the surface | one optical depth in | matter surface T vs. T_s | radiation / matter energy (hot plasma) |
|---|---|---|---|---|
| 1 | 1.005 | 1.002 | −0.5% | 0.9% |
| 2 | 1.006 | 1.003 | −0.6% | 0.9% |
| 3 | 1.005 | 1.002 | −0.5% | 1.0% |
| 4 | 1.007 | 1.002 | −0.6% | 1.0% |
| 5 | **1.065** | 1.019 | **−5.8%** | 0.6% |
| 6 | **1.047** | 1.025 | **−4.1%** | 0.4% |

What this shows:
- **Interior equilibrium holds everywhere.** One optical depth in from the surface, T_rad/T_mat is within 0.2–2.5% in every test.
- **At the surface, equilibrium holds in gold** (≤ 0.7% off) **and fails in copper and aluminum** (5–7% off), exactly as 𝒯_lb predicts: 2.3–2.6 vs. 0.46–0.56.
- **The radiation energy is negligible everywhere** (≤ 1% of the hot plasma's energy). Assumption 2 of §1.3 holds in all tests; it reaches 5–18% only in the outermost one or two cells.
- **Every test has a thin non-equilibrium layer at the heat front itself,** where T_rad and T_mat separate by more than 1% over the last few percent of the ablated mass. That is the structure of a steep front and does not depend on the material.

**The decisive test: the three baths.** If the surface is in equilibrium, matching the solver's surface temperature and matching its energy flux should be the same thing. If it is not, they should differ. Diffusion vs. the solver at 2 ns:

| | bath | heat front | shock | ablation p | energy delivered | matter surface T |
|---|---|---|---|---|---|---|
| Au (Test 1) | manuscript's | +1.2% | +0.6% | −2.1% | −2.0% | 0.997 |
| | surface T matched | +2.2% | +1.8% | −2.1% | 0.0% | 1.000 |
| | energy flux matched | +2.0% | +1.4% | −2.2% | −0.5% | 0.999 |
| Cu | manuscript's | −1.5% | −4.7% | −5.1% | −10.0% | 0.942 |
| | surface T matched | +20.3% | +13.2% | +8.6% | +23.8% | 1.000 |
| | energy flux matched | +3.8% | **−0.2%** | −1.8% | −2.2% | 0.956 |
| Al | manuscript's | −4.1% | −7.8% | −8.5% | −11.2% | 0.959 |
| | surface T matched | +14.3% | +6.6% | +2.8% | +15.4% | 1.000 |
| | energy flux matched | +2.3% | −2.9% | −2.8% | −2.3% | 0.974 |

(Test 1 at N = 400 here, so its fronts carry ~1–2% resolution error; see §4.4.)

- **In gold the three baths agree within 2%,** as expected for an equilibrium surface.
- **In copper and aluminum they differ by 25–35% in energy.** There, "the surface is at T_s" is ambiguous. The matter at T_s means radiation hotter than T_s and too much flux; the radiation at T_s means matter cooler and too little flux. The solver's one-temperature surface lies between.
- **Matching the energy flux brings both within ~4% of the solver,** with the matter settling between the extremes (0.956, 0.974). That is the operational meaning of the equilibrium assumption: the solver is a valid reference for a non-equilibrium material only when driven to its energy flux, not its temperature.

**The a-priori predictor works.** The solver's own 𝒯_hot matches the simulated one within 1–14% in every test: 8.48 vs. 9.01, 7.78 vs. 8.00, 8.70 vs. 8.80, 8.74 vs. 10.09, 2.84 vs. 3.22, 2.16 vs. 2.29. So both 𝒯_lb and 𝒯_hot can be computed before any simulation, and they separate the equilibrium tests (gold) from the non-equilibrium ones (copper, aluminum) cleanly.

**Verdict:** the equilibrium assumption holds in all four gold tests and fails at the surface in copper and aluminum, where 𝒯_lb < 1. The consequence is ~10% in the energy delivered with the manuscript's bath, which an energy-flux-calibrated bath removes.

### 4.3 The equilibrium problem vs. the black curve: how good is the patch?

**Prediction:**
- The patch is exact when the cut lies at the piston (m_f/m_s → 0).
- Otherwise, the shock solution at the cut has the pressure of a point inside its shocked slab. Momentum conservation in mass coordinates, ∂u/∂t = −∂p/∂m, gives that pressure as p(m_f) − p_f ≈ m_f·|du/dt|: the force needed to decelerate the slab between the piston and the cut. The real flow has no such slab: its ablated mass is hot plasma flowing *backwards*. So the real shocked material is pushed by less than p(m_f), and the solver's shock should run ahead of the codes', by an amount growing with Δp/p_f.

**Check 1: the hydro code reproduces the solver's shock exactly, without the patch.** A pure-hydro run with a piston at m = 0, driven by the solver's own p_f(t) = p₀ₛ t^(c₃) on a 1000-cell uniform-mass mesh, reproduces `PistonShock`:
- ω = −0.5: −0.17% at 1 ns, −0.07% at 2 ns;
- ω = 0: −0.18% and −0.11%;
- energy is conserved to 10⁻¹².

So the hydro and the shock solution agree when they solve the same problem.

**Check 2: the step at the cut, and its estimate from the slab's inertia** (`PistonShock.solve` at m_f; rigid-slab estimate m_f·|du_piston/dt| with u_piston ∝ t^k):

| Test | m_f/m_s | p(m_f)/p_f − 1 | inertia estimate | u(m_f)/u_piston |
|---|---|---|---|---|
| 3 | 0.5% | 0.0% | — | 1.000 |
| 2 | 8.1% | 0.6% | — | 1.002 |
| 1 | 7.0% | 2.7% | 2.5% | 1.043 |
| 5 | 8.2% | 3.0% | — | 1.046 |
| 6 | 16.1% | 5.9% | — | 1.058 |
| 4 | 28.0% | **20.1%** | 14.5% | 1.291 |

- **The inertia estimate accounts for the step.** It reproduces Test 1's 2.7% and three quarters of Test 4's 20% (the slab there is not rigid: u(m_f) = 1.29 u_piston).
- **The step is not proportional to m_f/m_s alone.** Test 2 has the same ratio as Test 1 but a 0.6% step, because its drive rises (τ > 0), so its piston barely decelerates. **Δp/p_f is the better measure.**
- **In Test 4 the step is visible** in the black pressure curve of `test4_full_profiles.png`, at the ablation front.

**Check 3: the shock gap once the drive is removed.** The flux-calibrated bath removes the drive difference of §4.2 (for gold, the manuscript's bath at 1000 cells does). What remains of the shock error should follow Δp/p_f:

| Test | Δp/p_f | shock, diffusion vs. black curve |
|---|---|---|
| 3 (fine mesh) | 0.0% | 0.0% |
| 2 (N = 1000) | 0.6% | −0.1% |
| 5 (flux bath) | 3.0% | −0.2% |
| 1 (N = 1000) | 2.7% | −1.4% |
| 6 (flux bath) | 5.9% | −2.9% |
| 4 (flux bath) | 20.1% | −6.4% (manuscript's bath: −12.5%) |

- **The ranking follows Δp/p_f.** The two lowest steps give no shock error, the 3–6% steps give 0–3%, and Test 4's 20% step gives 6–13%. The sign is always the predicted one: the codes' shock behind the black curve. (Copper's −0.2% is lower than its 3% step suggests; with six points this is a strong trend, not a law.)
- **In Test 4 no bath reconciles the solver.** The flux bath halves the shock gap but moves the heat front (+6% → +14%) and the ablation pressure (+10% → +15%) further off. A structural approximation behaves this way; a drive mismatch does not.
- **The simulations' ablation pressure in Test 4 (+9 … +10% of p_f) lies between p_f and p(m_f) = 1.20 p_f,** as the inertia argument predicts.
- **The error shrinks with time:** m_f/m_s ∝ t^(−0.42) for ω = −0.5, from 37% at 1 ns to 28% at 2 ns, and the shock gap from −14.3% to −12.5%.

**Verdict:** the patch is accurate to ≲ 3% in the shock where Δp/p_f ≲ 6%, and fails in Test 4 (Δp/p_f = 20%). This is an approximation of the solver's construction, independent of equilibrium: Test 4 is in equilibrium (§4.2).

### 4.4 Discretization

| Test | change of mesh | effect |
|---|---|---|
| 1 (Au) | 400 → 1000 cells (2.5×10⁻⁵ g/cm² per cell) | heat front vs. solver +5.6% → +1.7% (density-based front; temperature-based at 1000 cells: −0.5%) |
| 3 (ω = +0.5) | 347-cell graded → 1237 cells with a 0.3 mg/cm² cap (branch `imc/test3-fine-mesh`) | shock vs. solver −2.1% → 0.0%; heat front (−1.0%) and ablation pressure (−4.5%) unchanged |
| 4 (ω = −0.5) | 1397-cell graded → 1000-cell uniform-mass | surface 0.934 → 0.994 HeV; energy error 10⁻¹¹ → 10⁻¹⁴; shock −13.3% → −12.5% |
| 6 (Al) | 400 → 1000 cells | ≤ 1.1 percentage points (diffusion), ≤ 0.15 (IMC) |

- **The mesh matters at the few-% level,** most in the fronts and the shock position.
- **It does not explain the differences of §§4.2–4.3:** those survive refinement (Test 4's shock, Test 6's surface).
- **The resolution needed:** ≳ 50 cells across the ablated layer, and shocked-region cells ≲ 10⁻³ m_s.
- **The graded meshes for ω ≠ 0 had to be replaced or capped.** They crowd cells at the surface: Test 4 had 628 cells in the first 10⁻⁵ g/cm², and Test 3 had 4–12 mg/cm² cells at its shock.

### 4.5 Secondary assumptions, all checked to be small

| Assumption shared by all three | Check | Size |
|---|---|---|
| no radiation pressure | `rad_coupling="physical"` (P_rad = E/3 and its work), Tests 1, 2, 6 | ≤ 0.1% on fronts; ≤ 0.5% on the ablation pressure (Al) |
| no O(v/c) terms | max v/c; Σ τ·v/c over the heated cells | v/c ≤ 1×10⁻³ everywhere; τv/c ≤ 3×10⁻² in gold, **0.25 in aluminum** (marginal, see §5.2) |
| radiation energy ≪ matter energy | §4.2 | ≤ 1% |
| Fleck–Cummings O(Δt) | shared by both codes, so it cancels in IMC vs. diffusion | not separated from the solver comparison |
| IMC noise | 3 seeds | s.e. 0.01–0.1% on fronts |

---

## 5. Conclusions

### 5.1 The parameters that decide whether the curves match

In order of their effect on these six tests:

1. **The opacity of the hot ablated plasma, at the surface:** 𝒯_lb = m_f κ_R(T_s, ρ_s), or the solver's 𝒯_hot. It decides both whether IMC matches diffusion and whether the solver's equilibrium surface exists.
   - 𝒯_lb ≈ 2.3–2.6 (𝒯_hot ≈ 8–9), gold: IMC ≈ diffusion to ~2%, surface equilibrium to < 1%, energy delivered within 2–3% of the solver.
   - 𝒯_lb ≈ 0.5 (𝒯_hot ≈ 2–3), copper and aluminum: IMC ≈ diffusion to 4–5%, surface 5–7% out of equilibrium, energy 10–11% short with the manuscript's bath.
   - The materials' opacity (κ_R at 1 HeV: Au 2866, Cu 588, Al 272 cm²/g) and its steepness in T (α = 1.5, 2.21, 3.1) set it. Aluminum is the least opaque and the most sensitive.
2. **The pressure step of the patch, Δp/p_f ≈ m_f·|du/dt| / p_f:** computable from the solver alone.
   - ≲ 3% (Tests 1, 2, 3, 5): the patch costs ≲ 1–2% in the shock;
   - ≈ 6% (Test 6): ~3%;
   - ≈ 20% (Test 4): 6–13%.
   - It is large when the ablated layer is a sizeable fraction of a light, fast, decelerating shocked shell: ω < 0, low density near the surface.
3. **Mesh resolution of the ablated layer and of the shocked region:** a few % on fronts and shock until ≳ 50 cells span the ablated layer and the shock is resolved to ≲ 10⁻³ m_s.
4. **Second order everywhere here:** radiation energy (≤ 1%), radiation pressure (≤ 0.5%), O(v/c) (≤ 3% in gold; marginal in aluminum), Monte Carlo noise (≤ 0.1%).

### 5.2 The limits of the equilibrium (one-temperature) solver

The solver is the exact self-similar solution of one-temperature equilibrium diffusion with hydro. From the analysis above, it is a valid reference for the full (transport, two-temperature) problem when all of the following hold. Each comes with its measured threshold.

1. **The hot plasma is optically thick at the surface: 𝒯_lb ≳ 2.** Below that the surface is out of equilibrium. At 𝒯_lb ≈ 0.5 the radiation is 5–7% hotter than the matter, and the delivered energy is ill-defined by ~25–35% between temperature- and flux-matched drives. The manuscript's own check uses 𝒯 > 1 with the lower bound 𝒯_lb. **These runs suggest 𝒯_lb > 1 is necessary but not sufficient for ~1% agreement: gold, at 𝒯_lb ≈ 2.5, is in equilibrium within 1%, while copper and aluminum (0.5) are not.** Where the threshold lies between 0.5 and 2.5 is not resolved by these six tests.
2. **The drive is specified the way the solver specifies it.** The manuscript's bath law T_bath(t) is the solver's surface flux turned into a bath temperature through the equilibrium Marshak relation. It reproduces the solver's surface only in the equilibrium limit. Outside it, the solver is matched by the energy flux (`bath_drive="calibrated_flux"`), not by the bath law or by the surface temperature.
3. **Radiation energy ≪ matter energy in the hot plasma.** It is ≤ 1% in all six tests. This condition depends on density, not opacity: a hotter or more rarefied plasma could break it while still opaque.
4. **Transport is close to diffusion: 𝒯_hot ≳ 8 for ~2%.** This is not an equilibrium condition, since the solver also assumes diffusion. It is a separate limit, and it degrades together with 1 because both follow from the opacity.
5. **Dynamic diffusion is negligible: Σ τ·v/c ≪ 1.** The solver, like the codes, has no O(v/c) terms. In gold it is ≤ 0.03; in aluminum it reaches 0.25, where mixed-frame corrections start to matter at the percent level (plan A9).

Independent of equilibrium, the *patched* solver also needs:

6. **A small pressure step at the cut: Δp/p_f ≲ 5%.** This fails for ω = −0.5 (20%) and improves with time as m_f/m_s ∝ t^(−0.42).

### 5.3 What this means for the paper

- **Tests 1–3 (gold, ω ≥ 0):** the black curve is verified by both codes within ~1–2% at the paper's resolution. The equilibrium solver is appropriate.
- **Test 4 (ω = −0.5):** both codes agree with each other (≤ 2%) and are in equilibrium. The 6–13% gap to the black curve is the patch (Δp/p_f = 20%). Options:
  - state the validity condition and report Δp/p_f with the test;
  - use a reference with a numerical shock region driven at the moving ablation front;
  - improve the patching to match pressure and velocity at m_f, a theory question for Menahem.
- **Tests 5–6 (copper, aluminum):** with 𝒯_lb ≈ 0.5 the solver's equilibrium surface does not exist. Compared with the energy-flux-matched drive, diffusion lies within ~4% of it, and IMC within 4–5% of diffusion. Present these two tests as the edge of the equilibrium solution's validity, with 𝒯_lb next to each, rather than as verifications of it.
- **Report 𝒯_lb and Δp/p_f for every test.** Both come from the solver alone and predict, before any simulation, how well it will agree.

---

## 6. Data and reproduction

- **Comparison runs:** `uv run python -m rad_hydro_sim.verification.imc_vs_diffusion --cases test1_full … test6_full`:
  - `--bath paper|calibrated|calibrated_flux` (results in `results/imc_vs_diffusion[_<bath>]/`);
  - `--calibrate-bath --calibrate-target temperature|flux` (tables in `src/rad_hydro_sim/data/bath_correction/`).
- **At 1000 cells** for Tests 1–2: `--hydro-n 1000 --out results/imc_vs_diffusion_n1000`.
- **Test 3's fine mesh** (`test3_smooth`, `max_cell_mass`) is on the branch `imc/test3-fine-mesh`.
- **One-off checks** (scripts in the session scratchpad, numbers recorded here):
  - the pure-hydro piston run against `PistonShock` (§4.3, check 1);
  - `PistonShock.solve` at m_f for p(m_f) and u(m_f) (check 2);
  - T_rad/T_mat, the radiation-to-matter energy and 𝒯 from the stored diffusion profiles (§4.2);
  - the solver's 𝒯_lb and 𝒯_hot from `AblationSolver.solve` on a dense mass grid (§3).
- **Run tracker:** https://claude.ai/artifact/22cVx9Mpve9F8FHj65H2iw (private until shared).
