# Misalignment: why IMC and diffusion miss the semi-analytic solver, and what fixes it

**Start here.** This page is the whole investigation (3–20 Nov 2026) at summary level: the causes, how they were found, and where everything is. The topic docs are numbered by when their question first came up; each elaborates one part of this page without repeating the others, with its figures in place:

| doc | began → last added | elaborates |
|---|---|---|
| [01_surface_equilibrium.md](01_surface_equilibrium.md) | 3 Nov → 20 Nov | A: matter and radiation out of equilibrium at the surface; the baths; the criterion t/t_eq; Be and Al, seeded early and late |
| [02_patching.md](02_patching.md) | 3 → 20 Nov | B: the solver's rigid-front patch; Test 4; the Al residual; fixes B and C |
| [03_startup.md](03_startup.md) | 7 → 18 Nov | C–D: the early drive error, frozen into the entropy; the surface mesh; χ; ω |
| [04_analytic_start.md](04_analytic_start.md) | 7 → 15 Nov | the seeded start for gold: method, fix A, the improved start, what remains |
| [05_shocked_region.md](05_shocked_region.md) | 10 → 18 Nov | E–H: hydro-only and rad-hydro-vs-hydro checks; 300 K; growing cells; the pressure diagnostic; the flicker |
| [06_runs.md](06_runs.md) | reference | every run, mesh, cost and command |

**The answer in four lines.**
1. The codes are right. IMC and diffusion agree with each other, and the hydro reproduces the solver's own shock to 0.1% when given the same problem.
2. The solver is a **one-temperature** (radiative-equilibrium: T_rad = T_m), self-similar, **patched** solution. The simulations are two-temperature. Each discrepancy breaks one of the solver's assumptions: radiative equilibrium at the surface (Cu, Al, Be, and every material at early times), or the rigid ablation front of the patch (Test 4, part of Al).
3. An early-time error is not forgotten. The shocked region is adiabatic, so each element keeps its shock entropy. A ~15% drive deficit in the first ~0.1 ns becomes a permanent ~17% entropy (density) error in the inner ~20% of the shocked mass.
4. Everything else was numerics (surface mesh at t → 0, 300 K target, growing cells, seeding artifacts). With these removed, Test 3 sits on the solver within 1–2% and Al within ~2% (the patch).

---

## 1. The two models

| | solver (`AblationSolver`, the black curve) | simulations (`rad_hydro_sim`) |
|---|---|---|
| ablated region 0 < m < m_f | subsonic heat wave; one temperature; radiation only as a diffusive heat flux; T = T_s at the surface | matter T_m and radiation E_r, coupled by absorption–emission (both in LTE: Planckian emission at T_m); gray diffusion or IMC; Marshak bath |
| front m_f | rigid wall at rest (ũ = ṽ = 0) | moves with the matter |
| shocked region | `PistonShock` with the piston at m = 0, driven by p_f(t), cold target, **cut at m_f** | continuous Lagrangian hydro; target at 300 K (30 K from 14 Nov) |
| start | self-similar from t = 0 (p_f ∝ t^−0.448 diverges) | cold target at t = 0, or the solver's profile seeded at t₀ |

Code: `src/menahem_new/ablation_solver_og.py` (`AblationSolver.solve`, walked through in [02_patching.md § 1](02_patching.md)); `src/rad_hydro_sim/simulation/{iterator,radiation_step}.py`.

## 2. How the understanding developed

1. **3–6 Nov. Tests 1–3 agree to 1–2%, Tests 4–6 miss by 5–15%.** Cu and Al have T_rad/T_m = 1.05–1.07 at the surface, so no bath reproduces both the solver's temperature and its flux. A flux-calibrated bath gets within ~4% → **A**. Test 4's shock is 12.5% behind with transport and drive excluded; the size follows m_f/m_s → **B**. Patch fixes B and C were tried and dropped.
2. **7 Nov. Test 3 has the right pressure but the wrong density below 30 mg/cm², differently in IMC and diffusion.** The density records the entropy, so the error is old: the early shock was overdriven while the layer was thinner than a cell (**D**). The codes differ through diffusion's face rule. A separate uniform −6% is the 300 K target (**E**).
3. **10 Nov. Seed the solver's profile at 0.1 ns ("fix A").** It removes the band in both codes, which confirms that the error is the start.
4. **10–14 Nov. Is the shocked region really pure hydro?** Yes: hydro alone with the run's front pressure reproduces it. The tables' "ablation pressure −4.5%" was a diagnostic artifact (**H**).
5. **14–15 Nov. Can a cold start sit on the solver?** On a resolved surface mesh the excess turns into a ~17% *deficit*: matter and radiation are out of equilibrium for ~0.1 ns (**C**). Multiplying the coupling by χ ≥ 100, or the 1T mode, cures it. IMC has no χ, so for IMC the seeded start stays. The smaller causes **F** and **G** were found along the way.
6. **15 Nov. Would a flatter target equilibrate faster?** No. The heat wave in mass coordinates does not see ρ₀(m), so ω only changes how much mass records the early phase.
7. **17–18 Nov. Why does opaque gold fail to equilibrate, and why does it still matter at 2 ns?** The ablated plasma is optically thin early (𝒯 ∝ t^0.41) and the matter, with 10³× the radiation's heat capacity, lags. The shocked region has no radiative conduction (rad-hydro = hydro to 4×10⁻⁹), so the drive error is written into the entropy for ever. The criterion: t/t_eq on the solver's own profile. Be and Al seeded at 0.1 ns (t/t_eq < 1) relax back out of equilibrium.
8. **20 Nov. Al seeded where t/t_eq = 7.5 (2 ns), run to 50 ns.** The non-equilibrium part disappears with Al's own coupling. The ~2% left at any χ is the patch.

## 3. The causes

| | cause | where it shows | cure / status | doc |
|---|---|---|---|---|
| **A** | surface out of radiative equilibrium (thin plasma, slow coupling) | Cu, Al: energy −10%, shocks −5…−8%; Be; every material for t ≲ t_* | flux-calibrated bath; seed where t/t_eq ≳ 7; Be: nothing reaches the solver's state | [surface_equilibrium](01_surface_equilibrium.md) |
| **B** | rigid-front patch: the shock solution is cut m_f/m_s into its own slab | Test 4: shock −12.5%; Al: −2% shock, −5% entropy | none by drive, start or χ; state m_f/m_s ≪ 1; compare late | [patching](02_patching.md) |
| **C** | early non-equilibrium (t/t_eq ≲ 1), frozen into the entropy | Test 3 cold start: entropy 0.83 (IMC 0.79) at 12–20 mg/cm² | χ ≥ 100 or 1T (diffusion); seeded start (IMC) | [startup § 3–4](03_startup.md) |
| **D** | unresolved surface at t → 0; diffusion's face mean of D | old mesh: entropy ×1.3–8 below 20 mg/cm² | graded `log_mass` mesh | [startup § 2](03_startup.md) |
| **E** | 300 K target (Mach 11.5) | ρ −6%, entropy +8%, u −0.7% | 30 K | [shocked_region § 4](05_shocked_region.md) |
| **F** | growing cells in the shocked region | entropy +0.9(g − 1) | 0.3% growth | [shocked_region § 5](05_shocked_region.md) |
| **G** | seeding artifacts (sharp jump; 2T not at rest in a 1T profile; patched step) | seam ×1.5; pulse marks ±1–7%; Al seam 0.83 | smeared seed; partly remain | [analytic_start § 3–4](04_analytic_start.md), [patching § 4](02_patching.md) |
| **H** | `p_abl` read on the hot side of the front | "ablation pressure −4.5%" | read the dense side: within 0.6% | [shocked_region § 3](05_shocked_region.md) |

## 4. Three ideas that carry the explanation

- **Validity of the one-temperature solution.** It is self-consistent only where t/t_eq ≫ 1 *on its own profile*, with t_eq = ρc_v/(4aT³cσ_abs) the matter's relaxation time to T_rad. That gives t_* ≈ 10 ps for Au, ≈ 0.2 ns for Al, and ~6 ns for Be. Before t_* the plasma is optically thin and the matter lags the radiation.
- **Entropy memory.** Beyond the front the matter is cold, opaque and adiabatic. It keeps its shock entropy, so ρ = (p/s)^(1/γ) carries any early drive error for ever. With m_s ∝ t^0.52 the early matter is a large share of the mass, and the front always sits in it.
- **Depth of the cut.** The patch keeps a shock solution whose pressure at m_f is that a fraction m_f/m_s into a shocked slab: p(m_f)/p_f = 1.00, 1.03, 1.06 and 1.20 for Tests 3, 1, 6 and 4.

## 5. What to run for a fair comparison

| material | diffusion | IMC | what remains |
|---|---|---|---|
| Au, ω ≥ 0 | cold, 30 K, graded fine mesh, 2T χ = 100 (`cold_2T_30K_chi100_fine`) or 1T | seeded at 0.1 ns, 30 K, smeared (`seeded_IMC_30K_smeared_seed*`) | IMC u −1%, shock −0.6% (unexplained) |
| Au, ω = −0.5 | any | any | patch ~12%: quote m_f/m_s |
| Al | seeded at 2 ns, to ≥ 20 ns (`al_long`) | same | patch ~2%; seam mark |
| Cu | flux-calibrated bath | same | 2–4% (not run late) |
| Be | none reaches the one-temperature solution | | |

## 6. Code map

Paths relative to `src/rad_hydro_sim/` unless they start with `src/`.

| option / function | file | cause |
|---|---|---|
| `bath_drive="paper"/"calibrated"/"calibrated_flux"`, `bath_correction_for`; tables `data/bath_correction/` | `simulation/iterator.py` | A |
| `initial_condition="analytic_ablation"`, `t_sec_start` (`_seed_ablation_profile`) | `simulation/iterator.py` | C, D |
| `seed_shock_on_face`, `seed_shock_smeared` | `simulation/iterator.py`, `src/hydro_sim/problems/simulation_config.py` | G |
| `mesh="log_mass"`, `log_mass_first`, `log_mass_segments` | same | D, F |
| `force_black="conduction"` (1T) | `simulation/radiation_step.py` | C |
| `RadHydroCase.chi` | `problems/RadHydroCase.py` | C |
| `HARMONIC_MEAN` | `simulation/radiation_step.py` | D |
| `bc_type="reflective"` | `simulation/radiation_step.py` | (rad-hydro twin) |
| `T_initial_Kelvin` | `problems/RadHydroCase.py`, presets | E |
| `fronts_of_profile` (`p_abl`) | `verification/imc_vs_diffusion.py` | H |
| studies `test3_hydro_only`, `test3_on_the_solver`, `test3_omega_scan`, `test3_transparency`, `light_analytic_start`, `al_long` (+ `*_figures`) | `src/test_misalignment/` (`python -m test_misalignment.<study>`) | [06_runs.md](06_runs.md) | reference |
| Phase C under three baths; patch fixes B, C | `verification/bath_calibration.py`, `verification/legacy/` | A, B |

## 7. For the paper

- Give the validity conditions next to each test: t ≫ t_* (radiative equilibrium on the solution's own profile) and m_f/m_s ≪ 1. A draft paragraph is in [03_startup.md § 6](03_startup.md).
- Report Tests 5–6 as the limit of the one-temperature solution, and Test 4 as the limit of the patch.
- Compare IMC with the solver where both hold: Au seeded or at χ = 100, Al seeded at 2 ns at late times.

## 8. Open

- IMC's velocity −1% and shock −0.6%, with equal starts, unchanged by 4× packets.
- The pressure-to-entropy lag is described, not derived.
- 1T energy conservation is not measured; the 1T run was not repeated on the fine mesh.
- Al: the patch residual and the seam were not tested directly.
- The heat front runs 6–16% ahead of the rigid solution for ω < 0 (suspect: ṽ(ξ_f) = 0).
