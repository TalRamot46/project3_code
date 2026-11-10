# Test 3: the startup imprint, and starting from the analytic solution

Branch `feat/patching-poc`, 2026-11-07. Test 3 is gold with ω = +0.5 under the paper bath, on its fine mesh (`test3_smooth`, 1237 cells).

**In one paragraph.** In `paper_bath/test3_full/full_semilogx.png`, diffusion and IMC miss the solver's density between the heat front and ~30 mg/cm², and they miss it by different amounts, while their pressure, velocity and temperature lie on it. The cause is the start. Until ~0.6 ps the analytic ablated layer is thinner than one surface cell, so the first cell is heated as a block, and the early shock is overdriven. The matter that shock crosses keeps the extra entropy for the rest of the run, and that shows as low density at the right pressure. How strongly each code overdrives depends on how its radiation solver heats those unresolved cells, and that is why the two codes differ. Starting both codes from the solver's profile at 0.1 ns (`initial_condition="analytic_ablation"`) removes the imprint: after 20 ps the shocked matter's entropy is within 0.3% of the solver's in both codes, against 1.3–4× from the cold start. A second, smaller offset is unrelated to the start: the 300 K initial temperature lowers the shock compression by 5.7%.

---

## 1. The meshes

Diffusion and IMC (3 seeds; 2000 emission and 2000 bath packets per step, census capped at 20 000) run on the same 1237 cells:

| mass range | cells | cell mass |
|---|---|---|
| surface to ~3 mg/cm² | 139 | 0.0216 mg/cm² (~67 across the ablated layer at 2 ns) |
| ~3 to ~10 mg/cm² | 67 | growing 5% per cell |
| ~10 to 319 mg/cm² | ~1030 | 0.3 mg/cm² |

The default mesh (`test3_full`, 347 cells) has the same surface cells; the fine mesh refines only the shocked region beyond 3 mg/cm². That is why diffusion's density below 10 mg/cm² is the same on both meshes (0.345 vs. 0.343 of the solver's at 1.5 mg/cm², 1 ns).

## 2. What the density records

In the shocked layer the pressure is nearly uniform and follows the drive at each moment: it has no memory. The temperature there is ~0.1 eV, invisible on a HeV axis. The density at a given pressure is fixed by the entropy, ρ = (p/A)^(1/γ) with A = p/ρ^γ and γ = 1.25. Each element gets its A when the shock crosses it and keeps it afterwards (the shocked flow is adiabatic). The density profile is therefore a record of the shock's history, and the log-mass axis is a log-time axis for that history:

| m [mg/cm²] | 1.5 | 3 | 5 | 12 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| shocked at [ps] | 0.08 | 0.29 | 0.78 | 4.2 | 11 | 67 | 255 |
| ablated layer then, in surface cells | 0.36 | 0.71 | 1.2 | 2.8 | 4.7 | 12 | 23 |
| A / A_solver, diffusion | 3.78 | 2.46 | 1.93 | 1.29 | 1.14 | 1.07 | 1.07 |
| A / A_solver, IMC | 2.28 | 1.58 | 1.34 | 1.01 | 0.96 | 0.98 | 1.02 |

- **The record is frozen.** Diffusion's ratio at 3 mg/cm² is 2.459 at 10 ps and 2.461 at 0.5, 1, 1.5 and 2 ns; IMC's is 1.565 at 10 ps and 1.566–1.590 later. No heating reaches that matter after the first ~10 ps. The three IMC seeds agree to 0.1%, so the IMC–diffusion gap is not noise.
- **It is the whole density gap.** ρ/ρ_solver = (A/A_solver)^(−0.8): 2.46^(−0.8) = 0.49, diffusion's density ratio at 3 mg/cm².
- **The start overdrives the shock.** The solution is self-similar from t = 0: the ablation pressure goes as t^(−0.448) and the paper bath as T_bath = (1 + 0.169 t_ns^(−0.411))^(1/4) T_s(t), both diverging as t → 0. At 0.03 ps the simulated heat front is at 0.028 mg/cm² (the solver's: 0.0044) and the ablation pressure is 1.69× the solver's in diffusion, 1.16× in IMC. Between ~1 and ~100 ps both run below it (diffusion 0.89–0.97, IMC 0.81–0.91), and they settle at 0.955 and 0.94.
- **Why Test 3 shows it.** The shock sweeps ~200× the ablated mass (m_f/m_s = 0.5%; ~13× in Tests 1–2), so each unresolved early moment marks 200× as much shocked matter. The log-mass axis then gives that matter (~7% of the shocked mass, shocked in the first 0.5% of the run) half of the panel. The mass-weighted profile error is 3.4%.

## 3. Why IMC and diffusion differ

The first 10 ps, run again on the default mesh (same surface cells). A / A_solver at 10 ps:

| m [mg/cm²] | 1.5 | 3 | 5 | 8 | 12 |
|---|---|---|---|---|---|
| diffusion, arithmetic face mean of D (the code's) | 3.78 | 2.46 | 1.93 | 1.55 | 1.33 |
| diffusion, harmonic face mean | 3.63 | 2.11 | 1.43 | 0.95 | 0.66 |
| IMC | 2.27 | 1.57 | 1.32 | 1.16 | 1.05 |
| diffusion, 33× lighter surface cells (N = 100: 1397 cells, 0.00066 mg/cm²) | 1.26 | 1.08 | 0.99 | 0.93 | 0.91 |

- The 10 ps values equal the 0.5–2 ns ones of § 2: the imprint is complete by 10 ps.
- **The face rule sets diffusion's imprint.** Changing only the face average of D (`radiation_step.HARMONIC_MEAN`) turns the overdrive into an underdrive. From 5 mg/cm² on, the two rules bracket IMC. The gap between the codes is the discretization of an unresolved front in diffusion, against transport, which cannot carry more than the physical flux into the cold cell. It is not a preheat of the shocked matter ahead of the front, as `phase_c_tests3to6_results.md` § 5 suggested: nothing heats that matter after 10 ps.
- **Resolving the surface removes most of it,** which confirms the cause. It is not affordable: those 10 ps took 2.5 h against 45 s, and since m_s/m_f is constant, every refinement leaves a band, only a narrower one.
- Setting the flag needs `importlib.import_module("rad_hydro_sim.simulation.radiation_step")`: `import rad_hydro_sim.simulation.radiation_step as rs` returns the function that the package `__init__` re-exports under the same name.

## 4. A second offset: the 300 K target

Past ~50 mg/cm² diffusion's density is 5–6% low and uniform (A × 1.071–1.077). The solver's shock assumes a cold target. The simulated gold starts at 300 K, and at these densities (2700–25 000 g/cm³) its pressure is 0.68–0.69% of the shock pressure at every mass. At γ = 1.25 that lowers the compression from (γ+1)/(γ−1) = 9 to 8.49, which predicts A × 1.076–1.077. IMC sits at 0.98–1.06 there because its drive ran a few % weaker. This offset is not a startup effect: a run started from the solution still has it for the matter it shocks itself. Lowering `T_initial_Kelvin` (to ~10 K, which cuts that pressure ~230×) would remove it. Not done here.

## 5. The fix: start from the solution at t₀ = 0.1 ns

`initial_condition="analytic_ablation"` with `t_sec_start = t₀` (`iterator._seed_ablation_profile`) starts the run from the `AblationSolver` profile instead of a cold target at t = 0:

- **The mesh keeps its cells' masses.** The nodes behind the shock move to the solver's positions and velocities, so each cell's volume, and its density, is the solver's; the surface node goes to the solver's boundary. Ahead of the shock the cold target is untouched.
- **Energy.** The specific energy is the solver's mass average over each cell (8 points). Node velocities are mass averages over each node's half-cells, which keeps the momentum. The surface node carries the whole first cell in the momentum update but stands for half of it, so it gets the velocity that gives that half's kinetic energy. Point values at the nodes would have added 47% to the energy, all of it in the blow-off. The seeded energy is 1.486×10¹⁰ erg/cm², against 1.484×10¹⁰ in a fine quadrature of the solver's profile. The solver's closed form E(t₀) is 2.6% above its own profile.
- **Why 0.1 ns.** The ablated layer then spans 14 surface cells (0.311 mg/cm²). In the cold start, the matter shocked once the layer had ≥ 12 cells (m ≥ 50 mg/cm²) carries the solver's entropy up to § 4's offset, so from 0.1 ns on the codes can follow the solution. The shock is at 61.6 mg/cm²: everything inside starts with the solver's entropy.
- **Cases.** `test3_smooth_analytic` (fine mesh) and `test3_analytic` (default mesh) in `imc_vs_diffusion.HYDRO_CASES`; tests in `tests/test_analytic_ablation_start.py`.

**The first 20 ps** (default mesh, 0.10 → 0.12 ns):

| | diffusion | IMC |
|---|---|---|
| A / A_solver at 0.12 ns, 1–55 mg/cm² | 1.000–1.003 | 1.000–1.003 |
| heat front / shock / ablation pressure vs. solver at 0.12 ns | +0.8% / −1.3% / −5.5% | +0.9% / −1.3% / −7.6% |
| energy error | 3×10⁻¹⁵ | 5×10⁻¹¹ |

The densities still ring by a few % at 0.12 ns (up to +16% near 40 mg/cm²), following the pressure: the acoustic adjustment from the seed. The entropy does not ring.

## 6. Results at 1–2 ns

**Default mesh, diffusion** (`test3_analytic`, 347 cells, 1.41 million steps, 33 min, energy error 7×10⁻¹⁴). A / A_solver against the cold start on the same mesh:

| m [mg/cm²] | 2 | 3 | 5 | 8 | 12 | 20 | 30 | 50 | 70 | 100 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cold start, 1–2 ns | 3.14 | 2.46 | 1.93 | 1.55 | 1.34 | 1.19 | 1.09 | 1.07 | 1.07 | 1.12 | 1.12 |
| analytic start, 1–2 ns | 1.001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.002 | 1.002 | 1.09 | 1.13 | 1.13 |

- Inside the seeded region (< 61.6 mg/cm²) the entropy stays the solver's for the whole run, to 0.2%. Beyond it, the matter the run shocks itself gets the same excess as from the cold start: § 4's 7.6% plus this mesh's coarse shocked cells.
- The fronts do not move: heat front −0.8% (cold start −1.0%), shock −2.1% (−2.1%), ablation pressure −4.8% (−4.5%) at 2 ns. The start only changed what the matter remembers.
- On this mesh the density still wanders by up to ±8% in the seeded region. That is the pressure: the shocked layer rings by ±10% on its 1–6 mg/cm² cells, in the cold start as well, while on the fine mesh it stays within 0.5% of the solver's. At equal entropy, ρ/ρ_solver = (p/p_solver)^0.8.

**Fine mesh, diffusion** (`test3_smooth_analytic`, 1237 cells, 1.41 million steps, 56 min, energy error 2×10⁻¹⁴). ρ / ρ_solver, the same at 1, 1.5 and 2 ns to within ±1%:

| m [mg/cm²] | 2 | 3 | 5 | 8 | 12 | 20 | 30 | 50 | 70 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold start | 0.40 | 0.49 | 0.59 | 0.70 | 0.82 | 0.90 | 0.94 | 0.95 | 0.94 | 0.94 | 0.94 | 0.94 |
| analytic start | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.93 | 0.94 | 0.94 |

- **The startup band is gone.** From the heat front to the seeded shock position (61.6 mg/cm²) the density is the solver's within 1%, and the entropy within 0.2%. The cold start had 0.40–0.95 there.
- **Past 62 mg/cm² the 300 K offset remains:** ρ 6–7% low, A × 1.077–1.088, which is § 4's 1.076 plus a little near 70–100 mg/cm², the matter shocked just after the seeding. A colder target would remove it (§ 4).
- **The fronts at 2 ns are unchanged:** heat front −0.8% (cold start −1.0%), shock +0.1% (0.0%), ablation pressure −4.8% (−4.5%).
- At 2 ns the cell at 1.5 mg/cm² is next to the heat front (1.46 mg/cm²) and already heated (A × 1.31). In the cold start it was A × 4.69.
- **One cell at the seam.** The cell that held the solver's shock at t₀ (61.75 mg/cm²; the shock was at 61.61) starts half shocked, half cold. The shock re-forms from inside it, and it keeps A × 1.58, ρ −31%; its neighbours are within 1% on the near side and at the 300 K offset on the far side. It is the narrow dip at ~62 mg/cm² in the density panels. Moving the seeded shock to a cell face would remove it.

Figures: `results/phase_c/figures/paper_bath/test3_analytic_start/<region>_<scale>.png` (the twelve views of `test3_full/`) and `paper_bath/test3_analytic_start_profiles.png`; their titles say "analytic start at 0.1 ns". They are drawn from `bath_calibration --cases test3_smooth_analytic --baths paper --views` (and `--profiles`).

**Fine mesh, IMC** (seeds 1–3, 16.7 h each, energy error 1.2×10⁻⁹; added 2026-11-10). ρ / ρ_solver at 2 ns, mean of the seeds:

| m [mg/cm²] | 2 | 3 | 5 | 8 | 12 | 20 | 30 | 50 | 70 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold start | 0.59 | 0.68 | 0.80 | 0.86 | 0.98 | 1.01 | 1.04 | 1.05 | 1.00 | 0.97 | 0.97 | 0.96 |
| analytic start | 1.01 | 0.99 | 0.97 | 0.99 | 1.00 | 1.01 | 1.01 | 0.99 | 0.93 | 0.93 | 0.96 | 0.95 |

- IMC's startup band is gone as well. In the seeded region its entropy is the solver's within 0.1% beyond 12 mg/cm² and 1–2% above it at 2–8 mg/cm² (1.007 → 1.012 at 2 mg/cm² between 1 and 2 ns); the seeds agree to 0.5%. The density scatters by ±3% around the solver's with the pressure.
- Past 62 mg/cm² IMC has the same 300 K offset as diffusion (A × 1.05–1.09).
- The fronts at 2 ns: heat front −1.1%, shock −0.4%, ablation pressure −4.5%.

The figures now show diffusion and IMC (mean of the 3 seeds). Summary of the fix: [test3_fix_A.md](test3_fix_A.md).

## 7. Reproduce

```
uv run python -m rad_hydro_sim.verification.bath_calibration --run-one test3_smooth_analytic paper 0   # diffusion
uv run python -m rad_hydro_sim.verification.bath_calibration --run-one test3_smooth_analytic paper 1   # IMC seed 1 (also runs diffusion if missing)
uv run python -m rad_hydro_sim.verification.bath_calibration --cases test3_smooth_analytic --baths paper --views
uv run pytest tests/test_analytic_ablation_start.py
```
