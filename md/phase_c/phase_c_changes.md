# Phase C: every change for IMC in Lagrangian rad-hydro

This is the complete list of what Phase C of the IMC plan (`md/plans/full_plan_imc_comparison.md`, milestone M5 plus its Addendum sub-step M5b and the A9 diagnostic) changed in `src/imc`, `src/rad_hydro_sim` and `src/hydro_sim`, why, and how each change was checked. The comparisons are in `md/phase_c/phase_c_results.md`; the IMC adapter as a whole is described in `md/imc_code/imc_coupling.md`.

**The rule from Phase B still holds:** every change is opt-in. With the default `SimulationConfig` the code is bit-identical to the original. The Phase B golden-file test (`tests/test_phase_b_diffusion.py::test_defaults_are_bit_identical_to_the_original_code`, produced by the code *before* Phase B) still passes unchanged, and the radiation-only IMC runs are unchanged because the census is only moved when the mesh moved.

---

## 1. Summary

| # | Where | What | Switch (default = original) | Plan item |
|---|---|---|---|---|
| 1 | `imc/core/mesh.py` → `Mesh.carry` | moves packets with their cells: same cell, same relative position ξ, same weight | — (new method) | M5 |
| 2 | `imc/coupling/rad_hydro.py` → `IMCSolver` | full rad-hydro accepted; census carried across the hydro step; per-step energy ledger | `radiation_solver="imc"` with `scenario="full_rad_hydro"` | M5 |
| 3 | `integrator.step_rad_hydro` | guards lifted: IMC and `subtract_radiation_energy` in full rad-hydro | — | M5 |
| 4 | `integrator.step_rad_hydro` | compatible branch inverts the EOS with the case's own material when IMC runs | — (bug fix on a new path) | M5 |
| 5 | `iterator.simulate_rad_hydro` | no IMC solver for hydro-only runs | — | M5 gate |
| 6 | `presets_config.py` | `PRESET_PAPER_TEST1`, `PRESET_PAPER_TEST2` (full rad-hydro) | new keys only | M5 |
| 7 | `SimulationConfig`, `RadHydroState`, `hydro_steps`, `integrator` | radiation pressure and its p dV work | `rad_coupling="matched"` / `"physical"` | M5b (A7) |
| 8 | `iterator` → `relativistic_diagnostic` | max v/c and Σ τ·v/c per step in the energy budget | `energy_diagnostics=True` | A9 |
| 9 | `verification/imc_vs_diffusion.py` | cases `test1_full`, `test2_full` | new cases only | M5 gate |

Nothing else changed. In particular `get_e_star_from_hydro`, `update_nodes_from_pressure`, `radiation_step` and the IMC core's transport, sources and stepper are untouched.

---

## 2. Why: what "the same problem" means once the mesh moves

### 2.1 The operator split, and where the radiation lives during the hydro step

`integrator.step_rad_hydro` (full rad-hydro) does, per step:

1. **hydro predictor** (`get_e_star_from_hydro`): half-kick the nodes, move them, new volumes and densities, predict e*;
2. **radiation** (`radiation_step` or `IMCSolver.step`) on the moved mesh, from e*;
3. **close** (`update_nodes_compatible` with `hydro_energy="compatible"`): the end-of-step pressure, the second half-kick, and the energy update that makes the cells' work equal the nodes' kinetic-energy change.

Between two radiation steps, the radiation has to ride through step 1. The manuscript (§"Gray diffusion setup", Eq. `gray_diff`) evolves ∂(vE_r)/∂t: without coupling and flux, the radiation energy *per unit mass* is unchanged by the motion. In the diffusion code that is `rad_energy_per_mass=True` (E_rad ← E_rad V_old/V_new, Phase B switch 4). The IMC counterpart is exact and has no free choice in it:

$$
x_{\rm new} = x_j^{\rm new} + \xi\,\Delta x_j^{\rm new},\qquad \xi = \frac{x - x_j^{\rm old}}{\Delta x_j^{\rm old}},\qquad w_{\rm new} = w .
$$

A packet stays in its Lagrangian cell at the same relative position, and its energy weight doesn't change, so each cell's radiation energy E_rad·V is the same before and after the motion. That is `Mesh.carry`. The census therefore needs no new storage (the plan suggested storing (j, ξ)): `x` and `cell` plus the mesh of the previous step determine ξ. `IMCSolver` remembers that mesh and carries the census only when the edges actually changed. Radiation-only runs are therefore unaffected bit for bit.

**Consequence: IMC requires `rad_energy_per_mass=True` in full rad-hydro.** The original diffusion behavior (E_rad density unchanged, so E_rad·V ∝ V) has no natural transport counterpart, and comparing against it would mix a convention difference into the transport-vs-diffusion comparison. `IMCSolver` raises a `ValueError` naming the flag.

### 2.2 The energy ledgers with hydro

The standalone IMC keeps a run-long ledger: matter + census = initial + bath − leakage. With hydro, the matter's energy also changes by p dV work between radiation steps, so that ledger no longer closes, and it shouldn't. `IMCSolver` now opens a fresh ledger every step: before = Σ u_m(T*)Δx + census (after the carry); after = Σ u_m(Tⁿ⁺¹)Δx + census; the difference must be bath − leakage. That checks the radiation step exactly. The rad-hydro budget of Phase B (`energy_diagnostics`) checks the whole step including the hydro, and the two together check everything. In radiation-only runs the per-step ledger is equivalent to the old one: the run-long ledger is the sum of the per-step ledgers, and the T ↔ e round trip between steps is covered by the rad-hydro budget.

### 2.3 The pressure when aT⁴ is subtracted (a decision recorded for you)

Phase B refused `subtract_radiation_energy` outside radiation-only, because it is not obvious what the hydro pressure should be when the stored material energy is e_m = e − aT⁴/ρ (manuscript Eq. `um_def`). The two candidates:

- **(a) p = rρe_m:** the ideal-gas EOS applied to the stored material energy. This is what the code does with no change, because `get_e_star_from_hydro` and the compatible close compute p from `e_material`.
- **(b) p = rρe(T, ρ) = rρ(e_m + aT⁴/ρ):** the EOS of the manuscript's §2 (Eq. `ideal_gas_eos`), applied to Rosen's total energy at the material temperature.

**Decision: (a), and the guard is lifted.** The difference between the two is r·aT⁴, about 3×10⁻⁵ of p in dense gold at 1 HeV and about 8×10⁻³ in the ablated plasma at ρ ≈ 0.03 g/cm³. That is below the comparison's other uncertainties for Tests 1–2, but not negligible in principle. (a) keeps the compatible scheme's exact energy conservation without a new EOS path. (b) would need the pressure to be a function of T rather than e in the predictor and the close. If the manuscript's convention is to be followed literally in light materials (aluminum, where aT⁴/u is larger), (b) is the one to implement. It touches `pressure_ideal_gas` calls in two functions. Both solvers share the choice, so it cannot bias the IMC-vs-diffusion comparison.

### 2.4 Radiation pressure (M5b, plan item A7)

The paper's equations have no radiation pressure; neither does the diffusion code, nor the matched coupling above. The advisor's point A7 asks for the physical coupling as a separate mode, to measure what the paper's assumption costs. `rad_coupling="physical"`:

1. **P_rad = E_rad/3 in the momentum update.** It enters `update_nodes_compatible` as an extra pressure, `P_rad_new`, at tⁿ⁺¹; tⁿ's is the previous step's, stored on the state as `RadHydroState.P_rad`. Only the momentum sees it: the matter's work is still computed from p + q.
2. **Its work is charged to the radiation.** With M_i(uⁿ⁺¹ − uⁿ) = Δt F̄_i and F̄ now including P̄_rad, summation by parts gives the kinetic-energy change as the work of p + q + P_rad. So the radiation of cell j must lose exactly

$$
W_j = -\Delta t\,\bar P_{{\rm rad},j}\,(\bar u_{j+1}-\bar u_j),\qquad \bar P_{\rm rad} = \tfrac12(P_{\rm rad}^n + P_{\rm rad}^{n+1}),\ \bar u = \tfrac12(u^n+u^{n+1}),
$$

   which is `hydro_steps.radiation_work`. It is the discrete form of d(E_r V) = −P_rad dV, i.e. of E_r V^{4/3} conserved for a γ = 4/3 photon gas, the plan's "(V_old/V_new)^{1/3}" scaling, written so that total energy is conserved to round-off rather than to O(Δt²).
3. **Diffusion:** E_rad ← E_rad + W/V after the close.
4. **IMC:** `IMCSolver.add_radiation_work` scales the census of cell j by 1 + W_j/C_j (C_j = census energy of the cell). Where that factor would differ from 1 by more than 0.5, or the cell has no census, the work goes to the matter of that cell instead. This happens in thick cells, where packets live ≪ Δt and the census is a few late-emitted packets. There radiation and matter are in equilibrium within the step anyway, so it is the physically neutral choice, and total energy is conserved either way. The fraction rerouted is reported (`meta["imc"]["radiation_work_to_matter"]`); on Tests 1–2 it is 0.1–0.2% of the total work.

Details:
- **Which E_rad for P_rad.** Diffusion uses its cell value. IMC uses the track-length (step-averaged) E_rad, as A7 suggests: the census snapshot is far too noisy in thick cells (`md/imc_code/imc_coupling.md` §4). Conservation doesn't depend on that choice, because the same P̄_rad enters the momentum and W.
- **Boundaries.** The radiation pressure is part of the pressure at the boundaries too: the vacuum ablation surface keeps p_left = 0 for the total, and the right outflow face takes the total of the last cell. The boundary work recovered by `boundary_work` now uses the pressure the momentum update actually used (`_momentum_pressure`), so the budget stays exact. A more physical left boundary would apply the bath's momentum flux; it is O(aT_bath⁴/6) and left for M11 with the rest of the momentum coupling (Wollaber Eq. 149).
- **Requirements.** `rad_coupling="physical"` needs `hydro_energy="compatible"` (the work identity above holds only for the compatible close) and `rad_energy_per_mass=True` (physical = matched + P_rad + its work). Otherwise it raises.

### 2.5 The relativistic diagnostic (plan item A9)

`relativistic_diagnostic(state, case)` returns max|u|/c and Σ_j τ_j|u_j|/c over the heated cells (T > 0.1 T_max), with τ_j = ρκ_RΔx_j. Mixed-frame O(v/c) corrections matter only in the dynamic-diffusion regime τv/c ≳ 1. With `energy_diagnostics=True` both numbers are recorded every step in `meta["energy_budget"]` (`max_v_over_c`, `tau_v_over_c`). On Tests 1–2: v/c ≤ 4×10⁻⁴, Στv/c ≤ 3×10⁻². This confirms the plan's estimate, so M11's mixed-frame terms are not needed for the paper.

---

## 3. The changes in detail

### 3.1 `src/imc/core/mesh.py`

`Mesh.carry(x, cell, new)`: ξ = (x − x_j)/Δx_j clipped to [0, 1], then x_j^new + ξΔx_j^new. It raises if the meshes have different cell counts. Test: `test_carry_moves_packets_with_their_cells`, including packets sitting exactly on faces.

### 3.2 `src/imc/coupling/rad_hydro.py`

- `IMCSolver.__init__`: accepts `scenario="full_rad_hydro"`; refuses it without `rad_energy_per_mass` (§2.1); remembers the mesh.
- `IMCSolver.step`: carries the census if the mesh moved; opens a per-step ledger (§2.2).
- `IMCSolver.add_radiation_work(work, max_fraction=0.5)`: §2.4; it returns the new E_rad and the work charged to the matter per cell.
- `summary()`: adds `radiation_work_total` and `radiation_work_to_matter`.

### 3.3 `integrator.py`, `iterator.py`, `hydro_steps.py`, `state.py`, `simulation_config.py`

- `step_rad_hydro`:
  - The Phase B guards on IMC and on `subtract_radiation_energy` in full rad-hydro are removed; subtracting without radiation (hydro-only) still raises.
  - New `rad_coupling` validation, and the physical branch of §2.4.
  - **Bug fix on the new path.** The compatible branch recomputed T with `radiation_step.material_temperature`, which reads radiation_step's *module-level* material parameters (`f_Kelvin`, `beta_Rosen`, `mu`). Only the diffusion solve sets them. With IMC they were undefined, or worse, left over from another case run earlier in the same process. With IMC the branch now uses the IMC's `PowerLawMaterial`, which is bound to the case. With diffusion nothing changes. The compatible branch also passes `subtract_radiation_energy` instead of a hard-coded `False`, which only matters now that subtracting is allowed with hydro.
- `simulate_rad_hydro`: doesn't build an `IMCSolver` for hydro-only runs, so a hydro-only run with `radiation_solver="imc"` is the hydro-only run (gate: bit-identical, `test_hydro_only_is_bit_identical_with_imc_selected`). The budget records the A9 diagnostic.
- `hydro_steps`: `update_nodes_compatible(..., P_rad_new=None)`; new `radiation_work`; `boundary_work` uses `_momentum_pressure`. With `P_rad_new=None` the arithmetic is the original's (the radiation pressure is the float 0.0 added to p).
- `RadHydroState.P_rad = None` (new optional field).
- `SimulationConfig.rad_coupling = "matched"`.

### 3.4 Presets: the manuscript's Test 1 and Test 2

`PRESET_PAPER_TEST1` is `PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK` under the paper's name (snapshots at 1, 1.5, 2 ns); `PRESET_PAPER_TEST2` is the same with τ = 158/1285. They were checked against the manuscript's §"Test cases" (`test_paper_presets_match_the_manuscript`):

| | Manuscript | Preset (SubsonicHeatWave) |
|---|---|---|
| Test 1 bath | [1 + 0.169141 t^(−79/192)]^{1/4} T_s | 0.169139, −0.411458 |
| Test 1 m_front | 1.01889×10⁻³ t^(33/64) | 1.0189×10⁻³, 0.515625 |
| Test 2 bath | [1 + 0.204154 t^(−632/1285)]^{1/4} T_s | 0.204152, −0.491829 |
| Test 2 m_front | 8.34319×10⁻⁴ t^(193/257) | 8.3432×10⁻⁴ |
| material, EOS | f = 3.4×10¹³, β = 1.6, μ = 0.14, κ = 7200 T^−1.5 ρ^0.2, γ = 5/4, ρ₀ = 19.32 | the same |

The domain is 2.5×10⁻² g/cm². The analytic shock reaches 2.08×10⁻² (Test 1) and 1.74×10⁻² g/cm² (Test 2) at 2 ns, so it stays inside. The cell count is `SimulationConfig.N`: the mesh is uniform in mass, 2.5×10⁻⁵ g/cm² per cell at the paper's N = 1000.

### 3.5 `verification/imc_vs_diffusion.py`

The new cases `test1_full` and `test2_full` (N = 400, dt ramp to 2 ps, the Phase B flags plus `rad_energy_per_mass` and `compatible`) each run: diffusion, diffusion with `rad_coupling="physical"`, IMC with seeds 1–3, and IMC with the physical coupling (seed 1). Each run stores:
- profiles at 0.5, 1, 1.5, 2 ns, with the per-cell optical depth;
- the heat-front, shock-front and ablation-pressure series at every 20th step;
- the thinned energy budget, including A9.

The analysis compares them with Menahem's `AblationSolver` (profiles, and the closed-form m_f(t), m_s(t), p_f(t)) and with the energy that entered the self-similar solution, ∫s(0,t)dt = 2σ_SB T_s⁴ B t^{p+1}/(p+1).

**Front definitions.** `run_comparison`'s `detect_front` (steepest log-drop of T and p) locks onto the shock on coarse meshes, so it returned the same mass for both fronts. The fronts are read off the density instead, which is smooth in both solvers:
- the **heat (ablation) front** is where ρ first climbs back to ρ₀ from the surface: expanded ablated matter behind it, compressed matter ahead;
- the **shock** is where the compression ρ − ρ₀ falls through half its peak, searching from the cold side;
- the **ablation pressure** is p at the heat front.

`--imc-only`, `--no-physical`, `--hydro-n` and `--hydro-t-end` let the seeds run in parallel processes, sharing the cache.

---

## 4. Findings recorded, not changed

1. **Module-level material parameters in `radiation_step.py`.** `radiation_step`, `conduction_step` and the black-body step set `global alpha, beta_Rosen, mu, f_Kelvin, chi, lambda_, g_Kelvin` on every call, and the helpers (`material_temperature`, `fleck_beta`, `calculate_sigma_…`) read them. A helper called before any diffusion solve, or after a solve for a different case in the same process (the verification scripts run several cases), silently uses the wrong material. Phase C avoids that path for IMC (§3.3). The robust fix is to pass the case (or a material object) to the helpers.
2. **The right boundary node is not accelerated with outflow.** `_get_boundary_pressures` maps `"outflow"` to `None`, and `compute_acceleration_nodes` then sets a[−1] = 0: the right face is effectively a wall. For Tests 1–2 the shock stays inside the domain, so it doesn't matter there, but a domain that is too short reflects the shock.
3. **The Phase B finding about the left boundary node mass (m₀ instead of m₀/2) still stands** (`md/phase_b/phase_b_diffusion_changes.md` §5.1). It affects both solvers equally.

---

## 5. Tests

`tests/imc/test_rad_hydro_lagrangian.py` holds 11 tests:
- `Mesh.carry`;
- hydro-only bit-identity with IMC selected;
- energy budgets of IMC full rad-hydro, matched and physical (< 10⁻¹¹; measured ~10⁻¹²), and of diffusion matched and physical (< 10⁻¹²);
- the refusals;
- `subtract_radiation_energy` in full rad-hydro, diffusion and IMC;
- A9 at early time;
- the paper presets against the manuscript.

`tests/imc/test_rad_hydro_coupling.py::test_imc_refuses_what_it_does_not_model` and `tests/test_phase_b_diffusion.py::test_subtract_is_refused_without_radiation` were updated for the lifted guards.

```bash
uv run pytest tests/imc tests/test_phase_b_diffusion.py
```

---

## 6. Phase C½ additions (Tests 3–6; plan Addendum B)

Everything below is opt-in as well; the defaults are the Phase C code. Results: `md/phase_c/phase_c_tests3to6_results.md`. Why the semi-analytic solution differs: `md/phase_c/why_the_solver_differs.md`.

| # | Where | What | Switch (default = Phase C) |
|---|---|---|---|
| 12 | `iterator.initialize_problem` | mesh with equal initial cell masses for ρ₀x^(−ω), planar | `SimulationConfig.mesh="default"` / `"uniform_mass"` |
| 13 | `iterator.compute_boundary_temperatures`, `bath_correction*`; `imc.coupling.rad_hydro.power_law_bath(factor=)` | the self-similar bath times a calibration table c(t) | `SimulationConfig.bath_drive="paper"` / `"calibrated"` / `"calibrated_flux"` |
| 14 | `RadHydroCase.bath_correction`; `presets_config` | where each paper test's table lives (`src/rad_hydro_sim/data/bath_correction/<key>.txt`, `…_flux.txt`) | read only with a calibrated `bath_drive` |
| 15 | `verification/imc_vs_diffusion.py` | Tests 3–6 (`test3_full` … `test6_full`); `--bath`, `--calibrate-bath`, `--calibrate-target`; temperature-based heat front; the opacity and bath-comparison figures; profiles in the style of the ICTT figures | new options and cases |

### 6.1 The bath calibration (options (a) and (c))

`bath_drive="paper"` (option (c), the default) is the manuscript's Marshak relation T_bath(t). A calibrated drive multiplies it by a factor c(t) read from a two-column table (t in ns, c), interpolated in log t and held constant outside the table. The same factor goes to the diffusion bath (`compute_boundary_temperatures`) and to the IMC bath (`power_law_bath`), where it multiplies the exact closed-form step energy by c⁴ at mid-step. So both solvers keep identical drives.

`calibrate_bath(label, target)` makes the table by iterating diffusion runs on a log time grid, from 0.2 ns to t_end:
- `target="temperature"` (`bath_drive="calibrated"`) matches the matter's surface temperature to T_s(t);
- `target="flux"` (`bath_drive="calibrated_flux"`) matches the net energy flux into the slab to the solution's s(0, t) = 2σ_SB T_b⁴ B t^(4τ+p).

Why there are two targets, and why the flux is the right one for copper and aluminum, is in `why_the_solver_differs.md` §2.

Bugs found and fixed on the way:
- The table reader was cached by path alone, so successive iterations, which rewrite one file, kept reading the first table. It is now cached by path and modification time (`test_bath_table_is_reread_after_a_rewrite`).
- With a time grid starting at 0.02 ns the iteration could not converge: there the heated layer is thinner than the first cell. It now starts at 0.2 ns.

### 6.2 The uniform-mass mesh

x_i = ((1 − ω) m_i / ρ₀)^(1/(1−ω)) with m_i uniform. For ω = −0.5 the default graded mesh (`make_nonuniform_nodes` with `refine_by_mass`) put 628 of 1397 cells in the first 10⁻⁵ g/cm² and 0.06 mg/cm² cells across the shock. The uniform mesh gives 0.0073 mg/cm² everywhere at N = 1000, and is 20× cheaper (the default's first cells were 10⁻¹⁰ cm thick). Test: `test_uniform_mass_mesh_gives_equal_cell_masses`.

### 6.3 Analysis

- **The heat front** is the first point from the surface where T < 0.3 T_s (`fronts_of_profile`). Tables recompute it from the stored profiles at 1, 1.5 and 2 ns (`front_values`), so every run is analyzed with the same definition.
- **The optical depth reported** is that of the hot plasma, T > T_s/2 (`layer_optical_depth`). The sum over the whole layer behind the front is dominated by the cold, dense front cells and changes with the mesh (gold: 43 at N = 400, 13 at N = 1000); the hot-plasma value does not (8.9, 9.0).

### 6.4 Tests

`tests/imc/test_paper_tests_options.py`, 7 tests:
- the uniform-mass mesh, and that the default mesh is unchanged;
- the `bath_drive` switch and its refusals;
- the table re-read after a rewrite;
- the IMC bath factor (c⁴ on the step energy and on the flux);
- the front detector on a synthetic profile.
