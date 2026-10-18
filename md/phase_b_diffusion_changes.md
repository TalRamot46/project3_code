# Phase B: every change to the diffusion rad-hydro code

This is the complete list of what Phase B of the IMC plan (`md/full_plan_imc_comparison.md`, milestones M3 and M4 plus Addendum A1) changed in `src/rad_hydro_sim` and `src/hydro_sim`, why, and how each change was checked.

**The rule followed throughout:** every change is opt-in. With the default `SimulationConfig` the code is bit-identical to the code before Phase B. That was checked on five cases by comparing final states with `np.array_equal`, and it is frozen into `tests/data/phase_b_original_code_golden.npz`, which was produced by the *original* code (test `test_defaults_are_bit_identical_to_the_original_code`).

The IMC side (`src/imc/coupling/`) is described in `md/imc_coupling.md`; the comparisons are in `md/phase_b_results.md`.

---

## 1. Summary

| # | Where | What | Switch (default = original) | Plan item |
|---|---|---|---|---|
| 1 | `hydro_sim/problems/simulation_config.py` | 12 new `SimulationConfig` fields; `with_output_paths` now keeps all fields | — | M3, M4 |
| 2 | `radiation_step.py` → `radiation_step` | conservative material update | `material_update="u_r"` / `"conservative"` | M3(a) |
| 3 | `radiation_step.py` (helpers) + `iterator.initialize_problem` | material energy u_m = u − aT⁴ (manuscript convention) | `subtract_radiation_energy=False` | M3(b) |
| 4 | `integrator.step_rad_hydro` | radiation energy carried per unit mass through the hydro step | `rad_energy_per_mass=False` | M3(c) |
| 5 | `radiation_step.py` → `calculate_abcd` | vacuum Marshak right boundary | `right_rad_bc="ghost"` / `"marshak_vacuum"` | M3(d) |
| 6 | `iterator.simulate_rad_hydro`, `integrator`, `hydro_steps` | per-step total-energy budget | `energy_diagnostics=False` | M3(e) |
| 7 | `hydro_steps.update_nodes_compatible` + `integrator` | energy-conserving ("compatible") hydro update | `hydro_energy="standard"` / `"compatible"` | A1 |
| 8 | `iterator.compute_adaptive_dt` | dt ramp that ignores E_rad/T_rad changes | `dt_ramp_max=None` | M4 |
| 9 | `RadHydroCase` + `iterator.compute_boundary_temperatures` | tabulated Marshak bath | `bath_table=None` | M4 (T1) |
| 10 | `integrator._radiate` + `iterator` | IMC as the radiation solver | `radiation_solver="diffusion"` / `"imc"` | M4 |
| 11 | `presets_config.py` | 3 new presets: SiO₂ foam, paper Test 1 and Test 2 radiation-only | new keys only | M4 |

Nothing else changed. In particular, none of these changed: `get_e_star_from_hydro`, `update_nodes_from_pressure`, the `black` / `gray` / `conduction` modes, `calculate_flux`, the Marshak left boundary, the hydro-only retry logic, and the history format.

### Measured effect: maximum relative energy error

The error is relative to the larger of the total energy and the energy that crossed the boundaries; runs are short (N = 120, to 0.05–0.2 ns).

| Run | Original | With the Phase B flags |
|---|---|---|
| radiation-only, Marshak bath (ω = 0.01) | 5.2×10⁻¹ (early), 7×10⁻³ at 0.2 ns | 6.5×10⁻¹² (`conservative`) |
| radiation-only, Dirichlet | — | 5×10⁻¹⁵ (`conservative`) |
| radiation-only SiO₂, full 3 ns (T1) | 1.4×10⁻¹ | 6.8×10⁻¹⁵ (`conservative`, `marshak_vacuum`) |
| hydro-only | 3.4×10⁻⁴ | 4.0×10⁻¹⁵ (`compatible`) |
| full rad-hydro, Fig. 8 Marshak (Test 1 setup) | 4.3×10⁻² | 6.7×10⁻¹⁵ (`conservative` + `rad_energy_per_mass` + `compatible`) |
| … same, `conservative` only | | 1.2×10⁻³ |
| … same, `conservative` + `rad_energy_per_mass` | | 1.1×10⁻³ |

---

## 2. Why: the diffusion step is Fleck–Cummings diffusion

This is the plan's Learn exercise P3. It explains why each switch exists.

### 2.1 `calculate_abcd` is backward-Euler diffusion of F&C Eq. (2.1)

Take the IMC transport equation (2.1) with physical scattering, as in `md/imc_mc_derivation.md` §1.7. Integrate over angle, using ∫I dμ = cE and ∫μI dμ = F:

$$
\frac{\partial E}{\partial t} + \frac{\partial F}{\partial x} = f\,\chi c\sigma\,(aT_n^4 - E),\qquad f = \frac{1}{1 + \chi\beta c\sigma\Delta t}.
$$

Here χσ is the absorption opacity. Effective and physical scattering drop out of the zeroth moment, because they only redirect radiation. The first moment with the P1/diffusion closure gives Fick's law, F = −(c/3σ_t)∂E/∂x with σ_t = σ. Backward Euler in time and a finite-volume discretization in space give, for cell j,

$$
\frac{E_j^{n+1}-E_j^{n}}{\Delta t} - \frac{1}{\Delta x_j}\Big[D_{j+\frac12}\frac{E_{j+1}-E_j}{\Delta x_{j+\frac12}} - D_{j-\frac12}\frac{E_j-E_{j-1}}{\Delta x_{j-\frac12}}\Big]^{n+1} = B_j\,(aT_j^{*4} - E_j^{n+1}),\qquad B = \frac{\chi c\sigma}{1+A},\ A = \chi\beta c\sigma\Delta t .
$$

This is exactly the tridiagonal system of `calculate_abcd`: `lagrangian_coeff = 1/dx`, `flux_coeff = D_face/dx_face`, `B = chi*c*sigma/(1+A)`, `d = B*UR* + E_old/dt`. So the diffusion code and IMC share the F&C time linearization and the Fleck factor. Comparing them isolates transport vs. diffusion, plus their spatial and boundary treatments.

### 2.2 The material update must be the conservative one (switch 2)

Energy conservation requires the material to gain exactly what the radiation lost through the coupling term, which is F&C (1.18):

$$
u_m^{n+1} = u_m^{*} + B\,\Delta t\,(E^{n+1} - aT^{*4}).
$$

The original code instead set aT^{n+1,4} = (A E^{n+1} + aT*⁴)/(1+A) and took the temperature from it. That updates u_r = aT⁴ as if it were the stored energy. It equals the conservative update only if β = ∂u_r/∂u_m is constant. Wollaber (2016) makes this remark after his Eq. (26). For a power-law EOS β varies like T^(4−β_EOS), so the original update creates or destroys energy every step. That is the 52% / 14% / 4.3% in the table above.

### 2.3 Rosen's e is matter + radiation (switch 3)

The manuscript (§"Gray diffusion setup", Eq. `um_def`, restated in §"Transport setup" as u_m = u − aT⁴) reads e = fT^βρ^(−μ) as the total specific energy in LTE. The material alone then has u_m = fT^βρ^(1−μ) − aT⁴. In gold at 1 HeV the difference is ~3×10⁻⁵ of u_m. It matters in light, low-density materials, and it has to be the same in the IMC and in the diffusion code.

### 2.4 Radiation moves with the mass (switch 4)

The manuscript evolves ∂(vE_r)/∂t in Lagrangian coordinates: without coupling and flux, radiation energy per unit mass is unchanged by compression. The original hydro step kept the energy *density* E_rad unchanged when a cell's volume changed, so the radiation energy E_rad·V changed in proportion to V. The switch sets E_rad ← E_rad·V_old/V_new. The physical alternative is adiabatic compression of a γ = 4/3 photon gas, E_rad·V^(4/3) conserved, plus radiation pressure. That is plan item A7, Phase C.

### 2.5 The Marshak boundary and its transport counterpart (switch 5)

The incident isotropic intensity I(0, μ>0) = (ac/4π)T_bath⁴ carries the half-range current F⁺ = cE_bath/4. In P1, F⁺ = cE/4 + F/2 at the face. Equating the two gives F = (c/2)(E_bath − E), which is the manuscript's Eq. `marshak_bc_general`. The existing left boundary implements this with the cell-center E₀: `cooling_left = c/(2 dx0)` adds (c/2)(E_bath − E₀)/dx₀ to cell 0. A vacuum on the right is the same relation with E_bath = 0: F_out = cE/2. The original right boundary instead kept the edge-padded diffusion coefficient, i.e. a sink E = 0 one face-spacing away. Switch 5 makes the right face the vacuum Marshak face, the counterpart of the IMC's vacuum boundary.

### 2.6 χ and the transport problem

In the diffusion code χ multiplies only the coupling term. In transport that means absorption k_a = χρκ_R and physical scattering k_s = (1−χ)ρκ_R (manuscript Eq. `scatt`), with the total ρκ_R unchanged. A transport equivalent therefore exists only for χ ≤ 1. The IMC refuses χ > 1, so e.g. `PRESET_FIG_9_CONSTANT_FLUX`, which has χ = 1000, has no transport counterpart.

---

## 3. The changes in detail

### 3.1 `SimulationConfig` (`src/hydro_sim/problems/simulation_config.py`)

New fields, all defaulting to the original behavior:

| Field | Default | Values |
|---|---|---|
| `material_update` | `"u_r"` | `"conservative"` |
| `subtract_radiation_energy` | `False` | `True` |
| `rad_energy_per_mass` | `False` | `True` |
| `right_rad_bc` | `"ghost"` | `"marshak_vacuum"` |
| `hydro_energy` | `"standard"` | `"compatible"` |
| `energy_diagnostics` | `False` | `True` → `meta["energy_budget"]` |
| `dt_ramp_max` | `None` | seconds |
| `radiation_solver` | `"diffusion"` | `"imc"` |
| `imc_n_emission`, `imc_n_bath`, `imc_max_census`, `imc_seed` | 20000, 20000, 200000, 1 | IMC packet counts and seed |

`with_output_paths` used to rebuild the config field by field, so it would silently drop any new field. It now uses `dataclasses.replace(self, save_path=..., gif_path=...)`. For the existing fields the result is identical (test `test_with_output_paths_keeps_phase_b_fields`).

### 3.2 `radiation_step.py`

- **New helpers** (after `calculate_A`): `material_energy_density`, `material_temperature` (closed form, or Newton when aT⁴ is subtracted, with an error if ∂u_m/∂T ≤ 0), `fleck_beta` (β for either convention; it calls the original `calculate_beta_from_temperature_and_density` when not subtracting), and `face_flux_coefficients` (the same D_face/dx_face as `calculate_abcd`, used for diagnostics).
- **`calculate_abcd(..., right_rad_bc="ghost")`:** with `"marshak_vacuum"`, `b[-1] += -flux_coeff[-1]/dx[-1] + c/(2 dx[-1])`. This removes the ghost leak and adds c E/(2Δx). The docstring used to say "Applies Marshak vacuum leakage on the right boundary", but that code was commented out; the docstring now describes what the code does. Test: `test_right_marshak_vacuum_replaces_the_ghost_leak` checks that only `b[-1]` changes, and by exactly that amount.
- **`radiation_step(..., *, material_update, subtract_radiation_energy, right_rad_bc, diagnostics)`.** These are keyword-only, so existing positional calls are unaffected. Behavior:
  - Non-default options are accepted only in the time-dependent gray mode (`force_black=None`); other modes raise `ValueError`.
  - With `subtract_radiation_energy`, T* and β come from the new helpers.
  - With `material_update="conservative"`, the update is u = ρe* + B Δt (E^{n+1} − aT*⁴) and T = `material_temperature(u)`.
  - The u_r branch is the original code. With `subtract_radiation_energy` it stores e = (fT^βρ^(1−μ) − aT⁴)/ρ.
  - If a `diagnostics` dict is given, it receives the boundary fluxes *as the matrix imposed them*: F_left = (c/2)(E_bath − E₀) for Marshak or flux_coeff₀(E_left − E₀) for Dirichlet, and F_right = flux_coeff_N·E_{N−1} (ghost) or (c/2)E_{N−1} (vacuum). It also receives the per-cell exchange.
- **The default path keeps the original expressions**, line for line.

### 3.3 `hydro_steps.py` (new functions only)

- `node_masses(m)`: the masses implied by `compute_acceleration_nodes`. Interior nodes get (m_{j−1}+m_j)/2 and the right node m_{N−1}/2. The left node gets **m₀**, because the code accelerates it by (p_L − P₀)/m₀ (see §5).
- `kinetic_energy`, `boundary_forces` and `boundary_work`. The boundary pressures are recovered from the accelerations actually used, so the work is consistent even at the first step, where the initial accelerations use p_left = p[0].
- `update_nodes_compatible(...)`: the A1 energy-conserving close of the step (§4).

### 3.4 `integrator.py`

`step_rad_hydro(..., *, imc_solver=None, diagnostics=None)`:
- **`_radiate`** dispatches to `radiation_step` (diffusion) or `IMCSolver.step` (IMC), passing the Phase B options.
- **`rad_energy_per_mass`**, in `full_rad_hydro`: `state_star.E_rad = state.E_rad * state.V / state_star.V` before the radiation step.
- **`hydro_energy="compatible"`:**
  - `full_rad_hydro` calls `update_nodes_compatible` with the radiation exchange `new_e − e*`, then recomputes T from the final e.
  - `hydro_only` has its own branch, without the dt-halving retry, because the compatible close needs one dt throughout.
- **Guards:**
  - `subtract_radiation_energy` outside `radiation_only` raises: how the hydro pressure should treat the subtracted aT⁴ is a Phase C decision.
  - `radiation_solver="imc"` outside `radiation_only` raises: hydro coupling is M5.
- **`diagnostics`** receives `F_left`, `F_right` and `W_boundary` (zero for radiation-only).

The original branches are textually unchanged inside their `else` paths.

### 3.5 `iterator.py`

- `initialize_problem`: with `subtract_radiation_energy`, the "temperature, density" initial condition gets e −= aT⁴/ρ. Other initial conditions raise.
- `compute_adaptive_dt(..., dt_ramp_max=None)`: if set, and the scenario is not hydro-only, dt = min(1.1·dt_prev, dt_ramp_max, t_end − t), still capped by CFL with hydro. The first steps (1e-10·t_end) are unchanged. Diffusion and IMC then take identical steps. IMC's E_rad is noisy, so the relative-change controller would make its dt depend on the noise.
- `compute_boundary_temperatures`: if `case.bath_table` is set, T_bath is interpolated from the table (t in ns, T in eV, converted with rad_hydro's eV→K). Otherwise the original `SubsonicHeatWave.Tbath`. `simulate_rad_hydro` skips building the SubsonicHeatWave solver when a table is given. New helpers: `tabulated_bath_temperature`, `_bath_table` (cached).
- `simulate_rad_hydro`:
  - creates the `IMCSolver` when `radiation_solver="imc"`;
  - with `energy_diagnostics`, records every step: material energy Σ m e, kinetic energy Σ ½Mu², radiation energy Σ E_rad V, and the cumulative radiation in at the left, radiation out at the right and boundary work;
  - adds `meta["energy_budget"]`, built by the new `energy_budget_summary` (which also gives `total`, `expected` and `relative_error`), and `meta["imc"]` (IMC statistics).

### 3.6 `RadHydroCase` and presets

- `RadHydroCase.bath_table: str | None = None`.
- `PRESET_SIO2_FOAM_RADIATION_ONLY`: the SiO₂ problem of `src/imc/sio2/` in rad_hydro units, f = 8.78×10¹³/T₁₀₀^1.10 and g = 1/(9175·T₁₀₀^3.53), with T₁₀₀ = 100 eV in K. It uses the tabulated bath, 1 mm, ρ = 0.05 g/cm³, 3 ns.
- `PRESET_PAPER_TEST1_RADIATION_ONLY` and `PRESET_PAPER_TEST2_RADIATION_ONLY` (τ = 158/1285): the Fig. 8 Marshak gold case (the manuscript's material) with `scenario="radiation_only"`.

---

## 4. A1: the compatible hydro energy update

**The problem.** The original step updates the internal energy in the predictor with the pressure (pⁿ + p*)/2, where p* is the pre-radiation pressure (Eq. 16 of `get_e_star_from_hydro`). The nodes then move with (aⁿ + aⁿ⁺¹)/2, where aⁿ⁺¹ uses the *post*-radiation pressure, and the velocity in the work term is u^{n+½} rather than the time-centered (uⁿ + uⁿ⁺¹)/2. So the cells' work and the nodes' kinetic-energy change don't match, and total energy drifts: 3.4×10⁻⁴ hydro-only, and part of the 4.3% in full rad-hydro.

**The fix** (`update_nodes_compatible`). Keep the leapfrog momentum update. Then set

$$
m_j\,(e_j^{n+1}-e_j^n) = -\Delta t\,\bar P_j\,(\bar u_{j+1}-\bar u_j) + m_j\,\Delta e^{\rm rad}_j,\qquad \bar P = \tfrac12(P^n+P^{n+1}),\ \bar u = \tfrac12(u^n+u^{n+1}),\ P = p+q .
$$

With M_i(u^{n+1} − u^n) = Δt F̄_i, summation by parts gives exactly ΔKE + Σ_j m_j Δe_j = W_boundary + Σ m Δe^rad. Because p^{n+1} depends on e^{n+1}, the update is a fixed-point iteration. It contracts like (c_s Δt/Δx)², converges in a few iterations to 10⁻¹³, and raises an error if it doesn't. Δe^rad is the radiation exchange computed from e*, as in the original.

**Checks:**
- `test_compatible_hydro_conserves_energy_to_round_off`: standard > 10⁻⁸, compatible < 10⁻¹².
- `test_full_rad_hydro_budget_needs_all_three_flags`: < 10⁻¹² with all three flags, and more than 100× worse without `rad_energy_per_mass`.

---

## 5. Findings in the diffusion code that were *not* changed

These are recorded for you to decide on; none of them was altered.

1. **The left boundary node's mass.** `compute_acceleration_nodes` accelerates node 0 by (p_L − P₀)/m₀, i.e. with mass m₀ rather than m₀/2. Energy bookkeeping is consistent either way (`node_masses` uses m₀), but the ablation surface's inertia is doubled. This may matter for the ablation-surface dynamics in Phase C.
2. **The hydro-only retry mixes time steps.** When the CFL cap or the retry shrinks dt, `get_e_star_from_hydro` uses `attempt_dt`, but `update_nodes_from_pressure` and the clock (`new_state.t += dt`) use the original dt. The compatible branch doesn't retry.
3. **The Marshak flux uses the cell-center E₀.** F = (c/2)(E_bath − E₀) has no half-cell diffusion resistance between the face and the center. It is first-order accurate and relevant when cell 0 is optically thick.
4. **`calculate_flux` returns a different boundary flux** (D_bath/(dx₀/2)·(E_bath − E₀)) from the one the matrix imposes ((c/2)(E_bath − E₀)). `F_rad[0]` is therefore not the flux that entered the slab. The new `diagnostics` gives the imposed one.
5. **The radiation-only branch hides solver failures.** If the solver fails with "infs or NaNs", the state is silently left unchanged and the run continues.

---

## 6. How to use the switches

```python
from hydro_sim.problems.simulation_config import SimulationConfig
cfg = SimulationConfig(
    N=400, material_update="conservative", right_rad_bc="marshak_vacuum",
    rad_energy_per_mass=True, hydro_energy="compatible",     # full rad-hydro
    energy_diagnostics=True, dt_ramp_max=2e-12,
)
x, state, meta, history = simulate_rad_hydro(case, cfg)
meta["energy_budget"]["relative_error"]     # per step
```

For IMC, add `radiation_solver="imc"` (radiation-only cases with a Marshak bath).

## 7. Tests

`tests/test_phase_b_diffusion.py` holds 16 tests, 4 of them the slow bit-identity checks: `uv run pytest tests/test_phase_b_diffusion.py`. The IMC coupling tests are in `tests/imc/test_rad_hydro_coupling.py`.
