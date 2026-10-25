# `src/imc/coupling/rad_hydro.py`: IMC inside rad_hydro_sim

This module makes the IMC core (`src/imc/core/`, see `md/imc_code/imc_core.md`) a drop-in replacement for rad_hydro_sim's gray diffusion solver. It is milestone M4 of `md/plans/full_plan_imc_comparison.md`. The diffusion-side changes it relies on are in `md/phase_b/phase_b_diffusion_changes.md`; the results are in `md/phase_b/phase_b_results.md`.

```python
from hydro_sim.problems.simulation_config import SimulationConfig
from rad_hydro_sim.problems import presets_config as P
from rad_hydro_sim.simulation.iterator import simulate_rad_hydro

case = P.PRESET_TEST_CASES[P.PRESET_SIO2_FOAM_RADIATION_ONLY]
cfg = SimulationConfig(N=100, radiation_solver="imc", dt_ramp_max=1e-12, energy_diagnostics=True,
                       imc_n_emission=2500, imc_n_bath=4000, imc_seed=1)
x, state, meta, history = simulate_rad_hydro(case, cfg)
meta["imc"]               # steps, packets, collisions, capped histories, ledger errors
meta["imc_E_rad_track"]   # low-noise E_rad at the stored frames (see Sec. 4)
```

## 1. How it plugs in

rad_hydro_sim calls the radiation solver once per step as `radiation_step(state, dt, case, T_left)`, which returns `(T_material, e_material, T_rad, E_rad, F, LHS)`. `IMCSolver.step` has the same signature and the same return tuple.

- `simulate_rad_hydro` creates an `IMCSolver` when `radiation_solver="imc"`.
- `step_rad_hydro` passes it to `integrator._radiate`, which calls it in place of the diffusion solve.

The solver object holds what diffusion doesn't need between steps: the census (the radiation packets alive at the end of a step), the energy ledger, and statistics.

One step:
1. Build the mesh from the state's node positions (`Mesh(state.x)`, planar).
2. Get T from the state's material energy with `PowerLawMaterial.temperature`. This respects `subtract_radiation_energy`.
3. Run `imc.core.stepper.imc_step`: emission from Tⁿ with the Fleck factor, the bath packets, the census, the random walk to t+Δt, and the material update u_m^{n+1} = u_m^n + (deposited − emitted)/Δx.
4. Return:
   - T and e = u_m/ρ;
   - E_rad = census energy per cell / Δx, and T_rad = (E_rad/a)^{1/4};
   - F = net current through faces 0…N−1, averaged over the step (the diffusion code's index convention);
   - LHS = 0.
5. If `diagnostics` is given, fill in F_left = (bath energy − energy leaving at the left)/Δt and F_right = energy leaving at the right/Δt, for the energy budget.

## 2. Conventions: why this is the same problem as the diffusion run

| | Diffusion (rad_hydro_sim) | IMC (this module) |
|---|---|---|
| units, c, a | CGS, `radiation_step.c`, `a_Kelvin` | the same (`PowerLawMaterial.from_case`) |
| opacity | σ = ρκ_R; coupling χσ | k_t = ρκ_R, k_a = χk_t, k_s = (1−χ)k_t; χ > 1 refused |
| time linearization | F&C, α = 1, Fleck factor 1/(1+A) | the same |
| material update | `material_update="conservative"` | F&C (1.18), always |
| material energy | `subtract_radiation_energy` | the same switch |
| left boundary | Marshak: F = (c/2)(E_bath − E₀) | isotropic blackbody bath at T_bath(t) |
| right boundary | `right_rad_bc="marshak_vacuum"`: F = cE/2 | vacuum: packets leave |
| bath T_bath(t) | `SubsonicHeatWave.Tbath`, or `case.bath_table` | the same function, integrated over the step |
| time step | `dt_ramp_max`: ramp by 1.1 up to a cap | identical sequence |

**The bath, exactly.** The self-similar Marshak bath is T_bath = T_b t^τ (1 + B t^p)^{1/4}, so T⁴ = T_b⁴ t^{4τ}(1 + B t^p) and the energy entering in a step has a closed form:

$$
E_b = \sigma_{SB}T_b^4\Big[\frac{t^{4\tau+1}}{4\tau+1} + B\,\frac{t^{4\tau+p+1}}{4\tau+p+1}\Big]_{t_0}^{t_1}.
$$

`power_law_bath` uses it. This matters in the paper's Test 1, where p = −79/192 makes the flux singular at t = 0. Quadrature would misjudge the first steps; the closed form is exact (checked against adaptive quadrature to 10⁻⁹). Birth times are sampled with density ∝ flux. A tabulated bath (`bath_table`) uses Gauss–Legendre quadrature of the interpolated table.

**Why a ramped dt.** rad_hydro_sim's default radiation dt control reacts to relative changes of E_rad and T_rad. IMC's E_rad is noisy, so the step size would follow the noise. With `dt_ramp_max` both solvers take exactly the same steps. That also makes the comparison like-for-like in its time discretization, since both use the same Fleck factor per step.

## 3. What the IMC does not do (yet)

- **Radiation-only, static mesh.** The full rad-hydro branch refuses `radiation_solver="imc"`. The Lagrangian coupling (census positions carried with the mesh, the energy convention under compression, radiation pressure) is Phase C, milestones M5 and M5b.
- **Planar geometry only.**
- **A Marshak bath is required.** A Dirichlet surface temperature has no transport counterpart, so the solver refuses `bc_type="Dirichlet"`. The paper's transport setup uses the bath.

## 4. Census vs. track-length radiation energy

`state.E_rad` is the end-of-step census divided by the cell width. That keeps the energy ledger exact: matter + census = initial + bath − leakage, to ~10⁻¹². In optically thick cells, though, a packet lives only ~1/(f c σ) ≪ Δt, so nearly all packets are absorbed within the step, and the census holds only the few emitted in its last instant. It is unbiased but extremely noisy: in thick gold, cells jump between 0 and 1.4 HeV in T_rad.

The **track-length estimate**, Σ (weight × time spent in the cell) / (Δt·Δx), is the step-averaged radiation energy density. It uses every packet's whole flight and is far smoother. The solver keeps it as `IMCSolver.E_rad_track`, and `simulate_rad_hydro` stores it at every history frame as `meta["imc_E_rad_track"]`. Use it for radiation-temperature profiles. It lags the snapshot by O(Δt).

## 5. Validation

| Check | Result |
|---|---|
| Energy ledgers (`tests/imc/test_rad_hydro_coupling.py::test_adapter_closes_both_energy_ledgers`) | rad_hydro budget < 10⁻¹⁰ (measured 1×10⁻¹²); IMC ledger and per-cell ledger ~10⁻¹²; no capped histories |
| T1 SiO₂ through rad_hydro_sim vs. standalone `imc.problems.sio2` (3 seeds each) | χ²/dof = 1.39, mean difference +0.09% after 1.5 ns |
| T1 vs. the S_N twin (slow test) | mean difference +0.04%, rms 0.35% |
| Bath energy closed form vs. adaptive quadrature, across the t = 0 singularity | 10⁻⁹ |
| Refusals: hydro scenario, Dirichlet bath, χ > 1 | raise with an explanation |

Agreeing with the standalone SiO₂ IMC checks the whole chain at once: the CGS conversion of the material (the preset vs. the θ-unit script), the tabulated bath, the dt ramp, the coupling and the diagnostics.

## 6. Cost

On this machine:

| Case | Cells | Steps | Packets per step | Time per seed |
|---|---|---|---|---|
| T1 (SiO₂) | 100 | 3223 | ~17 000 | 16 s |
| Gold Test 1/2, radiation-only | 400 | 1234 | 20 000 + 20 000 | 11–14 s |
| Gold with opacity ÷100 | 400 | 1234 | 20 000 + 20 000 | 75–105 s |

Diffusion takes 6–8 s for the same gold runs. Gold is cheap despite its opacity: in cold gold f → 1, so packets are absorbed within a few collisions. Thinner material means longer histories, which is why the ÷100 case costs more.
