# `src/imc/core/`: the generic gray IMC solver

This folder holds the solver that came out of the SiO₂ script (`src/imc/sio2/`), stripped of everything specific to that problem. It solves the gray, 1-D slab Fleck & Cummings (F&C) equations on any nonuniform mesh, in any consistent units, for any power-law material. The math behind each step is in `md/imc_mc_derivation.md`; this file explains the code.

## 1. Layout

| Module | What it holds |
|---|---|
| `material.py` | `PowerLawMaterial`: opacities, EOS and its inverse, Fleck factor |
| `mesh.py` | `Mesh`: cell edges, widths, centers, masses |
| `particles.py` | `Particles`: the struct-of-arrays packet storage |
| `sources.py` | emission, bath and initial-radiation packets; the drive table |
| `transport.py` | `transport_step`: the numba random walk for one time step |
| `population.py` | `comb`: census population control |
| `tallies.py` | `StepTallies` and `EnergyLedger`: what a step measured, and the energy checks |
| `stepper.py` | `imc_step`: one complete time step, built from all of the above |

Problem drivers live outside this folder: `src/imc/sio2/problem.py` (the SiO₂ slab) and `src/imc/problems/{linear_step,zero_d}.py` (verification problems).

## 2. Conventions used everywhere

- **Units are the caller's choice.** The core never assumes CGS or any other system. The material carries the radiation constant `a` and the speed of light `c` in the caller's units, and every other quantity (time, length, temperature, energy) just has to be consistent with them.
  - The SiO₂ problem uses cm, ns, MJ and θ = T/(100 eV), so `a` = a·(100 eV)⁴.
  - `PowerLawMaterial.from_case` uses CGS with the same `c` and `a` as `rad_hydro_sim`.
  - The 0-D problem uses Wollaber's scaled units with a = c = 1.
- **Energies are per unit transverse area** (slab geometry). A packet's weight `w` is energy per area. Energy *densities* (per volume) are weights divided by cell width.
- **Every packet carries the index of its cell.** The kernel never searches for a packet's cell, and a face crossing places the packet exactly on the face. The frozen script instead nudged packets by 10⁻¹⁰ Δx.
- **Random numbers come from two generators:** numpy's (bath times, initial radiation, combing) and numba's (emission positions, the random walk). They are seeded separately, so always seed both with `stepper.seed_all(seed)`.
- **The left face is where the bath enters.** Each face is either vacuum (packets leave) or reflecting (μ → −μ).

## 3. One time step (`stepper.imc_step`)

```
T^n, census ─┐
             ▼
 material.py   k_a, k_s, Fleck factor f, u_r = aT⁴, u_m(T^n)          (frozen over the step)
             ▼
 sources.py    emission  E_j = f k_a c u_r Δx_j Δt  → packets uniform in cell, μ, t
               bath      E_b = ∫ σ_SB T_bath⁴ dt     → packets at x = 0, density 2μ,
                                                       birth times ∝ incident flux
               census    packets left over from the previous step
             ▼
 transport.py  random walk to t^{n+1}: deposition, face currents, leakage, new census
             ▼
 stepper.py    u_m^{n+1} = u_m^n + (deposited_j − E_j)/Δx_j     (F&C 1.18)
               T^{n+1} = inverse EOS;  error if any u_m^{n+1} ≤ 0
             ▼
 population.py comb the census if it exceeds max_census
             ▼
T^{n+1}, census, StepTallies
```

`imc_step(state, mesh, rho, material, dt, bath, settings)` returns the new `IMCState` and the step's `StepTallies`.
- `bath` may be `None`; that is used for the 0-D problem.
- `rho` is a scalar or one value per cell.
- `IMCSettings` holds the packet counts (`n_emission`, `n_bath`), the comb trigger `max_census`, the safety cap `max_events`, the two boundary types, and `fleck_mode`.

## 4. The modules

### `material.py`: `PowerLawMaterial`

The Hammer–Rosen power laws, in the form `rad_hydro_sim` uses:

- u_m(T, ρ) = f T^β ρ^(1−μ), minus aT⁴ when `subtract_radiation=True`. That is the manuscript's convention, which reads f T^β ρ^(−μ) as matter plus equilibrium radiation.
- Total opacity k_t = ρ^(1+λ) / (g T^α). Absorption k_a = χ k_t; physical scattering k_s = (1−χ) k_t. χ > 1 is refused, because then k_s < 0 and no transport problem corresponds to it.
- `beta_fc` = ∂u_r/∂u_m = 4aT³/(∂u_m/∂T), the β of F&C.
- `fleck(T, ρ, dt, mode)`:
  - `"implicit"` gives f = 1/(1 + β c k_a Δt), with α = 1.
  - `"explicit"` gives f = 1: emission frozen at Tⁿ and no effective scattering. This mode exists for milestone M2a, to show the instability.
- `temperature(u, ρ)` inverts the EOS: in closed form, or by Newton iteration when aT⁴ is subtracted. It raises an error if ∂u_m/∂T ≤ 0.
- `from_case(case)` builds the material from a `RadHydroCase` (CGS units).

### `mesh.py`: `Mesh`

Strictly increasing cell `edges`, with derived `widths`, `centers`, `n_cells` and `masses(rho)`. Build one with `Mesh.uniform(L, N)`, or with `Mesh.from_masses(m_cells, rho)` for a Lagrangian mesh (Δx = m/ρ), which the rad-hydro coupling will need.

### `particles.py`: `Particles`

Five parallel arrays:
- `x`: position;
- `mu`: direction cosine;
- `w`: energy weight per area;
- `t`: birth time, or the start of the step for census packets;
- `cell`: the cell index.

Helpers: `energy()`, `energy_per_cell(n)`, `concatenate([...])`, `empty()`.

### `sources.py`

- `allocate_emission(E, target)` gives each emitting cell at least one packet and shares out the rest by largest remainder. It is the frozen script's scheme, kept for the regression test; the advisor's smarter allocation is milestone M7a.
- `emission_particles(edges, E, counts, t0, dt)` creates the emission packets. Position, μ and time are uniform; the weight is E_j / count_j (F&C 4.13, 4.16b, 4.19, 4.20, 4.23).
- `BathSource(flux, energy_integral=None, n_sub=16)` describes the bath.
  - `step_energy(t0, t1)` integrates the incident flux over the step with 8-point Gauss–Legendre quadrature. Pass `energy_integral` instead when the flux is singular at t = 0, as in the paper's Test 1.
  - `sample_times(t0, t1, n)` draws birth times with density proportional to the flux, averaged over `n_sub` sub-intervals. The frozen script drew them uniformly.
- `bath_particles(edges, E_b, n, times)` creates the bath packets at the left face, with μ = √(1−r), which has density 2μ (F&C 4.21).
- `uniform_radiation_particles(edges, E, per_cell, t0)` creates isotropic radiation, uniform in each cell. It is used as an initial condition.
- `TabulatedDrive(path, time_scale, temperature_scale, T_pad)` reads a T_bath(t) table and interpolates it linearly.

### `transport.py`: `transport_step`

The numba kernel for one step, with the signature `transport_step(x, mu, w, t, cell, edges, k_a, k_s, fleck, c, t1, left_bc, right_bc, max_events)`. For each packet it repeats:

1. Draw the distance to collision, −ln(r)/k_t with k_t = k_a + k_s; compute the distance to the next face along μ, and the distance to census, c(t1 − t).
2. Move to whichever is nearest. Ties go to census, then to the face.
3. **At census:** store the packet.
4. **At a face:** tally the energy crossing that face, then move to the next cell, reflect, or leak out.
5. **At a collision:** with probability f k_a / k_t the packet is absorbed and its weight is deposited. Otherwise it scatters isotropically. This merges the F&C effective scattering with the physical scattering k_s.

It returns the census arrays and the tallies listed below. `max_events` stops a runaway history by moving it to census; the number of such histories is reported as `n_capped`, which should always be 0.

### `population.py`: `comb`

A stratified comb. It replaces a census larger than `max_packets` with exactly `max_packets` packets of equal weight W/M: one tooth per stratum, each with its own random offset. Energy is conserved exactly, and each input packet survives in proportion to its weight, so the census is unbiased. Stratified (rather than single-offset, systematic) resampling is what Theorem 3 of the derivation needs. At the SiO₂ defaults it never triggers.

### `tallies.py`

`StepTallies` holds everything one step measured, all per unit area:
- `emitted`, `deposited` (per cell);
- `bath_energy`;
- `face_plus` and `face_minus` (energy crossing each face in each direction; the bath counts as crossing face 0);
- `track` (∫ w dt per cell, so `mean_energy_density(widths)` is the step-averaged radiation energy density; it will be needed for radiation pressure);
- leakage at each face, the census energy per cell before and after the step, and counters.

There are two energy checks, and both must close to round-off in every run:
- `cell_balance_residual()`, the per-cell ledger: the change in a cell's radiation energy equals emission − deposition + net inflow through its two faces.
- `EnergyLedger.relative_error(total)`, the global ledger: matter + radiation = initial + bath − leakage.

### `stepper.py`

`IMCSettings`, `IMCState` (T, census, t, ledger), `initial_state(...)`, `total_energy(...)`, `imc_step(...)` and `seed_all(seed)`. The material-update guard raises `RuntimeError` if any u_m^{n+1} ≤ 0. With the implicit Fleck factor that cannot happen for power-law EOSs with β < 4 (Proposition 1 of the derivation). In explicit mode it is how the instability ends.

## 5. Minimal use

```python
import numpy as np
from imc.core import sources, stepper
from imc.core.material import PowerLawMaterial
from imc.core.mesh import Mesh

mat = PowerLawMaterial(f=8.78, beta=1.1, mu=0.09, g=1/9175, alpha=3.53, lam=0.75,
                       a=1.372e-3, c=29.98, T_floor=1e-12)      # SiO2 in cm, ns, MJ, theta
mesh = Mesh.uniform(0.1, 100)
bath = sources.BathSource(lambda t: 0.25 * mat.c * mat.a * 1.5**4 * np.ones_like(t))
state = stepper.initial_state(np.full(100, 2.6e-4), mesh, 0.05, mat)
settings = stepper.IMCSettings()
stepper.seed_all(1)
for _ in range(100):
    state, tallies = stepper.imc_step(state, mesh, 0.05, mat, 1e-3, bath, settings)
print(state.T.max(), state.ledger.relative_error(stepper.total_energy(state, mesh, 0.05, mat)))
```

## 6. Tests

`uv run pytest tests/imc` runs 27 tests, one of them marked slow. Those that check the core directly:

- `tests/imc/test_basics.py` checks:
  - material, allocation and bath energy identical to the frozen script;
  - EOS inversion with and without −aT⁴;
  - `from_case` against the rad_hydro formulas;
  - exponential attenuation, exact free streaming, a reflecting box;
  - both ledgers to round-off on a nonuniform mesh with scattering and a bath;
  - comb conservation and unbiasedness.
- `tests/imc/test_linear_vs_sn.py` (milestone M1) checks one step with frozen coefficients against the S_N solver, for σΔx = 0.1, 1 and 10.
- `tests/imc/test_zero_d.py` (milestones M2 and M2a) checks the 0-D problem against Wollaber's Eqs. 120 and 122, including explicit-mode divergence.

## 7. Where later milestones plug in

- **M4/M5:** a `step.py` adapter that calls `imc_step` with a `Mesh.from_masses` mesh inside `rad_hydro_sim`'s integrator.
- **M6:** continuous absorption as a second kernel mode in `transport.py`.
- **M7/M7a:** per-cell allocation and per-cell combing in `sources.py` and `population.py`.
- **M9:** `ddmc.py`.
- **MP:** parallel versions of `transport_step`.

The tallies (`face_plus`, `face_minus`, `track`) already give the face fluxes and time-averaged energy densities those milestones need.
