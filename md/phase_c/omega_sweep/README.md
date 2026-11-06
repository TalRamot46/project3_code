# Gold ω sweep: heat front and shock against time

Gold, 1 HeV, τ = 0 (Test 4's material and drive) with the density ρ₀x^(−ω) for ω = −0.8 … +0.8 in steps of 0.2 (simulated up to +0.6; +0.8 has the solver only, §2). For each ω: the heat front (x_f, m_f) and the shock (x_s, m_s) against time, from diffusion with the paper bath, diffusion with the flux-calibrated bath, IMC with the paper bath (ω ≤ +0.2), and the semi-analytic `AblationSolver` (the same for both baths). Background on the baths: `../bath_calibration.md`.

| file | content |
|---|---|
| `fronts_omega<ω>.png` | one per ω (`m0.8` = −0.8, `p0.2` = +0.2): x_f, m_f, x_s, m_s against t, with each simulation's deviation from the solver at 2 ns |
| `fronts_vs_omega.png` | the deviations at 2 ns against ω, one panel per quantity |

## Deviation from the solver at 2 ns

Simulation vs. `AblationSolver`, in %; + means the simulation is ahead.

| ω | run | x_f | m_f | x_s | m_s |
|---|---|---|---|---|---|
| −0.8 | diffusion, paper | −6.7 | +17.6 | −9.0 | −15.8 |
| | IMC, paper | −7.2 | +17.0 | −9.5 | −16.6 |
| | diffusion, flux-calibrated | −0.2 | +34.1 | −2.7 | −5.0 |
| −0.6 | diffusion, paper | −7.2 | +9.6 | −9.4 | −14.9 |
| | IMC, paper | −7.8 | +9.0 | −10.0 | −15.7 |
| | diffusion, flux-calibrated | −1.9 | +21.2 | −4.0 | −6.6 |
| −0.4 | diffusion, paper | −5.4 | +3.8 | −6.8 | −9.7 |
| | IMC, paper | −6.0 | +3.3 | −7.6 | −10.6 |
| | diffusion, flux-calibrated | −2.4 | +8.9 | −3.6 | −5.2 |
| −0.2 | diffusion, paper | −3.1 | +1.0 | −3.3 | −4.2 |
| | IMC, paper | −4.1 | +0.9 | −4.5 | −5.7 |
| | diffusion, flux-calibrated | −1.8 | +3.1 | −1.7 | −2.3 |
| 0 | diffusion, paper | −2.4 | −0.1 | −0.7 | −0.8 |
| | IMC, paper | −3.5 | −0.2 | −2.4 | −2.5 |
| | diffusion, flux-calibrated | −1.7 | +0.7 | +0.2 | 0.0 |
| +0.2 | diffusion, paper | −4.7 | −0.5 | +0.7 | +0.5 |
| | IMC, paper | −6.0 | −0.6 | −1.4 | −1.2 |
| | diffusion, flux-calibrated | −4.6 | −0.2 | +1.2 | +0.9 |
| +0.4 | diffusion, paper | −8.9 | −1.3 | −0.8 | −3.3 |
| | diffusion, flux-calibrated | −8.7 | −1.2 | −0.5 | −3.2 |
| +0.6 | diffusion, paper | not resolved | 0.0 | +6.3 | −0.1 |

What it shows:
- **The shock follows the trend of `why_the_solver_differs.md` §3.** It agrees within ~1% for ω = 0 and +0.2 and falls behind steadily as ω decreases, to −15 … −16% in mass at ω ≤ −0.6, where the ablated mass is a large fraction of the shocked mass.
- **The flux-calibrated bath roughly halves or thirds the shock deficit for ω < 0** (−16 → −5% at ω = −0.8). It does so by driving more energy in, so the heat front's *mass* m_f runs further ahead (+17 → +34%); its *position* x_f then lands on the solver's. For ω ≥ 0 the two baths agree within ~1%.
- **IMC follows diffusion** within 0.1–2.1 percentage points on every quantity.
- **x_f for ω > 0 is limited by the cell at the front**, while m_f agrees within ~1%. The front's position is interpolated between cell centres, and the heated cell holding it is 0.1–0.5 µm wide, against x_f = 3.3 µm (+0.2), 1.0 µm (+0.4) and 0.19 µm (+0.6). Taking the position from the dense side of the front instead (the first cold cell, stepped back to m_f) gives −2.1% at +0.2 and −7.9% at +0.4, so part of the x_f deficit there is real and part is this interpolation; for ω ≤ 0 the two definitions agree within 0.7 points. At +0.6 the heated cell (0.53 µm) is wider than x_f itself, so x_f is not resolved and is left out of the summary.
- **ω = +0.6** (graded mesh, 1.14 million steps): the heat front's mass and the shock's mass agree with the solver to 0.1%; the shock position is +6.3% ahead, which fits the low density behind the ablation front that Test 3 shows on the same mesh (`../phase_c_tests3to6_results.md`).

## What was run, and what was skipped

| ω | mesh | paper bath | flux calibration | IMC |
|---|---|---|---|---|
| −0.8 … −0.2 | 400 cells, uniform in mass | diffusion, 39–49 s | 7 runs (flux matched to 0.2–1.0%), then diffusion, 19–35 s | 1 seed, paper bath |
| 0 | 657, uniform in mass | 76 s | converged after 4 runs; then diffusion, 32 s | 1 seed |
| +0.2 | 1510, uniform in mass | 186 s | converged after 3 runs; then diffusion, 100 s | 1 seed, 420 s |
| +0.4 | 347, graded (Test 3's) | 555 s | converged after 3 runs (0.16%); then diffusion | skipped |
| +0.6 | 347, graded | ~1 h of compute, 1.14 million steps, with the first step at 10⁻¹² t_end (below) | skipped | skipped |
| +0.8 | — | **not simulated** (below); the solver exists | — | — |

The uniform-mass meshes put ≥ 40 cells across the ablated layer at 2 ns (cells = 40 × 1.15 m_s/m_f, 400 to 2000). From ω = +0.4 the shocked mass is ≥ 100× the ablated one, so the graded mesh of Test 3 is used. Its shocked region has coarse cells, which costs a few % in the shock (Test 3: −2% on this mesh).

### ω = +0.6 and +0.8

- **The first time step at ω = +0.6.** The first two rad-hydro steps are fixed at 10⁻¹⁰ t_end and not limited by the CFL, because the drive is singular as t → 0. At ω = +0.6 the surface cell is 1.2×10⁻¹⁵ cm thick at ρ = 4×10¹⁰ g/cm³. The first step heats it ~2000×, its sound speed then gives c_s dt/dx ≈ 6, and the compatible energy update, a fixed point contracting like (c_s dt/dx)², diverges at t = 0. A smaller CFL or step cap does not help; `SimulationConfig.dt_first_frac = 1e-12` does (default 1e-10, unchanged elsewhere).
- **The solver at ω ≥ 0.75.** Menahem's `PistonShock` starts `fsolve` from a fixed guess where the shock-root function is flat at −1, and fails although the root exists (ω = 0.8: ξ_s = 4.137, piston pressure matched). The sweep uses `omega_sweep.BracketedPistonShock`, which finds it with `brentq` over a scanned bracket and gives the same root as `fsolve` to 10⁻⁶ for ω ≤ 0.7.
- **Why ω = +0.8 is not simulated.** With x(m) = (0.2 m/ρ₀)⁵ the whole ablated layer, 1.46 mg/cm², lies within 10⁻²⁴ cm of the surface, so any mesh that resolves the heat front has densities above 10²⁰ g/cm³. With first steps of 10⁻¹², 10⁻¹⁴ and 10⁻¹⁶ t_end the run fails at once: the energy update, then a collapsing cell, then non-finite opacities. Running it would need a regularized density near the surface, which is no longer the problem the solver solves.

## Reproduce

```
uv run python -m rad_hydro_sim.verification.omega_sweep --run=-0.8 --imc       # likewise -0.6 ... 0.0
uv run python -m rad_hydro_sim.verification.omega_sweep --run=0.2
uv run python -m rad_hydro_sim.verification.omega_sweep --run=0.4
uv run python -m rad_hydro_sim.verification.omega_sweep --run=0.6 --no-flux
uv run python -m rad_hydro_sim.verification.omega_sweep --figures
```

Runs are cached in `results/omega_sweep{,_calibrated_flux}/`, the flux tables in `src/rad_hydro_sim/data/bath_correction/SWEEP_*_flux.txt`. The solver for each ω reuses Test 4's heat solver, which does not depend on ω, and builds only its own `PistonShock`, in about a second. At ω = −0.5 it reproduces Test 4's solver exactly.
