# A. The surface out of radiative equilibrium (Cu, Al, Be; every material at early times)

Elaborates cause A of the [README](README.md). The criterion t/t_eq and the optical depth 𝒯 are defined here and used by [03_startup.md](03_startup.md).

**Terminology.** *LTE* means the matter is locally thermal at T_m and emits a Planckian at T_m. The 2T gray model assumes it, and so does IMC. The solver assumes much more: **radiative equilibrium**, T_rad = T_m, E_r = aT⁴, with the radiation entering only as a diffusive heat flux (equilibrium diffusion). "Out of equilibrium" below always means T_rad ≠ T_m, never a failure of LTE.

---

## 1. What goes wrong in Tests 5–6

Diffusion at 2 ns, finest mesh, against the solver (IMC agrees with diffusion to ≤ 5%):

| test | T_rad / T_m at the surface | shock: paper bath / T-calibrated | energy in: paper / T-calibrated |
|---|---|---|---|
| 1 (Au) | 1.003 | +0.6% / +1.8% | −2.0% / 0.0% |
| 4 (Au, ω = −0.5) | 1.007 | −12.5% / −10.9% | −10.7% / −8.1% (that one is the [patch](02_patching.md)) |
| 5 (Cu) | **1.065** | −4.7% / +13.2% | −10.0% / **+23.8%** |
| 6 (Al) | **1.047** | −7.8% / +6.6% | −11.2% / **+15.4%** |

"The surface is at 1 HeV" has one meaning in equilibrium and two out of it. The two meanings give different fluxes:

- **Paper bath** (the Marshak relation, derived assuming equilibrium): the *radiation* at the surface is at 1.003 HeV and the *matter* settles at 0.942 HeV (Cu). The matter sets the ablation pressure, so the shock lags and the energy falls ~10% short.
- **T-calibrated bath** (`bath_drive="calibrated"`): raises T_bath until the *matter* is at 1 HeV. The radiation is then at 1.063 HeV, the flux follows the radiation, and the energy overshoots by 15–24%.
- **Flux-calibrated bath** (`bath_drive="calibrated_flux"`): delivers the solver's surface flux s(0, t), so the total energy matches E(t) = −s(0,t) t/(2c_u + c_m). The matter settles in between:

| | heat front | shock | ablation p | energy | matter T(0) [HeV] |
|---|---|---|---|---|---|
| Cu, flux-calibrated | +3.8% | −0.2% | −1.8% | −2.2% | 0.956 |
| Al, flux-calibrated | +2.3% | −2.9% | −2.8% | −2.3% | 0.974 |

The remaining 2–4% is not decomposed further (energy before 0.2 ns, where the calibration does not act; the patch in Al). Figure: [bath_comparison.png](../../results/phase_c/figures/legacy/non_profile/bath_comparison.png); per-bath profiles in `results/phase_c/figures/<bath>/`; tables in [../phase_c/bath_calibration.md](../phase_c/bath_calibration.md). Code: `bath_correction_for` in `src/rad_hydro_sim/simulation/iterator.py`; the tables are made by `bath_calibration.py --calibrate-bath --calibrate-target {temperature,flux}`.

## 2. Why: a thin plasma and a slow coupling

**Optical depth.** The opacity is the test's Rosseland law, σ = κρ = ρ^(1+λ)/(g T^α). The optical depth of the plasma hotter than T_s/2 is 𝒯. At 2 ns it is 8–10 in gold, but only 3.2 in Cu and 2.3 in Al ([opacity_summary.png](../../results/phase_c/figures/legacy/non_profile/opacity_summary.png)). For constant T_s it grows in time, 𝒯 ∝ m_f (m_f/x_layer)^λ ∝ t^((1+λ)a_m − λ a_x). In Test 3 (a_m = 33/64, a_x ≈ 1) that gives 𝒯 ≈ (t / 10 ps)^0.41:

| t | 0.01 ps | 0.1 ps | 1 ps | 10 ps | 0.1 ns | 1 ns | 2 ns |
|---|---|---|---|---|---|---|---|
| 𝒯, Au solver (any ω) | 0.058 | 0.15 | 0.39 | 0.99 | 2.56 | 6.61 | 8.79 |
| 𝒯, 2T run (χ = 1) | 1.31 | 1.27 | 1.48 | 1.62 | 2.73 | 6.60 | 9.18 |
| 𝒯, 2T run (χ = 100) | 0.14 | 0.21 | 0.44 | 0.98 | 2.47 | 6.87 | 9.02 |

The simulation's layer at χ = 1 is opaque earlier than the solver's because its matter is cold (κ ∝ T^−1.5). That is opacity without equilibrium. Beyond the front the shocked gold has τ > 10³ within 0.1 m_f: the front is an opaque wall at every time. Figure: [tau_profiles.png](../../results/misalignment/test3_transparency/figures/tau_profiles.png).

**The coupling parameter.** At fixed E_r the matter relaxes to T_rad in

  t_eq = ρ c_v / (4 a T³ c σ_abs),  c_v = β f T^(β−1) ρ^(−μ),  σ_abs = χ ρ^(1+λ)/(g T^α).

With L ≈ vt, t/t_eq ≈ 𝒯 (c/v)(u_r/u_m), i.e. 𝒯 times a factor O(1) here. The solver is the χ → ∞ limit, so it has no t/t_eq of its own. Evaluated **on the solver's profile**, though, t/t_eq tests its self-consistency: could a two-temperature medium in that state stay in equilibrium? Median over the hot plasma, surface minimum in parentheses:

| | 10 ps | 0.1 ns | 0.2 ns | 0.5 ns | 1 ns | 2 ns | 10 ns | 50 ns |
|---|---|---|---|---|---|---|---|---|
| Au (Test 3) | 1.0 (0.52) | 6.7 (3.5) | | | 44 (23) | 79 (41) | | |
| Al (Test 6) | | 0.6 (0.5) | 1.07 | 2.3 (2.1) | 4.2 (3.7) | 7.5 (6.7) | 29 (26) | 113 (101) |
| Be | | 0.055 | 0.090 | 0.17 | 0.28 | 0.46 (0.37) | | |

So the one-temperature solution holds only for t ≫ t_*, where t/t_eq(t_*) = 1: t_* ≈ 10 ps for Au, ≈ 0.2 ns for Al, and ~6 ns for Be (extrapolated as t^0.71). For Be the solver's state is never self-consistent within the run.

Out of equilibrium the matter is colder and denser, which makes it more opaque and lowers its heat capacity. That is a negative feedback: the 2T run settles into its own non-equilibrium state, with t/t_eq above the solver's (Au 9.9 at 0.1 ns; Be 3.5 at 2 ns) but not high enough. Physically the heat wave runs in the expanded ablated layer (0.04–6 g/cm³ at 5 ps), not in dense gold. There the mean free path ℓ = g T^α ρ^−(1+λ) is long, and the matter holds ~10³ times the radiation's energy, so it must absorb and re-emit ~10³ times before it catches up ([03_startup.md § 3](03_startup.md) has the numbers).

## 3. Does an analytic start fix it? Only where t/t_eq(t₀) is large

The test: seed the solver's (equilibrium) profile at t₀ ([04_analytic_start.md](04_analytic_start.md) explains the method) and see whether the run stays in equilibrium.

**At t₀ = 0.1 ns** (`light_analytic_start.py`, 1000 cells; Be is a new case: Shussman's `Be.m`, ρ₀ = 1.85, ω = 0, preset `beryllium_const_temperature`). At 2 ns, over the solver's:

| | run | T_m(0) | T_r(0) [HeV] | heat front | shock | front p | 𝒯 (solver) |
|---|---|---|---|---|---|---|---|
| Al | cold, paper bath | 0.943 | 1.004 | 0.950 | 0.916 | 1.007 | 2.22 (2.18) |
| | seeded, paper bath | 0.937 | 1.004 | 0.973 | 0.934 | 1.002 | 2.21 |
| | seeded, T-calibrated | 1.008 | 1.077 | 1.144 | 0.997 | 1.296 | 2.30 |
| | seeded, flux-calibrated | 0.952 | 1.018 | 1.028 | 0.969 | 1.041 | 2.34 |
| Be | cold, paper bath | 0.613 | 1.061 | 0.897 | 0.869 | 0.891 | 1.48 (0.73) |
| | seeded, paper bath | 0.603 | 1.061 | 0.919 | 0.888 | 0.885 | 1.41 |
| | seeded, T-calibrated (cap) | 0.792 | 1.600 | 5.5 | 1.15 | 6.8 | 2.08 |
| | seeded, flux-calibrated | 0.630 | 1.124 | 1.128 | 0.998 | 1.081 | 2.85 |

- **The seed does not hold.** The surface leaves equilibrium at once. The matter first overshoots (Al 1.12, Be 1.23 HeV at ~0.2 ns) and by ~1 ns lies on the cold start's curve (Al 0.914 vs 0.919, Be 0.604 vs 0.609 HeV). The non-equilibrium is the state the 2T equations relax to, not a memory of the start. The fronts keep a little memory (Al shock −6.6% instead of −8.4%).
- **Forcing T_m = T_s overdrives.** In Al the radiation must then be at 1.07–1.18 HeV: heat front +14%, front pressure +30%. In Be, even with the bath at its cap, the matter reaches only 0.79 HeV.
- **The flux-calibrated bath gets the shock right and nothing else.** Be: shock 0.998, heat front +13–20%, front p +8–15%. Best Al combination: seeded + flux-calibrated, within 4.1% on every front.
- On 2500 cells the fronts move by ≤ 1.4 points. The first cell's T_m is not converged (Al 0.94 → 0.92, Be 0.61 → 0.56 HeV).

Figure: [light_materials.png](../../results/misalignment/test3_transparency/figures/light_materials.png); full table: [light_table.md](../../results/misalignment/test3_transparency/figures/light_table.md). Bath tables: `src/rad_hydro_sim/data/bath_correction/{aluminum,beryllium}_const_temperature_analytic*.txt`, `beryllium_const_temperature_flux.txt`.

**At t₀ = 2 ns for Al** (`al_long.py`: 30 K, paper bath, 190 mg/cm² domain, `log_mass` mesh with equal 0.02 mg/cm² cells to 20 mg/cm², to 50 ns). At 2 ns Al's t/t_eq = 7.5 (surface 6.7), gold's value when its 0.1 ns seed held, and the layer spans 100 cells. Over the solver's:

| start | run | t [ns] | heat front | shock | entropy (shocked, median) | T_m/T_r mid-layer |
|---|---|---|---|---|---|---|
| cold | 2T, χ = 1 | 2 / 50 | 0.952 / 0.984 | 0.912 / 0.968 | 0.814 / 0.921 | 0.981 / 0.999 |
| cold | 2T, χ = 1000 | 2 / 50 | 1.009 / 0.993 | 0.971 / 0.980 | 0.931 / 0.954 | 1.000 |
| cold | IMC (3 seeds) | 2 / 50 | 0.916 / 0.974 | 0.888 / 0.960 | 0.762 / 0.900 | 0.93 / 0.97 |
| seeded 2 ns | 2T, χ = 1 | 10 / 50 | 0.995 / 0.994 | 0.976 / 0.975 | 1.000 / 0.936 | 0.995 / 0.999 |
| seeded 2 ns | 2T, χ = 1000 | 10 / 50 | 1.007 / 0.999 | 0.982 / 0.980 | 1.001 / 0.950 | 1.000 |
| seeded 2 ns | IMC (3 seeds) | 10 / 50 | 0.981 / 0.985 | 0.971 / 0.970 | 0.992 / 0.921 | ~1 |

- **Seeded at 2 ns, Al's own coupling (χ = 1) does what χ = 1000 does.** Matter and radiation agree within 1% from the start, and the heat front is within 0.6% to 50 ns. The cold start carries the early non-equilibrium to 50 ns: shock −3.2%, entropy 0.92. That is the [startup](03_startup.md) mechanism in a light material.
- **IMC follows diffusion within about one point** from both starts. The Al IMC–diffusion gap (3–5% at 2 ns from a cold start) shrinks to that once both start in equilibrium.
- **What remains at any χ (shock −2%, entropy −5% in the matter shocked after 2 ns) is the patch**, and the mark at the seed's seam is too: see [02_patching.md § 4](02_patching.md).

Figures: [fronts.png](../../results/misalignment/al_long/figures/fronts.png), [profiles.png](../../results/misalignment/al_long/figures/profiles.png), [table.md](../../results/misalignment/al_long/figures/table.md); slider `results/misalignment/al_long/figures/slider/index.html` (serve the folder).

## 4. What follows

- **Rule for seeding:** seed at t₀ with t/t_eq(t₀) ≳ 7 on the solver's profile (surface ≳ 3.5). That means Au at 0.1 ns and Al at 2 ns. Be has no such time within the run.
- **For the paper:** report Tests 5–6 as the limit of the one-temperature solution. A surface with 𝒯 ≲ 3 costs ~10% of the delivered energy, in transport (IMC) and non-equilibrium diffusion alike, so it is physics, not a code artifact. Alternatively choose drives with larger 𝒯 (denser, longer, or lower T_s for light materials).

## 5. Open

- Al T-calibration from the analytic start stopped at 1.9% (its table starts at 0.4 ns, to step over the seeding transient).
- Be's EOS is a choice (r = 0.5529 from `Be.m`, not μ/(β−1)); ρ₀ = 1.85.
- Cu was not run long or seeded late.
