# Plan: from the SiO₂ IMC script to an IMC-vs-diffusion verification machine for the subsonic Marshak-wave paper (with DDMC, comb, continuous absorption, hydro coupling)

## Context

**Goal.** Produce transport (IMC) results that verify your gray diffusion rad-hydro code (`src/rad_hydro_sim`) and the semi-analytic solution of the paper `subsonic_self_similar_paper/manuscript.tex` ("Generalized self-similar solutions to the subsonic Marshak wave problem"). The transport solver is the IMC in `src/imc/archive/sio2_original/sio2_1mm_gray_imc_check_og.py`, grown step by step into a Lagrangian rad-hydro solver. Continuous absorption, comb population control and DDMC are added at the end. Each milestone has a **Learn** part (reading plus a derivation exercise), a **Build** part (code) and a **Gate** (a check that must pass before moving on).

### What the paper requires (manuscript §§2–4)

- **Physics:** Lagrangian 1-D rad-hydro (Eqs. `cont/mom/energy_lagrange_heat`) with an ideal gas p = rρe, r = 1/4, and the power-law model e = fT^βρ^{−μ}, 1/κ_R = gT^αρ^{−λ}. For gold: f = 3.4×10¹³ erg/g·HeV^{−1.6}, β = 1.6, μ = 0.14, κ_R = 7200 cm²/g·HeV^{−1.5}·ρ^{0.2}, ρ₀ = 19.32, t_end = 2 ns.
- **Test cases:**
  - **Test 1:** τ = 0, T_s = 1 HeV, T_bath = [1 + 0.169141 t_ns^{−79/192}]^{1/4}·T_s, m_front = 1.01889×10⁻³ t_ns^{33/64} g/cm².
  - **Test 2:** τ = 158/1285, T_bath = [1 + 0.204154 t_ns^{−632/1285}]^{1/4}·T_s, m_front = 8.34319×10⁻⁴ t_ns^{193/257}.
- **The transport setup (§"Transport setup") fixes the conventions the IMC must use:**
  1. Incident isotropic intensity I(0, μ>0, t) = (ac/4π)T_bath⁴(t), with T_bath from the Marshak relation (Eq. `Tbath_marshak`).
  2. Total opacity k_t = ρκ_R and effective physical scattering k_s = ρκ_R − k_a. With χ, the diffusion coupling χcσ corresponds to k_a = χρκ_R and k_s = (1−χ)ρκ_R. A transport equivalent exists **only for χ ≤ 1**, so the existing `PRESET_FIG_9_CONSTANT_FLUX` (χ = 1000) cannot be transported.
  3. Material energy density u_m = fT^βρ^{1−μ} − aT⁴, so that the equilibrium total equals the Hammer–Rosen e (Eq. `um_def`).
     - **Rosen's e is the total energy.** The power law e = fT^βρ^{−μ} is read as the **total** specific energy of the plasma in LTE, matter plus equilibrium radiation: e_tot = e_m + v·aT⁴. It is not the material energy alone. The current diffusion implementation uses it as the material energy, and M3(b) fixes that.
     - **Where this convention is stated.** In the manuscript, §"Gray diffusion setup", Eq. `um_def`: e_m = e − v·aT⁴, "since the total specific energy … represents the matter plus radiation energy at equilibrium". It is restated in §"Transport setup", Eq. `tr_mat`: u_m = u − aT⁴. The power-law fit itself is cited to Hammer & Rosen (2003), and that citation still carries the TODO "ensure correct reference here".
     - **Who decided it.** `git blame` in `subsonic_self_similar_paper` shows these lines were committed by you (TalRamot46): `um_def` in commit `b193f45` on 2026-07-09 ("added a section about LTE and optical thickness"), and `tr_mat`'s definition in commit `0477019` the same day. The repository doesn't record where the convention came from. The nearby text says parts of these setup sections were adapted from M. Krief's earlier supersonic Marshak-wave paper, so it most likely follows his convention. Neither Hammer & Rosen (2003) nor Krief's paper has been checked against this here. Confirm with M. Krief (corresponding author) that the original fit is for total energy before relying on it.
  4. Radiation is carried per unit mass: the diffusion equation evolves ∂(vE_r)/∂t (Eq. `gray_diff`), with no radiation pressure.
- **Figures and diagnostics needed for Tests 1–2:**
  - profiles of ρ, p, v, T at 1, 1.5, 2 ns;
  - Lagrangian tracer x–t paths (100 of 1000 cells) with the ablation and shock fronts;
  - T_bath(t) vs. T_s(t);
  - internal, kinetic and total energies vs. time (closed forms in §"The energy in the system");
  - the optical-depth check 𝒯(t).
- **Who produces what in the paper:** the IMC curves come from Steinberg's discrete-IMC code and the S_N curves from McClarren's code. Your IMC is therefore an **independent transport cross-check**, and possibly a source of the IMC curves, built on the same conventions.

### Findings from the code that shape the plan

1. **The diffusion step is F&C-linearized.** `radiation_step.py` uses A = χβσcΔt and B = χcσ/(1+A) = f·χcσ. It is backward-Euler diffusion applied to F&C (2.1), so IMC and diffusion share the time linearization.
2. **Diffusion vs. paper/IMC mismatches to fix behind flags (M3):**
   - (a) The u_r-form material update (`new_UR = A/(1+A)E + UR*/(1+A)`, then T = (UR/a)^{1/4}) is not conservative when β varies (Wollaber after Eq. 26). **Decision: add a conservative u_m option.**
   - (b) The material energy omits the paper's −aT⁴ term.
   - (c) `E_rad` passes through the hydro step unchanged as an energy *density*. The paper's ∂(vE_r)/∂t requires E_rad·V to be conserved: E_rad ← E_rad·V_old/V_new.
   - (d) The right boundary acts as an E = 0 ghost (the Marshak vacuum is commented out).
3. **Integration point.** `integrator.step_rad_hydro` performs hydro predictor → `radiation_step` → `update_nodes_from_pressure`. The IMC replaces `radiation_step` with the same return tuple. **Decision: energy coupling only, matching diffusion; momentum is optional at the end.**
4. **Bath drive.** `iterator.compute_boundary_temperatures` gives T_left = `SubsonicHeatWave.Tbath(t)` at the start of the step. The IMC must integrate T_bath⁴ over each step. In Test 1, T_bath⁴ ∝ t^{−79/192} as t → 0, which is integrable, so the first step needs the analytic integral.
5. **Preset mapping.**
   - Test 1 corresponds to `PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK` (τ = 0, Marshak, 2 ns, χ = 1). Verify its domain and mesh against the paper's 1000 cells.
   - Test 2 needs a **new preset**: τ = 158/1285, Marshak, χ = 1, 2 ns, a domain that contains the shock at 2 ns.
6. **Optical regime.**
   - In the hot, low-density ablated region, σΔx is O(0.1–10) per cell, which is fine for IMC.
   - In the cold dense gold ahead of the front, βσ ∝ T^{0.9} → 0, so f → 1. That is near-pure absorption, which is cheap but causes teleportation at the front.
   - So analog IMC is feasible for Tests 1–2. Emission tilt, DDMC and variance reduction improve accuracy and cost; they are not prerequisites.
7. **Reusable code:**
   - `RadHydroCase._get_params()`;
   - `hydro_sim/core/grid.py:make_nonuniform_nodes` (mass-resolved ablation layer);
   - `verification/radiation_data.py`, `compare_radiation_plots.py`;
   - in `run_comparison.py`: `_build_front_markers`, `_report_front_agreement`, `_mass_xlim`, `run_full_rad_hydro_comparison`;
   - `iterator.check_marshak`;
   - `src/sn/sn.py`, the deterministic S_N twin of the SiO₂ script (same problem, drive and Fleck linearization; `transport_step` for the linear-step check). Theory and plan: `md/theory/sn_derivation.md`, `md/plans/full_plan_sn_comparison.md`;
   - Wollaber's 0-D exact IMC solution (Eq. 120) and amplification factor (Eq. 122, which equals the md's §1.10 g with β ↔ 4q).

**Decision:** planar geometry only.

---

## Phase A: foundations

> **Status (2026-10-18): Phase A code milestones done** (M0, M1, M2, M2a). The Learn exercises are still yours (the A1–A4 ↔ Wollaber map, etc.).
> - **Code:** `src/imc/core/{material,mesh,particles,sources,transport,population,tallies,stepper}.py` (see `md/imc_code/imc_core.md`); drivers `src/imc/problems/{sio2,linear_step,zero_d}.py` (see `md/imc_code/imc_problem_*.md`); tests `tests/imc/` (27 tests, 1 marked slow; `uv run pytest tests/imc`).
> - **M0:** the packaged SiO₂ run gives breakout 1.47 ns, peak 0.957 ± 0.013 (frozen: 1.47, 0.954), energy error 1.4×10⁻¹³, and per-cell ledger 1.1×10⁻¹². Against S_N: mean bias −0.04%, rms 0.3% after 1.5 ns. The "known issue" below was a stale file: re-running the frozen script gives 5.3×10⁻¹⁴. The energy error of 80 came from `sio2_1mm_gray_imc_check.npz`, written before the `np.full(1, THETA0)` fix.
> - **M1:** census energy agrees with S_N cell by cell for σΔx ∈ {0.1, 1, 10}. Deposition is compared on sums over adjacent cell pairs, because S_N's diamond difference oscillates odd–even when σΔx ≳ 1. At σΔx = 10 the comparison is limited to cells 5–89: the boundary layers are 0.1 cell thick, and S_N's spatial error there isn't estimable on its fixed grid.
> - **M2:** every step matches Wollaber Eq. 120 within noise (p = 3, 5). Fig. 2 is reproduced (0.90 → 1.034 → 0.990 → 1.003 → 0.999), and so is ρ = −0.2868.
> - **M2a, correction to the A2 gate below:** the explicit scheme is unstable only when β = du_r/du_m > 1 (ρ → −β, i.e. −4q in Wollaber's variables), not merely when βcσΔt ≳ 1. In 0-D at q = 0.5 the deviation doubles every step and the material energy goes negative by step 6. SiO₂ at Δt = 1 ps runs stably with the explicit scheme, because β ≲ 0.05 there.

### M0. Package the SiO₂ IMC (keep the script frozen as a regression baseline)

- **Learn:** `md/theory/imc_mc_derivation.md` §§1–5. Wollaber §2.2 (his approximations 16, 19, 23 correspond to the md's A1, A2/A3, A4), §2.5–3.5 (energy-weights, Eqs. 61–67 and 94; tallies 87–93; local energy balance 93). *Exercise:* add the A1–A4 ↔ Wollaber map to the md.
- **Build:** new modules under `src/imc/`:
  - `material.py`: built from a `RadHydroCase` (k_t = ρκ_R, k_a = χk_t, k_s = (1−χ)k_t, rejecting χ > 1); u_m with an optional −aT⁴; β = ∂u_r/∂u_m; T(u_m) by Newton iteration when −aT⁴ is used. CGS units, with `c` and `a_Kelvin` taken from `radiation_step.py`.
  - `mesh.py`: nonuniform edges and cell masses.
  - `particles.py`: struct-of-arrays.
  - `sources.py`: emission, bath and census. Bath energy is the exact integral of σ_SB·T_bath⁴ over the step; birth times are sampled ∝ T_bath⁴.
  - `transport.py`: numba kernel with nonuniform cells, physical plus effective scattering, and vacuum, reflecting and bath boundaries.
  - `tallies.py`: deposition, census, partial currents, per-cell energy balance.
  - `problems/sio2.py`: the SiO₂ driver rebuilt on the package.
- **Gate:**
  - The SiO₂ regression matches the frozen script: breakout 1.47 ns, peak about 0.954 within ±2 s.e., energy error about 10⁻¹⁴.
  - **Known issue to resolve first:** the stored run of the frozen script (`sio2_1mm_gray_imc_check.npz`) has a relative energy error of about 80 up to t ≈ 0.01 ns, and still 5×10⁻⁵–10⁻⁴ for t ≥ 1 ns, far above round-off. `src/sn/sn.py` closes the same ledger to 5×10⁻¹⁴ (S_N plan S4a). Find the source by comparing the two ledgers step by step.
  - The packaged IMC also agrees with `src/sn/sn.py` on SiO₂ within its error bars (the frozen script already does: S_N breakout 1.4675 ns, peak 0.9506).
  - `tests/imc/` passes: attenuation, free streaming, pathwise energy conservation.

### M1. The linear step (the md's Theorem 2 in numbers)

- **Learn:** md §§2–4; `md/theory/sn_derivation.md` §§0, 3–7 and the `src/sn/sn.py` header.
- **Build:** `tests/imc/test_linear_vs_sn.py`. Freeze σ, f, q, k_s and solve (T) by IMC and by `sn.transport_step` (σ_t = σ, σ_s = (1−f)σ), with S_n sub-stepping. `transport_step` has no physical scattering k_s yet; adding it is part of S_N plan S0. This milestone is S_N plan S4.
- **Gate:** census E_r, deposition and leakage agree within 3 s.e. plus S_n error, for σΔx ∈ {0.1, 1, 10}.

### M2. 0-D nonlinear behavior

- **Learn:** Wollaber §4: O(Δt) error (Eq. 101), stability (Eq. 122, Figs. 1–2), maximum principle (Eq. 129, Fig. 3), diffusion limit (Eq. 138). Compare with the md's §1.10 and Theorem 4.
- **Build:** `src/imc/problems/zero_d.py`, one cell with reflecting walls.
- **Gate:** reproduces Eq. 120 within noise for σ ∝ T^{−p} (p = 3, 5), and the damped oscillation matches Eq. 122 (p = 5, q = 0.14, Δτ = 10 gives ρ ≈ −0.287).

---

## Phase B: make diffusion and IMC solve the same problem

> **Status (2026-10-18): Phase B done** (M3 incl. A1, M4). Details in `md/phase_b/phase_b_diffusion_changes.md` (every diffusion-code change), `md/imc_code/imc_coupling.md` (the IMC adapter) and `md/phase_b/phase_b_results.md` (comparisons).
> - **M3:** every change is opt-in through new `SimulationConfig` fields. With the defaults the code is bit-identical to the original (golden file from the original code, 4 slow tests).
>   - Maximum energy error, original → with flags: radiation-only 52% → 7×10⁻¹²; hydro-only 3×10⁻⁴ → 4×10⁻¹⁵ (A1 `compatible`); full rad-hydro 4.3% → 7×10⁻¹⁵.
>   - Paper Test 1 (full, N = 200): the flags move the heat front ≤ 0.02% (gate recorded).
> - **M4:** `src/imc/coupling/rad_hydro.py`, selected with `radiation_solver="imc"`; radiation-only; Marshak bath integrated exactly.
>   - T1 SiO₂ via rad_hydro_sim matches standalone IMC and S_N (+0.04%). Diffusion breaks out 0.05 ns early and peaks 5% high.
>   - Gold Test 1/2, radiation-only: diffusion and IMC agree to ≤ 1% in front position and temperature.
>   - Opacity ÷100: diffusion's front is 4% deep.
> - **Not changed, recorded for you:** the left node mass is m₀; the hydro-only retry mixes dt; the Marshak flux uses the cell-center E₀; `calculate_flux`'s boundary flux differs from the matrix's (`md/phase_b/phase_b_diffusion_changes.md` §5).
> - **Changed paths vs. the text below:** the adapter is `src/imc/coupling/rad_hydro.py` (not `src/imc/step.py`). T3 uses the new radiation-only Test 1/2 presets, on a 10⁻² g/cm² domain with 400 cells.

### M3. Bring the diffusion code to the paper's conventions (all behind flags; defaults unchanged)

- **Learn (P3, the key derivation):**
  - Angular moments of (2.1): P1, then Fick, then backward Euler. Show this equals `calculate_abcd` with B = f·χcσ.
  - Derive the Marshak bath BC (paper Eq. `marshak_bc_general`) from the half-range currents of the incident intensity (paper Eq. `Ibath_tr`).
  - Show that χ ↔ k_s (paper Eq. `scatt`).
  - Explain the u_r vs. u_m update.
  - Explain why the paper writes ∂(vE_r)/∂t.
  - *Exercise:* write this as a new md section. It also justifies the paper's "Transport setup" section.
- **Build** (`radiation_step.py`, `hydro_steps.py`, `SimulationConfig` in `hydro_sim/problems/simulation_config.py`):
  - (a) `material_update="conservative"`: e_new = e* + χf cσΔt(E^{n+1} − UR*)/ρ, then T from the EOS.
  - (b) `subtract_radiation_energy=True` (paper u_m).
  - (c) `rad_energy_per_mass=True`: E_rad ← E_rad·V_old/V_new across the hydro step.
  - (d) Optional right Marshak vacuum.
  - (e) A total-energy diagnostic: internal + kinetic + radiation vs. bath inflow minus leakage.
- **Gate:**
  - With the flags on, the diffusion energy budget closes to solver tolerance.
  - Existing verification (`run_comparison.py`, radiation-only, hydro-only, full) is unchanged with the flags off.
  - The flag-on vs. flag-off difference is recorded for Test 1.

### M4. IMC as a drop-in radiation solver (static mesh, radiation-only)

- **Build:**
  - `src/imc/step.py:imc_radiation_step(state_star, dt, case, T_left, imc_state)`. It returns `radiation_step`'s tuple: E_rad from census, F from net face currents, cells from `m_cells/rho`.
  - An `IMCSolver` created in `iterator.simulate_rad_hydro`, holding the census and the bath integrator.
  - Dispatch `radiation_solver: "diffusion" | "imc"` in `integrator.step_rad_hydro`.
  - Fixed dt or a smoothed controller for IMC runs, so that `compute_adaptive_dt` doesn't react to noise.
- **Test ladder:**
  1. **T1:** a new SiO₂-foam preset (intermediate optical depth).
  2. **T2:** reduced-opacity gold.
  3. **T3:** radiation-only versions of Test 1 and Test 2.
- **Gate:**
  - IMC and diffusion agree where the medium is thick; differences in thin regions are explained.
  - Energy check about 10⁻¹⁴.
  - Front agreement reported (`_report_front_agreement`).
  - The surface T vs. T_bath is checked with `check_marshak`.

---

## Phase C: hydro coupling, the critical path to the paper

### M5. Lagrangian rad-hydro with IMC (energy coupling, matching the paper's equations)

- **Learn:**
  - The operator split in `integrator.step_rad_hydro`.
  - Why the paper evolves vE_r.
  - Radiation-energy fraction aT⁴/u_m (about 3×10⁻⁵ in dense Au, larger in the ablated low-density gas).
  - Wollaber §6.3.6 (momentum, for later).
  - *Exercise:* write the split-scheme energy budget, and connect it to the paper's closed-form energies (§"The energy in the system", e.g. Eq. `total_energy_shock_t`).
- **Build:**
  - Census particles stored as (cell j, ξ ∈ [0,1)), so mesh motion maps them: x = x_j^{new} + ξΔx_j^{new}. Weights are unchanged, which is exactly the per-mass convention.
  - Wire `imc_radiation_step` into the `full_rad_hydro` branch with the actual hydro dt.
  - New preset `PRESET_PAPER_TEST2` (τ = 158/1285, Marshak, χ = 1, 2 ns).
  - Confirm that `PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK` matches the paper's Test 1, including 1000 cells.
- **Gate:**
  - Hydro-only runs are bit-identical with the IMC selected.
  - The full energy budget closes.
  - Test 1 and Test 2: IMC (analog) vs. diffusion (M3 flags on) vs. semi-analytic (Shussman/Menahem through `run_full_rad_hydro_comparison`) agree on m_front(t) vs. the paper's power laws, ablation pressure, shock position, and ρ, p, v, T profiles at 1, 1.5, 2 ns, within statistical bands.

---

## Phase D: accuracy and efficiency features (the "final stage" you asked for)

### M6. Continuous absorption (F&C (4.25b), (4.26c), (4.27))

- **Learn:** Wollaber §6.3.1–6.3.2 (Eqs. 141–147); md §5.3.
- **Build:** `absorption="continuous"`: d_col = |ln r|/((1−f)k_a + k_s), weight decay e^{−fk_a d}, track-length deposition, and a low-weight switch back to analog. Never kill and dump the energy.
- **Gate:** means agree with analog; energy check about 10⁻¹⁴; FOM = 1/(σ²·CPU) reported for T(m) and m_front.

### M7. Comb population control

- **Learn:** Wollaber §6.3.3 and §6.3.5 (Fig. 6; Booth 1996); md §7 (requirement R1; systematic resampling vs. stratified).
- **Build (`population.py`):** global stratified comb, per-cell comb (local energy conservation), energy-conserving roulette.
- **Gate:** unbiasedness (with vs. without combing); exact energy; FOM gain on Tests 1–2.

### M8. Emission tilt against teleportation

- **Learn:** Wollaber §5 (Eqs. 139–140, Fig. 5).
- **Build:** linear sub-cell T⁴ emission with a positivity limiter.
- **Gate:** on Test 1, the coarse-mesh front-speed error shrinks toward the 1000-cell result.

### M9. DDMC (hybrid IMC–DDMC)

- **Learn:**
  - Wollaber §6.2.
  - Densmore, Urbatsch, Evans & Buksas, *JCP* (2007), the gray hybrid transport–diffusion method; Densmore et al., *JCP* (2012); Cleveland & Gentile (2014); Fleck & Canfield (1984).
  - *Exercise:*
    - Derive the leakage opacities from the same finite-volume stencil as `calculate_abcd`, so that DDMC is a Monte Carlo solution of your discrete diffusion equation.
    - Derive the IMC↔DDMC interface conditions from the asymptotic diffusion-limit boundary condition (λ ≈ 0.7104; emergent angular shape ∝ μ(1 + 3μ/2)).
    - Check both against Densmore 2007 before coding.
- **Build (`ddmc.py`):**
  - Per-step region flags (σΔx ≥ threshold).
  - DDMC events: leak left or right (c·σ_L), effective absorption (c·f·k_a), census.
  - IMC↔DDMC conversions at region faces.
  - Identical tallies.
- **Gate:**
  - Pure DDMC vs. diffusion on a thick slab agree.
  - Hybrid vs. pure IMC on Tests 1–2 give the same answers, with a measured speedup.
  - Energy check about 10⁻¹⁴.

---

## Phase E: the comparison machine and the paper figures

### M10. Batch verification and paper outputs

- **Build:**
  - `VerificationMode.IMC_VS_DIFFUSION` with a sweep list (paper Tests 1–2 first; then the ω ∈ {0.01, 0.5, −0.5} presets and SiO₂) in `verification/verification_config.py`.
  - New `verification/imc_vs_diffusion.py`. It runs diffusion (M3 flags) and IMC (all M6–M9 options selectable) on an identical mesh and dt, over several seeds, plus the semi-analytic solution.
  - **Outputs per test:**
    - ρ, p, v, T at 1, 1.5, 2 ns (mean ± 2 s.e.);
    - x–t tracers (100 of 1000 nodes) with ablation and shock fronts;
    - T_bath(t) vs. T_s(t);
    - E_int, E_kin, E_rad, E_total vs. time against the paper's closed forms;
    - 𝒯(t);
    - m_front(t) vs. the power laws.
  - **Convergence tables:** N (1/√N), Δt (O(Δt)), Δx.
  - Output under `get_verification_output_dir()`.
- **Gate:** one command regenerates all of it. Every discrepancy is within the error bars or explained (transport regime, boundary layer, teleportation, Fleck O(Δt)).

### M11 (optional)

- Radiation momentum in both codes (Wollaber Eq. 149).
- Adaptive α or time-dependent Fleck factor (Wollaber Eqs. 53, 57).
- A diffusion flux limiter.

---

## Critical files

- **New:** `src/imc/core/{material,mesh,particles,sources,transport,tallies,population,ddmc,step}.py`, `src/imc/problems/sio2.py`, `src/imc/problems/zero_d.py`, `tests/imc/*.py`, `src/rad_hydro_sim/verification/imc_vs_diffusion.py`.
- **Modified:**
  - `src/rad_hydro_sim/simulation/{radiation_step,integrator,iterator,hydro_steps}.py`;
  - `hydro_sim/problems/simulation_config.py`;
  - `src/rad_hydro_sim/problems/presets_config.py` (Test 2 preset, SiO₂ preset);
  - `src/rad_hydro_sim/verification/verification_config.py`;
  - `md/theory/imc_mc_derivation.md` (P1/P3/P9 derivations).
- **Frozen:** `src/imc/archive/sio2_original/sio2_1mm_gray_imc_check_og.py` (with its drive `src/imc/data/back_pulse.txt`), and its S_N twin `src/sn/sn.py`.

## Step 0 (done)

This plan is versioned as `md/plans/full_plan_imc_comparison.md`, next to `md/theory/imc_mc_derivation.md` and the S_N pair `md/theory/sn_derivation.md` and `md/plans/full_plan_sn_comparison.md`.

## Verification

- `uv run pytest tests/imc -q` runs: attenuation, free streaming, energy conservation, linear IMC vs. `sn.py`, Wollaber 0-D (Eqs. 120, 122), continuous vs. analog mean, comb unbiasedness, DDMC vs. diffusion (thick), SiO₂ regression.
- Existing `run_comparison.py` modes are unchanged with `radiation_solver="diffusion"` and the flags off.
- `uv run python -m rad_hydro_sim.verification.imc_vs_diffusion` produces the Test 1/Test 2 figure set and convergence tables, with the energy budgets of both solvers reported.

---

## Addendum A: points from the advisor meeting (new; everything above is unchanged)

Each item says which existing milestone it amends. Read this section together with the milestone it names; the rest of the plan stays valid.

| # | Advisor's point | Where it goes | What changes |
|---|---|---|---|
| A1 | The current rad-hydro simulation does not conserve energy: the hydro does not solve with total energy. | **M3**, new item (f) | Beyond the budget diagnostic (M3e), measure the hydro scheme's own energy error. `hydro_steps.py` evolves internal energy, not total energy, and `update_nodes_from_pressure` uses a leapfrog. Then add a total-energy-conserving option (compatible discretization: the work p·dV on each cell equals the kinetic-energy change at its nodes), so that the IMC-vs-diffusion budget isn't polluted by hydro error. Gate: with radiation off, the total-energy drift on the hydro-only preset is at round-off level in the conservative mode. |
| A2 | Pedagogic order: an **explicit** MC exercise first (very noisy), then IMC, then the "semi-implicit" continuous-absorption version. | **M2**, new sub-step M2a | Add `fleck_mode="explicit"` (f = 1, emission frozen at Tⁿ, no effective scattering). Run it on the 0-D problem and on SiO₂ with a fixed grid. Gate: it shows the instability and oscillation for βcσΔt ≳ 1 predicted by md §1.10 and Wollaber Eq. 122, and the f-implicit run on the same Δt is stable. This is item 1/3/5 of the advisor's list; the fixed-grid code already exists (the SiO₂ script). |
| A3 | Opening benchmark: an optically **not**-thick problem, SiO₂ (the Back problem). Ask the colleague named in your notes for the typical system size. | **M4**, test ladder T1 | T1 becomes the first full benchmark, not just a sanity case. Set L, ρ and the drive from the colleague's numbers before building the preset. (The current script uses L = 1 mm, ρ = 0.05 g/cc, and a synthetic drive.) |
| A4 | Smart particle allocation. More particles where more energy is emitted. Weight each cell by **emissivity × optical depth of the cell**. A **minimum** per cell (never 1 particle in a cell per step). A per-cell population range of about **[20, 1000]**, with the upper bound depending on the cell's radiation energy relative to the total. Aim for a **uniform weight per particle** across the domain, so that particles don't blow up statistics when crossing cells. | **M7** (comb), new part M7a, done before the comb | `population.py:allocate_sources`: the number of emission particles in cell j is ∝ E_j·min(1, σ_jΔx_j)^γ_w (γ_w = 1 per the advisor; tunable), clipped to [N_min = 20, N_max,j], with N_max,j ∝ (E_rad,j + E_j)/E_tot. Boundary particles get a target equal to the uniform-weight value E_b/w̄. **Census control per cell:** comb each cell's census back into [N_min, N_max,j] at the target weight w̄ = E_tot/N_target, which conserves energy per cell. This replaces the current "≥1 per cell plus proportional share" in the SiO₂ code. Gate: FOM for T(m) and F_out improves on SiO₂ and Test 1; there is no population explosion in thin regions, and the census stays bounded by Σ N_max,j. |
| A5 | Continuous absorption: no absorption events; each move multiplies the weight by an exponential, which reduces noise. | **M6** | Already planned. Add the advisor's framing to M6's Learn part, together with the comparison against the explicit and analog runs from A2. |
| A6 | DDMC details. Use τ > 10 as the thick-cell threshold (Steinberg's experience, gray). A DDMC particle has no direction, only an energy and a cell. Per-face diffusion (leakage) coefficients; sample the time to leakage and change the cell index; continuous absorption inside DDMC cells. DDMC → IMC is placed on the face like blackbody emission with a random angle. Use the benchmarks in Densmore 2007/2012. | **M9** | Set the threshold default to σΔx ≥ 10. Within DDMC, use continuous absorption (a weight decay at rate c·f·k_a while waiting for leakage or census) instead of discrete absorption events. For DDMC → IMC, implement the advisor's blackbody face emission (μ ∝ 2μ) as the default and Densmore's asymptotic μ(1 + 3μ/2) as an option, then compare both on the Densmore 2007 benchmarks (gate). |
| A7 | Hydro coupling with full advection ("MMC" in your notes). Hydro step first, then the MC step. Add **radiation pressure** P_rad to the hydro pressure, P_tot = P_mat + P_rad. Compute E_rad per cell either by counting particles or, better, as a **time integral during the MC step**. While cell boundaries move, keep each particle's **relative position** in its cell. Scale particle energies by the factor of an **adiabatic radiation gas with γ = 4/3**. | **M5**, new sub-step M5b, after M5's matched-physics gate | Add `rad_coupling="physical"` alongside the paper/diffusion-matched mode of M5. It does three things. (1) A time-averaged E_rad tally per cell (track-length estimator: ∫w dt / (V·Δt)), giving P_rad = E_rad/3, which is passed to `update_nodes_from_pressure` as part of the pressure. (2) Census weights scaled by w ← w·(V_old/V_new)^{1/3}, the adiabatic γ = 4/3 compression, with the relative position ξ kept. (3) The same physics in diffusion behind the M3 flags: P_rad in the pressure, and E_rad·V^{4/3} conserved across the hydro step. Gate: the physical mode conserves total energy (including the P_rad work); for Tests 1–2 the physical-mode vs. matched-mode difference is reported. The advisor expects it to be small, and it shows the size of the paper's no-P_rad assumption. |
| A8 | Parallelize with shared memory (OpenMP); learn basic compiled code. The advisor's exercise is to write it in C++ with explicit memory handling; libraries share memory between Python and C++. | **New milestone MP**, a parallel track that starts after M4 and is applied again after M6, M7 and M9 | MP1: numba `parallel=True` with `prange` over particles, per-thread tally arrays reduced at the end, and independent per-particle RNG streams so results are reproducible regardless of thread count. MP2 (the advisor's exercise): port `transport.py` (later `ddmc.py`) to C++ with OpenMP, exposed through pybind11 or nanobind on the same numpy buffers (zero copy), with a structure-of-arrays particle layout and preallocated census buffers. Gate: bit-identical tallies serial vs. parallel for fixed streams; speedup measured on SiO₂ and Test 1. |

| A9 | Relativistic (O(v/c)) terms as part of the hydro coupling. | **M5b** (diagnostic) and **M11** (optional extension) | The O(v/c) terms that matter in a Lagrangian scheme, the comoving compression work −(1/3)E_r∇·v (γ = 4/3) and the radiation force P_rad, are already in A7. Full mixed-frame corrections (Lorentz transforms of particle energy and direction, 1 − μv/c and aberration, at emission and collision) matter only in the dynamic-diffusion regime τ·v/c ≳ 1. Add a per-step diagnostic of max v/c and of τ_cell·v/c, summed over the ablated region. Estimate for Tests 1–2: v/c ≲ 3×10⁻⁴, τ·v/c ≲ 3×10⁻². Implement the mixed-frame corrections in M11 only if the diagnostic exceeds about 0.1. Neither the paper's equations nor the diffusion code include them, so leaving them out keeps the comparison like-for-like. |

**Order of the additions.** M2a goes with Phase A. A3 goes with M4. A1 goes with M3. M5b goes with M5. M7a comes before the comb. MP runs alongside, from M4 on.

**Not changed.** The scope (planar, gray), the milestone gates above, and the final comparison machine (M10). M10 gains one flag: `rad_coupling ∈ {"matched","physical"}`.
