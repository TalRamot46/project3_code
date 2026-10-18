# Plan: from `src/sn/sn.py` to an $S_N$-vs-diffusion (and $S_N$-vs-IMC) verification machine for the subsonic Marshak-wave paper, Tests 1–6

## Context

**Goal.** Produce deterministic transport ($S_N$) results for the paper's
benchmarks, alongside the gray diffusion rad-hydro code (`src/rad_hydro_sim`),
the semi-analytic solution, and the IMC of `md/plans/full_plan_imc_comparison.md`.
The starting point is `src/sn/sn.py`, the deterministic twin of the SiO₂
IMC script: the same problem, drive and Fleck linearization, with cell-wise
coefficients, a bath boundary with exact discrete inflow, and exact energy
bookkeeping. The theory is in `md/theory/sn_derivation.md`
(cited below as "derivation §x"). Each milestone has a **Learn** part, a
**Build** part and a **Gate**, as in the IMC plan.

### Status (2026-10-17)

| Milestone | State |
|---|---|
| S0 per-step kernel | **Done for a uniform mesh**: `transport_step` (sweep + source iteration, angular census, boundary currents). Still needed: a non-uniform mesh and a shared material module. |
| S2 bath boundary | **Done for a tabulated drive**: step-averaged inflow, `--bath-norm current/naive`, `--quadrature gauss/double-gauss`. Still needed: the analytic power-law average for Tests 1–6. |
| S3 thermal coupling | **Done for the SiO₂ closed-form EOS.** Still needed: Hammer–Rosen with $\rho$ and $-aT^4$ (Newton inversion), `newton_outer`, Su–Olson and 0-D gates. |
| S4 $S_N$ vs. IMC | **Nonlinear SiO₂ comparison passed** (S4a below). The frozen linear-step test is still open. |
| S1, S5–S11 | Not started. |

### Why $S_N$ is worth building alongside the IMC

* It is **deterministic**: no seeds and no error bars. A difference from
  diffusion is either transport physics or discretization error, and
  refinement tells the two apart.
* It solves **the same linear equation as the IMC** (derivation §3: F&C (2.1)
  at $\alpha=1$). Once both exist, IMC-minus-$S_N$ is noise plus $S_N$
  discretization error, which makes it the best debugging tool for the IMC.
* In 1-D it is cheap: about 1000 cells × 16 ordinates × ~10 sweeps per step.
* The paper's $S_N$ curves come from McClarren's code (McClarren & Haut 2022).
  Your own $S_N$ is an independent check on them, built on the same
  conventions as your diffusion and IMC.

**Recommended order with the IMC plan.** Build $S_N$ through S6 (radiation
coupled to hydro) **before** IMC M4–M5. The two plans share `material.py`,
the bath integrator, the energy diagnostics and the hydro hook (IMC M0, M3,
M4). $S_N$ exercises all of them without noise. IMC M1 ("linear step vs.
`sn.py`") becomes S4 below and uses `transport_step`.

### The six tests

The manuscript defines only Tests 1–2 explicitly (`\subsubsection*{Test 1}`,
`{Test 2}`). It says "six benchmarks" and a comment notes that only the two
ICTT cases are written up. The mapping of Tests 3–6 below is **inferred from
the existing presets** in `src/rad_hydro_sim/problems/presets_config.py`.
**Confirm it before building presets.**

| Test | Material | $\tau$ | $\omega$ ($\rho_0\propto x^{-\omega}$) | Existing preset | Notes for transport |
|---|---|---|---|---|---|
| 1 | Au | 0 | 0 | `PRESET_FIG_8_CONSTANT_TEMPERATURE_MARSHAK` | Defined in the paper. $T_{\rm bath}^4$ singular as $t\to0$: use the step average (derivation §8.3). |
| 2 | Au | 158/1285 | 0 | **none with χ = 1** (`fig_9` and `menahem_ablation` have χ = 1000) | Defined in the paper. New preset needed, as in IMC M5. |
| 3 | Au | 0 | +0.5 | `CONSTANT_TEMPERATURE_OMEGA_0_5_FULL` | Density $\to\infty$ at $x=0$: an optically very thick surface cell; first-cell resolution matters. |
| 4 | Au | 0 | −0.5 | `CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL` | Density $\to0$ at $x=0$: an **optically thin surface layer**, so the largest transport-vs-diffusion difference among the Au tests. Check its `x_max` (1.1²·5.6419e-3): it is 100× the $\omega=+0.5$ value, while the comment copied from that preset quotes 5.6419e-5. Its title also says "radiation only" while `scenario="full_rad_hydro"`. |
| 5 | Cu | 0 | 0 | `PRESET_COPPER_CONST_TEMPERATURE` | $r=0.4$, $\alpha=2.21$, $\lambda=0.29$. |
| 6 | Al | 0 | 0 | `PRESET_ALUMINUM_CONST_TEMPERATURE` | Low Z and low opacity, yet subsonic. This is the regime your notes from Shay single out: hydro matters but diffusion may be inaccurate. **The test where $S_N$ matters most.** |

### What the paper requires from a transport code (manuscript §"Transport setup")

1. Incident isotropic intensity $I(0,\mu>0,t)=\frac{ac}{4\pi}T_{\rm bath}^4(t)$,
   with $T_{\rm bath}$ from `Tbath_marshak`. In `sn.py` variables:
   $\psi_b=acT_{\rm bath}^4$ (derivation Lemma 0).
2. $k_t=\rho\kappa_R$, $k_a=\chi k_t$, $k_s=(1-\chi)k_t$. All tests use
   $\chi=1$, so $k_s=0$.
3. $u_m=fT^\beta\rho^{1-\mu}-aT^4$.
4. Radiation carried per unit mass ($\partial(vE_r)/\partial t$), with no
   radiation pressure: the "matched" mode of IMC plan M5.
5. The same figures as for IMC: $\rho,p,v,T$ at 1, 1.5, 2 ns; x–t tracers;
   $T_{\rm bath}$ vs. $T_s$; energies; $\mathcal T(t)$; $m_{\rm front}(t)$.

### Findings from the code and the numbers that shape the plan

1. **`sn.py` solves the coupled SiO₂ problem** (derivation §10). What is
   left for the paper is the material model, the mesh, the analytic bath
   average and the hydro coupling (derivation §10.2). The sweep, the fixup
   and source iteration are correct and reusable (derivation §§6–7).
2. **Half-range current bias.** With Gauss–Legendre on $[-1,1]$, imposing
   $\psi_b=acT_{\rm bath}^4$ injects
   $\tfrac12\sum_{\mu>0}w\mu/\tfrac14-1$ too much energy: +4.3% ($S_4$),
   +1.15% ($S_8$), +0.30% ($S_{16}$). Fix by normalizing $\psi_b$ or by
   double-Gauss quadrature (derivation §5.2). Both are implemented, and the
   bias is visible on SiO₂: the $S_8$ Gauss peak is 0.9551 with the naive bath
   vs. 0.9383 with the exact inflow. $S_{16}$ and $S_{32}$ double-Gauss both
   give 0.9506.
3. **Source iteration is fast enough without acceleration** for $\chi=1$.
   Here $c_{\rm eff}=(1-f)k_a/(k_a+1/(c\Delta t))$, with
   $f=1/(1+\beta_Fck_a\Delta t)$. Estimates from the presets' EOS and opacity:

   | | $\rho$ | $T$ [HeV] | $\Delta t=10^{-13}$ | $10^{-12}$ | $10^{-11}$ s |
   |---|---|---|---|---|---|
   | Au | 19.32 | 1.0 | 0.06 | 0.37 | **0.86** (~120 its) |
   | Au | 1.0 | 1.0 | 0.02 | 0.18 | 0.69 |
   | Au | 0.01 | 1.0 | 0.00 | 0.02 | 0.29 |
   | Cu | 8.96 | 1.0 | 0.01 | 0.11 | 0.55 |
   | Al | 2.78 | 1.0 | 0.00 | 0.03 | 0.25 |

   DSA or DMD is optional (S10).
4. **The radiation energy fraction $aT^4/u_m$** is $10^{-6}$–$10^{-2}$ (largest in
   the low-density ablated plasma). The $-aT^4$ term and the per-mass
   convention are small, but not negligible at the $10^{-3}$ level of the
   front comparisons.
5. **Cold dense matter is enormously thick per cell**, so DD runs on its
   set-to-zero fixup ahead of the front (derivation §6.3). Expect a small
   front-position sensitivity. Measure it against LD (S9).
6. **Reusable from the IMC plan:** `material.py` (IMC M0), the bath integrator
   (IMC M0 `sources.py`), the diffusion M3 flags, the hook in
   `integrator.step_rad_hydro` (IMC M4), the census remap by relative
   position (IMC M5), and the verification outputs (IMC M10).

**Decisions:** planar geometry only; gray; $\chi=1$; energy-only coupling,
matching diffusion (the "matched" mode); physical mode (P_rad, $V^{4/3}$)
optional, as in IMC A7.

---

## Phase A: foundations

### S0. A reusable per-step kernel (done for a uniform mesh; keep `sn.py` as the SiO₂ baseline)

- **Learn:** derivation §§2, 4–7, 10.
  *Exercise:* turn Lemma 0 and the two normalization checks into tests on
  `transport_step`. Pure absorber: $f=1$, $B=0$ against
  $(1+\sigma c\Delta t)^{-n}$. Pure source: $f=1$ with $B$ chosen so that
  $f\sigma cB=S$, giving $\phi$ growing as $cSt$ when $\sigma c\Delta t\ll1$.
- **Done** in `src/sn/sn.py`:
  - `sweep`: cell-wise `sig_eff`, bath inflow for $\mu>0$, vacuum for
    $\mu<0$, outgoing currents returned;
  - `transport_step`: backward Euler with the angular census, source
    iteration, deposition taken from the last sweep;
  - `make_quadrature`: Gauss–Legendre and double-Gauss.
- **Still to build** (in a package next to the frozen `sn.py`):
  - cell widths `dx[i]` instead of the global `DX`;
  - per-ordinate inflow on both faces (`psi_left[m]`, `psi_right[m]`), for
    reflecting walls and the 0-D problem;
  - coefficients from a shared `material.py` instead of `material_coefficients`.
- **Gate:**
  - `sn.py` stays as the SiO₂ baseline: breakout 1.4675 ns, peak 0.9506 at
    1.9485 ns, energy error $\le10^{-13}$ ($S_{16}$, double-Gauss,
    $\Delta t=1$ ps). The generalized kernel must reproduce these to
    round-off with uniform `dx`.
  - `tests/sn/` passes: the two normalization checks, a symmetric problem
    stays symmetric, and $\psi\ge0$.
  - Energy balance (derivation Theorem 2) at round-off for any tolerance.
    Already true in `sn.py`: $5\times10^{-14}$.

### S1. Linear verification of the kernel

- **Learn:** derivation §§5–6 (orders of accuracy, positivity, the fixup).
- **Build:** `tests/sn/test_linear.py`:
  - **Steady pure absorber** with incoming isotropic flux: compare with the
    exact $\phi(x)=\tfrac12\int_0^1\psi_be^{-\Sigma x/\mu}d\mu$ (an $E_2$
    function) evaluated with the same quadrature. The DD error must be
    $O(\Delta x^2)$ for $\Sigma\Delta x<2\mu_{\min}$.
  - **Thick-cell behaviour:** the same problem at $\Sigma\Delta x\in\{1,10,100\}$
    with and without the fixup. Record where it triggers.
  - **Time order:** a time-dependent infinite medium with a source, giving
    $O(\Delta t)$.
  - **Method of manufactured solutions** with cell-wise $\sigma(x)$, on a
    non-uniform mesh.
- **Gate:** observed orders 2 (space, smooth) and 1 (time); exact discrete
  conservation; fixup triggers only where predicted.

### S2. The Marshak bath boundary

- **Learn:** derivation §§2.3–2.4 (the transport BC and why diffusion's
  Marshak BC is its P1 version) and §5.2 (the half-range defect). Paper
  `Tbath_marshak` and the appendix "Derivation of the radiation flux and bath
  temperature".
- **Build:**
  - `bath.py` (shared with the IMC `sources.py`): exact step average
    $\overline{F_{\rm inc}}=\Delta t^{-1}\int\sigma_{\rm SB}T_{\rm bath}^4dt$,
    analytic for the power law (needed at $t\to0$ in Test 1).
  - Already in `sn.py`: `--bath-norm {current,naive}` and
    `--quadrature {gauss,double-gauss}`, with a tabulated drive averaged by an
    8-point Gauss rule. Still to add: the analytic average of the power law.
- **Gate:**
  - With `"current"` or `"double_gauss"`, the discrete incoming current equals
    $\sigma_{\rm SB}T_{\rm bath}^4$ to round-off.
  - Steady half-space, pure absorber: reflected current zero; deposited
    energy equals the incoming current.
  - Report the `"naive"` bias for $N=4,8,16$ (expect the table in finding 2).

### S3. Thermal coupling: the F&C linearization in $S_N$

- **Learn:** derivation §3 (the linearization, the ledger L1–L5), §8.1
  (Theorem 2), §8.2 (positivity of $u_m$). Compare line by line with
  `imc_mc_derivation.md` §§1.2–1.8: it is the same algebra at $\alpha=1$.
- **Already in `sn.py`** for the SiO₂ EOS: steps 1–3 and 6 below, with the
  closed-form inversion instead of Newton.
- **Build:** `src/sn/thermal.py:sn_thermal_step(state, dt, bath)`:
  1. From $T^n$, $\rho$ (via the shared `material.py`), compute $k_t$, $k_a$,
     $\beta_F$, $f$, $\Sigma_s^{\rm eff}=(1-f)k_a$ and $S^{\rm eff}=fk_aacT_n^4$.
  2. `transport_step`.
  3. Material update (M): $u_m^{n+1}=u_m^n+\Delta tfk_a(\phi-acT_n^4)$.
  4. Newton inversion $T(u_m,\rho)$ including the $-aT^4$ term.
  5. Option `newton_outer=k`: re-linearize around $T^*$ $k$ times (removes
     L1/L2; fully implicit BE).
  6. Energy ledger: $\Delta(E_r+u_m)$ vs. $\Delta t(J_{\rm left}-J_{\rm right})$.
- **Gate:**
  - **Su–Olson benchmark** (Su & Olson 1997, already cited in the paper):
    $C_v\propto T^3$ makes the problem linear; compare $E_r$ and $u_m$ with
    the published tables at the published times.
  - **0-D (one cell, reflecting walls):** the discrete map matches the
    analytic backward-Euler-linearized recursion to round-off. Its
    amplification factor matches IMC M2's analysis at $\alpha=1$.
  - Energy closes to the SI tolerance in every step; $u_m>0$ always.

### S4a. $S_N$ vs. IMC on the full SiO₂ problem (done)

- `python src/sn/sn.py --imc sio2_1mm_gray_imc_check.npz` overlays the IMC
  mean and its 2 s.e. band.
- **Result** ($S_{16}$, double-Gauss, $\Delta t=1$ ps, 4 s wall time):
  - breakout 1.4675 ns vs. IMC 1.47 ns;
  - peak 0.9506 at 1.9485 ns vs. IMC $0.953\pm0.029$ at 1.95 ns;
  - in 99% of the 77 bins where the flux exceeds 5%, $S_N$ lies inside the
    IMC 2 s.e. band (mean offset −0.02 s.e.).
- **Open:** the IMC run on disk (`sio2_1mm_gray_imc_check.npz`) has a
  relative energy error of about 80 up to $t\approx0.01$ ns, 0.3 at 0.1 ns,
  and $5\times10^{-5}$–$10^{-4}$ for $t\ge1$ ns. $S_N$'s is $\le5\times10^{-14}$
  throughout. The late-time IMC value is far above round-off, so the IMC
  script's bookkeeping is not exact. That contradicts the "energy error about
  $10^{-14}$" gate in IMC M0. Find the cause in IMC M0, comparing step by step
  with `sn.py`'s ledger.

### S4. $S_N$ vs. IMC on a frozen linear step (this is IMC plan M1)

- **Learn:** derivation §0 (why the two differ only by L3–L5).
- **Build:** `tests/imc/test_linear_vs_sn.py` from IMC M1, using `transport_step`
  with cell-wise coefficients and $S_N$ sub-stepping in time.
- **Gate:** census $E_r$, deposition and leakage agree within 3 s.e. plus the
  $S_N$ error, estimated from $(N,\Delta x,\Delta t)$ refinement, for
  $\sigma\Delta x\in\{0.1,1,10\}$.

---

## Phase B: make $S_N$ and diffusion solve the same problem

### S5. Radiation-only (static mesh) Tests 1–6

- **Learn:** derivation §8.4 (the thick limit gives exactly the diffusion
  equation with $D=1/(3k_t)$, for every $N$). Paper §"Gray diffusion setup"
  (the optical-depth check $\mathcal T$). This milestone needs the diffusion
  M3 flags from the IMC plan (conservative $u_m$ update, $-aT^4$, per-mass
  $E_r$).
- **Build:**
  - `src/sn/problems/radiation_only.py`: build the mesh, $\rho_0(x)$
    (including $\omega$), and the bath from a `RadHydroCase`.
  - Run each test's radiation-only variant (hydro frozen) with $S_N$ and with
    diffusion on the identical mesh and $\Delta t$ sequence.
- **Gate:**
  - Where $\mathcal T\gg1$, $S_N$ and diffusion agree to the $S_N$
    discretization error.
  - The differences are localized: surface boundary layer, front.
  - The surface $T$ vs. $T_{\rm bath}$ is compared with `check_marshak`.
  - Convergence in $N\in\{4,8,16,32\}$ is shown.
  - Expected ordering of $S_N$-vs-diffusion differences: Test 6 (Al) ≳ Test 4
    ($\omega=-0.5$) > Tests 1, 2, 3, 5. If the ordering differs, explain why
    before going on.

---

## Phase C: hydro coupling, the critical path to the paper

### S6. Lagrangian rad-hydro with $S_N$

- **Learn:**
  - IMC plan M5 (operator split, per-mass convention).
  - The census remap: in Lagrangian coordinates each cell keeps its mass, so
    in the matched mode $\psi_{m,i}\leftarrow\psi_{m,i}V_i^{\rm old}/V_i^{\rm new}$
    for every ordinate (energy per mass conserved, the paper's
    $\partial(vE_r)/\partial t$). In the physical mode the factor is
    $(V^{\rm old}/V^{\rm new})^{4/3}$ (IMC A7).
  - $S_N$ in mass coordinates: $\mu\partial_x=\mu\rho\partial_m$, so the cell
    optical depth is $\kappa_i\Delta m_i$. Hydro moves faces but not the
    optical depth per unit mass.
- **Build:**
  - `src/sn/hydro_step.py:sn_radiation_step(state_star, dt, case, T_left, sn_state)`
    returns `radiation_step`'s tuple: $E_{\rm rad}=\phi/c$, $F$ from the DD
    face currents, and `cells`.
  - An `SNSolver` holding $\psi$ (the angular census) and the bath integrator,
    created in `iterator.simulate_rad_hydro`.
  - Dispatch `radiation_solver: "diffusion" | "imc" | "sn"` in
    `integrator.step_rad_hydro` (the same hook as IMC M4).
  - Time-step control: hydro CFL, plus a limit on the relative change of $T$
    per step, shared with diffusion so both see the same $\Delta t$.
  - The Test 2 preset (χ = 1, τ = 158/1285, Marshak, 2 ns).
- **Gate:**
  - Hydro-only runs are bit-identical with `"sn"` selected.
  - The full energy budget closes (internal + kinetic + radiation vs. bath
    inflow minus leakage), to the SI tolerance plus the hydro scheme's own
    error (IMC A1).
  - Tests 1 and 2: $S_N$ vs. diffusion (M3 flags on) vs. semi-analytic agree
    on $m_{\rm front}(t)$ (against the paper's power laws), the ablation
    pressure, the shock position, and $\rho,p,v,T$ at 1, 1.5, 2 ns. Every
    difference is explained by refinement.

### S7. Tests 3–6 (density profiles and other materials)

- **Learn:** paper §"The shocked region" ($m\propto x^{1-\omega}$,
  $\omega<1$). Shay's note in `ramot_meetings.tex` (subsonic but
  low-opacity materials such as Al).
- **Build:**
  - Confirm the Tests 3–6 mapping, fix the `x_max` of the $\omega=-0.5$ preset
    if needed, and add χ = 1 Marshak presets where missing.
  - Mesh for $\omega=+0.5$: resolve the singular density near $x=0$ in mass
    (`make_nonuniform_nodes`).
  - Mesh for $\omega=-0.5$: resolve the thin, low-density surface layer in
    *optical depth*, not only in mass.
- **Gate:**
  - All four tests run to 2 ns with energy closure.
  - For each test, report $\mathcal T(t)$ and the $S_N$-vs-diffusion
    difference in $m_{\rm front}$ and $T_s$.
  - The difference correlates with $\mathcal T^{-1}$ (the transport
    correction scales like one mean free path over the heated thickness).
    Tests 4 and 6 are the stress cases.

---

## Phase D: accuracy features

### S8. Angular quadrature and convergence in $N$

- **Build:**
  - Double-Gauss as the default for bath-driven problems.
  - $N\in\{4,8,16,32,64\}$ on Tests 1, 4 and 6.
- **Gate:** the front position and $T_s$ converge in $N$. The $N$ used for
  the paper figures changes them by less than the plotting resolution.

### S9. Spatial scheme in thick cells (DD + fixup vs. linear discontinuous)

- **Learn:**
  - Derivation §§6.3 and 8.4.
  - Larsen, Morel & Miller, *JCP* 69 (1987).
  - Adams, "Discontinuous finite element transport solutions in thick
    diffusive problems", *NSE* 137 (2001).
- **Build:** an LD sweep option (2 unknowns per cell per ordinate,
  lumped-mass variant for positivity).
- **Gate:**
  - On Test 1 at 250, 500 and 1000 cells, LD and DD+fixup converge to the
    same front.
  - The coarse-mesh front error of each is reported. This is the $S_N$
    analogue of the IMC's teleportation error (IMC M8).

### S10. Acceleration and full implicitness (optional)

- **Build:**
  - DSA consistent with DD (Alcouffe 1977), or the McClarren–Haut DMD
    acceleration, the method used for the paper's $S_N$ curves.
  - `newton_outer>0` from S3.
- **Gate:**
  - Iteration counts drop on a $\chi<1$ variant and at
    $\Delta t\sim10^{-11}$ s; answers are unchanged to tolerance.
  - With Newton on vs. off, the difference on Tests 1–2 is reported. It is
    the size of the L1/L2 lag, shared with IMC.

---

## Phase E: the comparison machine and the paper figures

### S11. Batch verification and paper outputs (merged with IMC M10)

- **Build:**
  - Generalize IMC M10's `VerificationMode.IMC_VS_DIFFUSION` into
    `TRANSPORT_VS_DIFFUSION`, with `solvers=["diffusion","sn","imc"]` and the
    sweep list Tests 1–6.
  - $S_N$ needs no seeds. Instead, run it at two resolutions in
    $(N,\Delta x,\Delta t)$ and plot the finer one, with the coarse-fine
    difference as an error band.
  - Outputs per test (identical to IMC M10):
    - $\rho,p,v,T$ at 1, 1.5, 2 ns;
    - x–t tracers;
    - $T_{\rm bath}$ vs. $T_s$;
    - energies against the paper's closed forms;
    - $\mathcal T(t)$;
    - $m_{\rm front}(t)$ vs. the power laws.
  - **Extra $S_N$-only diagnostic:** the angular shape at the surface,
    $\langle\mu\rangle$ and $\langle\mu^2\rangle$ (`sn.py` already computes
    them, `avg_mu_snap` and `avg_mu2_snap`). The deviation of $\langle\mu^2\rangle$
    from 1/3 shows directly where diffusion's assumption fails.
  - Convergence tables in $N$, $\Delta x$, $\Delta t$.
- **Gate:**
  - One command regenerates everything.
  - Every $S_N$-vs-diffusion difference is attributed to transport physics
    (it does not shrink with refinement) or to discretization (it shrinks).
  - Every $S_N$-vs-IMC difference is within IMC's error bars plus $S_N$'s
    discretization band.
  - If McClarren's curves are available, $S_N$-vs-McClarren is reported too.
    Ask him for: the quadrature and its order, the spatial scheme, $\Delta t$,
    whether he re-linearizes within a step, and how he normalizes the bath
    inflow.

---

## Critical files

- **New:**
  - `src/sn/{kernels,quadrature,bath,thermal,hydro_step}.py` (generalizing
    `sweep`, `transport_step`, `make_quadrature` and `Drive` from `sn.py`)
  - `src/sn/problems/radiation_only.py`
  - `tests/sn/*.py`
  - `bath.py` and `material.py` shared with `src/imc/`, placed where both
    packages can import them (e.g. `src/transport_common/`)
- **Modified:**
  - `src/rad_hydro_sim/simulation/{integrator,iterator}.py` (solver dispatch)
  - `src/rad_hydro_sim/problems/presets_config.py` (Test 2 with χ = 1; Tests
    3–6 checked)
  - `src/rad_hydro_sim/verification/verification_config.py`
  - `md/theory/sn_derivation.md` (exercises)
- **Frozen:** `src/sn/sn.py` as the SiO₂ $S_N$ baseline (S0 gate), next to
  the frozen IMC script it mirrors.

## Verification

- `uv run pytest tests/sn -q`:
  - normalization;
  - DD orders;
  - fixup triggers;
  - bath current;
  - Su–Olson;
  - 0-D map;
  - energy closure;
  - $S_N$ vs. IMC linear step.
- Existing `run_comparison.py` modes are unchanged with
  `radiation_solver="diffusion"`.
- `uv run python -m rad_hydro_sim.verification.transport_vs_diffusion`
  produces the Tests 1–6 figure set with the energy budgets of every solver.
