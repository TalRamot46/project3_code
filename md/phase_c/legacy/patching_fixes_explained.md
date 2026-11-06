# Fixes B and C to the patching: what they do, what they assume, and why neither is a consistent improvement

Extends `../why_the_solver_differs.md` §3 (the cut) and §3.3 (the suggested fixes). Both fixes are legacy: B is not analytic, and C is not consistently better. This page explains why, at the level of the equations and the code. Results tables: `patching_poc_fix_B.md`, `analytic_patching_impulse.md`.

**In one paragraph.** Both fixes answer one question: what pressure does the shocked target feel at an ablation front that moves *through* the matter? Equivalently, how much momentum leaves the target with the matter that the front swallows. The original patch answers implicitly: the swallowed matter rides on with the front (a *massive* front). Fix B answers explicitly. It solves the target numerically for a chosen answer, and with the simulation's own front history it reproduces the simulated shock to 0.3–1.8%, which proves the target side. Fix C answers in closed form with the other extreme: every swallowed element keeps the velocity it had when it was swallowed (a *massless* front). The simulations sit between the two. Measured as the momentum the target carries beyond the cut, the fraction θ of the way from massless (θ = 0) to massive (θ = 1) is 1.15 at ω = 0, 0.49 at ω = −0.25, 0.37 at ω = −0.5 and 0.29 at ω = −0.75. No patch that fixes θ can be right across ω, and C adds approximations of its own on top.

---

## 1. The question both fixes answer

### 1.1 A boundary that moves through the matter

Outside the ablated layer the flow is pure hydro in the Lagrangian mass m: ∂u/∂t = −∂p/∂m, ∂v/∂t = ∂u/∂m, with an ideal-gas energy equation. The target is the region m > m_f(t), where m_f(t) = A t^α is the heat solution's ablated mass. Its left edge is not a material boundary. Matter crosses it, from the target into the ablated layer, at the rate ṁ_f = α m_f / t.

The target's momentum per unit area, Π_T = ∫_{m_f}^{∞} u dm, then changes by

  dΠ_T/dt = p_b − ṁ_f u_f,                                   (1)

where p_b is the pressure on the target at the front and u_f the velocity of the matter crossing it. The second term is the momentum the target loses with the swallowed matter. Integrated over time:

  Π_T(t) = ∫₀ᵗ p_b dt′ − ∫₀ᵗ ṁ_f u_f dt′.                       (2)

The heat solution provides p_f(t), the pressure at its own front, and m_f(t). It doesn't say what p_b is, because it has no notion of the front moving: its front is a rigid wall (ũ = ṽ = 0 at ξ_f, `why_the_solver_differs.md` §1). So equation (1) needs a **closure**, a statement of how the ablated layer reacts to the front's motion.

### 1.2 The two closures

Write the ablated layer's lab-frame velocity as u = u′ + (offset), where u′ is the rigid heat solution's velocity, zero at its front.

- **Massive front:** the whole layer rides on the front, offset = u_f(t). In the frame of a decelerating front the layer then feels a body force −a_f per unit mass. Momentum balance of the layer gives the pressure it puts on the target: p_b = p_f − m_f a_f. For a decelerating front (a_f < 0) that is *more* than p_f.
- **Massless front:** each element keeps the offset it had when it was swallowed, offset = u_f(t_swallowed). The layer's internal pressure is then the rigid p′, unchanged, and p_b = p_f.

The two differ by how much momentum leaves the target. Integrating ṁ_f u_f by parts:

  ∫₀ᵗ ṁ_f u_f dt′ = m_f(t) u_f(t) − ∫₀ᵗ m_f a_f dt′.              (3)

- **Massive:** with p_b = p_f − m_f a_f in (2), the a_f terms cancel and Π_T = ∫p_f dt − m_f u_f(t). The swallowed matter takes away exactly the momentum it has *now*, moving with the front.
- **Massless:** Π_T = ∫p_f dt − m_f u_f(t) + ∫ m_f a_f dt. That is less, by ∫ m_f |a_f| dt for a decelerating front, because each element left with its larger, earlier velocity.

So the gap between the closures is the time integral of the inertial pressure m_f|a_f|. It is large when m_f is a large fraction of the shocked mass and the front decelerates hard: ω < 0, where u_p ∝ t^((ω+c₃)/(2−ω)) falls off fastest.

### 1.3 The original patch is the massive closure

`AblationSolver` drives PistonShock with p_f at m = 0 and cuts it at m_f. In PistonShock the slab 0 < m < m_f is shocked target matter between the piston and the cut, moving at about u_p. Momentum beyond the cut is then

  Π_T(original) = ∫₀ᵗ p_f dt′ − ∫₀^{m_f} u_PS dm,                  (4)

and the slab's momentum ∫₀^{m_f} u_PS dm ≈ m_f u_f(t). That is the massive closure's budget, to the extent the slab moves uniformly. The 20% pressure jump at the cut in Test 4 (`why_the_solver_differs.md` §3.1b) is the slab's deceleration, the term m_f|a_f|. The patch was never a mass-bookkeeping error: it assumes the ablated layer decelerates with the front.

---

## 2. Fix B: solve the target with the moving front, numerically

### 2.1 Theoretical basis

The restriction of any flow to m > m_f(t) is a solution of the same hydro equations on a domain whose left edge moves through the matter. When the front moves subsonically into the shocked matter (ṁ_f < ρc there), exactly one characteristic enters the domain through that edge. One boundary condition, the pressure p_b(t), then fixes the solution beyond it, given the initial state ρ₀x^(−ω) at rest. So:

- driven with the true p_b(t) and m_f(t), a numerical target reproduces the true target exactly, whatever the ablated layer does;
- driven with a closure's p_b, it gives that closure's target, with no further approximation.

B therefore tests closures, and the target model itself, separately from the self-similar PistonShock. It is not a new analytic solution.

### 2.2 What the code does

`src/menahem_new/legacy/moving_front_shock.py`, class `MovingFrontShock`:

- **Mesh and scheme.** The target ρ₀x^(−ω) on a mesh uniform in mass (3000 cells to 1.4 m_s), as a staggered von Neumann–Richtmyer leapfrog: node velocities u, cell volumes v and energies e, quadratic plus linear artificial viscosity q. The energy update is implicit in p for the ideal gas: e₁ = (e₀ − (½p₀ + q) Δv) / (1 + ½ r Δv / v₁). The right edge is a rigid wall far ahead of the shock.
- **The moving front.** Each step the front node is k = round(index of m_f(t)) (`_front_node`). Cells below it are removed from the integration, and the front node gets a = (p_b − P_k)/(node mass). In the massless closure the node mass is ½dm_k and p_b = p_f(t), and swallowed cells take their momentum with them. In the massive closure it is m_f(t) + ½dm_k, so p_b = p_f − m_f a_f follows from the front node's own inertia. When the front jumps cells, the new front node takes the swallowed nodes' mass-weighted mean velocity: m_f(t) already grows continuously, which adds the swallowed matter at the front velocity.
- **Time step.** The minimum of the acoustic CFL, an acceleration limit (cold matter has no sound speed), the front sweeping at most half a cell, and 0.01 t. The run starts at 10⁻⁶ t_end. Two bugs taught these limits: with a front pressure of 0 nothing limited dt and the run stepped over the push, and starting at 10⁻³ t_end missed 2% of the impulse.
- **The patched profile.** `PatchedAblationSolver` puts the rigid heat solution on the front: its positions shifted to the front node's x, its velocities by the front's u_f. For the massive closure its pressure is shifted by −m a_f, with a_f a smoothed derivative of the front's velocity history.

### 2.3 What B showed

- **The target model is right.** Driven by PistonShock's own pressure at the cut, p_PS(m_f(t), t), B reproduces PistonShock's shock to 0.1–0.2% (Tests 1, 3–6). Driven by the diffusion run's own m_heat(t) and front pressure, it reproduces the simulated shock to 0.3–1.8% in every gold case with a clean front history.
- **B‑massive ≈ the original**, to within 1–2% (§1.3, numerically).
- **B‑massless with the rigid heat drive** cuts Test 4's shock error from −12.5% to +5.9%, and ω = −0.75's from −16.2% to +3.1%. At ω = 0 it is 3.8% behind the simulation, where the original is at +0.6%.

### 2.4 Why it is legacy

It integrates the target's hydro in (m, t), so a curve built on it is a hybrid and can't benchmark IMC and diffusion. Even as a reference it needs a closure, and neither closure is right across ω (§4).

---

## 3. Fix C: the same question in closed form

`src/menahem_new/legacy/matched_piston_patch.py`, class `MatchedPistonPatch`.

### 3.1 The idea

Keep both pieces self-similar. The target stays a PistonShock with the piston at m = 0, but with an effective amplitude: p_eff(t) = P_eff(t) t^(c₃) in place of p_f(t). P_eff is chosen so that the target beyond the cut carries what a closure says it should. Two choices are implemented:

- **`match="impulse"` (C_impulse).** Momentum beyond the cut equals the massless budget (2) with p_b = p_f:
    P_eff t^(1+c₃)/(1+c₃) − ∫₀^{m_f} u_PS(P_eff) dm = ∫₀ᵗ p_f dt′ − ∫₀ᵗ ṁ_f u_f dt′.     (5)
  The left side is PistonShock(P_eff)'s total impulse minus the slab it discards, as in (4). On the right, u_f is taken as PistonShock(P_eff)'s velocity at the cut.
- **`match="pressure"` (C_pressure).** Quasi-static and local: p_PS(m_f; P_eff, t) = p_f(t), so the target starts at the heat layer's front pressure.

In both, the heat layer is shifted to PistonShock(P_eff)'s position *and velocity* at the cut (`solve`), so x and u are continuous at m_f.

### 3.2 How (5) is solved (`_march`)

The right side of (5) depends on the history of u_f, so it can't be solved at one time. `_march` steps along 60 log-spaced times from 10⁻³ t_end. At each time it:

1. adds the push increment p₀ₛ (t_i^(1+c₃) − t_{i−1}^(1+c₃))/(1+c₃);
2. adds the loss increment ½(u_f,prev + u_f)·Δm_f, with u_f from the candidate amplitude;
3. finds the P_eff at which the left side equals the running total (`brentq` in log P_eff; `_layer_momentum` is a trapezoid of u_PS over the slab).

Each step is a root-find over self-similar profiles: no field is integrated, which is what keeps C analytic.

**The early phase needs two guards.** At early times the heat front runs ahead of any shock its pressure could drive: m_f/m_s ∝ t^(−0.42) at ω = −0.5, so m_f > m_s(p_f) as t → 0. There the swallowed matter takes nearly all the impulse, and (5) as written has no monotone root. A larger P_eff raises u_f, and the loss overtakes the push. Hence:

- the loss increment is capped at the push increment;
- P_eff is bounded below by the amplitude that puts the shock exactly on the cut, from m_s ∝ P^((1−ω)/(2−ω)). Below it the target has received nothing.

With both, have − target is monotone in P_eff and the root is unique.

### 3.3 What C assumes, explicitly

1. **The massless closure** (the right side of (5)).
2. **The rigid heat solution's m_f(t) and p_f(t)** as the drive.
3. **Quasi-static self-similarity.** At each t the target is the PistonShock profile of a constant amplitude P_eff(t), as if the drive had always had that amplitude. The momentum beyond the cut is right; the profile shape is that of a different drive history.
4. **u_f = u_PS(m_f; P_eff).** The velocity of the matter crossing the front is the self-similar profile's velocity at the cut.
5. **Only momentum is matched.** The impulse match does not make p continuous: the target starts at p_PS(m_f; P_eff) ≠ p_f. The pressure match makes p continuous but not the momentum.

---

## 4. Why neither is a consistent improvement

### 4.1 The closure itself moves with ω

The decisive measurement is the target's momentum beyond the analytic cut m_f at 2 ns, from the paper-bath diffusion runs. They conserve momentum (vacuum surface, p = 0 at m = 0). Units: 10³ g cm⁻¹ s⁻¹ per cm². `python -m rad_hydro_sim.verification.legacy.patching_poc --cases test1_full test4_full --omega-scan -0.25 -0.75 --momentum`.

| case | ω | push ∫p_f dt | massive (original) | massless (C_impulse) | simulation | θ |
|---|---|---|---|---|---|---|
| Test 1 | 0 | 7.25 | 6.78 | 6.49 | 6.83 | 1.15 |
| ω scan | −0.25 | 7.25 | 6.23 | 5.22 | 5.71 | 0.49 |
| Test 4 | −0.5 | 7.25 | 5.39 | 3.56 | 4.24 | 0.37 |
| ω scan | −0.75 | 7.25 | 4.22 | 1.92 | 2.60 | 0.29 |

θ = (simulation − massless)/(massive − massless): 1 is the massive closure (the original), 0 the massless one (C).

- **At ω = 0 the target behaves massively**, i.e. the original patch is right. The gap between the closures there is only 4% of the push, comparable to other small effects, so θ ≈ 1 within errors.
- **As ω falls, the target moves toward the massless side**, but never reaches it: at ω = −0.75 it keeps 29% of the gap.

Physically, θ is the fraction of the deceleration impulse ∫ m_f |a_f| dt that the ablated layer hands back to the target through its pressure: how much of the layer stays dynamically coupled to the front. A fixed closure fixes θ, at 1 (the original, B‑massive) or 0 (B‑massless, C). Each is right at one end of the ω range and wrong at the other. That is the main reason for the pattern in the shock errors (2 ns, diffusion vs. patch, + = diffusion ahead):

| case | original (θ = 1) | B‑massless (θ = 0) | C_impulse (θ = 0) | C_pressure | B, simulation's own drive |
|---|---|---|---|---|---|
| Test 1 (ω = 0) | **+0.6** | +3.8 | +2.8 | +2.0 | +1.3 |
| ω = −0.25 | −5.5 | +5.1 | +3.4 | **−0.8** | +1.8 |
| Test 4 (ω = −0.5) | −12.5 | +5.9 | +6.7 | **−0.9** | +0.9 |
| ω = −0.75 | −16.2 | **+3.1** | +12.9 | +9.2 | +0.5 |

The massive patches leave the shock ahead of the simulation where θ < 1; the massless ones leave it behind where θ > 0. C_pressure's good Test 4 and ω = −0.25 values are not a better closure: it matches the cut's *pressure* to p_f, which at those ω happens to land the shock near the simulation's. At ω = −0.75 it is +9% at 2 ns and +26–32% at 0.5–1 ns, because it applies the current m_f/m_s to the whole history (the quasi-static assumption, §4.3).

Only B with the simulation's own drive is right everywhere. It needs no closure, because the simulation's front pressure already contains the true θ, and its heat front history too.

### 4.2 The drive itself is off for ω < 0

Both fixes keep the rigid heat solution's m_f(t) and p_f(t). For ω < 0 the simulated heat front runs ahead of it: m_f +6% in Test 4 and +16% at ω = −0.75 (more at early times). This is converged under mesh refinement (`../omega_sweep/README.md`). A front that has swallowed more mass changes both terms of (2), and no choice of θ can absorb that. It is a separate approximation of the solver: likely the heat solution's front condition ṽ(ξ_f) = 0, a front against an infinitely dense wall, where for ω < 0 the target near the surface is light. This is untested.

### 4.3 What C adds on top of the massless closure

Even granting θ = 0, C is only an approximation of B‑massless:

- **Quasi-static shape (assumption 3).** For ω ≥ −0.25 C tracks B‑massless within 1–2%, and at ω = −0.5 within 1–4%. At ω = −0.75 it falls 9–13% short, because most of the history there is the early phase.
- **The early phase is not modelled, only guarded.** The lower bound (shock on the cut) is active in 0 of the 60 march steps in Test 1, 14 at ω = −0.25 (to 0.009 ns), 34 in Test 4 (to 0.095 ns) and 46 at ω = −0.75 (to 0.39 ns). P_eff/p_f starts at 1.2, 2.6, 14 and 47 there, and falls to 0.96, 0.85, 0.72 and 0.63 at 2 ns. In that phase the target model is "PistonShock with whatever amplitude puts its shock on the cut", which is a bound, not physics. The original has the same problem: its own shock lies inside the ablated mass until the same times (0.095 ns in Test 4, 0.35 ns at ω = −0.75).
- **A pressure step at the cut (assumption 5).** The impulse match fixes momentum, not pressure. The target starts at p/p_f = 0.985, 0.931, 0.897 and 0.961 (Test 1, −0.25, Test 4, −0.75), against the original's 1.027, 1.086, 1.201 and 1.385. The original's step points up (the slab's inertia), C's points down. This is why C's front pressure is further from the simulations than the original's (+23% vs. +10% in Test 4), although its shock momentum is closer at that ω.
- **u_f from the profile (assumption 4)** makes the loss term depend on P_eff itself. That is what makes the early-time root non-monotone without the guards.

### 4.4 What a consistent improvement would need

1. **θ(ω, t) from theory:** the ablated layer's response to the front's deceleration. This is a first-order perturbation of the heat wave in the decelerating frame, ε = m_f|a_f|/p_f. The rigid solution gives θ = 0 for the layer's pressure; the massive closure assumes its full response. The simulations say the truth is in between and ω-dependent.
2. **The drive's own error (§4.2):** a heat wave with a finite-density inflow at the front.
3. **For C specifically:** a perturbation of PistonShock with a moving-mass boundary in place of the quasi-static amplitude, and an explicit model of the early, supersonic phase of the front.

Without (1) and (2), any fixed closure is right at one end of the ω range and wrong at the other. With the paper bath, the original is right where θ ≈ 1, i.e. ω ≥ 0. That is why the reference stays the original `AblationSolver`, with the bath chosen per material (`../bath_calibration.md`).
