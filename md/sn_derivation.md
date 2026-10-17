# Discrete ordinates ($S_N$) for the gray Marshak-wave problem: from the transport equation to `src/sn/sn.py`, with proofs

This document is the $S_N$ companion of `md/imc_mc_derivation.md`. It covers
five things:

1. **Where the method comes from** (Section 1): the foundational papers, and
   which of them plays the role that Fleck & Cummings play for IMC.
2. **How the paper's transport problem maps onto the variables of `sn.py`**
   (Section 2). `sn.py` is written in neutron-transport language (particle
   density, speed $v$, cross sections $\Sigma_t,\Sigma_s$). The paper is
   written in radiation language ($I$, $E_r$, $k_a$, $k_s$, $T_{\rm bath}$).
   The map between them is exact, and two numerical checks confirm the
   normalization.
3. **The chain of discretizations** (Sections 3–7): time linearization (the
   same as F&C with $\alpha=1$), backward Euler, the $S_N$ angular quadrature,
   diamond differencing in space with the set-to-zero fixup, and source
   iteration. Each step is marked as exact algebra or as an approximation.
   Each step is tied to the lines of `sn.py` that implement it.
4. **Theorems for the discrete scheme** (Section 8): convergence of source
   iteration, exact discrete energy conservation, positivity of the material
   energy, and the thick (diffusion) limit.
5. **What `sn.py` does not do yet** (Section 10), which is the bridge to
   `md/full_plan_sn_comparison.md`.

The spherical sweep (Lewis–Miller) is derived in Section 9 for completeness.
The paper is planar, so the sphere is not on the critical path.

Labels such as (R), (L), (T$_N$), (DD) and the numbering of lemmas and
theorems are this document's own. "Paper Eq. `x`" refers to the `\label{x}`
in `subsonic_self_similar_paper/manuscript.tex`.

---

## 0. Scope

```
  paper's transport problem: Eqs. Treq + tr_mat + Ibath_tr (nonlinear in T)
        │  (A) linearize the emission over one step            ── Section 3
        │      identical to F&C with α = 1 (approximations L1, L2)
        ▼
  linear transport equation (L), frozen coefficients, Fleck factor f
        │  (B) backward Euler in time  (approximation L3)       ── Section 4
        │  (C) S_N: Gauss–Legendre ordinates (approximation L4) ── Section 5
        │  (D) diamond difference in space + fixup (L5)         ── Section 6
        ▼
  algebraic system for ψ_{m,i}^{n+1}: one per step
        │  (E) source iteration = repeated transport sweeps     ── Section 7
        │      PROVED: converges geometrically, rate ≤ c_eff < 1
        ▼
  ψ^{n+1}, φ^{n+1}, then the material update (energy-conserving form)
        │  PROVED: total energy conserved to the SI tolerance;   ── Section 8
        │          u_m stays positive for β_EOS < 4
        ▼
  deterministic S_N solution (no statistical noise)
        │  N → ∞, Δx → 0, Δt → 0: converges to the solution of the
        │  nonlinear problem under smoothness hypotheses (the same open
        │  points as IMC's Section 8 in the cold, opaque front region)
        ▼
  true solution of paper Eqs. Treq + tr_mat
```

The key contrast with IMC is the middle box. IMC solves (L) **exactly in angle,
space and time within the step**, and then pays with noise. $S_N$ discretizes
(L) in all three variables and gets a deterministic answer with
discretization error. **Both start from the same linear equation (L).** So
once the linearization is shared, a disagreement between IMC and $S_N$
measures only noise plus $S_N$ discretization error. That is what makes $S_N$
the natural deterministic reference for the IMC.

---

## 1. Foundational papers: is there an "F&C of $S_N$"?

There is no single founding paper that does for $S_N$ what Fleck & Cummings
did for IMC. The method was assembled in three layers:

| Layer | What it contributed | Reference |
|---|---|---|
| **The idea of discrete ordinates** | Replace $\int d\mu$ by a Gauss quadrature and solve a coupled ODE system, one per direction. Developed analytically for steady stellar atmospheres. | G. C. Wick, *Z. Physik* **121**, 702 (1943); S. Chandrasekhar, *Astrophys. J.* **100**, 76 (1944) and the book *Radiative Transfer* (Oxford, 1950), Ch. II–III. |
| **The numerical $S_n$ method** | The name "$S_n$", the finite-difference cell balance, and the **transport sweep** in the direction of flow. Built at Los Alamos for neutron criticality. This is the closest thing to an "F&C of $S_N$". | B. G. Carlson, *Solution of the transport equation by $S_n$ approximations*, Los Alamos report LA-1599 (1953); LA-1891 (1955). The canonical review: B. G. Carlson & K. D. Lathrop, "Transport theory: the method of discrete ordinates", in *Computing Methods in Reactor Physics* (Gordon & Breach, 1968). |
| **The textbook form used by `sn.py`** | Diamond difference, the sphere's angular-redistribution coefficients $\alpha$, the starting direction, source iteration and its Fourier analysis. | E. E. Lewis & W. F. Miller, *Computational Methods of Neutron Transport* (Wiley, 1984), Ch. 3–4. |

The analysis that matters for this problem came later:

- **Positivity vs. accuracy of the spatial scheme**, and the set-to-zero
  fixup: K. D. Lathrop, *J. Comput. Phys.* **4**, 475 (1969).
- **The thick diffusion limit** (does the discrete scheme give the right
  diffusion equation when cells are many mean free paths thick?):
  E. W. Larsen, J. E. Morel & W. F. Miller, *J. Comput. Phys.* **69**, 283 (1987).
- **Acceleration of source iteration**: R. E. Alcouffe, *Nucl. Sci. Eng.*
  **64**, 344 (1977) (DSA); M. L. Adams & E. W. Larsen, *Prog. Nucl. Energy*
  **40**, 3 (2002) (review).
- **$S_N$ for nonlinear thermal radiation.** No new linearization was invented
  here. Deterministic codes use **the same Fleck-type linearization as IMC**,
  in its backward-Euler ($\alpha=1$) form; Section 3 shows why it is the same.
  Two references make this explicit: E. W. Larsen, "A grey transport
  acceleration method for time-dependent radiative transfer problems",
  *J. Comput. Phys.* **78**, 459 (1988); J. E. Morel, E. W. Larsen &
  M. K. Matzen, *JQSRT* **34**, 243 (1985).
- **The $S_N$ runs in the paper** use R. G. McClarren & T. S. Haut,
  "Data-driven acceleration of thermal radiation transfer calculations with
  the dynamic mode decomposition and a sequential singular value
  decomposition", *J. Comput. Phys.* **448**, 110756 (2022) (bib key
  `mcclarren2022data`). It is an $S_N$ thermal-radiation solver whose source
  iterations are accelerated by DMD.

> Citation details above are from memory, except `mcclarren2022data`, which is
> copied from `datab.bib`. Check volume and page numbers before any of them
> enter the manuscript.

`sn.py` itself is a port of a Fortran code (`sn1_spherical.f90`, referenced in
its docstrings). That source is not in the repository. The fixup and
negative-source schemes are copied from it line by line.

---

## 2. The problem and its normalization in `sn.py`

### 2.1 The paper's transport problem

Paper Eqs. `Treq`, `tr_mat`, `Ibath_tr`, planar:

$$
\Big(\frac1c\partial_t+\mu\partial_x\Big)I+(k_a+k_s)I=\frac{ac}{4\pi}k_aT^4+\frac12k_s\!\int_{-1}^1\!I\,d\mu',
\qquad
\partial_tu_m=k_a\Big[2\pi\!\int_{-1}^1\!I\,d\mu'-acT^4\Big],
$$

$$
I(0,\mu>0,t)=\frac{ac}{4\pi}T_{\rm bath}^4(t),\qquad
u_m=fT^{\beta}\rho^{1-\mu_{\rm EOS}}-aT^4 .
$$

$k_t=k_a+k_s=\rho\kappa_R$, and with the diffusion code's coupling factor
$\chi$: $k_a=\chi\rho\kappa_R$, $k_s=(1-\chi)\rho\kappa_R$ (paper Eq. `scatt`).
**All planned tests use $\chi=1$, so $k_s=0$.** The only scattering in the
discrete problem is the effective scattering created by the linearization.

Notation clash: the paper's $\beta$ (EOS exponent) and $\mu$ (density exponent)
collide with F&C's $\beta$ and the direction cosine $\mu$. In this document
$\beta_{\rm EOS}$ and $\mu_{\rm EOS}$ are the paper's exponents, $\beta_F$ is
F&C's ratio, and $\mu$ is always a direction cosine.

### 2.2 The equation `sn.py` solves

`sn.py` solves, for an angular flux $\psi(x,\mu,t)$ and scalar flux
$\phi=\tfrac12\int_{-1}^1\psi\,d\mu$ (`scalar_flux`, line 61):

$$
\frac1v\partial_t\psi+\mu\partial_x\psi+\Sigma_t\psi=\Sigma_s\,\phi+S(x).
\tag{N}
$$

Two features of the normalization are easy to get wrong:

* The factor $\tfrac12$ in $\phi$ means **an isotropic $\psi$ equals $\phi$**.
  The initial condition `np.tile(v * n0, (n_mu, 1))` (line 492) relies on this.
* The external source $S$ enters (N) **without** a factor $\tfrac12$.
  Integrating (N) over $\mu$ and dividing by 2 gives the balance law
  $$
  \frac1v\partial_t\phi+\partial_xJ+(\Sigma_t-\Sigma_s)\phi=S,\qquad J=\tfrac12\!\int\!\mu\psi\,d\mu .
  \tag{N0}
  $$
  So $n=\phi/v$ is a particle density and $S$ is particles per volume per time.

**Numerical check** (scratch script, not committed). In a 200-mfp slab with
8 ordinates and $\Delta t=0.01$:
(i) pure absorber, $\phi_{\rm centre}(t=1)=0.369713$, against the
backward-Euler value $(1+\Sigma_tv\Delta t)^{-100}=0.369711$;
(ii) pure scatterer with $S=1$, $\phi_{\rm centre}(1)=0.9999995$, against
$vSt=1$.

### 2.3 Lemma 0 (the dictionary)

Set

$$
\boxed{\;\psi=4\pi I,\quad \phi=cE_r,\quad v=c,\quad \Sigma_t=k_t,\quad \Sigma_s=k_s,\quad S=k_a\,acT^4\;}
$$

Then (N) is the paper's transport equation, $n=\phi/v=E_r$ is the radiation
energy density, $J=F$ is the radiative energy flux, and the material equation
reads $\partial_tu_m=k_a(\phi-acT^4)$.

*Proof.* Multiply `Treq` by $4\pi$. The emission becomes
$4\pi\cdot\frac{ac}{4\pi}k_aT^4=k_a\,acT^4=S$. The scattering term becomes
$4\pi\cdot\frac12k_s\int I=\frac12k_s\int\psi=k_s\phi$. Further,
$\phi=\frac12\int4\pi I\,d\mu=2\pi\int I\,d\mu=cE_r$ by the paper's definition of
$E_r$, and $J=\frac12\int\mu\,4\pi I=2\pi\int\mu I=F$. In `tr_mat`,
$2\pi\int I\,d\mu'=\phi$. ∎

The boundary condition becomes $\psi(0,\mu>0,t)=acT_{\rm bath}^4(t)$, and the
incoming current is

$$
J^+(0)=\tfrac12\!\int_0^1\!\mu\,acT_{\rm bath}^4\,d\mu=\tfrac14acT_{\rm bath}^4=\sigma_{\rm SB}T_{\rm bath}^4=F_{\rm inc},
$$

which is paper Eq. `f_inc_bath`. The initial condition (cold, equilibrium) is
$\psi=acT_0^4$, i.e. `n0` $=aT_0^4$.

### 2.4 Why the transport BC is the Marshak BC (intuition for the surface layer)

If $\psi$ near the surface is close to its P1 form $\psi\approx\phi+3\mu J$,
then $J^+=\frac12\int_0^1\mu(\phi+3\mu J)\,d\mu=\phi/4+J/2$. With $\phi=cE$
this is $\frac4cF_{\rm inc}=E+\frac2cF$, paper Eq. `marshak_bc_general`.
So the diffusion code's Marshak BC is the transport BC **under the P1
assumption at the surface**. $S_N$ makes no such assumption. Any
$S_N$-vs-diffusion difference in the surface temperature is the transport
boundary layer: a few mean free paths thick, and largest where the surface
region is optically thin. That happens in Test 4 ($\omega=-0.5$, density
$\to0$ at the surface) and Test 6 (Al, low opacity); see the plan.

### 2.5 Exact energy balance of the continuous problem

Integrating (N0) and the material equation over $[0,L]$:

$$
\frac{d}{dt}\int_0^L(E_r+u_m)\,dx=F(0,t)-F(L,t).
$$

Every discrete step below is checked against a discrete version of this
identity (Theorem 2).

---

## 3. Step A: linearize the emission over a step (the same as F&C with $\alpha=1$)

This is the step that makes the problem linear, so that (N) applies. It is
the same idea as `imc_mc_derivation.md` §§1.2–1.7. Here it is written
directly in backward-Euler form, since that is what an $S_N$ code uses.

**Step A1 (change of variable, exact).** $u_r=aT^4$,
$\beta_F=\partial u_r/\partial u_m=4aT^3/C_v$, with

$$
C_v=\frac{\partial u_m}{\partial T}\Big|_\rho=\beta_{\rm EOS}fT^{\beta_{\rm EOS}-1}\rho^{1-\mu_{\rm EOS}}-4aT^3 .
$$

**Step A2 (backward Euler for the material).**

$$
u_m^{n+1}-u_m^n=\Delta t\,k_a\big(\phi^{n+1}-c\,u_r^{n+1}\big).
$$

**Step A3 (linearize the EOS over the step; approximation L1).**
$u_r^{n+1}\approx u_r^n+\beta_F^n(u_m^{n+1}-u_m^n)$, with $\beta_F$ and $k_a$
frozen at $T^n$ (F&C's A1). Substituting A2:

$$
u_r^{n+1}=u_r^n+\beta_F\Delta t\,k_a(\phi^{n+1}-cu_r^{n+1})
\;\Rightarrow\;
u_r^{n+1}=\frac{u_r^n+\beta_F\Delta t\,k_a\phi^{n+1}}{1+\beta_Fck_a\Delta t}.
$$

Define the **Fleck factor** $f=1/(1+\beta_Fck_a\Delta t)$, F&C's $f$ at
$\alpha=1$. Then

$$
ck_a\,u_r^{n+1}=f\,ck_a\,u_r^n+(1-f)\,k_a\,\phi^{n+1}.
\tag{A}
$$

**Step A4 (use this emission in the transport equation; approximation L2).**
The emission $S=ck_au_r$ in the transport equation is taken at its
end-of-step value (A), with $\phi^{n+1}$ treated as the current $\phi$:

$$
\boxed{\frac1c\partial_t\psi+\mu\partial_x\psi+k_t\psi=\underbrace{\big[(1-f)k_a+k_s\big]}_{\Sigma_s^{\rm eff}}\phi+\underbrace{f\,k_a\,acT_n^4}_{S^{\rm eff}}}
\tag{L}
$$

with $\Sigma_t=k_t$. In `sn.py` terms:

$$
\texttt{sigma\_t}=k_t,\qquad \texttt{sigma\_s}=(1-f)k_a+k_s,\qquad \texttt{source\_density}=f\,k_a\,acT_n^4,\qquad v=c .
$$

This is **F&C (2.1) at $\alpha=1$**, term by term. The effective absorption is
$\Sigma_t-\Sigma_s^{\rm eff}=fk_a$.

**Step A5 (material update, energy-conserving form; exact given $\phi^{n+1}$).**
Substituting (A) into A2:
$k_a(\phi^{n+1}-cu_r^{n+1})=fk_a(\phi^{n+1}-cu_r^n)$, so

$$
\boxed{u_m^{n+1}=u_m^n+\Delta t\,f\,k_a\big(\phi^{n+1}-acT_n^4\big)}
\tag{M}
$$

and $T^{n+1}$ is obtained by inverting $u_m(T,\rho)$ with Newton's method.
This is F&C (1.18) with exact EOS inversion. Theorem 2 shows that pairing (M)
with the discrete transport step conserves energy exactly.

**Ledger.**

| | What | Same as IMC? |
|---|---|---|
| L1 | $\beta_F$, $k_a$, $k_t$ frozen at $T^n$ | Yes (F&C A1) |
| L2 | Linearized $u_r^{n+1}$, emission treated implicitly through $f$ | Yes (F&C A2–A4 with $\alpha=1$) |
| L3 | Backward Euler for $\partial_t\psi$ | **No.** IMC integrates (L) exactly in time within the step. |
| L4 | $N$ discrete directions | **No.** IMC is continuous in angle. |
| L5 | Diamond difference in space, with fixup | **No.** IMC is continuous in space; both share the cell-wise constant $T_n$ in $S^{\rm eff}$. |

L1 and L2 can be removed by Newton iteration within the step: re-linearize
around the latest $T^{*}$ and repeat until $T^*$ stops changing. That gives
fully implicit backward Euler. `sn.py` has no such loop; the plan makes it an
option (S10).

**Intuition.** $f$ is the fraction of absorbed energy that stays in the
material during the step. The remaining fraction $1-f$ is re-emitted within
the step, isotropically and at the same place, so it acts as scattering. The
estimates in the plan give $f\approx0.14$–$1$ for these materials and time
steps. Where $f$ is small, the effective scattering ratio
$c_{\rm eff}=\Sigma_s^{\rm eff}/\Sigma_{\rm eff}$ is close to 1. That number
controls source iteration (Section 7).

---

## 4. Step B: backward Euler in time (approximation L3)

Within the step, (L) has constant coefficients. Backward Euler gives

$$
\mu\partial_x\psi^{n+1}+\Big(\Sigma_t+\frac{1}{v\Delta t}\Big)\psi^{n+1}=\Sigma_s\phi^{n+1}+S+\frac{\psi^n}{v\Delta t}.
$$

So a time step is a **steady** transport problem with an augmented total cross
section and an extra source:

$$
\Sigma_{\rm eff}=\Sigma_t+\frac{1}{v\Delta t}\quad(\texttt{sig\_eff}, \text{line 490}),\qquad
\text{rhs}=q+\frac{\psi^n}{v\Delta t}\quad(\text{line 342}).
$$

The term $1/(v\Delta t)$ is **time absorption**. Particles that "do not make it
to the end of the step" are removed and then re-injected at the next step
through $\psi^n/(v\Delta t)$. Its size relative to $\Sigma_t$ is the number of
mean free paths a photon travels in one step, $(\Sigma_tc\Delta t)^{-1}$. It
makes every time step strictly subcritical (Section 7).

Two consequences for the comparison:

* **$\psi^n$ is stored per ordinate.** `psi_old` (line 324) is the
  angular census. The angular shape of the radiation is carried between
  steps, as IMC carries particle directions. A driver that restarts from an
  isotropic $\psi$ each step (as `run_sn`'s `n0` argument would) would throw
  this away.
* **The time error is $O(\Delta t)$ and diffusive.** Backward Euler damps the
  wave front. Its leading error is
  $\tfrac12v\Delta t\,\partial_t^2\psi$ per unit length, which smears a
  moving front over roughly $v_{\rm front}\Delta t$. IMC's in-step transport has
  no such smearing, only its L1/L2 error. That is a known, systematic
  IMC-vs-$S_N$ difference at large $\Delta t$.

---

## 5. Step C: discrete ordinates in angle (approximation L4)

### 5.1 The quadrature

`make_quadrature` (line 37): Gauss–Legendre nodes and weights on $[-1,1]$,
$N=$ `n_mu` even, sorted increasing, $\sum w_m=2$. The method collocates (N)
at $\mu_m$ and replaces $\phi$ by $\tfrac12\sum_mw_m\psi_m$.

Properties used below, all from exactness of Gauss quadrature for polynomials
of degree $\le2N-1$:

1. $\tfrac12\sum w_m=1$: an isotropic $\psi$ equals $\phi$, exactly.
2. $\tfrac12\sum w_m\mu_m=0$ and $\tfrac12\sum w_m\mu_m^2=\tfrac13$ (exact for
   $N\ge2$). The Eddington factor of an isotropic field is exactly $\tfrac13$,
   so the **diffusion coefficient in the thick limit is exactly $1/(3\Sigma_t)$**
   for every $N$ (Section 8.4).
3. In slab geometry, $S_N$ with Gauss quadrature is equivalent to the
   spherical-harmonics $P_{N-1}$ method (Lewis & Miller, Ch. 3). For example,
   $S_2$ ($\mu=\pm1/\sqrt3$) is the telegraph (P1) equation. Increasing $N$
   adds angular moments.
4. $N$ even means no ordinate at $\mu=0$. Every ordinate streams, so each
   ordinate can be swept from one boundary.

### 5.2 The half-range defect at the bath boundary

Gauss–Legendre quadrature on $[-1,1]$ is **not** exact for half-range
integrals. The incoming current of an isotropic bath is
$J^+=\tfrac12\sum_{\mu_m>0}w_m\mu_m\,\psi_b$, and $\tfrac12\sum_{\mu_m>0}w_m\mu_m$
is not exactly $\tfrac14$:

| $N$ | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| $\tfrac12\sum_{\mu>0}w\mu$ | 0.28868 | 0.26063 | 0.25288 | 0.25076 | 0.25020 | 0.25005 |
| relative error | +15.5% | +4.3% | +1.15% | +0.30% | +0.078% | +0.020% |

With $\psi_b=acT_{\rm bath}^4$ imposed naively, $S_8$ injects 1.15% too much
energy from the bath. That raises $T_s$ by about 0.3% and moves the front.
This is a systematic bias of the size of the effects the paper wants to
measure. There are two remedies (plan, S2):

* **(a) normalize the bath:** $\psi_b=F_{\rm inc}\big/\big(\tfrac12\sum_{\mu_m>0}w_m\mu_m\big)$,
  so the discrete incoming current is exactly $\sigma_{\rm SB}T_{\rm bath}^4$;
* **(b) half-range (double-Gauss) quadrature:** Gauss–Legendre on $[0,1]$ and
  on $[-1,0]$ separately. Half-range moments are then exact, which is the
  standard choice for problems driven at a surface.

### 5.3 The scattering source (general, and why it is trivial here)

`scattering_source` (line 83) implements the azimuthally integrated
Legendre expansion

$$
Q_s(\mu_m)=\Sigma_s\sum_{l=0}^{L}(2l+1)f_l\,P_l(\mu_m)\,\phi_l,\qquad
\phi_l=\tfrac12\sum_{m'}w_{m'}P_l(\mu_{m'})\psi_{m'} .
$$

*Derivation.* With a phase function $p(\mu_0)=\sum_l\frac{2l+1}{4\pi}f_lP_l(\mu_0)$,
the addition theorem $\int_0^{2\pi}P_l(\mu_0)\,d\varphi'=2\pi P_l(\mu)P_l(\mu')$
gives the slab kernel $K(\mu,\mu')=\sum_l\frac{2l+1}{2}f_lP_l(\mu)P_l(\mu')$.
In the normalization of (N), the isotropic kernel is
$\Sigma_s\phi=\Sigma_s\tfrac12\int\psi$, so
$Q_s=\Sigma_s\int K\psi\,d\mu'=\Sigma_s\sum_l(2l+1)f_lP_l(\mu)\phi_l$. ∎

**For this problem only $l=0$ is used.** Both physical scattering ($k_s$ is
Thomson-like and isotropic in the gray model) and the effective scattering
$(1-f)k_a$ are isotropic. So `scatter_model="isotropic"`, `lmax=0`, and
$Q_s=\Sigma_s\phi$.

**Lemma 1 (the negative-source fixups are no-ops here).** If $Q_s$ is
isotropic and $S\ge0$, then `fixup_scale_clip` (line 96) and
`fixup_min_blend` (line 116) leave $q$ unchanged, up to a
relative change of $10^{-80}$.

*Proof.* $q_m=q_{\rm iso}\ge0$ for all $m$. In scale_clip, $S_{\rm neg}=0$, so
`fac`$=1$ and $q_m\leftarrow\max(0,q_{\rm iso}+(q_m-q_{\rm iso}/(1+10^{-80})))=q_m(1+O(10^{-80}))$.
If $q_{\rm iso}\le10^{-40}$, `fac`$=0$ and $q_m\leftarrow\max(0,q_{\rm iso})=q_m$.
In min_blend, `src_neg` stays $0\ge\min(0,q_{\rm iso})=0$, so the cell is
skipped. If $q_{\rm iso}\le10^{-40}$, $q_m\leftarrow q_{\rm iso}=q_m$. ∎

The anisotropic machinery (HG, `scatter_fl`, the two Fortran fixups) can be
ignored for the paper.

---

## 6. Step D: diamond difference in space (approximation L5)

### 6.1 The cell balance (exact)

Integrate the $m$-th ordinate equation over cell $i=[x_{i-1/2},x_{i+1/2}]$ and
divide by $\Delta x_i$:

$$
\frac{\mu_m}{\Delta x_i}\big(\psi_{m,i+1/2}-\psi_{m,i-1/2}\big)+\Sigma_{\rm eff}\,\bar\psi_{m,i}=\bar q_{m,i}.
\tag{B}
$$

$\bar\psi$ and $\bar q$ are cell averages. (B) is **exact**: no
approximation yet. It has three unknowns per cell and direction: the two face
values and the average. The incoming face value is known from the upwind
neighbour (or the boundary). One more relation, the closure, is needed.

### 6.2 The diamond closure

$$
\bar\psi_{m,i}=\tfrac12\big(\psi_{m,i-1/2}+\psi_{m,i+1/2}\big).
\tag{DD}
$$

It is the trapezoidal rule, exact for $\psi$ linear in the cell, so the
scheme is second-order accurate for smooth solutions. For $\mu_m>0$, with
$\psi_{\rm in}=\psi_{m,i-1/2}$ and $\psi_{\rm out}=2\bar\psi-\psi_{\rm in}$
substituted into (B):

$$
\bar\psi=\frac{\bar q+2|\mu|\psi_{\rm in}/\Delta x}{\Sigma_{\rm eff}+2|\mu|/\Delta x},\qquad
\psi_{\rm out}=\frac{2\bar q+(2|\mu|/\Delta x-\Sigma_{\rm eff})\psi_{\rm in}}{\Sigma_{\rm eff}+2|\mu|/\Delta x}.
$$

This is exactly lines 147–149 of `sweep_slab`. One loop serves both
directions: for $\mu<0$ the cell index runs right to left (line 146).

### 6.3 Positivity, and the set-to-zero fixup

From the formula for $\psi_{\rm out}$: **diamond difference is positive iff**

$$
\frac{\Sigma_{\rm eff}\Delta x}{|\mu|}\le2\qquad\text{or}\qquad\bar q\text{ is large enough}.
$$

In a cell thicker than two mean free paths along the ray, with weak source, DD
extrapolates linearly through an exponential and overshoots below zero. For
$S_8$, the smallest $|\mu|$ is 0.183, so cells with $\Sigma\Delta x>0.37$ can go
negative. In the cold gold ahead of the front ($T\approx300$ K, where
$\kappa_R\propto T^{-1.5}$ is enormous), $k_a\Delta x\gg1$ (of order $10^3$ for a
1000-cell mesh), so **the fixup is active there all the time**.

The fixup (lines 150–152) sets $\psi_{\rm out}=0$ and keeps the balance (B):

$$
\frac{\mu}{\Delta x}(0-\psi_{\rm in})+\Sigma_{\rm eff}\bar\psi=\bar q
\;\Rightarrow\;
\bar\psi=\frac{\bar q+|\mu|\psi_{\rm in}/\Delta x}{\Sigma_{\rm eff}}.
$$

This is exactly the code. The fixup is

* **conservative**: (B) still holds, so Theorem 2 survives;
* **positive**: $\bar\psi\ge0$ whenever $\bar q,\psi_{\rm in}\ge0$;
* **nonlinear and locally first order**: the scheme now depends on the
  solution, and in the fixed-up cell it behaves like the step scheme.

`time_loop` asserts $\psi\ge0$ after each sweep (line 355). With the fixup on and
$S\ge0$, $\psi_{\rm in}\ge0$, this assertion can only fail through round-off.

### 6.4 The sweep

For $\mu_m>0$, (B)+(DD) give $\psi_{\rm out}$ from $\psi_{\rm in}$ cell by
cell, from left to right, starting from the left boundary value. For
$\mu_m<0$ the sweep runs right to left. One sweep **inverts the streaming–collision
operator $\mathcal L=\mu\partial_x+\Sigma_{\rm eff}$ exactly** (for the discrete
scheme) at cost $O(N\cdot N_x)$, with no matrix.

**Boundary conditions in the code.** `psi_face = 0.0` at the start of every
ordinate (line 144): vacuum on both sides. The bath BC needs
`psi_face = psi_b` for $\mu_m>0$ (Section 5.2). This is the first change the
plan makes.

---

## 7. Step E: source iteration

### 7.1 The iteration

Write one step as $\psi=\mathcal L^{-1}(\Sigma_s\phi+s)$ with
$s=S+\psi^n/(v\Delta t)$, and $\phi=\mathcal D\psi$, where
$\mathcal D\psi=\tfrac12\sum w_m\psi_m$. The scattering source depends on the
unknown $\phi$, so the code iterates (`time_loop`, lines 328–367):

$$
\phi^{(\ell+1)}=\mathcal D\,\mathcal L^{-1}\big(\Sigma_s\phi^{(\ell)}+s\big).
$$

In words: build the source from the last $\phi$, sweep all ordinates, and
recompute $\phi$. Physically, iterate $\ell$ contains exactly the photons that
have scattered at most $\ell$ times within the step. This is the
deterministic analogue of following IMC particles collision by collision.

### 7.2 Theorem 1 (source iteration converges geometrically)

*Setting.* The exact streaming operator (continuous in $x$, discrete in $\mu$),
any slab $[0,L]$, incoming boundary data fixed, $\Sigma_s<\Sigma_{\rm eff}$.

*Claim.* With $c_{\rm eff}=\Sigma_s/\Sigma_{\rm eff}$:

$$
\|\phi^{(\ell+1)}-\phi^\star\|_\infty\le c_{\rm eff}\,\|\phi^{(\ell)}-\phi^\star\|_\infty,
\qquad
c_{\rm eff}=\frac{(1-f)k_a+k_s}{k_t+1/(c\Delta t)}<1 .
$$

*Proof.* Let $e=\phi^{(\ell)}-\phi^\star$. Then
$e^{(\ell+1)}=\mathcal D\mathcal L^{-1}_0(\Sigma_se)$, where $\mathcal L^{-1}_0$
has zero incoming data. Along a ray with $\mu>0$,
$\mathcal L_0^{-1}g(x)=\int_0^{x}e^{-\Sigma_{\rm eff}(x-x')/\mu}\,g(x')\,dx'/\mu$.
This is positive and bounded by
$\|g\|_\infty\int_0^\infty e^{-\Sigma_{\rm eff}s/\mu}\,ds/\mu=\|g\|_\infty/\Sigma_{\rm eff}$.
The same holds for $\mu<0$. $\mathcal D$ is an average with positive weights
summing to 1. Hence
$\|e^{(\ell+1)}\|_\infty\le\Sigma_s\|e\|_\infty/\Sigma_{\rm eff}$. Finally,
$\Sigma_{\rm eff}=k_t+1/(c\Delta t)>k_t\ge(1-f)k_a+k_s$, because $f>0$. ∎

*Remarks.*

* **Time absorption alone makes every step converge.** Even with $f\to0$ (pure
  effective scattering), $c_{\rm eff}\le1/(1+1/(k_tc\Delta t))<1$.
* The bound is sharp in an infinite medium: Fourier analysis gives spectral
  radius $c_{\rm eff}$ for the slowest (flat) mode. Leakage only helps.
* **For DD with the fixup**, $\mathcal L^{-1}$ is not exactly positive (DD) or
  not linear (fixup), so the proof does not apply verbatim. In practice the
  same rate is observed. Larsen–Morel–Miller-type Fourier analysis covers the
  unfixed DD case.
* **Stopping rule and false convergence.** The code stops when
  $\max_i|\phi_i^{(\ell+1)}-\phi_i^{(\ell)}|/(\phi_i+10^{-3}\phi_{\max})<\texttt{conv\_tol}$.
  Because of geometric convergence, the true error is
  $\le\frac{c_{\rm eff}}{1-c_{\rm eff}}\times$ the last change. At
  $c_{\rm eff}=0.86$ that is a factor of 6. The floor $10^{-3}\phi_{\max}$
  makes the criterion absolute in cells ahead of the front, where $\phi$ is
  tiny. That is acceptable, because those cells carry no energy.

**Numbers for the paper's materials** (the plan has the full table). For
$\chi=1$ ($k_s=0$), $c_{\rm eff}\le0.37$ for $\Delta t\le10^{-12}$ s in all three
materials (and $\le0.11$ for Cu and Al). The worst case found is Au at $\rho_0$, $T=1$ HeV,
$\Delta t=10^{-11}$ s: $c_{\rm eff}=0.86$, about 120 iterations to $10^{-8}$.
**Unaccelerated source iteration is adequate for Tests 1–6.** DSA or
McClarren–Haut DMD would be needed only for $\chi<1$, or for much larger
steps.

---

## 8. Theorems for the full discrete step

Notation: one step consists of (i) evaluating $f,k_a,k_t,S^{\rm eff}$ at $T^n$;
(ii) SI on (B)+(DD)+fixup to tolerance; (iii) the material update (M) with the
converged $\phi^{n+1}$; (iv) Newton inversion of $u_m\to T^{n+1}$.

### 8.1 Theorem 2 (exact discrete energy conservation)

*Claim.* Let $J_{\pm}(x_b)=\tfrac12\sum_mw_m\mu_m\psi_m(x_b)$ be the quadrature
currents at the two boundary faces, computed from the face values of the sweep.
At SI convergence,

$$
\sum_i\Delta x_i\Big[(E_{r,i}^{n+1}+u_{m,i}^{n+1})-(E_{r,i}^n+u_{m,i}^n)\Big]=\Delta t\,\big[J(x_{1/2})-J(x_{N_x+1/2})\big].
$$

The left boundary current is the bath inflow minus the reflected outflow; the
right one is the leakage. If SI stops at a finite tolerance, the residual is
$\Delta t\,c\sum_i\Delta x_i\Sigma_{s,i}(\phi^{(\ell)}_i-\phi^{(\ell+1)}_i)$,
which is bounded by the stopping rule.

*Proof.* Sum (B) over cells for each $m$. The face terms telescope, because
DD and the fixup both use the same face value as the outgoing value of one
cell and the incoming value of the next:

$$
\mu_m\big(\psi_{m,N_x+1/2}-\psi_{m,1/2}\big)+\sum_i\Delta x_i\Sigma_{\rm eff}\bar\psi_{m,i}=\sum_i\Delta x_i\bar q_{m,i}.
$$

Multiply by $w_m/2$ and sum over $m$. With
$\bar q=\Sigma_s\phi^{(\ell)}+S+\psi^n/(v\Delta t)$ and
$\tfrac12\sum w_m=1$, at convergence ($\phi^{(\ell)}=\phi$):

$$
J_{\rm right}-J_{\rm left}+\sum_i\Delta x_i\Big[\frac{\phi_i-\phi^n_i}{c\Delta t}+(\Sigma_t-\Sigma_s)\phi_i-S_i\Big]=0 .
$$

Insert $\Sigma_t-\Sigma_s=fk_a$, $S=fk_aacT_n^4$ and $E_r=\phi/c$, and multiply
by $\Delta t$:
$\sum\Delta x\,(E_r^{n+1}-E_r^n)=-\Delta t\sum\Delta x\,fk_a(\phi-acT_n^4)+\Delta t(J_{\rm left}-J_{\rm right})$.
The update (M) adds exactly $+\Delta t\sum\Delta x\,fk_a(\phi-acT_n^4)$ to
$\sum\Delta x\,u_m$. ∎

*Consequences.*

* As in IMC, energy conservation holds **whatever** the errors L1–L5. A
  non-conserving run means a bug, never a discretization error.
* It requires (M) to use the **same** $f$, $k_a$ and $T_n$ as the transport
  step, and $\phi$ to be the DD cell average `psi[m,i]`. A material update in
  $u_r$ form (the diffusion code's current default, IMC plan finding 2a) breaks
  it when $\beta_{\rm EOS}$ varies.
* The code must **tally the boundary face currents**. `sn.py` does not
  return them yet (plan, S0).

### 8.2 Proposition 1 (the material energy stays positive)

*Claim.* If $\phi^{n+1}\ge0$ and $\beta_{\rm EOS}<4$, then
$u_m^{n+1}\ge(1-\beta_{\rm EOS}/4)\,e_v^n>0$, where
$e_v=fT^{\beta_{\rm EOS}}\rho^{1-\mu_{\rm EOS}}$.

*Proof.* From (M) with $\phi\ge0$:
$u_m^{n+1}\ge u_m^n-\Delta tfk_a\,acT_n^4$. Now
$\Delta tfk_a\,c\,u_r=\frac{k_ac\Delta t}{1+\beta_Fk_ac\Delta t}u_r<\frac{u_r}{\beta_F}=\frac{C_vT}{4}$.
With $C_vT=\beta_{\rm EOS}e_v-4aT^4$ and $u_m=e_v-aT^4$:
$u_m-\frac{C_vT}4=e_v(1-\frac{\beta_{\rm EOS}}4)$. ∎

Au ($\beta_{\rm EOS}=1.6$), Cu (1.35) and Al (1.2) all satisfy this, with room
to spare. $\phi\ge0$ is guaranteed by the fixup. The bound does not depend on
$\Delta t$: the linearized step can never drive $u_m$ negative.

### 8.3 Proposition 2 (bounded drive at the singular start of Test 1)

In Test 1, $T_{\rm bath}^4\propto1+0.169\,t^{-79/192}$, which is singular at
$t=0$. Backward Euler evaluates the BC at $t^{n+1}>0$, where it is finite.
The energy injected in the first step is then
$\Delta t\,F_{\rm inc}(t^1)$, whereas the exact value is
$\int_0^{t^1}F_{\rm inc}\,dt$. The two differ by the factor
$1/(1-79/192)\approx1.70$ in the singular part. Using the **step average**
$\overline{F_{\rm inc}}=\Delta t^{-1}\int_{t^n}^{t^{n+1}}F_{\rm inc}\,dt$
(analytic for the power law) makes the injected energy exact for every step.
This is the same as IMC plan finding 4. With the step average, Theorem 2
gives a total energy equal to the exact integral of the drive minus the
leakage.

### 8.4 The thick (diffusion) limit

Scale $k_t\to k_t/\varepsilon$, $k_a\to\varepsilon k_a$, $S\to\varepsilon S$,
$\partial_t\to\varepsilon\partial_t$ and let $\varepsilon\to0$. For the
continuous-space $S_N$ equations, a Hilbert expansion
$\psi=\psi^{(0)}+\varepsilon\psi^{(1)}+\dots$ gives
$\psi^{(0)}=\phi^{(0)}$ (isotropic) and
$\psi^{(1)}=-\mu\partial_x\phi^{(0)}/k_t$. The solvability condition at
$O(\varepsilon^2)$, using $\tfrac12\sum w\mu^2=\tfrac13$, is

$$
\frac1c\partial_t\phi-\partial_x\frac{1}{3k_t}\partial_x\phi+k_a(\phi-acT^4)=0,
$$

the paper's `gray_diff` in Eulerian form. **$S_N$ therefore contains the
diffusion code's equation as its thick limit, for every $N$.** The
$S_N$-vs-diffusion difference measures transport effects, not a mismatch in
the diffusion coefficient.

For the **spatially discrete** scheme, Larsen, Morel & Miller (1987) show that
DD keeps this limit in the interior when cells are thick. However, DD's
solutions in thick cells oscillate and lose positivity, and the fixup then
switches those cells to step-like behaviour, which does **not** have the
correct limit. Linear-discontinuous Galerkin (LD) keeps both robustness and
the limit. For this problem, the thick-cell region is the cold gold ahead of
the front, where the radiation carries almost no energy. The practical risk is
therefore a small error in the front position, not a global error. The plan
measures it (S9).

---

## 9. The spherical sweep (Lewis–Miller), for completeness

The conservative form of the spherical $S_N$ equation in cell $i$ (shell
$[r_{i-1/2},r_{i+1/2}]$, areas $A=4\pi r^2$, volume $V_i$):

$$
\mu_m\big(A_{i+1/2}\psi_{m,i+1/2}-A_{i-1/2}\psi_{m,i-1/2}\big)+\frac{A_{i+1/2}-A_{i-1/2}}{w_m}\big(\alpha_{m+1/2}\psi_{m+1/2,i}-\alpha_{m-1/2}\psi_{m-1/2,i}\big)+V_i\Sigma_{\rm eff}\bar\psi_{m,i}=V_i\bar q_{m,i}.
$$

The second term is the discretized angular derivative
$\frac1r\partial_\mu[(1-\mu^2)\psi]$: photons moving outward along a straight
line see $\mu$ increase. **The $\alpha$ recursion** (`lewis_miller_alpha`, line 158)
$\alpha_{m+1/2}=\alpha_{m-1/2}-w_m\mu_m$, with $\alpha_{1/2}=0$, comes from
demanding that a spatially constant isotropic $\psi$ be an exact solution with
$q=\Sigma_{\rm eff}\psi$. The streaming term is then
$\mu_m\psi(A_{i+1/2}-A_{i-1/2})$, and it must cancel the angular term. Since
$\sum w\mu=0$, the recursion closes with $\alpha_{N+1/2}=0$. The weights
$\alpha\ge0$ because the $\mu_m<0$ ordinates come first.

**Diamond in angle** gives $\psi_{m+1/2,i}=2\bar\psi_{m,i}-\psi_{m-1/2,i}$. The
angular sweep runs from $\mu=-1$ upwards, so the **starting direction**
$\mu=-1$ (lines 210–223) has no angular redistribution. It satisfies a
slab-like equation along a diameter:
$-\partial_r\psi+\Sigma_{\rm eff}\psi=q$. The code uses the source of the most
negative ordinate, $q_{0,k}$, for it (`q_total[0,k]`), and keeps its own time
census `psi_start_old`. The $\mu<0$ loop (lines 225–233) sweeps inward. The
$\mu>0$ loop (lines 235–242) sweeps outward, with the reflection
$\psi_{\mu}(0)=\psi_{-\mu}(0)$ at the centre (`psi_center[n_mu - 1 - j]`).
Both loops call `sphere_cell` (line 168), which solves one cell and ordinate;
only the roles of the inner and outer face areas swap between them.
Substituting the two diamond relations into the balance reproduces the
`denom`/`numer` of `sphere_cell` (lines 180–185) exactly.

The `iff`/`muff` flags are the set-to-zero fixup applied separately to the
spatial outgoing face (`iff`) and the angular outgoing edge (`muff`). When one
is set to zero, its diamond coefficient 2 becomes 1 and it drops out of the
denominator, as in Section 6.3.

---

## 10. Code map, and what `sn.py` cannot do yet

### 10.1 Line-by-line map

| Code | Equation | Section |
|---|---|---|
| `make_quadrature` (37) | Gauss–Legendre $\mu_m,w_m$, $\sum w=2$ | 5.1 |
| `legendre_table` (46) | $P_l(\mu_m)$ by the 3-term recurrence | 5.3 |
| `scalar_flux`, `angular_moments` (61, 71) | $\phi=\tfrac12\sum w\psi$, $\phi_l=\tfrac12\sum wP_l\psi$ | 2.2, 5.3 |
| `scattering_source` (83) | $Q_s=\Sigma_s\sum(2l+1)f_lP_l\phi_l$ | 5.3 |
| `fixup_scale_clip`, `fixup_min_blend` (96, 116) | no-ops for isotropic sources (Lemma 1) | 5.3 |
| `sweep_slab` (133) | (B)+(DD)+set-to-zero fixup, vacuum BCs | 6 |
| `lewis_miller_alpha`, `sphere_cell`, `sweep_sphere` (158, 168, 203) | Lewis–Miller | 9 |
| `sig_eff` (490), `rhs` (342) | backward Euler, time absorption | 4 |
| SI loop in `time_loop` (328–367) | $\phi^{(\ell+1)}=\mathcal D\mathcal L^{-1}(\Sigma_s\phi^{(\ell)}+s)$ | 7 |
| convergence test (357–364) | relative change with floor $10^{-3}\phi_{\max}$ | 7.2 |
| adaptive $\Delta t$ (375–382) | $\Delta t\leftarrow\min(\Delta t\cdot\texttt{dtfac}/\delta,\;1.1\Delta t,\;\Delta t_{\max})$ | — |
| `scattering_coefficients` (390) | $f_l$: isotropic, HG $g^l$, or user-given | 5.3 |
| `run_sn` (410) | validation, setup, one-line log | — |

### 10.2 Gaps between `sn.py` and the paper's problem

`sn.py` solves (N) with **constant scalars** $\Sigma_t,\Sigma_s,v$ and a
**constant-in-time** source. Each item below is a change in the plan:

1. **Cell-wise coefficients.** $k_t$, $\Sigma_s^{\rm eff}=(1-f)k_a$ and
   $S^{\rm eff}$ vary by many orders of magnitude across the front.
   `sig_eff` is already an array; `sigma_s` and the SI source are scalars.
2. **Non-uniform mesh.** `run_sn` rejects non-uniform edges (line 436). The
   slab sweep already takes `dx_arr`. Only the tracer lookup assumes uniform
   spacing.
3. **Incoming boundary data.** Both slab boundaries are hard-wired to vacuum
   (line 144). The bath needs $\psi_b$ for $\mu>0$ at the left, normalized as
   in Section 5.2.
4. **A single-step API with an angular census.** The driver owns the time
   loop and starts from an isotropic `n0`. The coupled problem needs
   `sn_step(psi_old, coeffs, dt, psi_b) -> (psi_new, phi, J_left, J_right, iters)`,
   called once per hydro step, with **no printing** inside the JIT.
5. **Boundary-current tallies** for Theorem 2.
6. **Material coupling** ($f$, (M), Newton $T(u_m)$). This belongs in a shared
   `material.py` (the same module as IMC plan M0).
7. **Small issues to fix while there:**
   - In adaptive mode, the last step can overshoot `t_final`, because `dt` is
     not clipped to `t_final - t`.
   - Snapshots are taken at the first step with $t\ge t_{\rm snap}$, not at
     $t_{\rm snap}$ itself.
   - `n_steps = int(round(t_final/dt))` silently changes the final time when
     `t_final/dt` is not an integer.
   - The adaptive formula reacts to the global maximum relative change. With
     the $10^{-3}\phi_{\max}$ floor, a moving front controls $\Delta t$. That
     is the desired behaviour, but it must be the same controller as in the
     diffusion run for a like-for-like comparison.

---

## 11. Status of each link

| Link | Status |
|---|---|
| Dictionary (N) ↔ paper transport problem (Lemma 0) | Exact; normalization checked numerically. |
| Linearization (L) = F&C (2.1) at $\alpha=1$ | Exact algebra given L1, L2; same as IMC. |
| Backward Euler, $S_N$, DD | Standard approximations, orders $O(\Delta t)$, $P_{N-1}$-equivalent in angle, $O(\Delta x^2)$ smooth / $O(\Delta x)$ where fixed up. |
| SI convergence (Theorem 1) | Proved for exact streaming; $c_{\rm eff}<1$ always; $\le0.86$ for the paper's materials. |
| Energy conservation (Theorem 2) | Proved for the discrete scheme including the fixup; needs (M) and boundary tallies. |
| $u_m>0$ (Proposition 1) | Proved for $\beta_{\rm EOS}<4$. |
| Thick limit (Section 8.4) | Formal asymptotics for $S_N$; DD+fixup degrades it in fixed-up cells (literature). |
| Convergence of the coupled nonlinear scheme as $\Delta t,\Delta x\to0$, $N\to\infty$ | Same status as IMC (`imc_mc_derivation.md` §8): conditional on smoothness; open in the cold front region. |

## Summary

$S_N$ and IMC share step A: the Fleck linearization turns the nonlinear
problem into the linear transport equation (L), with effective scattering
$(1-f)k_a$, effective source $fk_aacT_n^4$, and the energy-conserving material
update (M). From there, IMC solves (L) exactly by a random walk. $S_N$ discretizes
it: backward Euler in time, Gauss ordinates in angle, and diamond differences
in space. It then solves the resulting steady problem per step by source
iteration, i.e. repeated exact sweeps. That iteration converges at rate
$c_{\rm eff}<1$. The scheme conserves energy exactly and keeps $u_m$
positive. Its thick limit is the paper's diffusion equation with exactly
$D=1/(3k_t)$.

Three facts matter most for the paper comparison:

1. `sn.py` needs a bath boundary whose discrete current is normalized (a 1.15%
   energy bias at $S_8$ otherwise).
2. It needs cell-wise coefficients and a per-step API carrying the angular
   census.
3. For $\chi=1$, plain source iteration is fast enough. No acceleration is
   needed for Tests 1–6.
