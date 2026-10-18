# Implicit Monte Carlo for the gray slab problem: from Fleck & Cummings Eq. (1.1) to a proof that the code solves the linearized equation

This document is a self-contained account of five things:

1. **How Fleck & Cummings (F&C) turn the nonlinear radiation–matter system,
   their Eq. (1.1), into a linear transport equation that a Monte Carlo
   random walk can solve** (their Eq. 2.1). Each step is marked as exact
   algebra or as an approximation (Section 1).
2. **A theorem with proof** that the random walk implemented in
   `src/imc/sio2/sio2_1mm_gray_imc_check_og.py` produces, in expectation, the unique
   solution of that linear equation, and that its tallies are unbiased
   (Sections 2–5).
3. **Guarantees for the code as written** (Section 6): the material energy
   stays positive, a bound on the bias from the 20000-event cap, and a bound
   on the error from averaging the drive over each step.
4. **Convergence over many coupled steps as the number of packets grows**
   (Section 7).
5. **A conditional convergence theorem in $\Delta t$** for the F&C scheme,
   and an explanation of why full convergence to the true radiation field
   remains open (Section 8).

Section 9 summarizes the status of every link.

Appendix A maps every line of the algorithm to the corresponding F&C equation.

Primary source: J. A. Fleck Jr. and J. D. Cummings, "An Implicit Monte Carlo
Scheme for Calculating Time and Frequency Dependent Nonlinear Radiation
Transport", *J. Comput. Phys.* **8**, 313–342 (1971). A copy is in the
companion repository `project3_docs`, at `src_papers/Fleck and Cummings.pdf`. Equation numbers "F&C (x.y)"
refer to that paper. Labels such as (T), (Mild), (Mild-$\Pi$) and (P), and the numbering of
lemmas, propositions and theorems, are this document's own.

---

## 0. Scope

```
  original nonlinear system, F&C Eq. (1.1)
        │  (A) F&C linearization over one time step         ── Sections 1, 8
        │      Δt → 0, continuous in space: PROVED under smoothness hypotheses
        │      (Theorem 4); its constants are useless for the cold foam
        ▼
  linear transport equation F&C (2.1) = (T), frozen coefficients
        │  (B) packet random walk, one step                 ── Sections 2–6
        │      PROVED: mean packet density = unique solution of (T)
        ▼
  expected packet density / expected tallies of the code
        │  (C1) N → ∞ over many coupled steps, Δt and Δx fixed ── Section 7
        │       PROVED: convergence to the deterministic IMC scheme (Theorem 3)
        │  (C2) Δx → 0, and error bounds that do not depend on the opacity
        │       OPEN (Section 8.5)
        ▼
  true solution I of Eq. (1.1)
```

---

## 1. From F&C Eq. (1.1) to a Monte-Carlo-ready linear equation

### 1.1 The original system and why it cannot be simulated directly

F&C start from gray, one-dimensional slab transport, assuming local
thermodynamic equilibrium, with no physical scattering:

$$
\frac1c\frac{\partial I}{\partial t}+\mu\frac{\partial I}{\partial x}+\sigma I=\frac12\,\sigma\,acT^4,
\tag{F\&C 1.1a}
$$

$$
\frac{\partial u_m}{\partial t}=\sigma\Big(\int_{-1}^{1}I\,d\mu-acT^4\Big)+S .
\tag{F\&C 1.1b}
$$

* $I(x,\mu,t)$ is the azimuthally integrated specific intensity, normalized so
  that $\int I\,d\mu=cE_r$ ($E_r$ is the radiation energy density) and
  $\int\mu I\,d\mu=F$ (the flux).
* $u_m(T)$ is the material energy density, $\sigma(T)$ the absorption opacity,
  $a$ the radiation constant, and $S$ an external source.

For the SiO₂ problem in the code:
$S=0$, $u_m=E_{100}H\theta^{\beta_{\rm EOS}}$, $\sigma=\sigma_0\theta^{-\alpha_{\rm op}}$,
with $\theta=T/100\,$eV.

Three features block a direct Monte Carlo solution:

* **Nonlinearity.** The emission source $\tfrac12\sigma acT^4$ and the
  coefficient $\sigma(T)$ depend on the unknown $T$. Monte Carlo is a linear
  method: the mean of a random walk satisfies a linear equation.
* **Coupling.** $T$ evolves through (1.1b), driven by $\int I\,d\mu$.
* **Stiffness.** In optically thick, cold matter, emission and absorption
  nearly cancel. If emission is taken from $T$ at the start of the step
  (explicit), the scheme oscillates or blows up unless
  $\beta c\sigma\Delta t\lesssim1$ (F&C (4.30); see 1.10 below). With
  $\sigma\sim10^{5}$–$10^{9}\,$cm⁻¹ in the cold foam, that would need absurdly
  small $\Delta t$.

F&C's idea is to **linearize over one time step in a way that is implicit in
the emission**. Part of the absorbed energy is treated as re-emitted *within
the step*, as an "effective scattering". What remains is a linear transport
problem with known coefficients.

### 1.2 Step 1 — change of variable to $u_r=aT^4$ (exact)

Define the equilibrium radiation energy density and the ratio of derivatives:

$$
u_r=aT^4,\qquad \beta=\frac{\partial u_r}{\partial u_m}=\frac{4aT^3}{\partial u_m/\partial T}.
\tag{F\&C 1.2–1.3}
$$

$u_m$ depends only on $T$, because the density is fixed in a static medium. By
the chain rule, $\partial_t u_r=\beta\,\partial_t u_m$. Multiplying (1.1b) by
$\beta$ gives

$$
\frac1c\frac{\partial I}{\partial t}+\mu\frac{\partial I}{\partial x}+\sigma I=\frac12 c\sigma u_r,
\tag{F\&C 1.4a}
$$

$$
\frac{\partial u_r}{\partial t}=\beta\sigma\Big(\int I\,d\mu-cu_r\Big)+\beta S .
\tag{F\&C 1.4b}
$$

All of the nonlinearity now sits in the coefficients $\sigma(T)$ and
$\beta(T)$. Both equations are linear in the pair $(I,u_r)$.

*For the code:* $\beta=4\theta^{4-\beta_{\rm EOS}}/(H\beta_{\rm EOS})$. This is
the variable `gamma` in `material_coefficients` (line 103).

### 1.3 Step 2 — integrate the material equation over a step (exact)

Integrate (1.4b) from $t^n$ to $t^{n+1}=t^n+\Delta t$:

$$
u_r^{n+1}-u_r^n=\int_{t^n}^{t^{n+1}}\!dt\,\beta\sigma\int I\,d\mu-c\int_{t^n}^{t^{n+1}}\!dt\,\beta\sigma u_r+\int_{t^n}^{t^{n+1}}\!dt\,\beta S .
\tag{F\&C 1.7a}
$$

### 1.4 Step 3 — freeze coefficients and choose a time-centering (approximations A1, A2)

F&C "factor appropriate average values of the integrands outside the
integrals":

* **A1 (frozen coefficients).** Treat $\beta$ and $\sigma$ as constant over the
  step. F&C Sec. II, step 1: "From temperatures available at $t=t^n$, or by
  forward extrapolation, values of $\beta$ and $\sigma$ are determined for each
  spatial zone." The code uses the values at $T^n$.

  With A1, define the step-averaged intensity
  $I^\lambda=\Delta t^{-1}\int_{t^n}^{t^{n+1}}I\,dt$, and $S^\gamma$ likewise.
  This is a *definition*, not an approximation. Then
  $\int dt\,\beta\sigma\int I\,d\mu=\Delta t\,\beta\sigma\int I^\lambda d\mu$
  holds exactly.
* **A2 ($\alpha$-quadrature of the emission integral).**
  $\int_{t^n}^{t^{n+1}}u_r\,dt\approx\Delta t\,[\alpha u_r^{n+1}+(1-\alpha)u_r^n]$
  with $\tfrac12\le\alpha\le1$. Here $\alpha=1$ is backward Euler and
  $\alpha=\tfrac12$ the trapezoid rule.

Together these give

$$
u_r^{n+1}-u_r^n=\Delta t\,\beta\sigma\Big\{\int I^\lambda d\mu-c\,[\alpha u_r^{n+1}+(1-\alpha)u_r^n]\Big\}+\beta S^\gamma\Delta t .
\tag{F\&C 1.7b}
$$

### 1.5 Step 4 — solve for the time-centered $u_r$ (exact algebra)

Let $u_r^\gamma=\alpha u_r^{n+1}+(1-\alpha)u_r^n$. Then
$u_r^{n+1}-u_r^n=(u_r^\gamma-u_r^n)/\alpha$, and (1.7b) becomes
$(u_r^\gamma-u_r^n)/\alpha=\Delta t\beta\sigma(\int I^\lambda d\mu-cu_r^\gamma)+\beta S^\gamma\Delta t$.
Solving this linear equation,

$$
u_r^\gamma=\frac{\alpha\beta\sigma\Delta t}{1+\alpha\beta c\Delta t\sigma}\int I^\lambda d\mu+\frac{u_r^n}{1+\alpha\beta c\Delta t\sigma}+\frac{\alpha\beta\Delta t\,S^\gamma}{1+\alpha\beta c\Delta t\sigma}.
\tag{F\&C 1.9}
$$

Define the **Fleck factor**

$$
f=\frac{1}{1+\alpha\beta c\Delta t\sigma}\in(0,1],\qquad\text{so that}\qquad \frac{\alpha\beta c\Delta t\sigma}{1+\alpha\beta c\Delta t\sigma}=1-f .
$$

Then (1.9) reads

$$
c\sigma\,u_r^\gamma=(1-f)\,\sigma\!\int I^\lambda d\mu+f\,c\sigma\,u_r^n+(1-f)\,S^\gamma .
$$

### 1.6 Step 5 — use the time-centered emission in the transport equation (approximation A3)

In (1.4a), F&C replace the instantaneous emission $\tfrac12c\sigma u_r(t)$ by
its time-centered value $\tfrac12c\sigma u_r^\gamma$. The resulting equation is
"presumed to be accurate over a time $\Delta t$":

$$
\frac1c\frac{\partial I}{\partial t}+\mu\frac{\partial I}{\partial x}+\sigma I=\frac12(1-f)\sigma\int I^\lambda d\mu+\frac12 f c\sigma u_r^n+\frac12(1-f)S^\gamma .
\tag{F\&C 1.10}
$$

This is the **implicitness**. With $\alpha=1$, the emission used during the
step is that of the *end-of-step* temperature, expressed through (1.9) in terms
of the radiation absorbed during the step.

### 1.7 Step 6 — instantaneous re-emission: effective scattering (approximation A4)

(1.10) still contains the step average $I^\lambda$, which is non-local in time.
F&C now "substitute the instantaneous value of $I$ for $I^\lambda$":

$$
\boxed{\frac1c\frac{\partial I}{\partial t}+\mu\frac{\partial I}{\partial x}+\underbrace{f\sigma}_{\sigma_a}I+\underbrace{(1-f)\sigma}_{\sigma_s}I=\frac12\,\sigma_s\int_{-1}^{1}I\,d\mu'+\frac12\,f c\sigma u_r^n+\frac12(1-f)S^\gamma}
\tag{F\&C 1.10a = 2.1}
$$

**Interpretation.** The total opacity splits into an effective absorption
$\sigma_a=f\sigma$ (F&C (1.11a)) and an effective isotropic scattering
$\sigma_s=(1-f)\sigma$ (F&C (1.11b)). Equivalently, the emission at time $t$
uses

$$
u_r^{\rm eff}(t)=f\,u_r^n+(1-f)\,\frac1c\int I(x,\mu',t)\,d\mu' .
$$

That is, a fraction $f$ comes from the start-of-step temperature and a fraction
$1-f$ is absorbed radiation re-emitted instantly, at the same place, in an
isotropic direction. As $\Delta t\to\infty$, $f\to0$ and (2.1) becomes the
equilibrium transport equation (F&C (1.12)–(1.13)).

### 1.8 Step 7 — the material update, defined by energy conservation

F&C integrate (2.1) over $\mu$ (F&C (1.14)). They observe that the net
radiation loss must equal the material gain, which gives

$$
\frac{\partial u_m}{\partial t}=f\sigma\int I\,d\mu-fc\sigma u_r^n+fS^\gamma .
\tag{F\&C 1.17}
$$

The three terms are: effective absorption; effective emission ($f$ times the
start-of-step emission); and the fraction $f$ of the external source.
Integrating over the step:

$$
u_m^{n+1}=u_m^n+f\sigma\int_{t^n}^{t^{n+1}}\!dt\int I\,d\mu-fc\sigma\Delta t\,u_r^n+fS^\gamma\Delta t .
\tag{F\&C 1.18}
$$

F&C then linearize the EOS as $T^{n+1}=T^n+b^{-1}\{\cdots\}$ (F&C (1.19),
(2.3), (4.28)). The code instead inverts the power-law EOS exactly from
$u_m^{n+1}$. That is F&C (1.18) with exact inversion, which is what F&C
recommend for non-ideal EOS ("substitute $\partial u_m/\partial T$ for $b$").

Because (1.18) is **built from conservation**, the update conserves total
energy exactly, whatever the errors of A1–A4. F&C call this update "much
superior in practice" to the $u_r$-form (1.15b).

**Self-consistency (F&C (1.14)–(1.16)).** Integrating (2.1) gives an update
for $u_r$ (F&C (1.15b)). It coincides with the time-discrete material equation
(1.8) when $\int_{t^n}^{t^{n+1}}dt\int I\,d\mu$ is identified with
$\Delta t\int I^\lambda d\mu$. So, once $I^\lambda$ is *defined* as the step
average of the solution of (2.1), the transport solve and the discretized
material equation (1.7b) agree exactly. A4 changes *when within the step*
re-emission happens. It does not create an energy inconsistency.

### 1.9 The coefficient $\alpha$

The coefficient of $u_r^n$ in F&C (1.15b) is
$[1-(1-\alpha)\beta c\Delta t\sigma]/(1+\alpha\beta c\Delta t\sigma)$. It is
negative when $(1-\alpha)\beta c\Delta t\sigma>1$, which makes the solution
oscillate from step to step. Hence F&C: "for large values of
$\alpha\beta c\Delta t\sigma$, $\alpha$ must be set equal to 1" (F&C p. 319,
(3.12)). **The code uses $\alpha=1$:**

$$
f=\frac{1}{1+\beta c\sigma\Delta t}\quad(\text{line 104}).
$$

### 1.10 Illustration: 0-D stability and overheating (this document's analysis, not in F&C)

Consider an infinite homogeneous medium, so there is no $x$-dependence. Take
$\alpha=1$, frozen $\beta,\sigma$, and a linearized EOS. Write
$E=\frac1c\int I\,d\mu$ and $k=fc\sigma\Delta t$.

Integrating (2.1) over $\mu$ gives
$\dot E=fc\sigma(u_r^n-E)$. So during the step,
$E(t)=u_r^n+(E^n-u_r^n)e^{-fc\sigma(t-t^n)}$. By (1.18), the material gains
$(E^n-u_r^n)(1-e^{-k})$, so
$u_r^{n+1}=u_r^n+\beta(E^n-u_r^n)(1-e^{-k})$.

The imbalance $D=E-u_r$ is then multiplied each step by

$$
g=e^{-k}(1+\beta)-\beta .
$$

* **Upper bound.** $g<1$ always.
* **Lower bound.** $k<1/\beta$, because
  $k=c\sigma\Delta t/(1+\beta c\sigma\Delta t)$. Set $x=1/\beta$. Then
  $g>-1 \iff e^{-k}(1+\beta)>\beta-1$, which follows from
  $e^{-x}(1+x)\ge 1-x$ (true for all $x\ge0$: equality at $0$, and derivative
  $-xe^{-x}\ge-1$). So $|g|<1$ for **every** $\Delta t$. In this linear model
  the scheme is unconditionally stable.
* **Oscillation.** $g<0$ when $e^{-k}<\beta/(1+\beta)$. The material
  temperature then overshoots the equilibrium value within one step. This is
  the 0-D picture of the maximum-principle violation ("overheating") that
  Larsen & Mercier (1987) identified for large $\Delta t$.
* **Explicit comparison.** Taking emission explicitly (no Fleck factor,
  $f\to1$ in the source with emission frozen at $u_r^n$) gives
  $g=e^{-c\sigma\Delta t}(1+\beta)-\beta$. This tends to $-\beta$ when
  $c\sigma\Delta t\gg1$, so it is unstable whenever $\beta>1$.

### 1.11 Ledger of the linearization

| Step | Content | Status |
|---|---|---|
| 1 | $u_r=aT^4$, $\beta=\partial u_r/\partial u_m$ (1.2)–(1.4) | exact |
| 2 | time integration (1.7a) | exact |
| 3 | **A1**: $\beta,\sigma$ frozen at $T^n$ over the step | approximation, formally $O(\Delta t)$ |
| 3 | **A2**: $\alpha$-quadrature of $\int u_r\,dt$ (1.7b) | approximation; $\alpha=1$ gives $O(\Delta t)$ |
| 4 | solve for $u_r^\gamma$, define $f$ (1.9) | exact algebra |
| 5 | **A3**: emission in transport $\to$ $u_r^\gamma$ (1.10) | approximation (implicit emission) |
| 6 | **A4**: $I^\lambda\to I$ (1.10a = 2.1) | approximation (instantaneous re-emission) |
| 7 | material update from conservation (1.17)–(1.18) | exact energy conservation by construction |
| — | spatial: $T,\sigma,\beta,f$ constant per cell (F&C Sec. II) | approximation, $\Delta x\to0$ |

**What F&C establish:**

* consistency with the discrete energy balance (1.14)–(1.16);
* the formal limits $\Delta t\to0$ (with $f\to1$, Eq. (1.1) is recovered) and
  $\Delta t\to\infty$ (equilibrium transport (1.12));
* agreement with time-differencing (1.1) directly (F&C p. 317);
* numerical comparisons (Sec. V).

**What they do not establish:** an error bound, or a convergence proof of
(2.1)+(1.18) to the solution of (1.1). On stability they say only that the
method "appears to be unconditionally stable". Densmore & Larsen (2004, *J.
Comput. Phys.*) analyzed IMC's asymptotic time accuracy. Wollaber (2016, *J.
Comput. Theor. Transport*, "Four decades of implicit Monte Carlo") reviews the
later theory. Section 8 of this document proves a conditional first-order
convergence result (Theorem 4) and explains why it does not cover the SiO₂
problem.

### 1.12 The Monte-Carlo-ready problem

Set $S=0$, $\alpha=1$, and change to the angular energy density $\phi=I/c$, so
that $\int\phi\,d\mu=E_r$. Substituting $I=c\phi$ into (2.1) changes nothing
else:

$$
\boxed{\;
\partial_t\phi + c\mu\,\partial_x\phi + c\sigma\,\phi
= (1-f)\,c\sigma\cdot\frac12\int_{-1}^{1}\phi(x,\mu',t)\,d\mu' \;+\; q(x),
\qquad q=\tfrac12 f c\sigma u_r^n
\;}
\tag{T}
$$

with boundary and initial data for the Back problem

$$
\phi(0,\mu,t)=\phi_L(t)=\tfrac12 aT_D(t)^4\ (\mu>0),\qquad
\phi(L,\mu,t)=0\ (\mu<0),\qquad
\phi(\cdot,t^n)=\phi^n .
\tag{BC/IC}
$$

The left condition is an isotropic blackbody bath. Its incoming flux is
$\int_0^1c\mu\,\phi_L\,d\mu=\tfrac{ac}{4}T_D^4=\sigma_{SB}T_D^4$, which is
F&C (4.12). The right condition is vacuum.

**Why (T) is ready for Monte Carlo:**

* it is **linear** in $\phi$;
* its coefficients $\sigma$, $f$ and its source $q$ are **known at $t^n$** and
  constant over the step, and piecewise constant in $x$;
* it is exactly the transport equation of a medium with absorption
  $\sigma_a=f\sigma$, isotropic scattering $\sigma_s=(1-f)\sigma$ and a known
  isotropic volume source. Packets can simulate such a medium literally.

**One time step (F&C Sec. II).**

1. Compute $\sigma$, $\beta$, $f$ in each cell from $T^n$.
2. Solve (T) by Monte Carlo with census plus new source packets, recording the
   energy deposited by effective absorption in each cell. This is F&C (2.2),
   $\frac{\sigma}{1+\beta c\Delta t\sigma}\int dt\int I\,d\mu$.
3. Update $u_m$ by (1.18) and invert the EOS to get $T^{n+1}$.

The rest of this document proves that step 2, as implemented, is correct.

---

## 2. Intuition: where the derivatives go

Along the straight line $x(s)=x_0+\mu s$, $t(s)=t_0+s/c$, the chain rule gives

$$
\frac{d}{ds}I\big(x(s),\mu,t(s)\big)=\mu\,\partial_x I+\tfrac1c\,\partial_t I .
$$

So the differential operator of (2.1) **is the derivative along a straight
flight at speed $c$**. (2.1) is an ODE along each ray, $dI/ds+\sigma I=Q$,
with solution

$$
I(x,\mu,t)=I(x_b,\mu,t_b)\,e^{-\tau(s_b)}+\int_0^{s_b}Q\big(x-\mu s',\mu,t-s'/c\big)\,e^{-\tau(s')}\,ds',
\qquad \tau(s)=\int_0^s\sigma(x-\mu s'')\,ds'' .
$$

Here $s_b$ is the distance back to $t^n$ (census) or to the boundary (inflow).
Each term corresponds to something a packet does in the code:

| Term | Packet action in the code |
|---|---|
| straight ray, speed $c$ | `xp += mup*distance`, `tp += distance/C` |
| $e^{-\tau}$ | no-collision probability, `-log(r)/sig` (line 271) |
| $I$ at $t^n$ / at $x=0$ | census packets / boundary packets |
| $\tfrac12 fc\sigma u_r^n$ | emission packets |
| $\tfrac12\sigma_s\int I\,d\mu'$ | non-absorbing collision: new $\mu\sim U[-1,1]$ |

Sections 3–4 turn this table into a proof. The proof uses the integral
("mild") form of the equation rather than the differential form. The mild form
is the standard notion of solution for transport with discontinuous $\sigma$,
for which classical $C^1$ solutions generally do not exist. It also needs no
boundary traces or regularity assumptions.

---

## 3. Mild formulation, existence and uniqueness

### 3.1 Notation

Phase space: $\bar D=[0,L]\times[-1,1]$, points $z=(y,\mu)$. All measures are
finite signed Borel measures on $\bar D$ with total-variation norm
$\|\nu\|=\sup_{|h|\le1}\langle h,\nu\rangle$, where $h$ ranges over bounded
Borel functions and $\langle h,\nu\rangle=\int h\,d\nu$.

Opacities are finite: the code's temperature floor keeps
$\sigma_{\max}=\max_j\sigma_j<\infty$.

**Free flight.**

* Position after flying for time $s$: $X_s(z)=y+c\mu s$.
* Exit time: $\tau(z)=\inf\{s>0:X_s(z)\notin(0,L)\}$. It is $0$ for a point on
  a face pointing out, $L/(c\mu)$ for $z=(0,\mu>0)$, and $+\infty$ if $\mu=0$.
* Attenuation: $A_s(z)=\exp\!\big(-\int_0^s c\,\sigma(X_{s'}(z))\,ds'\big)$.

**Streaming–attenuation operator.**

$$
\langle h,S(s)\nu\rangle=\int \mathbf 1\{s<\tau(z)\}\,A_s(z)\,h\big(X_s(z),\mu\big)\,\nu(dz).
$$

The kernel has mass at most $1$, so
$$\|S(s)\nu\|\le\|\nu\|. \tag{P1}$$

**Scattering operator.**

$$
\langle h,R\nu\rangle=\int (1-f(y))\,c\sigma(y)\,\Big[\tfrac12\!\int_{-1}^1 h(y,\mu')\,d\mu'\Big]\nu(dy,d\mu),
$$

which satisfies
$$\|R\nu\|\le c\sigma_{\max}\|\nu\|. \tag{P2}$$

**Sources.**

* Volume: $Q=q(y)\,dy\,d\mu$.
* Surface, at time $u$: $\beta_u=\delta_0(dy)\otimes c\mu\,\phi_L(u)\mathbf 1\{\mu>0\}\,d\mu$.
  (The name $\beta_u$ for the surface source is unrelated to F&C's $\beta$.)

> **Drive approximation used by the code.** The code gives boundary packets
> uniform birth times within the step, and their total energy is the exact
> step integral. It therefore realizes $\phi_L$ replaced by the step average
> $\bar\phi_L=\tfrac12 a\,\overline{T_D^4}$, with
> $\overline{T_D^4}=\frac1{\Delta t}\int_{t^n}^{t^{n+1}}T_D^4\,dt$. Everything
> below uses $\bar\phi_L$. The error from this is $O(\Delta t)$ in the timing
> inside a step.

### 3.2 Definition (mild solution of (T))

A family $(\phi_t)_{t\in[t^n,t^{n+1}]}$ of measures is a **mild solution** if

* $\sup_t\|\phi_t\|<\infty$, and
* for every $t$ and every bounded Borel $h$, the integrand below is Borel in
  $u$ and

$$
\langle h,\phi_t\rangle=\langle h,S(t-t^n)\phi^n\rangle+\int_{t^n}^{t}\big\langle h,\,S(t-u)\big(R\phi_u+Q+\beta_u\big)\big\rangle\,du .
\tag{Mild}
$$

### 3.3 Lemma A (classical solutions are mild)

Let $\phi$ be a classical solution of (T)–(BC/IC): $C^1$ along characteristics
and bounded. Then $\phi\,dx\,d\mu$ is a mild solution.

*Proof.* Integrating the ODE of Section 2 with the integrating factor gives, at
each point $(x,\mu,t)$, the backward Duhamel formula. In it, $s$ runs over
$[0,\min(t-t^n,\tau_b(x,\mu))]$, where $\tau_b$ is the backward exit time.
Multiply by $h(x,\mu)$ and integrate in $x$ and $\mu$.

* **Volume, scattering and census terms.** For fixed $\mu$ and $s$, substitute
  $y=x-c\mu s$. The Jacobian is $1$. The condition $s<\tau_b(x,\mu)$ becomes
  $s<\tau(y,\mu)$, and the backward attenuation becomes $A_s(y,\mu)$. This
  turns each term into the corresponding term of (Mild), with $u=t-s$.
* **Inflow term.** For $\mu>0$ substitute $x=c\mu(t-u)$, so
  $dx=c\mu\,du$. Then
  $\int dx\,h\,\phi_L(t-x/c\mu)A_{x/c\mu}(0,\mu)=\int du\,c\mu\,\phi_L(u)\,h(X_{t-u}(0,\mu),\mu)A_{t-u}(0,\mu)$,
  which is the $\beta_u$ term. $\square$

The same proof applies unchanged in two more general cases. First, (T) may
carry an additional bounded measurable source $r(x,\mu,t)$; that source then
appears in (Mild) as the source measure $r\,dx\,d\mu\,du$ (see (Mild-$\Pi$) in
Section 4.3). Second, $\sigma$, $f$ and $q$ may be any bounded measurable
functions of $x$, not only piecewise constant ones. Sections 3–4 use only
boundedness and measurability of the coefficients.

### 3.4 Theorem 1 (uniqueness and existence)

(Mild) has at most one solution. A solution exists.

*Proof of uniqueness.* Let $\phi,\psi$ be two mild solutions and
$\delta_t=\phi_t-\psi_t$. Subtracting the two copies of (Mild), the census,
source and inflow terms cancel:

$$
\langle h,\delta_t\rangle=\int_{t^n}^{t}\langle h,S(t-u)R\delta_u\rangle\,du .
$$

Let $M=\sup_t\|\delta_t\|<\infty$. We show by induction on $k$ that

$$
\|\delta_t\|\le M\,\frac{\big(c\sigma_{\max}(t-t^n)\big)^k}{k!}\qquad\text{for all }t .
$$

* The case $k=0$ is the definition of $M$.
* If the bound holds for $k$, then for $|h|\le1$, by (P1) and (P2),
  $|\langle h,S(t-u)R\delta_u\rangle|\le c\sigma_{\max}\|\delta_u\|\le c\sigma_{\max}M(c\sigma_{\max}(u-t^n))^k/k!$.
  The integrand is Borel by the definition of a mild solution. Integrating
  gives $|\langle h,\delta_t\rangle|\le M(c\sigma_{\max}(t-t^n))^{k+1}/(k+1)!$,
  and taking the supremum over $h$ gives the bound for $k+1$.

Letting $k\to\infty$ gives $\delta_t=0$ for every $t$. $\square$

This argument never needs the measurability of $u\mapsto\|\delta_u\|$, nor any
regularity of $\delta$. It works for discrete census measures and for
discontinuous $\sigma$.

*Existence.* Theorem 2 below constructs a mild solution: the mean of the
particle process. Alternatively, Picard iteration converges, with the same
factorial bounds.

> **Corollary.** If (2.1) has a classical solution, then by Lemma A it
> coincides with the unique mild solution. In all cases, "the solution of
> (2.1)" means the mild solution.

The factorial bound is enough for uniqueness. Used as a *stability* estimate
it would carry the factor $e^{c\sigma_{\max}\Delta t}$, which is astronomically
large here: $\sigma\approx2.2\times10^{14}\,$cm⁻¹ in the cold foam. Lemma B
(Section 4.3) gives stability with constant $1$ instead.

### 3.5 Lemma P (changing the absorption coefficient)

Let $\sigma,\sigma'$ be bounded measurable, and let $S,S'$ be the corresponding
streaming–attenuation operators. Then for every finite measure $\nu$ and every
$s\ge0$,

$$
S'(s)\nu=S(s)\nu+\int_0^s S(s-v)\Big[c(\sigma-\sigma')\cdot S'(v)\nu\Big]\,dv,
$$

where $g\cdot\lambda$ denotes the measure $g(y)\lambda(dy,d\mu)$.

*Proof.* Fix a ray from $z$ and write $a(v)=c\sigma(X_v(z))$ and
$a'(v)=c\sigma'(X_v(z))$. Differentiating
$v\mapsto e^{-\int_v^s a}\,e^{-\int_0^v a'}$ and integrating over $[0,s]$ gives
the scalar Duhamel identity

$$
e^{-\int_0^s a'}-e^{-\int_0^s a}=\int_0^s e^{-\int_v^s a}\,(a-a')(v)\,e^{-\int_0^v a'}\,dv .
$$

The flow property gives $e^{-\int_v^s a}=A_{s-v}(X_v(z))$ and
$\mathbf 1\{s<\tau(z)\}=\mathbf 1\{v<\tau(z)\}\,\mathbf 1\{s-v<\tau(X_v(z))\}$.
Multiply the identity by $\mathbf 1\{s<\tau(z)\}\,h(X_s(z),\mu)$ and integrate
against $\nu(dz)$; by Fubini, this is the claim. $\square$

**Consequence.** Suppose $\psi'$ is a mild solution with coefficients
$(\sigma',f',q')$, census $\nu$ and inflow $\beta$. Apply Lemma P to every $S'$
in its mild equation and change the order of integration ($w=u+v$). Then
$\psi'$ is also a mild solution with coefficients $(\sigma,f,q)$ and the
additional source

$$
\rho_u=c(\sigma-\sigma')\cdot\psi'_u+(R'-R)\psi'_u+(Q'-Q). \tag{P}
$$

---

## 4. Theorem 2: the mean packet density is the mild solution

### 4.1 The idealized particle process

Fix the start-of-step state: temperatures, and the census measure
$\nu^n=\sum w_p\delta_{Z_p}$. Each packet has a state $Z=(x,\mu)$ and a weight
$w$.

**Flight.** Between events, $\dot x=c\mu$ and $\dot\mu=0$.

**Collisions.** A collision clock runs with hazard $c\sigma(x(t))$. In the
code, the collision distance is drawn with the current cell's $\sigma$ and
redrawn after each face crossing. By the memorylessness of the exponential,
this realizes exactly the survival probability $A_s$ along the path (line 271
and the face branch). At a collision:

* with probability $f$, the packet is **absorbed**: removed and added to
  `deposited`;
* otherwise it **scatters**: $\mu\leftarrow\mu'\sim U[-1,1]$, independent of
  the past (line 313).

**Removal.** A packet is removed if it exits $(0,L)$ (escape), or stored if it
reaches $t^{n+1}$ (census).

**Births** (lines ~188–208):

* Volume: in cell $j$, $N_j\ge1$ packets of weight $E_j/N_j$, where
  $E_j=f_jc\sigma_ju_{r,j}^n\Delta x\Delta t$. Position, direction and time are
  i.i.d. uniform on the cell, on $[-1,1]$ and on the step.
* Surface: $N_b$ packets of weight $E_b/N_b$ at $x=0^+$, with $\mu$ having
  density $2\mu$ and $t$ uniform on the step.

The 20000-event cap and the $10^{-10}\Delta x$ face nudge are not part of this
process. Proposition 2 (Section 6.2) bounds the effect of the cap.

Let $\nu_t=\sum_{\text{alive}}w_p\delta_{Z_p(t)}$ and define the mean measure
by $\langle h,m_t\rangle=\mathbb E\langle h,\nu_t\rangle$.

### 4.2 Statement and proof

> **Theorem 2.** $(m_t)$ is a mild solution of (T) with initial measure
> $\nu^n$, volume source $q$ and inflow $\bar\phi_L$. By Theorem 1, it is
> *the* solution.

*Proof.*

**(a) Segment decomposition.** Every packet alive at time $t$ is on a flight
segment that started at a **start event** $(z,u)$ with $u\le t$. A start event
is one of: census at $t^n$, a volume birth, a surface birth, or a scattering.
Since that start, the packet has had no collision and no exit, and
$Z(t)=X_{t-u}(z)$. Hence

$$
\langle h,\nu_t\rangle=\sum_{\text{start events }(z,u,w),\,u\le t} w\,\mathbf 1\{\text{segment survives to }t\}\,h(X_{t-u}(z),\mu).
$$

**(b) Survival probability.** Condition on the history up to and including a
start event. By the strong Markov property of the piecewise-deterministic
process (Davis, *Markov Models and Optimization*, 1993) and the collision clock
of 4.1, the segment survives to $t$ with probability
$\mathbf 1\{t-u<\tau(z)\}A_{t-u}(z)$. Define
$K_sh(z)=\mathbf 1\{s<\tau(z)\}A_s(z)h(X_s(z),\mu)$, so that
$\langle K_sh,\nu\rangle=\langle h,S(s)\nu\rangle$. Then

$$
\mathbb E\langle h,\nu_t\rangle=\mathbb E\sum_{\text{start events},\,u\le t} w\,K_{t-u}h(z).
$$

Exchanging the expectation and the sum is justified because there are
finitely many packets, scattering creates no packets, and each packet has at
most $\mathrm{Poisson}(c\sigma_{\max}\Delta t)$-dominated collisions. So
$\mathbb E[\#\text{events}]<\infty$, and the summand is bounded by $w|h|_\infty$.

**(c) Expected start-event measures.** For any bounded Borel $F(z,u)$:

* **Census:** contributes $\langle F(\cdot,t^n),\nu^n\rangle$.
* **Volume births:**
  $\mathbb E\sum wF=\sum_jN_j\frac{E_j}{N_j}\int\!\!\int\!\!\int F\,\frac{dy}{\Delta x}\frac{d\mu}{2}\frac{du}{\Delta t}=\int du\,\langle F(\cdot,u),Q\rangle$,
  since $E_j/(2\Delta x\Delta t)=q_j$.
* **Surface births:**
  $\mathbb E\sum wF=\int du\int_0^1 \frac{E_b}{\Delta t}\,2\mu\,F(0,\mu,u)\,d\mu=\int du\,\langle F(\cdot,u),\beta_u\rangle$,
  since $\frac{E_b}{\Delta t}2\mu=2\mu\sigma_{SB}\overline{T_D^4}=c\mu\bar\phi_L$
  (using $\sigma_{SB}=ac/4$).
* **Scatterings.** For each packet, the marked point process of scatterings
  (time, position, new direction) has compensator
  $(1-f)c\sigma(x_p(u^-))\mathbf 1\{\text{alive}\}\,du\otimes\tfrac12d\mu'$.
  The rate comes from the collision hazard thinned by $1-f$, and the mark from
  the independent uniform redirection. For the predictable integrand
  $F(x_p(u^-),\mu',u)$ the compensator identity (Brémaud, *Point Processes and
  Queues*, 1981; Jacod & Shiryaev, *Limit Theorems for Stochastic Processes*,
  ch. II.1) gives
  $\mathbb E\sum_{\text{scatterings}}wF=\int du\,\langle F(\cdot,u),Rm_u\rangle$.

**(d) Assembly.** Insert $F(z,u)=\mathbf 1\{u\le t\}K_{t-u}h(z)$ into (b) and
use (c):

$$
\langle h,m_t\rangle=\langle h,S(t-t^n)\nu^n\rangle+\int_{t^n}^t\big\langle h,S(t-u)(Rm_u+Q+\beta_u)\big\rangle\,du .
$$

* Borel measurability of the integrand follows from Fubini, because it is the
  expectation of a jointly measurable functional of càdlàg paths.
* $\sup_t\|m_t\|\le\langle1,\nu^n\rangle+\sum_jE_j+E_b<\infty$.

So $(m_t)$ is a mild solution. $\square$

**Unconditional version.** The equation is linear in its data. Averaging over
a random census (the previous step's output) gives the same statement with
$\nu^n$ replaced by $\mathbb E\nu^n$, provided the coefficients are held
fixed.

### 4.3 Lemma B (positivity, and stability with constant 1)

It is convenient to allow general data. Let $\nu$ be a finite signed census
measure on $\bar D$, and let $\Pi$ be a finite signed source measure on
$\bar D\times[t^n,t^{n+1}]$. $\Pi$ can contain $Q\,du$, $\beta_u\,du$, or a
residual $r\,dx\,d\mu\,du$. The general mild equation is

$$
\langle h,\phi_t\rangle=\langle h,S(t-t^n)\nu\rangle+\int_{\bar D\times[t^n,t]}K_{t-u}h(z)\,\Pi(dz,du)+\int_{t^n}^t\langle h,S(t-u)R\phi_u\rangle\,du .
\tag{Mild-$\Pi$}
$$

Here $K_s$ is the survival kernel of Section 4.2(b), with
$\langle K_sh,\nu\rangle=\langle h,S(s)\nu\rangle$. The proof of Theorem 1
applies unchanged, because the sources cancel in the difference of two
solutions. So (Mild-$\Pi$) has at most one solution.

> **Lemma B.**
> 1. If $\nu\ge0$ and $\Pi\ge0$, the solution exists and is a nonnegative
>    measure.
> 2. For signed data, $\displaystyle\|\phi_t\|\le\|\nu\|+|\Pi|\big(\bar D\times[t^n,t]\big)$
>    for every $t$ in the step.
> 3. Define a tally of the solution as the expected tally of the particle
>    representation used in the proof: deposition in a set of cells, escape
>    at either face, or the census mass. For nonnegative data each tally is
>    at most $\|\nu\|+\|\Pi\|$. For signed data (extended by linearity), the
>    absolute value of each tally is at most $\|\nu\|+|\Pi|$.
>
> The constant is $1$, independent of $\sigma$, $f$ and $\Delta t$.

*Proof.*

1. **Nonnegative data.** Build a particle process as in 4.1. Draw $N$ census
   packets i.i.d. from $\nu/\|\nu\|$, each with weight $\|\nu\|/N$. Draw $N'$
   source packets i.i.d. from $\Pi/\|\Pi\|$, each with weight $\|\Pi\|/N'$.
   Each source packet starts at its sampled point $(z,u)$. Steps (a)–(d) of
   Theorem 2 go through unchanged. The only modification is in (c), where the
   births now give $\mathbb E\sum wF=\int F\,d\Pi$. So the mean measure solves
   (Mild-$\Pi$). It is nonnegative, being the expectation of a nonnegative
   measure. No packet is ever created except by a birth, and weights never
   increase, so pathwise
   $\langle1,\nu_t\rangle\le\|\nu\|+\Pi(\bar D\times[t^n,t])$. Taking
   expectations gives item 2 for nonnegative data.
2. **Signed data.** Use the Jordan decompositions $\nu=\nu^+-\nu^-$ and
   $\Pi=\Pi^+-\Pi^-$, and let $\phi^\pm$ be the solutions for
   $(\nu^\pm,\Pi^\pm)$. By linearity, $\phi^+-\phi^-$ solves (Mild-$\Pi$)
   with data $(\nu,\Pi)$, so it is *the* solution. Then
   $\|\phi_t\|\le\|\phi^+_t\|+\|\phi^-_t\|\le\|\nu^+\|+\|\nu^-\|+\Pi^+(\cdot)+\Pi^-(\cdot)=\|\nu\|+|\Pi|(\cdot)$.
3. **Tallies.** Pathwise, every unit of weight ends up in exactly one of:
   deposition, escape left, escape right, or census (Section 5.2). So for
   nonnegative data each tally is at most the input mass. The signed case
   follows from the Jordan decomposition as in item 2. $\square$

For a measure with a density, the total-variation norm is the $L^1$ norm, so
Lemma B is an $L^1$ stability estimate.

---

## 5. Tallies, energy conservation, and F&C's continuous absorption

### 5.1 Tallies

**Deposition.** The same compensator argument applies, with absorption
intensity $fc\sigma$:

$$
\mathbb E[\mathcal E_j]=\int_{t^n}^{t^{n+1}} f_jc\sigma_j\;m_u\big(\text{cell }j\times[-1,1]\big)\,du .
$$

When $m_u$ has a density $\phi$, this equals
$\Delta x\,\frac{\sigma}{1+\beta c\Delta t\sigma}\int dt\int I\,d\mu$, which
is F&C (2.2).

**Census.** $\mathbb E[\text{census measure}]=m_{t^{n+1}}$, which is the
initial datum of the next step.

**Material update.** Line 409, $U_j^{n+1}=U_j^n+(\mathcal E_j-E_j)/\Delta x$,
is F&C (1.18) with $S=0$ (Section 1.8). The expected update equals the update
driven by the exact solution of (T).

**Right-face leakage.** By (a)–(c) with $h$ replaced by the probability of
exiting at $x=L$ before $t^{n+1}$:

$$
\mathbb E[\text{escaped\_right}]=\langle G,\nu^n\rangle+\int_{t^n}^{t^{n+1}}\langle G(\cdot,u),Rm_u+Q+\beta_u\rangle\,du,
$$
$$
G(z,u)=\mathbf 1\{\text{the ray from }z\text{ exits at }x=L\text{ before }t^{n+1}-u\}\,A_{\tau(z)}(z).
$$

If the solution has a density with a trace at $x=L$, the change of variables
of Lemma A turns this into $\int dt\int_0^1c\mu\,\phi(L,\mu,t)\,d\mu=\int F(L,t)\,dt$.
In that case `Fout` estimates the step-averaged outgoing flux without bias.
That trace regularity is assumed here, not proved.

### 5.2 Exact energy conservation in every run

Every packet's weight ends in exactly one of `deposited`, `escaped_left`,
`escaped_right` or census. The material loses exactly $E_j$ to emission, and
the surface adds exactly $E_b$. So the energy balance holds **in every
realization**, not only in expectation. This is the discrete counterpart of
Section 1.8, and it is why the F&C energy check (Eq. 4.29) reads about
$10^{-14}$, i.e. round-off. F&C quote about $10^{-12}$ on the CDC 7600.

### 5.3 Continuous absorption (F&C (4.25b), (4.26c), (4.27)) has the same mean

In F&C's version, collisions are sampled only for scattering, at hazard
$(1-f)c\sigma$, and the weight decays as $w\,e^{-\int fc\sigma}$. A segment's
expected surviving weight is therefore
$w\,e^{-\int(1-f)c\sigma}e^{-\int fc\sigma}=w\,A_s$, which gives the same $S$.
The scattering compensator, weighted by the current weight, is the same $R$
applied to the weighted measure. Steps (a)–(d) go through unchanged, so both
algorithms are unbiased for the same unique solution. They differ only in
variance and cost.

---

## 6. Guarantees for the code as written

### 6.1 Proposition 1 (the material energy stays positive in every run)

> **Proposition 1.** Assume $d\ln u_m/d\ln T<4$ at the temperatures that
> occur. This holds at every temperature when $u_m\propto T^{b}$ with $b<4$.
> Then, with $\alpha=1$, in every realization, every cell and every step,
> $$U_j^{n+1}>U_j^n-\frac{T}{4}\frac{du_m}{dT}\Big|_{T_j^n}>0 .$$
> For the code's EOS ($b=\beta_{\rm EOS}=1.10$) this gives
> $U_j^{n+1}>(1-b/4)\,U_j^n=0.725\,U_j^n$, and
> $\theta_j^{n+1}>0.725^{1/b}\,\theta_j^n\approx0.7465\,\theta_j^n$.

*Proof.* The update (line 409) is $U^{n+1}=U^n+(\mathcal E_j-E_j)/\Delta x$
with $\mathcal E_j\ge0$, so $U^{n+1}\ge U^n-E_j/\Delta x$.

* By definition, $E_j/\Delta x=fc\sigma\Delta t\,u_r^n$.
* $fc\sigma\Delta t=\frac{c\sigma\Delta t}{1+\beta c\sigma\Delta t}<\frac1\beta$,
  since $x/(1+\beta x)<1/\beta$ for every $x\ge0$. Hence $E_j/\Delta x<u_r/\beta$.
* $u_r/\beta=u_r\frac{du_m}{du_r}=u_r\frac{du_m/dT}{4aT^3}=\frac T4\frac{du_m}{dT}$,
  and this is less than $u_m$ by hypothesis.
* For $u_m=E_{100}H\theta^b$, $\frac T4\frac{du_m}{dT}=\frac b4u_m$. The
  temperature bound follows from $\theta\propto U^{1/b}$. $\square$

**Consequences.**

* The code's `RuntimeError` branch (non-positive material energy) cannot be
  reached for any $\Delta t$ or any number of packets, apart from
  floating-point round-off.
* Positive material energy bounds the total packet weight (Lemma D in
  Section 7).
* The per-step temperature bound is too weak to bound temperatures from below
  over a whole run: $0.7465^{3000}$ is negligible.

### 6.2 Proposition 2 (the 20000-event cap)

> **Proposition 2.** Fix a step and assume $f_j\ge f_{\min}>0$ in every cell.
> Let
> $$K^*=\Big\lceil\tfrac12\Big(19999-\frac{c\Delta t}{(1-10^{-10})\,\Delta x}\Big)\Big\rceil .$$
> Then for every packet, $\Pr[\text{the cap is reached}]\le(1-f_{\min})^{K^*-1}$.
> Consequently, for every tally, the expected difference between the code and
> the idealized process is at most $W(1-f_{\min})^{K^*-1}$, where $W$ is the
> total packet weight in the step.

*Proof.*

* **Counting passes.** Each pass through the loop in `transport_step` ends in
  exactly one of: a collision, a face crossing, census, or escape. Census and
  escape end the history, so at most one pass is terminal. With $K$
  collisions and $F$ face crossings there are at most $K+F+1$ passes. The cap
  is checked at the top of pass 20001, so reaching it requires $K+F\ge20000$.
* **Bounding face crossings.** The history consists of at most $K+1$ straight
  segments separated by collisions. After a crossing, the nudge places the
  packet $\epsilon=10^{-10}\Delta x$ past the face without using time. So on
  one segment, consecutive crossings are separated by horizontal flight of at
  least $\Delta x-\epsilon$, and a segment with horizontal flight $e_i$ has at
  most $e_i/(\Delta x-\epsilon)+1$ crossings. Time advances by flight
  length$/c$ and the step lasts at most $\Delta t$, so $\sum_ie_i\le c\Delta t$.
  Hence $F\le c\Delta t/(\Delta x-\epsilon)+K+1$.
* **Combining.** $K+F\ge20000$ then forces
  $2K\ge19999-c\Delta t/(\Delta x-\epsilon)$, i.e. $K\ge K^*$.
* **Probability.** A history reaches $K^*$ collisions only if its first
  $K^*-1$ collisions were all scatterings. At each collision, the code uses a
  fresh uniform (line 313), so conditionally on the past, absorption happens
  with probability $f_{\rm cell}\ge f_{\min}$. Therefore
  $\Pr[K\ge K^*]\le(1-f_{\min})^{K^*-1}$.
* **Tally bias.** A capped packet differs from the idealized one only in
  where its weight $w$ is tallied, so each tally changes by at most $w$.
  Summing over packets gives the bound. $\square$

**Numbers for the default run.** Here $c\Delta t/\Delta x=29.98$, so
$K^*=9985$. In this model
$\beta\sigma=\frac{4\sigma_0}{Hb}\theta^{4-b-\alpha_{\rm op}}=\frac{4\sigma_0}{Hb}\theta^{-0.63}$
grows as $\theta$ falls. So $f$ is smallest in the coldest cell:

| Coldest $\theta$ in the step | $\beta c\sigma\Delta t$ | $f_{\min}$ | Bound on $\Pr[\text{cap}]$ |
|---|---|---|---|
| $\theta_0$ (300 K) | 2.30 | 0.303 | $10^{-1567}$ |
| $\theta_0/1000$ (0.3 K) | 178 | 0.0056 | $6\times10^{-25}$ |
| $\theta_{\rm floor}=10^{-12}$ | $4.6\times10^{5}$ | $2.2\times10^{-6}$ | 0.98 (no information) |

The condition on the coldest temperature is **not proved**. Proposition 1
alone allows a 25 % temperature drop per step. It can, however, be checked
at runtime: recording $\min_j\theta_j$ at every step would certify the bound
for a given run. The script does not record it. The final profile of the
default run has a minimum of 110.9 eV, which says nothing about earlier steps.

### 6.3 Proposition 3 (the step-averaged drive)

> **Proposition 3.** Fix a step's coefficients and census. Let $m_t$ solve (T)
> with the exact inflow $\phi_L(t)=\tfrac12aT_D(t)^4$, and let $\bar m_t$
> solve it with the step-averaged inflow $\bar\phi_L$ used by the code. Then
> for every $t$ in the step,
> $$\|m_t-\bar m_t\|\le\sigma_{SB}\int_{t^n}^{t^{n+1}}\big|T_D(u)^4-\overline{T_D^4}\big|\,du\le\frac{\sigma_{SB}\,\mathrm{Lip}(T_D^4)}{3}\,\Delta t^2 .$$
> Every expected tally (deposition in any cell, escape at either face, census
> mass) differs by at most the same amount. The total energy delivered in the
> step is identical.

*Proof.*

* The difference $m_t-\bar m_t$ is a solution of (Mild-$\Pi$) with zero
  census, zero volume source and the signed inflow source
  $\Pi=(\beta_u-\bar\beta_u)\,du$.
* Its total variation is
  $\int du\int_0^1c\mu\cdot\tfrac12a\,|T_D^4-\overline{T_D^4}|\,d\mu=\sigma_{SB}\int|T_D^4-\overline{T_D^4}|\,du$.
* Lemma B(2) gives the first inequality, and Lemma B(3) the statement for
  tallies.
* For the second inequality, let $g=T_D^4$, which is Lipschitz on the step.
  Then $|g(u)-\bar g|\le\frac1{\Delta t}\int|g(u)-g(v)|\,dv$, so
  $\int|g-\bar g|\,du\le\frac{\mathrm{Lip}(g)}{\Delta t}\iint_{[t^n,t^{n+1}]^2}|u-v|\,du\,dv=\frac{\mathrm{Lip}(g)\,\Delta t^2}{3}$.
* The drive table is interpolated linearly, so
  $\mathrm{Lip}(T_D^4)\le4T_{D,\max}^3\max|dT_D/dt|$. $\square$

The bound holds per step, at fixed coefficients. How the error propagates
through later steps is not quantified.

---

## 7. Theorem 3: many coupled steps as the number of packets grows

### 7.1 Setting

* **What is fixed:** the cells (number $J$), $\Delta t$, the number of steps
  $n_s$, and the drive.
* **The packet-count parameter $N$:** `n_boundary` $=\lceil a_bN\rceil$, the
  material target is $\lceil a_mN\rceil$, and `max_census`
  $=M_N=\lceil a_cN\rceil$, where $a_c=\infty$ means no resampling.
* **The process:** the idealized process of 4.1, without the cap; Proposition
  2 bounds the difference. Random numbers are idealized as independent
  uniforms.
* **Resampling:** one of the following holds.
  * **(R0)** resampling is never triggered; or
  * **(R1)** the resampling is unbiased, $\mathbb E[\langle g,\nu_{\rm post}\rangle\mid\nu_{\rm pre}]=\langle g,\nu_{\rm pre}\rangle$,
    and has conditional variance at most $W^2\|g\|_\infty^2/M$, where $W$ is
    the total weight and $M$ the number of output packets. Multinomial
    resampling satisfies (R1). Stratified resampling has conditional variance
    no larger than multinomial (Douc, Cappé & Moulines 2005), so it satisfies
    (R1) too.

  The code's systematic resampling is unbiased (Section 9). However, no
  general variance bound is known for systematic resampling; Douc, Cappé &
  Moulines discuss this. If it is triggered, Theorem 3 does not cover it. In
  the default runs it was never triggered: the census peaked at about 17,600
  packets against `max_census` = 120,000. Replacing the single comb offset by
  one independent offset per tooth turns it into stratified resampling.

**The deterministic IMC scheme $\mathfrak D$.** The state is
$x=(\theta,\nu)$: cell temperatures $\theta\in[\theta_{\rm floor},\infty)^J$
and a finite nonnegative census measure $\nu$. The map $\mathfrak D(x)$ does
the following:

* compute the coefficients from $\theta$;
* solve (T) exactly with census $\nu$, the sources $Q(\theta)$ and the
  step-averaged drive;
* take the new census $\nu'$ as the solution at $t^{n+1}$, and the
  depositions $\bar{\mathcal E}_j$ as in Section 5.1;
* update $U'_j=U_j(\theta_j)+(\bar{\mathcal E}_j-E_j(\theta))/\Delta x$ and
  $\theta'=\mathrm{EOS}^{-1}(U')$, with the floor.

Then $\bar x^n=\mathfrak D^n(x^0)$, with $x^0=(\theta_0\mathbf 1,0)$. In words:
exact transport, exact deposition, F&C (1.18), on the code's grid, with no
Monte Carlo noise. Proposition 1 holds for $\mathfrak D$ as well.

**History functionals.** A history functional $\Phi$ is a bounded measurable
function of one packet's start $(z,u)$ and its path during the step,
including its fate. For coefficients $\theta$, let
$\hat\Phi^\theta(z,u)=\mathbb E^\theta_{z,u}[\Phi]$. Examples:

* $\Phi=\mathbf 1\{\text{absorbed in cell }j\}$: deposition;
* $\Phi=\mathbf 1\{\text{escapes at }x=L\}$: leakage;
* $\Phi=g(Z_{t^{n+1}})\mathbf 1\{\text{in census}\}$: the census functional
  $\langle g,\cdot\rangle$.

### 7.2 Lemma C (the transport kernel depends continuously on the coefficients)

For coefficient sets $\theta,\theta'$, let $\hat\sigma=\max(\sigma_{\max},\sigma'_{\max})$.
Then

$$
\sup_{z,u}\big|\hat\Phi^\theta(z,u)-\hat\Phi^{\theta'}(z,u)\big|\le2\|\Phi\|_\infty\Big(c\Delta t\max_j|\sigma_j-\sigma'_j|+c\hat\sigma\Delta t\max_j|f_j-f'_j|\Big).
$$

Since $\sigma_j$ and $f_j$ are continuous in $\theta_j\ge\theta_{\rm floor}$,
the map $\theta\mapsto\hat\Phi^\theta$ is continuous in the sup norm.

*Proof (coupling by thinning).*

* Let $\Lambda=c\hat\sigma$. Run a Poisson process of candidate events with
  rate $\Lambda$ along the flight.
* At each candidate, at position $x$ in cell $j$, draw independent uniforms
  $U_1,U_2,U_3$:
  * it is a collision if $U_1<c\sigma_j/\Lambda$;
  * a collision is an absorption if $U_2<f_j$;
  * otherwise the packet scatters to $\mu=2U_3-1$.
* Accepted candidates form a point process with intensity $c\sigma(x(t))$
  (Lewis–Shedler thinning). So this construction has the law of the process
  in 4.1.
* Drive the $\theta$ and $\theta'$ processes with the same Poisson process and
  the same uniforms. Their paths coincide until the first candidate at which
  the decisions differ.
* At a given candidate, the decisions differ with probability at most
  $c|\sigma_j-\sigma'_j|/\Lambda+|f_j-f'_j|$. The uniforms at a candidate are
  independent of whether that candidate exists.
* The expected number of candidates in the step is at most $\Lambda\Delta t$.
  A union bound over candidates gives
  $\Pr[\text{paths differ}]\le c\Delta t\max|\sigma-\sigma'|+\Lambda\Delta t\max|f-f'|$.
* When the paths coincide, so does $\Phi$. Hence
  $|\hat\Phi^\theta-\hat\Phi^{\theta'}|\le2\|\Phi\|_\infty\Pr[\text{differ}]$,
  uniformly in the start. $\square$

### 7.3 Lemma D (weights are $O(1/N)$ and total energy is bounded)

Let $E_{\max}=\sum_jU_j^0\Delta x+\int_0^{t_{\rm end}}\sigma_{SB}T_D^4\,dt$.
For $N>2J/a_m$, at every step:

* the total packet weight is at most $E_{\max}$;
* every packet weight is at most $C_w/N$, with
  $C_w=E_{\max}\max(1/a_b,\,2/a_m,\,1/a_c)$.

*Proof.*

* **Total weight.** Energy is conserved in every run (Section 5.2), and the
  material energy is positive (Proposition 1). So the radiation energy, i.e.
  the packet weight, is at most $E_{\max}$. The births in one step total
  $E_b+\sum_jE_j$, where $\sum_jE_j<\sum_j(b/4)U_j\Delta x$. Both are also
  at most $E_{\max}$.
* **Boundary packets.** Their weight is $E_b/n_b\le E_{\max}/(a_bN)$.
* **Material packets.** The code sets $N_j=1+\lfloor\text{remaining}\cdot E_j/E_{\rm tot}\rfloor$,
  possibly plus one leftover. With remaining $\ge a_mN-J\ge a_mN/2$, this
  gives $E_j/N_j\le E_{\rm tot}/\text{remaining}\le2E_{\max}/(a_mN)$.
* **Census packets.** They keep their birth weight (absorption is analog), or
  have weight $W/M_N\le E_{\max}/(a_cN)$ after resampling. $\square$

### 7.4 Theorem 3

> **Theorem 3.** Under the setting of 7.1, for every $n\le n_s$, as
> $N\to\infty$:
> 1. $\theta^n\to\bar\theta^n$ in probability;
> 2. $\langle g,\nu^n\rangle\to\langle g,\bar\nu^n\rangle$ in probability, for
>    every bounded Borel $g$ on $\bar D$;
> 3. every tally of step $n$ (deposition per cell, `escaped_left`,
>    `escaped_right`, and hence `Fout`) converges in probability to the
>    corresponding value of $\mathfrak D$.
>
> All tallies are bounded by $E_{\max}$. So by bounded convergence the
> expectations converge too: **the bias of the multi-step Monte Carlo
> estimator tends to $0$.** No rate is claimed.

*Proof.* By induction on $n$. For $n=0$ everything is deterministic and
equal. Assume items 1–2 hold for step $n$. Let $\mathcal F_n$ be the history
up to the start of step $n$. Fix a history functional $\Phi$ and let
$Y=\sum_pw_p\Phi(\text{history}_p)$, summed over all packets of step $n$
(census and births).

1. **Conditional mean.** By 4.2(c), and independently of the packet counts,
   $$\mathbb E[Y\mid\mathcal F_n]=\langle\hat\Phi^{\theta^n}(\cdot,t^n),\nu^n\rangle+B_\Phi(\theta^n),$$
   where $B_\Phi(\theta)=\sum_jE_j(\theta)\,\mathrm{avg}_j[\hat\Phi^\theta]+E_b\,\mathrm{avg}_b[\hat\Phi^\theta]$
   averages over the birth distributions. $B_\Phi$ is continuous, by Lemma C
   and the continuity of $E_j$.
2. **Conditional variance.** Given $\mathcal F_n$, the histories are
   independent. So
   $\mathrm{Var}(Y\mid\mathcal F_n)\le\sum_pw_p^2\|\Phi\|_\infty^2\le\|\Phi\|_\infty^2\,(C_w/N)\cdot3E_{\max}$
   by Lemma D. (The census weighs at most $E_{\max}$, and the births at most
   $2E_{\max}$.) Hence $Y-\mathbb E[Y\mid\mathcal F_n]\to0$ in $L^2$, and so in
   probability.
3. **Deterministic value.** Let
   $\bar y=\langle\hat\Phi^{\bar\theta^n}(\cdot,t^n),\bar\nu^n\rangle+B_\Phi(\bar\theta^n)$.
   This is the corresponding quantity of $\mathfrak D$: apply Theorem 2 to a
   census sampled from $\bar\nu^n$, as in the proof of Lemma B.
4. **Comparison.** Split the difference:
   $$|\mathbb E[Y\mid\mathcal F_n]-\bar y|\le\|\hat\Phi^{\theta^n}-\hat\Phi^{\bar\theta^n}\|_\infty E_{\max}+|\langle\hat\Phi^{\bar\theta^n},\nu^n-\bar\nu^n\rangle|+|B_\Phi(\theta^n)-B_\Phi(\bar\theta^n)|.$$
   * The first term tends to $0$ in probability, by item 1 of the hypothesis,
     Lemma C and the continuous mapping theorem.
   * The second tends to $0$ by item 2 of the hypothesis, applied to the
     *fixed* bounded Borel function $\hat\Phi^{\bar\theta^n}(\cdot,t^n)$.
   * The third tends to $0$ by continuity of $B_\Phi$.

   Therefore $Y\to\bar y$ in probability, which proves item 3 for step $n$.
5. **Census.** Take $\Phi=g(Z_{t^{n+1}})\mathbf 1\{\text{census}\}$; this gives
   item 2 before resampling. Under (R1), resampling adds a term with zero
   conditional mean and conditional variance at most
   $E_{\max}^2\|g\|_\infty^2/(a_cN)\to0$. Under (R0) there is no change. So
   item 2 holds at step $n+1$.
6. **Temperatures.** $U_j^{n+1}=U_j(\theta_j^n)+(\mathcal E_j-E_j(\theta^n))/\Delta x\to\bar U_j^{n+1}$
   by item 4 and continuity, and $\mathrm{EOS}^{-1}$ is continuous. So item 1
   holds at step $n+1$. $\square$

### 7.5 What Theorem 3 does and does not say

* **Bias and rate.** The bias provably tends to $0$. The rate, heuristically
  $O(1/N)$, is not proved. A rate would need
  Lipschitz rather than mere continuous dependence, plus a CLT-type
  argument.
* **Order of limits.** The convergence holds for a fixed $\Delta t$, $\Delta x$
  and number of steps. Nothing is claimed uniformly as $\Delta t\to0$, which
  would mean $n_s\to\infty$.
* **Resampling.** The code's systematic resampling is covered only while it
  is not triggered (R0).

---

## 8. Convergence in $\Delta t$: a conditional theorem, and why the general problem is open

This section isolates link (A), the F&C time linearization. Monte Carlo
noise and spatial cells are removed. The question is whether the
deterministic IMC scheme converges to the solution of Eq. (1.1) as
$\Delta t\to0$.

### 8.1 Setting and hypotheses

**The semi-discrete scheme $\mathfrak D_{\Delta t}$** (continuous in $x$ and
$\mu$). Given $(\phi^n,u_m^n)$, with temperature $T^n(x)$:

1. compute the coefficients $\sigma^n(x)$, $\beta^n(x)$, $f^n(x)$ and
   $u_r^n(x)$ from $T^n(x)$, pointwise;
2. solve (T) exactly over the step (mild solution, Section 3), with the exact
   inflow $\phi_L(t)$;
3. update $u_m^{n+1}(x)=u_m^n+f^nc\sigma^n\int_{t^n}^{t^{n+1}}\!\int\phi\,d\mu\,dt-f^nc\sigma^n\Delta t\,u_r^n$,
   which is F&C (1.18).

**Error norm.** $e^n=\|\phi^n-\phi(t^n)\|_{L^1(D)}+\|u_m^n-u_m(t^n)\|_{L^1(0,L)}$.

**Hypotheses.** Fix $K=[T_{\rm lo},T_{\rm hi}]$ with $T_{\rm lo}>0$ and
$\Delta t\le\Delta t_0$.

* **(H1) Material data.** $u_m$ is $C^1$ on $K$ with $du_m/dT\ge\kappa>0$.
  $\sigma$ and $\beta$ are Lipschitz on $K$. Write $\sigma_{\max}$,
  $\beta_{\max}$, $u_{\max}=aT_{\rm hi}^4$ for the bounds on $K$. Let
  $\mathcal L$ be a common Lipschitz constant on $K$ (in $T$) of $\sigma$,
  $\sigma u_r$, $f\sigma$ and $f\sigma u_r$, uniform in
  $\Delta t\le\Delta t_0$. It exists because $f\sigma=\sigma/(1+c\Delta t\beta\sigma)$
  has bounded derivatives in $(\sigma,\beta)$ on bounded sets.
* **(H2) Exact solution.** Eq. (1.1) with the boundary and initial data has a
  solution on $[0,t_{\rm end}]$ that is classical along characteristics, with
  $0\le\phi\le\Phi_{\max}$, $T(x,t)\in K$, and
  $|T(x,t)-T(x,s)|\le M_T|t-s|$.
* **(H3) The scheme stays in $K$.** Every scheme step used below, started
  either from the scheme's own state or from the exact state $x(t^n)$, has
  temperatures in $K$.

**Maximum principle for one step (so no a priori bound on $\phi$ is needed).**
Let $U\ge\max(u_r^n,\ 2\|\phi^n\|_\infty,\ 2\sup\phi_L)$. The constant
$\tfrac12U$ is a classical solution of (T) with inflow $\tfrac12U$ and source
$\tfrac12fc\sigma U\ge q$. Then $\tfrac12U-\phi$ has nonnegative census,
inflow and source $\tfrac12fc\sigma(U-u_r^n)$. By Lemma A and Lemma B(1),
$0\le\phi\le\tfrac12U$ throughout the step.

Inductively, under (H3), all scheme solutions satisfy
$\phi\le\Phi_{\max}:=\tfrac12\max(2\|\phi(0)\|_\infty,\,aT_{D,\max}^4,\,aT_{\rm hi}^4)$.
Enlarge $\Phi_{\max}$ in (H2) to this value if necessary.

### 8.2 Lemma E (local error)

Start one scheme step from the exact state at $t^n$. Then

$$
\|\tilde\phi(t^{n+1})-\phi(t^{n+1})\|_{L^1}+\|\tilde u_m^{n+1}-u_m(t^{n+1})\|_{L^1}\le C_{\rm loc}\,\Delta t^2 .
$$

*Proof.*

**Residual.** Write $\sigma(t)=\sigma(T(x,t))$ and $u_r(t)=u_r(T(x,t))$ along
the exact solution. In $\phi$-form, the exact transport equation is
$\partial_t\phi+c\mu\partial_x\phi+c\sigma(t)\phi=\tfrac12c\sigma(t)u_r(t)$,
since (1.1a) has no scattering. Rewritten with the frozen scheme coefficients,
it reads

$$
\partial_t\phi+c\mu\partial_x\phi+c\sigma^n\phi-(1-f^n)c\sigma^n\tfrac12\!\int\phi\,d\mu'-q^n=r,
$$
$$
r=c(\sigma^n-\sigma(t))\phi+\tfrac12c\big[\sigma(t)u_r(t)-\sigma^nu_r^n\big]+\tfrac12(1-f^n)c\sigma^n\Big[u_r^n-\int\phi\,d\mu'\Big].
$$

**Bound on the residual.** Use $|\sigma^n-\sigma(t)|\le\mathcal LM_T\Delta t$,
$|\sigma u_r(t)-\sigma^nu_r^n|\le\mathcal LM_T\Delta t$, and
$1-f^n=\frac{\beta c\sigma\Delta t}{1+\beta c\sigma\Delta t}\le\beta_{\max}c\sigma_{\max}\Delta t$.
This gives

$$
|r|\le C_r\Delta t,\qquad C_r=c\mathcal LM_T\big(\Phi_{\max}+\tfrac12\big)+\tfrac12c^2\beta_{\max}\sigma_{\max}^2\,(u_{\max}+2\Phi_{\max}).
$$

**Transport error.** By Lemma A (with a source, Section 3.3), $\phi$ solves
(Mild-$\Pi$) with the scheme coefficients, the same census and inflow, and the
additional source $r\,dx\,d\mu\,du$. The scheme solution $\tilde\phi$ solves
it without $r$. By uniqueness and Lemma B(2),

$$
\|\phi_t-\tilde\phi_t\|_{L^1}\le\int_{t^n}^t\|r_u\|_{L^1(D)}\,du\le2L\,C_r\,\Delta t\,(t-t^n).
$$

**Material error.** Pointwise in $x$, the exact update is
$u_m(t^{n+1})-u_m(t^n)=\int c\sigma(t)\big(\int\phi\,d\mu-u_r(t)\big)\,dt$. The
difference from the scheme's update is at most

$$
\int\!|c\sigma(t)-f^nc\sigma^n|\!\int\!|\phi|\,d\mu\,dt+f^nc\sigma^n\!\int\!\Big|\!\int(\phi-\tilde\phi)\,d\mu\Big|dt+\int\!|c\sigma(t)u_r(t)-f^nc\sigma^nu_r^n|\,dt .
$$

Here $|c\sigma(t)-f^nc\sigma^n|\le C_\sigma\Delta t$ with
$C_\sigma=c\mathcal LM_T+c^2\beta_{\max}\sigma_{\max}^2$, and
$|c\sigma(t)u_r(t)-f^nc\sigma^nu_r^n|\le(c\mathcal LM_T+c^2\beta_{\max}\sigma_{\max}^2u_{\max})\Delta t$.
Integrating over $x\in(0,L)$:

$$
\|\tilde u_m^{n+1}-u_m(t^{n+1})\|_{L^1}\le2L\Phi_{\max}C_\sigma\Delta t^2+c\sigma_{\max}LC_r\Delta t^3+L\big(c\mathcal LM_T+c^2\beta_{\max}\sigma_{\max}^2u_{\max}\big)\Delta t^2 .
$$

Adding the two errors gives $C_{\rm loc}=2LC_r+2L\Phi_{\max}C_\sigma+L(c\mathcal LM_T+c^2\beta_{\max}\sigma_{\max}^2u_{\max})+c\sigma_{\max}LC_r\Delta t_0$. $\square$

### 8.3 Lemma F (one step is Lipschitz with constant $1+L_s\Delta t$)

Let $x=(\phi^0,u_m)$ and $x'=(\phi'^0,u'_m)$ be states whose scheme steps
satisfy (H3). Then

$$
\|\mathfrak D_{\Delta t}x-\mathfrak D_{\Delta t}x'\|\le(1+L_s\Delta t)\,\|x-x'\|,\qquad L_s=c\sigma_{\max}+\frac{C_\rho+(2\Phi_{\max}+1)c\mathcal L+c\sigma_{\max}\Delta t_0C_\rho}{\kappa},
$$

with $C_\rho=(6\Phi_{\max}+1)c\mathcal L$.

*Proof.* By (H1), $|T-T'|\le|u_m-u'_m|/\kappa$ pointwise.

**Transport.** Let $\psi,\psi'$ be the transport solutions with the
coefficients from $T$ and $T'$. By Lemma P and its consequence (P), $\psi'$
solves the equation with the coefficients of $T$ plus the source $\rho$. With
$0\le\psi'\le\Phi_{\max}$:

* $\|c(\sigma-\sigma')\psi'\|\le2c\Phi_{\max}\mathcal L\|T-T'\|_{L^1}$;
* $\|(R'-R)\psi'\|\le2c\Phi_{\max}\cdot2\mathcal L\|T-T'\|_{L^1}$, since
  $(1-f)\sigma=\sigma-f\sigma$;
* $\|Q'-Q\|\le c\mathcal L\|T-T'\|_{L^1}$.

So $\|\rho_u\|\le C_\rho\|T-T'\|_{L^1}\le(C_\rho/\kappa)\|u_m-u'_m\|_{L^1}$.
The difference $\psi-\psi'$ solves (Mild-$\Pi$) with census $\phi^0-\phi'^0$
and source $-\rho$. By Lemma B(2), for $u$ in the step,

$$
\|\psi_u-\psi'_u\|\le\|\phi^0-\phi'^0\|+(u-t^n)\,(C_\rho/\kappa)\,\|u_m-u'_m\| .
$$

**Material.** Pointwise,

$$
|u_m^{\rm new}-u'^{\rm new}_m|\le|u_m-u'_m|+|fc\sigma-f'c\sigma'|\iint|\psi'|+fc\sigma\iint|\psi-\psi'|+\Delta t\,|fc\sigma u_r-f'c\sigma'u'_r| .
$$

Integrating over $x$, and using $\iint|\psi'|\,d\mu\,dt\le2\Phi_{\max}\Delta t$
at each $x$, gives

$$
\|u_m^{\rm new}-u'^{\rm new}_m\|\le\|u_m-u'_m\|\big(1+\Delta t\,[(2\Phi_{\max}+1)c\mathcal L+c\sigma_{\max}\Delta tC_\rho]/\kappa\big)+c\sigma_{\max}\Delta t\,\|\phi^0-\phi'^0\| .
$$

Adding the transport bound at $u=t^{n+1}$ gives the claim. $\square$

### 8.4 Theorem 4 (conditional convergence in $\Delta t$)

> **Theorem 4.** Under (H1)–(H3), for $\Delta t\le\Delta t_0$,
> $$\max_{t^n\le t_{\rm end}}e^n\le C_{\rm loc}\,t_{\rm end}\,e^{L_st_{\rm end}}\,\Delta t .$$
> The F&C semi-discrete scheme, (2.1) with (1.18), therefore converges to the
> solution of (1.1) at first order in $\Delta t$.

*Proof.* Split the error into a propagated part and a local part:

$$
e^{n+1}\le\|\mathfrak Dx^n-\mathfrak Dx(t^n)\|+\|\mathfrak Dx(t^n)-x(t^{n+1})\|\le(1+L_s\Delta t)\,e^n+C_{\rm loc}\Delta t^2,
$$

by Lemmas F and E. Since $e^0=0$,
$e^n\le C_{\rm loc}\Delta t^2\sum_{k<n}(1+L_s\Delta t)^k\le C_{\rm loc}\,t^n\,e^{L_st^n}\,\Delta t$. $\square$

### 8.5 Why this does not settle the question for the SiO₂ problem, and why it is open

Theorem 4 is a correct statement, but it is **vacuous for the problem the code
solves**. Closing the gap is a research problem, for the following reasons.

1. **The constants depend on the opacity, which is enormous.** $K$ must
   contain the cold foam at 300 K, where $\sigma=2.2\times10^{14}\,$cm⁻¹.
   * $L_s\ge c\sigma_{\max}$ gives $e^{L_st_{\rm end}}\ge e^{2\times10^{16}}$
     for $t_{\rm end}=3\,$ns.
   * $C_{\rm loc}$ contains $c^2\beta_{\max}\sigma_{\max}^2$.
   * The bound is meaningful only for $\Delta t\ll1/(c\sigma_{\max})\approx1.5\times10^{-16}\,$ns.
     That is exactly the regime that implicit Monte Carlo exists to avoid.

   What is needed is an error bound that does not depend on $\sigma$
   ("uniform in stiffness", or asymptotic-preserving). Such a bound would have
   to use the diffusion-limit structure of optically thick transport. The
   available analyses of IMC in this limit (e.g. Densmore & Larsen 2004) are
   formal asymptotics. I do not know of a rigorous uniform estimate.
2. **The solution is not smooth.** The Marshak front raises $T$ from 300 K to
   about 100 eV across a thin layer, so $M_T$ in (H2) is huge. Also,
   $\sigma\propto\theta^{-3.53}$ has Lipschitz constant growing like
   $\theta^{-4.53}$ as $T\to0$. The hypotheses are not violated, since
   $T_{\rm lo}=300\,$K$>0$, but the constants are useless. A
   convergence proof that survives fronts cannot rely on Lipschitz-in-time
   bounds.
3. **No monotonicity.** For degenerate nonlinear problems, the usual
   rigorous route to convergence without smoothness is a monotone scheme
   combined with a comparison principle. IMC is not monotone for large
   $\Delta t$: it can overheat (Larsen & Mercier 1987; the $g<0$ regime in
   Section 1.10). So that route is closed.
4. **Well-posedness of (1.1) is assumed.** (H2) assumes existence and bounds
   for the exact nonlinear system with these degenerate coefficients; they
   are not proved here.
5. **Spatial discretization is not covered.** Theorem 4 is continuous in $x$,
   while the code uses cells with constant temperature and uniform emission
   inside each cell. In the cold foam a cell is
   $\sigma\Delta x\approx2\times10^{11}$ mean free paths thick. Emitting
   uniformly across such a cell moves energy unphysically far (the "teleportation"
   error described in the IMC literature; see Wollaber 2016). An analysis
   uniform in $\sigma\Delta x$ would be needed, and none is known to me. In
   the hot region $\sigma\Delta x\approx0.007$, and this problem does not
   arise.
6. **The limits must be taken jointly.** Theorem 3 (N → ∞) holds for a fixed
   number of steps and gives no rate. Theorem 4 holds without noise.
   Combining them as $\Delta t\to0$ and $N\to\infty$ together would need
   Monte Carlo error rates that do not degrade with the number of steps.
   Those are not available.

In short, a theorem of the form "the simulation converges to the true $I$
for this problem" would need opacity-independent error bounds for the IMC
time and space discretization of a degenerate nonlinear transport system,
together with Monte Carlo rates uniform in the number of steps. To my
knowledge, none of these exists in rigorous form.

---

## 9. Status of each link

| Item | Status |
|---|---|
| One step of the idealized random walk vs. (T) | **Proved**: Theorems 1–2, Section 5. |
| Stability of (T) with respect to data | **Proved** with constant 1: Lemma B. |
| Material energy positivity; the code never aborts | **Proved**: Proposition 1. |
| 20000-event cap | **Bounded**: Proposition 2. Negligible ($10^{-1567}$) if no cell is colder than 300 K. That condition is checkable at runtime but not proved. |
| Drive averaged over each step | **Bounded**: Proposition 3, at most $\sigma_{SB}\mathrm{Lip}(T_D^4)\Delta t^2/3$ per step and per tally. |
| Face nudge ($10^{-10}\Delta x$) | Accounted for in the counting of Proposition 2. Its effect on tallies is not analyzed. |
| Census resampling (`merge_census`) | **Unbiased.** A comb of spacing $W/M$ with a uniform offset hits an input packet of weight $w_j$ on average $Mw_j/W$ times, each copy with weight $W/M$, and energy is conserved exactly. Its variance is not controlled in general; Theorem 3 needs (R0) or (R1). Default runs: never triggered. |
| Many steps, $N\to\infty$, with $\Delta t$ and $\Delta x$ fixed | **Proved**: Theorem 3. Convergence in probability, and the bias tends to 0. No rate. |
| $\Delta t\to0$ for the F&C scheme, continuous in space | **Proved conditionally**: Theorem 4, first order. Vacuous constants for the cold foam (Section 8.5). |
| Spatial discretization, $\Delta x\to0$ | **Open** (Section 8.5, item 5). |
| Simulation → true $I$ for this problem | **Open** (Section 8.5). |

---

## Appendix A — Code ↔ Fleck & Cummings correspondence

Line numbers refer to `src/imc/sio2/sio2_1mm_gray_imc_check_og.py`.

| Algorithm element | F&C | Code |
|---|---|---|
| $\beta=\partial u_r/\partial u_m$ | (1.3) | `gamma = 4θ^(4−BETA)/(H·BETA)`, line 103 |
| Fleck factor, $\alpha=1$ | (1.9), (4.1d) | `fleck = 1/(1 + C·σ·dt·gamma)`, line 104 |
| Cell emission energy $E_j=fc\sigma u_r^n\Delta x\Delta t$ | (4.13) | `emission = C·fleck·σ·B·DX·dt`, line 106 |
| Surface energy $(ac/4)T^4\Delta t$ | (4.12) | `SIGMA_SB·T_D^4`, Gauss–Legendre step average |
| Packets per source | (4.15) | ≥1 per emitting cell + energy-proportional share (largest remainder); fixed `n_boundary` (differs in bookkeeping) |
| Packet weight $E/N$ | (4.16) | `emission[i]/count`, `boundary_energy/n_boundary` |
| Position uniform in cell | (4.19) | `left + DX·r` |
| Isotropic $\mu=1-2r$ | (4.20) | `2r − 1` |
| Surface $f(\mu)=2\mu$ | (4.21)–(4.22) `max(r1,r2)` | `sqrt(r)` (same distribution) |
| Birth time uniform in step | (4.23) | `t0 + dt·r` |
| $d=\min(d_B,d_{COL},d_{CEN})$ | (4.24)–(4.25) | line 286 |
| Collision sampling | (4.25b) $d_{COL}=\lvert\ln r\rvert/((1-f)\sigma)$, continuous absorption (4.26c) | $d=\lvert\ln r\rvert/\sigma$, absorb w.p. $f$ else scatter (lines 271, 313). Same mean (Section 5.3) |
| 1 % weight cutoff | p. 329 | not needed with analog absorption; `cutoff` computed but unused (line 397) |
| Census | (4.26), $d=d_{CEN}$ | packets frozen at $t^{n+1}$; carried to next step |
| Census population control | (4.9)–(4.11) limit $N_S$ | systematic resampling `merge_census` (not in F&C; unbiased; see Sections 7.5 and 9) |
| Material update | (1.18), (4.28) | `U_new = U + (deposited − emission)/DX`, line 409; exact EOS inversion |
| Energy check | (4.29) | `energy_error`; about $10^{-14}$ |

**Code notes found during the review (the first two are fixed):**

* The initial temperature array was `np.full(1, THETA0)`. It was fixed to
  `np.full(N_CELLS, THETA0)`. Before the fix, the energy check read about 80
  (a wrong baseline), and on the first step only cell 0 emitted.
* The module docstring said "continuous absorption" and called the drive
  "measured". It now describes analog absorption and a synthetic drive,
  matching the code and the header of `back_pulse.txt`.
* `cutoff` (line 397) is computed but never used. This is left as is.

---

## Summary

1. **F&C linearization (Section 1).** Changing variable to $u_r=aT^4$ and
   integrating over a step are exact. Freezing $\sigma,\beta$ (A1), the
   $\alpha$-quadrature (A2), implicit emission (A3) and instantaneous
   re-emission (A4) are approximations. They turn Eq. (1.1) into the linear
   equation (2.1): absorption $f\sigma$, effective isotropic scattering
   $(1-f)\sigma$ and a known source $\tfrac12fc\sigma u_r^n$. The material is
   updated by the conservation law (1.18). F&C show this is consistent and
   conserves energy, but not that it converges.
2. **Monte Carlo solves (2.1) (Sections 2–5).** Packets flying straight at
   speed $c$ realize $\tfrac1c\partial_t+\mu\partial_x$. Their exponential
   collision clock is $\sigma I$. Isotropic redirection is
   $\tfrac12\sigma_s\int I\,d\mu'$. Births and escapes reproduce the source,
   inflow and outflow. Theorem 2 shows the mean packet measure satisfies the
   mild form of (2.1), and Theorem 1 (a Volterra/factorial estimate in total
   variation) shows that form has a unique solution. So for one step, the
   idealized algorithm is unbiased for (2.1) and for F&C's (2.2) and (1.18),
   and it conserves energy exactly in every run.
3. **The code as written (Section 6).** The material energy stays positive in
   every run, so the code's abort branch is unreachable (Proposition 1). The
   20000-event cap has negligible bias provided no cell drops below 300 K
   (Proposition 2). Averaging the drive over a step costs
   $O(\Delta t^2)$ per step and per tally (Proposition 3). These rest on
   Lemma B: stability of (2.1) with constant 1, independent of the opacity.
4. **Many steps (Section 7).** With $\Delta t$ and $\Delta x$ fixed, the
   coupled multi-step simulation converges in probability, as $N\to\infty$,
   to the deterministic IMC scheme, and its bias tends to 0 (Theorem 3). No
   rate is proved, and the code's systematic resampling is covered only
   while it is not triggered.
5. **$\Delta t\to0$ (Section 8).** Under smoothness and positivity hypotheses,
   the continuous-in-space F&C scheme converges to Eq. (1.1) at first order
   (Theorem 4). Its constants grow like $e^{c\sigma_{\max}t}$, so it says
   nothing for the cold foam. A convergence proof for this problem would
   need opacity-independent bounds, robustness to the Marshak front, a
   treatment of the spatial cells, and Monte Carlo rates uniform in the
   number of steps. None of these is available, so the question is open.
