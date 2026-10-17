# Implicit Monte Carlo for the gray slab problem: from Fleck & Cummings Eq. (1.1) to a proof that the code solves the linearized equation

This document is a self-contained account of three things:

1. **How Fleck & Cummings (F&C) turn the nonlinear radiation–matter system,
   their Eq. (1.1), into a linear transport equation that a Monte Carlo
   random walk can solve** (their Eq. 2.1). Each step is marked as exact
   algebra or as an approximation (Section 1).
2. **A theorem with proof** that the random walk implemented in
   `src/sio2_1mm_gray_imc_check_og.py` produces, in expectation, the unique
   solution of that linear equation, and that its tallies are unbiased
   (Sections 2–5).
3. **What is *not* established**: the rigor of the linearization itself, and
   convergence of the full simulation to the true radiation field (Section 6).

Appendix A maps every line of the algorithm to the corresponding F&C equation.

Primary source: J. A. Fleck Jr. and J. D. Cummings, "An Implicit Monte Carlo
Scheme for Calculating Time and Frequency Dependent Nonlinear Radiation
Transport", *J. Comput. Phys.* **8**, 313–342 (1971). A copy is in the
companion repository `project3_docs`, at `src_papers/Fleck and Cummings.pdf`. Equation numbers "F&C (x.y)"
refer to that paper. Labels such as (T), (Mild), (G) are this document's own.

---

## 0. Scope

```
  original nonlinear system, F&C Eq. (1.1)
        │  (A) F&C linearization over one time step        ── Section 1
        │      exact algebra + 4 approximations; consistent, NOT proved convergent
        ▼
  linear transport equation F&C (2.1) = (T), frozen coefficients
        │  (B) packet random walk                          ── Sections 2–5
        │      PROVED: mean packet density = unique solution of (T)
        ▼
  expected packet density / expected tallies of the code
        │  (C) N → ∞, Δt → 0, Δx → 0 over many coupled steps ── Section 6
        │      NOT proved
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
later theory.

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
process (Section 6).

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

**Material update.** Line 408, $U_j^{n+1}=U_j^n+(\mathcal E_j-E_j)/\Delta x$,
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

## 6. What is not covered

| Item | Status |
|---|---|
| Link B for the idealized process (one step) | **Proved**: Theorems 1–2, Section 5. The statistical error of a tally is $O(N^{-1/2})$ by the law of large numbers for the i.i.d.-sourced, independent packet histories. |
| Census resampling (`merge_census`) | **Unbiased.** A comb of spacing $W/M$ with a uniform offset hits an input packet of weight $w_j$ on average $Mw_j/W$ times, each copy with weight $W/M$. Its expected measure is therefore preserved, and energy is conserved exactly. Applied after transport, it preserves the conditional mean. |
| Drive averaged over the step | The code solves (T) with $\bar\phi_L$ instead of $\phi_L(t)$. This is an $O(\Delta t)$ timing error inside the step. |
| 20000-event cap | A deviation from the process. A capped history goes to census before $t^{n+1}$, which is a bias. Its size is not quantified, but it should be rarely triggered. |
| Face nudge ($10^{-10}\Delta x$) | A geometric perturbation, not analyzed. |
| Many steps with a finite number of packets | **Not proved.** $\sigma^{n+1},f^{n+1},u_r^{n+1}$ depend on the random $U^{n+1}$ through nonlinear maps, so the multi-step run is *not* an unbiased estimator of the deterministic semi-discrete scheme. One expects the bias to vanish as $N\to\infty$, heuristically $O(1/N)$, but no proof is given. |
| Link A: (2.1)+(1.18) vs. (1.1) | **Not proved** (Section 1.11). The error is formally first order in $\Delta t$. Overheating occurs for large $\Delta t$ (Section 1.10; Larsen & Mercier 1987). |
| Spatial discretization | Temperature is constant per cell and emission is uniform per cell (F&C (4.19)). Convergence as $\Delta x\to0$ is expected but not proved here. |
| Convergence of the full scheme to the solution of (1.1) | **Not proved.** As far as I know, no complete proof exists. The literature provides consistency, asymptotic-diffusion and stability analyses. |

---

## Appendix A — Code ↔ Fleck & Cummings correspondence

Line numbers refer to `src/sio2_1mm_gray_imc_check_og.py`.

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
| Census population control | (4.9)–(4.11) limit $N_S$ | systematic resampling `merge_census` (not in F&C; unbiased, Section 6) |
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
3. **Open (Section 6).** It is not proved that (2.1)+(1.18) approximates (1.1),
   nor that the coupled multi-step simulation converges to the true $I$.
