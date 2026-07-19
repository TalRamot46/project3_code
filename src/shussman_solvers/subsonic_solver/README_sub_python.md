# Subsonic solver – Python conversion

Python port of the MATLAB `subsonic_solver/` (self-similar profiles with 5-component state: V, V', P, P', u). Mirrors the structure of the supersonic solver.

---

## Solver structure

The solver computes a **self-similar solution** in the similarity variable ξ: state is volume V, pressure P, and velocity u with power laws in time. The profile is found by solving a 5-D ODE and normalizing via binary shooting (P(1) and xsi).

### Data flow

```
  materials_sub (MaterialSub, material_al/au/be/cu/pb)
           │
           ▼
  manager_sub(mat, tau)  ──►  (m0, mw, e0, ew, P0, Pw, V0, Vw, u0, uw, xsi, z, Ptilda, utilda, B, t, x)
           │
           ├──► solve_normalize_sub(alpha, beta, lambda_, mu, r, tau)  ──►  (t, x)  [x: V, V', P, P', u]
           │           ├──► F_sub.F(t, x, ...)   [ODE RHS]
           │           └──► utils_sub.integrate_ode(...)
           └──► utils_sub.trapz
  profiles_for_report_sub(mat, tau)  ──►  m_heat, P_heat, T_heat, u_heat, rho_heat, derivatives, ...
```

### File ↔ module mapping

| MATLAB | Python module | Main public API |
|--------|----------------|------------------|
| Al.m, Au.m, Be.m, Cu.m, Pb.m | `materials_sub` | `MaterialSub`, `material_al()`, `material_au()`, … |
| F.m | `F_sub` | `F(t, x, alpha, beta, lambda_, mu, r, tau)` |
| mid.m | `utils_sub` | `mid(x)`, `trapz(y, x)`, `integrate_ode(...)` |
| solve_for_tau.m | `solve_for_tau_sub` | `solve_for_tau(etta1, etta2, etta3, mat)` |
| solve_normalize.m | `solve_normalize_sub` | `solve_normalize(alpha, beta, lambda_, mu, r, tau, ...)` |
| manager.m | `manager_sub` | `manager_sub(mat, tau, ...)` |
| profiles_for_report_sub.m | `profiles_for_report_sub` | `compute_profiles_for_report(mat, tau, ...)` |
| — | `run_sub` | `run_sub(mat, tau=None, ...)` (entry point) |

## How to run

From the project root:

```bash
python -m shussman_solvers.subsonic_solver.run_sub
```

Or in Python:

```python
from shussman_solvers.subsonic_solver import material_al, run_sub, compute_profiles_for_report

mat = material_al()
tau = 0.3
m0, mw, e0, ew, P0, Pw, V0, Vw, u0, uw, xsi, z, Ptilda, utilda, B, t, x = run_sub(mat, tau)
data = compute_profiles_for_report(mat, tau, T0=7.0)
```

## Numerical note

The subsonic ODE is stiff; the code uses `scipy.integrate.solve_ivp` with **method='Radau'**. The MATLAB uses two outer iterations (xsi calibration) and `iternum` inner iterations (P(1) shooting); default `iternum=3000` can be reduced for faster (less accurate) runs.
