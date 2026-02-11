# Supersonic solver – Python conversion

Python port of `matlab_super/` (self-similar radiation-diffusion profiles), following the style of `project_3/shussman_shock_solver`.

---

## Solver structure

The solver computes a **self-similar solution** for radiation-diffusion: temperature and areal mass scale with a power law in time; the profile in the similarity variable ξ (xi) is found by solving an ODE and normalizing so T(ξ=0) = 1.

### Data flow

```
  materials_super (MaterialSuper, material_*)
           │
           ▼
  manager_super(mat, tau)  ──────────────────────────────────────►  (m0, mw, e0, ew, xsi, z, A, t, x)
           │                                                                    │
           │  calls                                                              │
           ▼                                                                    │
  solve_normalize_super(alpha, beta, tau)  ──►  (t, x)  ─────────────────────────┘
           │                                        (used by profiles_for_report_super)
           │  uses
           ├──► F_super.F(t, x, alpha, beta, tau)   [ODE RHS: dT/dξ, d²T/dξ²]
           └──► utils_super.integrate_ode(...)      [solve_ivp, Radau]
  manager_super also uses: utils_super.trapz(y, x) for z = -∫ T^β dξ
```

- **materials_super**: No solver dependencies. Defines materials used by manager and profiles.
- **F_super**: ODE only; used by solve_normalize via integrate_ode.
- **utils_super**: trapz (manager), mid (optional), integrate_ode (solve_normalize).
- **solve_normalize_super**: Called only by manager_super; depends on F_super, utils_super.
- **manager_super**: Entry point for “get solution”; depends on materials_super, solve_normalize_super, utils_super.
- **profiles_for_report_super**: Post-processes manager output; depends on materials_super, manager_super.
- **run_super**: Script/API entry; calls manager_super and optionally profiles_for_report_super.

### Main symbols (manager output and power laws)

| Symbol | Meaning |
|--------|--------|
| **ξ (xi)** | Similarity variable; profile is T(ξ). Integration from ξ = xsi (front) down to 0. |
| **t** | 1D array of ξ values (same as “coordinate xi” in MATLAB). |
| **x** | (n, 2): x[:,0] = dimensionless T(ξ), x[:,1] = dT/dξ. |
| **xsi** | max(t) = front coordinate ξ_front. |
| **A** | 3·f·β/(16·σ·g). |
| **z** | Dimensionless energy: -∫ T^β dξ. |
| **mw, ew** | Power-law exponents: m = m0·T^mw(2)·t^mw(3) [g/cm²], e = e0·T^ew(2)·t^ew(3) [J/cm²]. |
| **m0, e0** | Dimensional constants (with unit conversions; see below). |

### File ↔ module mapping

| MATLAB | Python module | Main public API |
|--------|----------------|------------------|
| Al.m, Au.m, Be.m, Pb.m, SiO2.m | `materials_super` | `MaterialSuper`, `material_al()`, `material_au()`, … |
| F.m | `F_super` | `F(t, x, alpha, beta, tau)` |
| mid.m | `utils_super` | `mid(x)`, `trapz(y, x)`, `integrate_ode(...)` |
| solve_normalize.m | `solve_normalize_super` | `solve_normalize(alpha, beta, tau, ...)` |
| manager.m | `manager_super` | `manager_super(mat, tau, ...)` |
| profiles_for_report_super.m | `profiles_for_report_super` | `compute_profiles_for_report(mat, tau, ...)` |
| — | `run_super` | `run_super(mat, tau=None, ...)` (entry point) |

## Unit conversions (MATLAB author – highlighted in code)

- **HEV_IN_KELVIN = 1160500**: 1 HeV in Kelvin; used in opacity/EOS formulas (f, g) and in m0/e0 scaling.
- **m0 (g/cm²)**: Formula uses `10^(-9*(-mw(2)*tau))`, `1160500^mw(2)`, `(1e-9)^mw(3)` — time in ns and temperature in HeV.
- **e0 (J/cm²)**: Same time/HeV factors plus **/1e9/100** — exact meaning of these two factors is not spelled out in the MATLAB; likely relating an internal energy/time scale to J/cm². Left as in original; verify if you need consistent units elsewhere.
- **T_heat** in `profiles_for_report_super`: **factor 100** in `100*T0*times^tau*T_tilde` — unit/scaling choice by the MATLAB author (e.g. display or K vs HeV). If your pipeline expects different units, adjust or expose this factor.

## Numerical note

The self-similar ODE is stiff; the Python code uses `scipy.integrate.solve_ivp` with **method='Radau'** and small `first_step`/`max_step`. Each shooting iteration can take on the order of 10–20 seconds; reduce `iternum` or relax `tol` in `solve_normalize_super` / `manager_super` for faster (less accurate) runs.

## Ambiguities / questions for you

1. **e0 formula**: The division by `1e9` and `100` in the e0 expression — do you know the intended unit convention (e.g. time in ns, area in cm² vs m²)?
2. **T_heat factor 100**: Should this remain 100 for your plots/reports, or be configurable (e.g. to get temperature in a specific unit)?
3. **Al.m header**: The MATLAB file says `%Au.m:` but the data (Yair’s, rho0=2.78) is for Aluminum; the Python function is `material_al()` and the docstring notes the mismatch.
