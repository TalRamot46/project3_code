# Shussman-style self-similar solvers

This directory contains two semi-analytic solvers (converted from MATLAB):

- **shock_solver** — Hydro (shock) self-similar solution. Compares with `hydro_sim` and is used by `rad_hydro_sim/verification` for hydro-only verification.
- **supersonic_solver** — Radiation-diffusion self-similar solution. Used by `rad_hydro_sim/verification` for radiation-only verification (with 1D Diffusion and run_rad_hydro).

## Layout

```
shussman_solvers/
├── README.md           (this file)
├── shock_solver/       (F_shock, manager_shock, materials_shock, run_shock_solver, …)
└── supersonic_solver/  (F_super, manager_super, materials_super, matlab_super/, …)
```

Run from repo root with `project_3` on `sys.path`; import e.g. `from project_3.shussman_solvers.shock_solver.run_shock_solver import compute_shock_profiles`.
