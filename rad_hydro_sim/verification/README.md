# Rad-Hydro Verification

This directory compares `rad_hydro_sim` to reference solutions to verify correctness when only radiation or only hydrodynamics is active.

## Comparisons

1. **Radiation-only**  
   - **Rad-Hydro**: `run_rad_hydro.py` with preset `radiation_only_constant_temperature_drive`.  
   - **References**:  
     - 1D Diffusion self-similar solution (`1D Diffusion self similar in gold/figures/1D Diffusion self similar.py`) with matching parameters (constant temperature drive, same domain and time).  
     - **Supersonic solver** (radiation self-similar, `project_3/shussman_solvers/supersonic_solver/`): same physics (opacity/EOS from the preset); used to verify the radiation-only case.  
   - **Quantities compared**: Temperature \(T\) and radiation energy density \(E_{\mathrm{rad}}\) vs position \(x\).

2. **Hydro-only**  
   - **Rad-Hydro**: `run_rad_hydro.py` with preset `hydro_only_power_law_pressure_drive`.  
   - **Reference**: `hydro_sim` `run_hydro.py` with a matching driven shock case (same \(P_0\), \(\tau\), \(\gamma\), domain, and time).  
   - **Quantities compared**: \(\rho\), \(P\), \(u\), \(e\) vs \(x\) (same as `verification_hydro_sim_with_shock`).

Later, the hydro case can also be compared to Shussman’s semi-analytic solver (hydro only).

## How to run

From the **repository root** (e.g. `WeaponGroup`):

```bash
python project_3/rad_hydro_sim/verification/run_comparison.py
```

Or as a module:

```bash
python -m project_3.rad_hydro_sim.verification.run_comparison
```

In `run_comparison.py`, set the mode in `main()`:

- `MODE = VerificationMode.RADIATION_ONLY` — run rad_hydro + 1D Diffusion + Supersonic solver and compare T, E_rad.  
- `MODE = VerificationMode.HYDRO_ONLY` — run rad_hydro + hydro_sim and compare rho, P, u, e.

Output figures are written under `project_3/rad_hydro_sim/verification/figures/` (PNG and optionally GIF).

## Dependencies

- **Radiation-only**:  
  - The 1D Diffusion script must be at  
    `1D Diffusion self similar in gold/figures/1D Diffusion self similar.py`  
    relative to the repo root. It is run with parameters matched to the rad_hydro preset.  
  - The **Supersonic solver** (`project_3/shussman_solvers/supersonic_solver/`) is run with a material built from the same preset (alpha, beta=gamma, rho0, f, g, etc.). Use `skip_supersonic=True` in `run_radiation_only_comparison()` to omit it (e.g. if the solver is slow or unavailable).

- **Hydro-only**: Uses `project_3.rad_hydro_sim.verification.hydro_shock` (compare_shock_plots, presets) and `project_3.hydro_sim.simulations.lagrangian_sim`.

## Files

- `verification_config.py` — Verification modes and output paths.  
- `run_comparison.py` — Main entry; runs one of the two comparisons (radiation-only or hydro-only).  
- `run_diffusion_1d.py` — Runs the 1D Diffusion script with given parameters and returns (times, x, T, E_rad).  
- `radiation_data.py` — Data types for radiation comparison (RadiationData).  
- `compare_radiation_plots.py` — Plot T and E_rad vs x (single time + slider).  
- `hydro_data.py` — Converts RadHydroHistory and HydroHistory to SimulationData for hydro comparison.  
- `hydro_shock/` — Hydro-only comparison vs Shussman shock solver (compare_shock_plots, presets, run_comparison).  
- `figures/` — Output directory for PNG/GIF (created automatically).
