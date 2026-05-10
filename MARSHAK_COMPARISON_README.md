# Marshak Radiation Diffusion Comparison

## Overview

This document describes the Marshak comparison framework that compares three solutions for 1D radiation diffusion with Marshak boundary conditions:

1. **Shussman Shock Solver** - Piston-shock solution from the Shussman legacy code
2. **Subsonic Heat Wave** - 1D self-similar diffusion solution (menahem_new)
3. **Rad-hydro Simulation** - Radiation-only rad_hydro simulation with improved T_bath calculation

## Changes Made

### 1. Enhanced `radiation_step.py`

Added two new functions to `rad_hydro_sim/simulation/radiation_step.py`:

#### `get_T_bath(state_star, rad_hydro_case) -> float`
Calculates the proper bath temperature at the left boundary using the subsonic heat wave (1D self-similar radiation diffusion) solution.

**Algorithm:**
1. Creates/retrieves a cached `SubsonicHeatWave` solver for the given case
2. Evaluates the self-similar profiles at the left boundary using `get_self_similar_profiles()`
3. Extracts the dimensionless radiation boundary flux from `S[0]`
4. Calls `calc_T_bath_from_dimensionless_boundary_flux()` to compute the actual bath temperature
5. Falls back to simple `T_surface = T0 * t^tau` if the solver fails

**Key Features:**
- Caches solver instances to avoid expensive re-initialization
- Physically accurate boundary condition for Marshak BC
- Graceful fallback for robustness

#### `_get_or_create_subsonic_heat_wave_solver(rad_hydro_case)`
Lazily creates and caches a `SubsonicHeatWave` solver instance keyed by case parameters.

### 2. Modified `radiation_step()` Function

Updated the main `radiation_step()` function to use the improved T_bath calculation:

**Before:**
```python
T_left = T0_left * (t_drive/(10**-9))**rad_hydro_case.tau
```

**After:**
```python
if bc_type == "Marshak":
    T_bath = get_T_bath(state_star, rad_hydro_case)
    T_left = T_bath
else:
    t_drive = max(state_star.t, dt)
    T0_left = rad_hydro_case.T0_Kelvin if rad_hydro_case.T0_Kelvin is not None else 0.0
    T_left = T0_left * (t_drive/(10**-9))**rad_hydro_case.tau
```

This ensures that when using Marshak boundary conditions, the true radiation bath temperature is computed, leading to more accurate physical solutions.

### 3. New File: `marshak_comparison.py`

Created `comparison/marshak_comparison.py` that implements a comprehensive comparison framework.

**Main Components:**

- `subsonic_heat_wave_profiles()` - Evaluates 1D self-similar diffusion profiles
- `shussman_shock_profiles()` - Evaluates Shussman shock solver profiles  
- `rad_hydro_profiles()` - Runs rad_hydro simulation and extracts profiles
- `main()` - Orchestrates comparison and generates visualization

**Output:**
Generates a 4-panel comparison plot showing:
1. Temperature vs self-similar coordinate (heat wave & shock)
2. Density vs self-similar coordinate (heat wave & shock)
3. Pressure vs self-similar coordinate (heat wave & shock)
4. Temperature vs position from rad_hydro simulation

Results are saved to `results/` directory with timestamp.

## Usage

### Running the Comparison

```bash
cd /path/to/project3_code
python comparison/marshak_comparison.py
```

This will:
1. Load the Marshak radiation-only preset case
2. Run all three solvers at the same evaluation time
3. Generate comparison plots
4. Save results to `results/menahem_reproduction/`

### Using the Improved T_bath in Simulations

The improved T_bath calculation is automatically used whenever:
- `bc_type="Marshak"` is set in the `RadHydroCase`
- Running a radiation step with Marshak boundary conditions

Example:
```python
from rad_hydro_sim.problems.presets_config import PRESET_CONSTANT_T_RADIATION
from rad_hydro_sim.problems.presets_utils import get_preset
from rad_hydro_sim.simulation.iterator import simulate_rad_hydro

case, config = get_preset(PRESET_CONSTANT_T_RADIATION)
# case.bc_type is already "Marshak"

# Run simulation - will use improved T_bath
states, _, _, history = simulate_rad_hydro(case, config)
```

## Physical Justification

The improved `T_bath` calculation is based on the observation that in radiation diffusion with a prescribed boundary temperature, the actual radiation energy flux at the boundary is determined by the self-similar heat wave solution. 

Instead of simply using `T_surface = T0 * t^tau`, we:
1. Solve the 1D self-similar radiation diffusion problem
2. Extract the dimensionless boundary radiation flux
3. Use the Stefan-Boltzmann relation to compute the actual bath temperature that would produce this flux

This provides a more consistent boundary condition that accounts for the coupling between radiation transport and material properties.

## Cached Solver Performance

The `SubsonicHeatWave` solver is expensive to initialize (requires root finding for xsi_f and Pf). The caching mechanism stores solver instances keyed by case parameters, so repeated calls with the same case reuse the same solver instance. This dramatically improves performance for time-dependent simulations.

## Related Files

- `rad_hydro_sim/simulation/radiation_step.py` - Enhanced radiation step with get_T_bath()
- `menahem_new/subsonic_heat_wave.py` - Self-similar heat wave solver
- `comparison/marshak_comparison.py` - Comparison framework
- `comparison/rt_menahem_vs_radhydro.py` - Original comparison script (reference)

## Future Improvements

1. Add support for time-dependent visualization (animation)
2. Implement quantitative error metrics between solvers
3. Add support for variable opacity models
4. Extend to 2D/3D geometries if needed
