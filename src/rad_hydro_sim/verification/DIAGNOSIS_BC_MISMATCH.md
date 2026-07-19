# Diagnosis: Rad-Hydro vs 1D Diffusion BC Mismatch

## Observed Symptoms
- **Rad-Hydro**: High T (~2.3e7 K) at both boundaries, peak ~3.1e7 K in the middle
- **1D Diffusion**: T near 0 at both boundaries, peak ~1.5e6 K
- **Expected**: T = T0 (~1.16e6 K) at left (driven boundary), T → 0 at right (vacuum)

## Root Causes Identified

### 1. **Formulation Mismatch: Mass vs Position Coordinates**
- **Rad-Hydro** (`radiation_step.py`): Uses **mass-coordinate** diffusion
  - `coeff = rho[1:-1] / (m_cells[1:-1]**2)` with `D_face = rho * (D_left + D_right)/2`
  - Operator acts in mass space; for constant ρ, dm = ρ dx
- **1D Diffusion**: Uses **position-space** diffusion  
  - `a_i = -D_imh / dz**2`, uniform grid z = linspace(0, L, Nz)
- These give different discrete operators even for the same physics.

### 2. **Grid Convention Difference**
- **1D Diffusion**: `z = np.linspace(0, L, Nz)` — first point at z=0 (left boundary)
- **Rad-Hydro**: `x_cells = 0.5*(x_nodes[:-1] + x_nodes[1:])` — cell centers, first point at dx/2
- Rad-Hydro has no point exactly at x=0; 1D Diffusion has T at z=0.

### 3. **f_Kelvin / g_Kelvin Consistency**
- 1D Diffusion default: `g_Kelvin = 1/(7200 * KELVIN^1.6 * 19.32^0.14)` (includes rho exponent)
- Preset: `g_Kelvin = 1/(7200 * KELVIN^1.5)` (no rho, different exponent)
- `run_diffusion_1d` patches from case, but verify both use identical formulas.

### 4. **Magnitude Discrepancy (20x)**
- T0 = 1.16e6 K → Rad-Hydro shows 2.3e7 K at boundaries (20× too high)
- Check for unit mix-ups (Kelvin vs HeV) or double application of BC.

## Recommended Actions

1. **Add BC diagnostics** in `run_radiation_only_comparison` to print T[0], T[-1], x[0], x[-1] for both sim and ref at one time.
2. **Align formulation**: Either convert 1D Diffusion to mass coordinates, or add a position-space radiation option in rad-hydro for this verification.
3. **Verify f/g**: Ensure `run_diffusion_1d` passes `f_Kelvin` and `g_Kelvin` from the case (not None) so they match exactly.
4. **Check right BC**: Confirm rad-hydro right boundary stays cold (E_rad[-1]=0, T[-1]≈0) and is not overwritten.
