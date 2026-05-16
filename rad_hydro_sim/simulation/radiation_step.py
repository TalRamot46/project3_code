# physical constants
from importlib import machinery
from importlib import machinery
c = 3e10  # speed of light [cm/s]
a_Kelvin = 7.5646e-15  # Radiation constant in erg cm^-3 K^-4
eV_to_erg = 1.60218e-12  # electron energy in CGS [erg/eV]
Hev_to_erg =  100 * eV_to_erg  # electron energy in CGS [erg/Hev]
k_B = 1.380649e-16  # Boltzmann constant in CGS [erg/K]
KELVIN_PER_HEV = Hev_to_erg / k_B  # Conversion factor from keV to Kelvin
a_Hev = a_Kelvin * KELVIN_PER_HEV**4  # Radiation constant in keV cm^-3 keV^-4
HARMONIC_MEAN = False

import numpy as np
from dataclasses import dataclass
from typing import Tuple

def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray: return 2 * a * b / (a + b)
def arithmetic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray: return (a + b) / 2

from project3_code.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project3_code.hydro_sim.core.state import RadHydroState

# Global cache for SubsonicHeatWave solver instances (keyed by case repr)
_subsonic_heat_wave_cache = {}

def calculate_temperature_from_specific_energy(
    e_material: np.ndarray, rho: np.ndarray, f: float, gamma: float, mu: float
) -> np.ndarray:
    return ((e_material / f) * rho**mu) ** (1/gamma)


def calculate_beta_from_temperature_and_density(T_material: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """beta = dU_R/dU_m; at equilibrium: 4a/(f*gamma) * T^(4-gamma) * rho^(mu-1)."""
    return 4*a_Kelvin / (f_Kelvin * beta_Rosen) * T_material**(4 - beta_Rosen) * rho**(mu - 1)

def calculate_sigma_from_temperature_and_density(T_material: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Opacity from Rosen's formula evaluated at temperature T."""
    return 1.0 / (g_Kelvin * T_material**alpha * rho**(-lambda_ - 1))

def calculate_D_from_sigma(sigma: np.ndarray) -> np.ndarray:
    return c / (3 * sigma)

def calculate_A(beta: np.ndarray, sigma: np.ndarray, dt: float) -> np.ndarray:
    return chi * beta * sigma * dt * c

def calculate_black_abcd(
    D: np.ndarray,
    m_cells: np.ndarray,
    rho: np.ndarray,
    e_old: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build tridiagonal coefficients for the black-physics diffusion update:
        a_j e_{j-1}^{n+1} + b_j e_j^{n+1} + c_j e_{j+1}^{n+1} = e_j^n
    using Eq. 340-352 from the docs.
    """
    if HARMONIC_MEAN:
        D_face = harmonic_mean(D[:-1], D[1:])
    else:
        D_face = arithmetic_mean(D[:-1], D[1:])

    rho_j = rho[1:-1]
    dm_j = m_cells[1:-1]
    dm_left = m_cells[:-2]
    dm_right = m_cells[2:]
    D_left = D_face[:-1]
    D_right = D_face[1:]

    coeff = dt * rho_j / dm_j
    a = -coeff * (D_left / dm_left)
    b = 1.0 + coeff * ((D_left + D_right) / dm_j)
    c_coeff = -coeff * (D_right / dm_right)
    d = e_old[1:-1].copy()

    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c_coeff)) or np.any(np.isnan(d)):
        raise ValueError("NaN encountered in black-radiation tridiagonal coefficients.")
    if np.any(b <= 0):
        raise ValueError("Non-positive diagonal encountered in black-radiation tridiagonal system.")

    return a, b, c_coeff, d

def calculate_abcd(
    sigma: np.ndarray,
    D: np.ndarray,
    A: np.ndarray,
    m_cells: np.ndarray,
    rho: np.ndarray,
    E_rad: np.ndarray | None,
    T_material: np.ndarray,
    dt: float,
    bc_type: str = "Dirichlet",
    T_left: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Face-weighted coefficient builder (only supported scheme).

    Produces full-length a,b,c,d arrays (length N = len(rho)). Applies
    Marshak vacuum leakage on the right boundary and either a Marshak left
    modification or a Dirichlet subtraction depending on `bc_type`.
    """
    # Face-weighted implementation only
    if HARMONIC_MEAN:
        D_face = harmonic_mean(D[:-1], D[1:])
        rho_face = harmonic_mean(rho[:-1], rho[1:])
        m_face = harmonic_mean(m_cells[:-1], m_cells[1:])
    else:
        D_face = arithmetic_mean(D[:-1], D[1:])
        rho_face = arithmetic_mean(rho[:-1], rho[1:])
        m_face = arithmetic_mean(m_cells[:-1], m_cells[1:])

    flux_coeff = (D_face * rho_face) / m_face
    flux_coeff = np.concatenate(([flux_coeff[0]], flux_coeff, [flux_coeff[-1]]))
    lagrangian_coeff = rho / m_cells

    B = chi * c * sigma / (1 + A)
    a = -lagrangian_coeff * flux_coeff[:-1]
    b = lagrangian_coeff * (flux_coeff[:-1] + flux_coeff[1:]) + 1 / dt + B
    c_coeff = -lagrangian_coeff * flux_coeff[1:]

    UR_star = a_Kelvin * T_material**4
    term_coupling = B * UR_star
    term_E = (1 / dt) * E_rad if E_rad is not None else 0.0
    d = term_coupling + term_E

    # Check for non-finite contributions (Inf can arise from divisions by zero)
    if not np.all(np.isfinite(UR_star)):
        j = np.where(~np.isfinite(UR_star))[0][0]
        raise ValueError(f"Non-finite UR_star at index {j}: UR_star={UR_star[j]}")
    if not np.all(np.isfinite(B)):
        j = np.where(~np.isfinite(B))[0][0]
        raise ValueError(f"Non-finite coupling B at index {j}: B={B[j]}, sigma={sigma[j]}, A={A[j]}")
    if E_rad is not None and not np.all(np.isfinite(E_rad)):
        j = np.where(~np.isfinite(E_rad))[0][0]
        raise ValueError(f"Non-finite E_rad at index {j}: E_rad={E_rad[j]}")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Bad dt passed to calculate_abcd: dt={dt}")
    # Check for overflows in the constructed RHS `d` will be done
    # after boundary modifications to catch any contributions from
    # Marshak/Dirichlet boundary terms.

    # Always apply Marshak vacuum leakage on the right boundary
    # if len(b) >= 2:
    #     rho_star_right = float(rho[-2])
    #     dm_right = float(m_cells[-2])
    #     if not np.isfinite(dm_right) or dm_right <= 0.0:
    #         cooling_right = 0.0
    #     else:
    #         cooling_right = c * rho_star_right / (2.0 * dm_right)
    #     b[-1] += cooling_right

    # Left boundary handling
    if bc_type == "Marshak":
        if T_left is None:
            raise ValueError("T_left must be provided when bc_type='Marshak'.")
        E_bath = a_Kelvin * (T_left ** 4)
        if len(b) >= 2:
            rho_star_left = float(rho[1])
            dm_left = float(m_cells[1])
            if not np.isfinite(dm_left) or dm_left <= 0.0:
                cooling_left = 0.0
            else:
                cooling_left = c * rho_star_left / (2.0 * dm_left)
            b[0] = b[0] + a[0] + cooling_left
            a[0] = 0.0
            d[0] += cooling_left * E_bath
    elif bc_type == "Dirichlet":
        if T_left is not None:
            E_left = a_Kelvin * (T_left ** 4)
            d[0] -= a[0] * E_left

    # Final safety: replace any non-finite RHS entries introduced by
    # boundary handling with large finite placeholders and warn.
    if not np.all(np.isfinite(d)):
        j = np.where(~np.isfinite(d))[0][0]
        import warnings
        # Choose a conservative finite cap based on existing energies to avoid
        # injecting astronomically large values that destabilize subsequent
        # temperature calculations.
        if E_rad is not None and np.any(np.isfinite(E_rad)):
            base = float(np.nanmax(np.abs(E_rad)))
        else:
            base = float(np.nanmax(a_Kelvin * (T_material ** 4))) if np.any(np.isfinite(T_material)) else 1.0
        cap = max(base * 1e2, 1.0)
        warnings.warn(f"Non-finite RHS 'd' at index {j} after boundary handling: d={d[j]}. Replacing with cap={cap}.")
        # Replace: NaNs -> 0, +inf -> cap, -inf -> -cap
        d = np.where(np.isnan(d), 0.0, d)
        d = np.where(d == np.inf, cap, d)
        d = np.where(d == -np.inf, -cap, d)
        # Also sanitize any remaining non-finite via nan_to_num as fallback
        d = np.nan_to_num(d, nan=0.0, posinf=cap, neginf=-cap)

    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c_coeff)) or np.any(np.isnan(d)):
        j = np.where(np.isnan(a))[0][0] if np.any(np.isnan(a)) else (np.where(np.isnan(b))[0][0] if np.any(np.isnan(b)) else (np.where(np.isnan(c_coeff))[0][0] if np.any(np.isnan(c_coeff)) else np.where(np.isnan(d))[0][0]))
        raise ValueError(
            f"NaN value encountered in coefficients at index {j}: a={a[j] if j < len(a) else 'NA'}, b={b[j]}, c={c_coeff[j] if j < len(c_coeff) else 'NA'}, d={d[j]}"
        )

    if np.any(b <= 0):
        j = np.where(b <= 0)[0][0]
        raise ValueError(
            f"Non-positive diagonal at index {j}: b={b[j]}, a={a[j] if j < len(a) else 'NA'}, c={c_coeff[j] if j < len(c_coeff) else 'NA'}"
        )

    return a, b, c_coeff, d




# Right boundary helpers removed: Marshak vacuum leakage is always applied
# within `calculate_abcd` and Dirichlet left handling is performed there too.

def black_radiation_step(
    state_star: RadHydroState,
    dt: float,
    rad_hydro_case: RadHydroCase,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Black-physics radiation update:
    - no matter-radiation coupling source term
    - solve a single diffusion equation for e where e = E_rad = e_material.
    Returns:
        new_e: updated shared energy variable
        new_T: updated shared temperature from E = a T^4
    """
    global alpha, beta_Rosen, mu, f_Kelvin, chi, lambda_, g_Kelvin
    alpha, beta_Rosen, mu, f_Kelvin, chi, lambda_, g_Kelvin = rad_hydro_case._get_params()

    rho = state_star.rho
    m_cells = state_star.m_cells
    e_old = state_star.E_rad if state_star.E_rad is not None else state_star.e_material

    T_material_star = calculate_temperature_from_specific_energy(e_old, rho, f_Kelvin, beta_Rosen, mu)
    sigma = calculate_sigma_from_temperature_and_density(T_material_star, rho)
    D = calculate_D_from_sigma(sigma)

    a, b, c_coeff, d = calculate_black_abcd(D, m_cells, rho, e_old, dt)

    t_drive = max(state_star.t, dt)
    T0_left = rad_hydro_case.T0_Kelvin if rad_hydro_case.T0_Kelvin is not None else 0.0
    T_left = T0_left * (t_drive / (10**-9)) ** rad_hydro_case.tau
    e_left = a_Kelvin * T_left**4
    # classical Dirichlet-like handling for left cell
    d[0] -= a[0] * e_left
    new_e = solve_tridiagonal(a, b, c_coeff, d)
    new_e[0] = e_left

    new_T = (new_e / a_Kelvin) ** (1 / 4)
    return new_e, new_T


def _get_or_create_subsonic_heat_wave_solver(rad_hydro_case: RadHydroCase):
    """
    Lazily create and cache a SubsonicHeatWave solver for the given case.
    This avoids expensive re-initialization across multiple radiation steps.
    """
    global _subsonic_heat_wave_cache
    
    # Create a unique key for this case based on its parameters
    case_key = (
        rad_hydro_case.T0_Kelvin,
        rad_hydro_case.tau,
        rad_hydro_case.g_Kelvin,
        rad_hydro_case.alpha,
        rad_hydro_case.lambda_,
        rad_hydro_case.f_Kelvin,
        rad_hydro_case.beta_Rosen,
        rad_hydro_case.mu,
        rad_hydro_case.r,
    )
    
    if case_key not in _subsonic_heat_wave_cache:
        from project3_code.menahem_new.subsonic_heat_wave import SubsonicHeatWave
        
        # Initialize the solver with case parameters
        # Note: Tb is T0_Kelvin, and gamma = r + 1
        solver = SubsonicHeatWave(
            Tb=float(rad_hydro_case.T0_Kelvin),
            tau=float(rad_hydro_case.tau),
            g=float(rad_hydro_case.g_Kelvin),
            alpha=float(rad_hydro_case.alpha),
            lambdap=float(rad_hydro_case.lambda_),
            f=float(rad_hydro_case.f_Kelvin),
            beta=float(rad_hydro_case.beta_Rosen),
            mu=float(rad_hydro_case.mu),
            gamma=float(rad_hydro_case.r) + 1.0,
        )
        
        # Find the self-similar front: this is an expensive operation
        # that computes xsi_f and Pf via root finding.
        solver.find_xsi_f()
        
        _subsonic_heat_wave_cache[case_key] = solver
    
    return _subsonic_heat_wave_cache[case_key]


def get_T_bath(
    state_star: RadHydroState,
    rad_hydro_case: RadHydroCase,
) -> float:
    """
    Calculate the bath temperature at the left boundary using the 
    subsonic heat wave (1D self-similar radiation diffusion) solution.
    
    This function:
    1. Creates/retrieves a cached SubsonicHeatWave solver for the given case
    2. Evaluates the self-similar profiles at the left boundary (xsi=0 region)
    3. Extracts the dimensionless boundary flux S[0]
    4. Calculates and returns the true bath temperature T_bath
    
    Parameters:
        state_star: Current hydro state with rho, m_cells, time
        rad_hydro_case: Problem configuration
        
    Returns:
        T_bath: Bath temperature in Kelvin
    """
    import logging
    logger = logging.getLogger("get_T_bath")
    
    try:
        # Get or create the SubsonicHeatWave solver
        heat_solver = _get_or_create_subsonic_heat_wave_solver(rad_hydro_case)
        
        # Get current time and mass grid
        t_sec = max(state_star.t, 1e-300)
        m_cells = np.asarray(state_star.m_cells, dtype=float)
        
        # Evaluate self-similar profiles at current state
        profiles = heat_solver.get_self_similar_profiles(xsi_vec=m_cells)
        
        # Extract the dimensionless boundary flux at the leftmost point
        S = profiles["S"]
        dimensionless_boundary_flux = S[0] if len(S) > 0 else 0.0
        
        # Calculate the actual bath temperature from the dimensionless flux
        T_bath = heat_solver.calc_T_bath_from_dimensionless_boundary_flux(
            dimensionless_boundary_flux=dimensionless_boundary_flux,
            time=t_sec
        )
        
        return float(T_bath)
    
    except Exception as e:
        logger.warning(
            f"Failed to compute T_bath from subsonic heat wave: {e}. "
            f"Falling back to T_surface = T0 * t^tau."
        )
        # Fallback to the simple T_surface approximation
        t_drive = max(state_star.t, 1e-9)
        T0_left = rad_hydro_case.T0_Kelvin if rad_hydro_case.T0_Kelvin is not None else 0.0
        T_left = T0_left * (t_drive / (10**-9)) ** rad_hydro_case.tau
        return float(T_left)

def solve_tridiagonal(
    a: np.ndarray, 
    b: np.ndarray, 
    c: np.ndarray, 
    d: np.ndarray, 
    use_scipy: bool = True
) -> np.ndarray:
    """Solves the tridiagonal system Ax = d where A has sub-diagonal a, diagonal b, and super-diagonal c."""
    # Defensive checks: ensure inputs are finite and shapes align
    for name, arr in ("a", a), ("b", b), ("c", c), ("d", d):
        if not np.all(np.isfinite(arr)):
            idx = np.where(~np.isfinite(arr))[0][0]
            raise ValueError(f"Non-finite value in '{name}' at index {idx}: {arr[idx]}")

    if use_scipy:
        from scipy.linalg import solve_banded
        N = len(b)
        if not (len(a) == N and len(c) == N and len(d) == N):
            raise ValueError(f"Tridiagonal vector length mismatch: len(a)={len(a)}, len(b)={len(b)}, len(c)={len(c)}, len(d)={len(d)}")
        ab = np.zeros((3, N))
        ab[0, 1:] = c[:N-1]  # super-diagonal
        ab[1, :] = b[:]    # diagonal
        ab[2, :-1] = a[1:]  # sub-diagonal
        if not np.all(np.isfinite(ab)):
            # locate first non-finite entry
            nonfin = np.argwhere(~np.isfinite(ab))[0]
            raise ValueError(f"Non-finite value in banded matrix at {tuple(nonfin)}: {ab[tuple(nonfin)]}")
        E_rad = solve_banded((1, 1), ab, d[:])
        return np.asarray(E_rad)
    
    else:
        n = len(b)
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n-1):
            denom = b[i] - a[i-1] * c_prime[i-1]
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom

        d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])

        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x
        

# Marshak-specialized tridiagonal assembler removed: calculate_abcd now
# returns full-length system and `solve_tridiagonal` is used directly.

def radiation_step(
    state_star: RadHydroState, 
    dt: float, 

    rad_hydro_case: RadHydroCase,
    T_left: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Updates the material specific energy & radiation energy density based on the coupling between matter and radiation.
    
    Parameters:
        state_star: Post-hydro state with e_material, rho, T_rad, E_rad
        dt: Time step in seconds
        rad_hydro_case: Problem configuration
        
    Returns:
        new_T_material: Updated material temperature in K
        new_e_material: Updated material specific energy in erg/g
        new_T_rad: Updated radiation temperature in K
        new_E_rad: Updated radiation energy density in erg/cm^3
    """
    global alpha, beta_Rosen, mu, f_Kelvin, chi, lambda_, g_Kelvin
    alpha, beta_Rosen, mu, f_Kelvin, chi, lambda_, g_Kelvin = rad_hydro_case._get_params()
    e_star, rho, m_cells, E_rad, T_rad_current = (
        state_star.e_material,
        state_star.rho,
        state_star.m_cells,
        state_star.E_rad,
        state_star.T_rad,
    )

    mode = getattr(rad_hydro_case, "force_black", None)
    # New naming: None | "gray" | "black"
    valid_modes = (None, "gray", "black")
    if mode not in valid_modes:
        raise ValueError(f"Invalid force_black mode '{mode}'. Expected one of {valid_modes}.")

    if mode == "black":
        new_e, new_T = black_radiation_step(state_star, dt, rad_hydro_case)
        return new_T, new_e, new_T, new_e

    # Material temperature from e_star (match working: use T_material for beta and sigma)
    T_material_star = calculate_temperature_from_specific_energy(e_star, rho, f_Kelvin, beta_Rosen, mu)
    beta = calculate_beta_from_temperature_and_density(T_material_star, rho)
    sigma = calculate_sigma_from_temperature_and_density(T_material_star, rho)
    D = calculate_D_from_sigma(sigma)
    A = calculate_A(beta, sigma, dt)

    # Calculate tridiagonal system coefficients.
    # If Marshak BC requested, compute the left bath drive first and let
    # calculate_abcd_marshak apply the Marshak modifications to the system.
    bc_type = getattr(rad_hydro_case, "bc_type", "Marshak")
    # Build the full system; for "gray" mode we pass E_rad=None so the
    # implicit E_old contribution is omitted (gray approximation).
    if mode == "gray":
        a, b, c_coeff, d = calculate_abcd(
            sigma, D, A, m_cells, rho, None, T_material_star, dt, bc_type=bc_type, T_left=T_left
        )
    else:
        a, b, c_coeff, d = calculate_abcd(
            sigma, D, A, m_cells, rho, E_rad, T_material_star, dt, bc_type=bc_type, T_left=T_left
        )

    # Solve for radiation energy density and temperature
    new_E_rad = solve_tridiagonal(a, b, c_coeff, d)
    new_T_rad = (new_E_rad / a_Kelvin) ** (1 / 4)

    # Solve for material energy density and temperature
    UR_star = a_Kelvin * T_material_star**4
    new_UR = (A / (1 + A)) * new_E_rad + (1 / (1 + A)) * UR_star
    new_T_material = (new_UR / a_Kelvin) ** (1 / 4)
    new_e_material = f_Kelvin * new_T_material**beta_Rosen * rho**(-mu)

    return new_T_material, new_e_material, new_T_rad, new_E_rad
