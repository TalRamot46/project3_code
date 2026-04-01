# physical constants
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
    coeff_scheme: str = "legacy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the coefficients a, b, c, d for the implicit update of material energy and radiation energy density.
    a = [a1, a2, ..., aN], a_i^n = a[i-1] at time step n

    T_material is the material temperature, used for UR_star = a*T_material^4 (coupling drives E toward material equilibrium).
    """
    if coeff_scheme == "legacy":
        if HARMONIC_MEAN:
            D_face = harmonic_mean(D[:-1], D[1:])
        else:
            D_face = arithmetic_mean(D[:-1], D[1:])
        D_face_left = D_face[:-1] # stands for D_i
        D_face_right = D_face[1:] # stands for D_{i+1}
        F = chi * c * sigma[1:-1] / (1 + A[1:-1])
        coeff = rho[1:-1] / (m_cells[1:-1] ** 2)
        a = -coeff * D_face_left
        c_coeff = -coeff * D_face_right
        b = coeff * (D_face_right + D_face_left) + 1 / dt + F
    elif coeff_scheme == "face_weighted":
        if HARMONIC_MEAN:
            D_face = harmonic_mean(D[:-1], D[1:])
            rho_face = harmonic_mean(rho[:-1], rho[1:])
            m_face = harmonic_mean(m_cells[:-1], m_cells[1:])
        else:
            D_face = arithmetic_mean(D[:-1], D[1:])
            rho_face = arithmetic_mean(rho[:-1], rho[1:])
            m_face = arithmetic_mean(m_cells[:-1], m_cells[1:])

        rho_face_left = rho_face[:-1] # stands for rho_i
        rho_face_right = rho_face[1:] # stands for rho_{i+1}
        D_face_left = D_face[:-1] # stands for D_i
        D_face_right = D_face[1:] # stands for D_{i+1}
        m_face_left = m_face[:-1] # stands for m_i
        m_face_right = m_face[1:] # stands for m_{i+1}

        left_flux_coeff = (rho_face_left * D_face_left) / m_face_left
        right_flux_coeff = (rho_face_right * D_face_right) / m_face_right
        coeff = rho[1:-1] / m_cells[1:-1] # this is not a mistake! It comes from
                                   # the first lagrangian derivative of the flux.

        F = chi * c * sigma[1:-1] / (1 + A[1:-1])
        a = -coeff * left_flux_coeff
        c_coeff = -coeff * right_flux_coeff
        b = coeff * (left_flux_coeff - right_flux_coeff) + 1 / dt + F
    else:
        raise ValueError(
            f"Invalid radiation coefficient scheme '{coeff_scheme}'. "
            "Expected 'legacy' or 'face_weighted'."
        )

    UR_star = a_Kelvin * T_material[1:-1]**4
    if E_rad is not None:
        d = F * UR_star + (1/dt) * E_rad[1:-1]
    else: # attempt to force black physics under LTE (T_m = T_rad)
        E_rad = a_Kelvin * T_material**4
        d = F * UR_star + (1/dt) * E_rad[1:-1]

    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c_coeff)) or np.any(np.isnan(d)):
        j = np.where(np.isnan(a))[0][0] if np.any(np.isnan(a)) else (np.where(np.isnan(b))[0][0] if np.any(np.isnan(b)) else (np.where(np.isnan(c_coeff))[0][0] if np.any(np.isnan(c_coeff)) else np.where(np.isnan(d))[0][0]))
        raise ValueError(
            f"NaN value encountered in coefficients at interior index {j}: "
            f"a={a[j]}, b={b[j]}, c={c_coeff[j]}, "
            f"coef={coeff[j]}, dt={dt}, F={F[j]}, UR_star={UR_star[j]}"
        )
    
    if np.any(b <= 0):
        j = np.where(b <= 0)[0][0]
        raise ValueError(
            f"Non-positive diagonal at interior index {j}: "
            f"b={b[j]}, a={a[j]}, c={c_coeff[j]}, "
            f"coef={coeff[j]}, dt={dt}, F={F[j]}"
        )
    return a, b, c_coeff, d


def _get_right_bc_mode(rad_hydro_case: RadHydroCase) -> str:
    """Return normalized right-BC mode name."""
    mode = str(getattr(rad_hydro_case, "right_BC", "Neuman"))
    valid_modes = {"Dirichlet", "Neuman", "free"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid right_BC '{mode}'. Expected one of {sorted(valid_modes)}.")
    return mode


def _apply_right_bc_to_system(
    b: np.ndarray,
    c_coeff: np.ndarray,
    d: np.ndarray,
    rad_hydro_case: RadHydroCase,
) -> float | None:
    """
    Modify the last interior equation according to selected right BC.
    Returns fixed right boundary value for Dirichlet, else None.
    """
    right_mode = _get_right_bc_mode(rad_hydro_case)

    if right_mode == "Dirichlet":
        if rad_hydro_case.T_initial_Kelvin is None:
            raise ValueError("T_initial_Kelvin must be provided when right_BC='Dirichlet'.")
        e_right = a_Kelvin * float(rad_hydro_case.T_initial_Kelvin) ** 4
        d[-1] -= c_coeff[-1] * e_right
        return e_right

    if right_mode == "Neuman":
        # Right "Vaccum" BC (trivial Neumann): dE/dx = 0 -> E_N = E_{N-1}.
        b[-1] += c_coeff[-1]
        return None

    # "free": leave last equation unconstrained at right edge (no explicit right BC in solve).
    return None


def _apply_right_bc_to_solution(
    E: np.ndarray,
    rad_hydro_case: RadHydroCase,
    e_right_dirichlet: float | None,
) -> None:
    """Write the boundary cell from selected right BC after interior solve."""
    right_mode = _get_right_bc_mode(rad_hydro_case)
    if right_mode == "Dirichlet":
        if e_right_dirichlet is None:
            raise ValueError("Internal error: missing Dirichlet right boundary value.")
        E[-1] = e_right_dirichlet
    elif right_mode == "Neuman":
        E[-1] = E[-2]
    else:  # "free"
        # Free right edge: linear extrapolation from interior for a smooth open boundary.
        E[-1] = 2.0 * E[-2] - E[-3] if len(E) >= 3 else E[-2]

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
    e_right_dirichlet = _apply_right_bc_to_system(b, c_coeff, d, rad_hydro_case)

    d[0] -= a[0] * e_left
    new_e = solve_tridiagonal(a, b, c_coeff, d)
    new_e[0] = e_left
    _apply_right_bc_to_solution(new_e, rad_hydro_case, e_right_dirichlet)

    new_T = (new_e / a_Kelvin) ** (1 / 4)
    return new_e, new_T

def solve_tridiagonal(
    a: np.ndarray, 
    b: np.ndarray, 
    c: np.ndarray, 
    d: np.ndarray, 
    use_scipy: bool = True
) -> np.ndarray:
    """Solves the tridiagonal system Ax = d where A has sub-diagonal a, diagonal b, and super-diagonal c."""
    if use_scipy:
        from scipy.linalg import solve_banded
        N = len(b) + 2
        ab = np.zeros((3, N - 2))
        ab[0, 1:] = c[:N-3]  # super-diagonal
        ab[1, :] = b[:N-2]    # diagonal
        ab[2, :-1] = a[1:N-2]  # sub-diagonal
        E_rad_interior = solve_banded((1, 1), ab, d[:N-2])
        E_rad = np.zeros(N)
        E_rad[1:N-1] = E_rad_interior
        return E_rad
    
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
        

def radiation_step(
    state_star: RadHydroState, 
    dt: float, 
    rad_hydro_case: RadHydroCase
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
    coeff_scheme = getattr(rad_hydro_case, "radiation_coeff_scheme", "legacy")
    valid_modes = (None, "gray corrected", "full black")
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid force_black mode '{mode}'. Expected one of {valid_modes}."
        )
    valid_coeff_schemes = ("legacy", "face_weighted")
    if coeff_scheme not in valid_coeff_schemes:
        raise ValueError(
            f"Invalid radiation_coeff_scheme '{coeff_scheme}'. Expected one of {valid_coeff_schemes}."
        )

    if mode == "full black":
        new_e, new_T = black_radiation_step(state_star, dt, rad_hydro_case)
        return new_T, new_e, new_T, new_e

    # Material temperature from e_star (match working: use T_material for beta and sigma)
    T_material_star = calculate_temperature_from_specific_energy(e_star, rho, f_Kelvin, beta_Rosen, mu)
    beta = calculate_beta_from_temperature_and_density(T_material_star, rho)
    sigma = calculate_sigma_from_temperature_and_density(T_material_star, rho)
    D = calculate_D_from_sigma(sigma)
    A = calculate_A(beta, sigma, dt)

    # Calculate tridiagonal system coefficients
    if mode == "gray corrected":
        a, b, c_coeff, d = calculate_abcd(
            sigma, D, A, m_cells, rho, None, T_material_star, dt, coeff_scheme=coeff_scheme
        )
    else:
        a, b, c_coeff, d = calculate_abcd(
            sigma, D, A, m_cells, rho, E_rad, T_material_star, dt, coeff_scheme=coeff_scheme
        )

    # Apply boundary conditions
    t_drive = max(state_star.t, dt)
    T0_left = rad_hydro_case.T0_Kelvin if rad_hydro_case.T0_Kelvin is not None else 0.0
    T_left = T0_left * (t_drive/(10**-9))**rad_hydro_case.tau
    E_left = a_Kelvin * T_left**4
    d[0] -= a[0] * E_left

    e_right_dirichlet = _apply_right_bc_to_system(b, c_coeff, d, rad_hydro_case)

    # Solve for radiation energy density and temperature
    new_E_rad = solve_tridiagonal(a, b, c_coeff, d)
    new_E_rad[0] = E_left
    _apply_right_bc_to_solution(new_E_rad, rad_hydro_case, e_right_dirichlet)
    new_T_rad = (new_E_rad / a_Kelvin) ** (1 / 4)

    # Solve for material energy density and temperature
    UR_star = a_Kelvin * T_material_star**4
    new_UR = (A / (1 + A)) * new_E_rad + (1 / (1 + A)) * UR_star
    new_T_material = (new_UR / a_Kelvin) ** (1 / 4)
    new_e_material = f_Kelvin * new_T_material**beta_Rosen * rho**(-mu)

    return new_T_material, new_e_material, new_T_rad, new_E_rad