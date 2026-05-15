# physical constants
from _typeshed import _type_checker_internals
from _typeshed import _type_checker_internals
c = 3e10  # speed of light [cm/s]
a_Kelvin = 7.5646e-15  # Radiation constant in erg cm^-3 K^-4
eV_to_erg = 1.60218e-12  # electron energy in CGS [erg/eV]
Hev_to_erg =  100 * eV_to_erg  # electron energy in CGS [erg/Hev]
k_B = 1.380649e-16  # Boltzmann constant in CGS [erg/K]
KELVIN_PER_HEV = Hev_to_erg / k_B  # Conversion factor from keV to Kelvin
a_Hev = a_Kelvin * KELVIN_PER_HEV**4  # Radiation constant in keV cm^-3 keV^-4

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.hydro_sim.core.state import RadHydroState

def calculate_temperature_from_specific_energy(
    e_material: np.ndarray, rho: np.ndarray, f: float, gamma: float, mu: float
) -> np.ndarray:
    return ((e_material / f) * rho**mu) ** (1/gamma)
     

def calculate_beta_from_temperature_and_density(T: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return 4*a_Kelvin / (f_Kelvin * gamma) * T**(4-gamma) * rho**(mu - 1)

def calculate_sigma_from_temperature_and_density(T: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return 1.0 / (g_Kelvin * T**alpha * rho**(-lambda_ - 1))

def calculate_D_from_sigma(sigma: np.ndarray) -> np.ndarray:
    return c / (3 * sigma)

def calculate_A(beta: np.ndarray, sigma: np.ndarray, dt: float) -> np.ndarray:
    return chi * beta * sigma * dt * c



def calculate_abcd(sigma: np.ndarray, D: np.ndarray, A: np.ndarray, m_cells: np.ndarray, rho: np.ndarray, E_rad
                   : np.ndarray, T_star: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the coefficients a, b, c, d for the implicit update of material energy and radiation energy density.
    a = [a1, a2, ..., aN], a_i^n = a[i-1] at time step n
    """
    D_face_left = rho[:-2] * (D[:-2] + D[1:-1]) / 2 # Left face
    D_face_right = rho[2:] * (D[1:-1] + D[2:]) / 2 # Right face
    F = chi*c*sigma[1:-1]/(1 + A[1:-1])
    coeff = rho[1:-1] / (m_cells[1:-1]**2)
    a = -coeff * D_face_left
    c_coeff = -coeff * D_face_right
    # b = coeff * (D_face_right - D_face_left) + 1/dt + F # My version
    b = coeff * (D_face_right + D_face_left) + 1/dt + F # Corrected version with positive diffusion coefficients on the diagonal
    UR_star = a_Kelvin * T_star[1:-1]**4
    d = F * UR_star + (1/dt) * E_rad[1:-1]

    # check for nan values in the coefficients which would indicate a problem with the input parameters or the state    if np.any(np.isnan(a)):
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

def calculate_abcd_Marshak(sigma: np.ndarray, D: np.ndarray, A: np.ndarray, m_cells: np.ndarray, rho: np.ndarray, E_rad: np.ndarray, T_star: np.ndarray, dt: float, T_bath: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the coefficients a, b, c, d for the implicit update with Marshak boundary conditions."""
    M = len(rho)
    a = np.zeros(M)
    b = np.zeros(M)
    c_coeff = np.zeros(M)
    d = np.zeros(M)

    # Interior cells (1 to M-2)
    D_face_left_interior = rho[:-2] * (D[:-2] + D[1:-1]) / 2 
    D_face_right_interior = rho[2:] * (D[1:-1] + D[2:]) / 2 
    F_interior = chi*c*sigma[1:-1]/(1 + A[1:-1])
    coeff_interior = rho[1:-1] / (m_cells[1:-1]**2)

    a[1:-1] = -coeff_interior * D_face_left_interior
    c_coeff[1:-1] = -coeff_interior * D_face_right_interior
    b[1:-1] = coeff_interior * (D_face_right_interior + D_face_left_interior) + 1/dt + F_interior
    UR_star_interior = a_Kelvin * T_star[1:-1]**4
    d[1:-1] = F_interior * UR_star_interior + (1/dt) * E_rad[1:-1]

    sigma_sb = a_Kelvin * c / 4

    # Boundary cell 0 (Left Marshak)
    D_face_right_0 = rho[1] * (D[0] + D[1]) / 2 
    coeff_0 = rho[0] / (m_cells[0]**2)
    F_0 = chi*c*sigma[0]/(1 + A[0])
    UR_star_0 = a_Kelvin * T_star[0]**4

    c_coeff[0] = -coeff_0 * D_face_right_0
    b[0] = 1/dt + coeff_0 * D_face_right_0 + F_0 + c * rho[0] / (2 * m_cells[0])
    d[0] = E_rad[0]/dt + 2 * rho[0] * sigma_sb * T_bath**4 / m_cells[0] + F_0 * UR_star_0
    a[0] = 0.0

    # Boundary cell M-1 (Right Marshak)
    D_face_left_M1 = rho[-2] * (D[-2] + D[-1]) / 2
    coeff_M1 = rho[-1] / (m_cells[-1]**2)
    F_M1 = chi*c*sigma[-1]/(1 + A[-1])
    UR_star_M1 = a_Kelvin * T_star[-1]**4

    a[-1] = -coeff_M1 * D_face_left_M1
    b[-1] = 1/dt + coeff_M1 * D_face_left_M1 + F_M1 + c * rho[-1] / (2 * m_cells[-1])
    d[-1] = E_rad[-1]/dt + F_M1 * UR_star_M1
    c_coeff[-1] = 0.0

    # check for nan values
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c_coeff)) or np.any(np.isnan(d)):
        raise ValueError("NaN value encountered in Marshak coefficients")
    if np.any(b <= 0):
        raise ValueError("Non-positive diagonal encountered in Marshak coefficients")

    return a, b, c_coeff, d

def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, use_scipy: bool = True, bc_type: str = "dirichlet") -> np.ndarray:
    """Solves the tridiagonal system Ax = d where A has sub-diagonal a, diagonal b, and super-diagonal c.
    If bc_type is 'dirichlet', the returned array is padded with zeros at the boundaries (length N+2).
    If bc_type is 'marshak', the returned array is the exact solution (length N).
    """
    n = len(b)
    if use_scipy:
        from scipy.linalg import solve_banded
        ab = np.zeros((3, n))
        ab[0, 1:] = c[:-1]  # super-diagonal
        ab[1, :] = b        # diagonal
        ab[2, :-1] = a[1:]  # sub-diagonal
        x = solve_banded((1, 1), ab, d)
    else:
        # Thomas algorithm
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        # Forward sweep
        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            if i < n - 1:
                c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
            
        # Back substitution
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

    if bc_type == "dirichlet":
        E_rad = np.zeros(n + 2)
        E_rad[1:-1] = x
        return E_rad
    elif bc_type == "marshak":
        return x
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")
        

def radiation_step(state_star: RadHydroState, dt: float, rad_hydro_case: RadHydroCase, bc_type: str = "dirichlet") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Updates the material specific energy & radiation energy density based on the coupling between matter and radiation.
    
    Parameters:
        state_star: RadHydroState
        dt: Time step in seconds
        rad_hydro_case: RadHydroCase
        bc_type: "dirichlet" or "marshak"
        
    Returns:
        new_e_material: Updated material specific energy in erg/g
        new_T: Updated temperature
        new_E_rad: Radiation energy density in erg/cm^3
    """
    global alpha, gamma, mu, f_Kelvin, chi, lambda_, g_Kelvin
    alpha, gamma, mu, f_Kelvin, chi, lambda_, g_Kelvin = rad_hydro_case._get_params()
    e_star, rho, m_cells, E_rad, T_star = (
        state_star.e,
        state_star.rho,
        state_star.m_cells,
        state_star.E_rad,
        state_star.T,
    )

    # calculating the opacity & specific energy from Rosen's model
    T_star = calculate_temperature_from_specific_energy(e_star, rho, f_Kelvin, gamma, mu)
    beta = calculate_beta_from_temperature_and_density(T_star, rho)
    sigma = calculate_sigma_from_temperature_and_density(T_star, rho)
    D = calculate_D_from_sigma(sigma)
    A = calculate_A(beta, sigma, dt)

    # calculating the coefficients for the implicit update
    a, b, c_coeff, d = calculate_abcd(sigma, D, A, m_cells, rho, E_rad, T_star, dt)

    t_drive = max(state_star.t, dt) # avoid t=0 for the power law drive
    T_left = rad_hydro_case.T0_Kelvin * (t_drive/(10**-9))**rad_hydro_case.tau
    E_left = a_Kelvin * T_left**4

    if bc_type == "marshak":
        a, b, c_coeff, d = calculate_abcd_Marshak(sigma, D, A, m_cells, rho, E_rad, T_star, dt, T_left)
        new_E_rad = solve_tridiagonal(a, b, c_coeff, d, bc_type="marshak")
    else:
        a, b, c_coeff, d = calculate_abcd(sigma, D, A, m_cells, rho, E_rad, T_star, dt)
        d[0] -= a[0] * E_left
        new_E_rad = solve_tridiagonal(a, b, c_coeff, d, bc_type="dirichlet")
        new_E_rad[0] = E_left  # Left boundary condition: E_rad[0] = a_Kelvin * T_left^4 (consistent with the boundary condition for material energy)
        new_E_rad[-1] = 0 # Right boundary condition: E_rad[-1] = 0 (vacuum)

    # updating UR, T and material specific energy based on the new radiation energy density
    UR_star = a_Kelvin * T_star**4   # length N (same as A and new_E_rad)
    if bc_type == "dirichlet":
        UR_star[0]  = E_left          # left boundary consistent
        UR_star[-1] = 1e-10             # right boundary vacuum
    new_UR = (A / (1 + A)) * new_E_rad + (1 / (1 + A)) * UR_star
    new_Um = new_UR / gamma

    new_T = (new_UR / a_Kelvin)**(1/4)  # calculating the temperature from the updated effective radiation energy density
    new_e_material = f_Kelvin * new_T**gamma * rho**(-mu)  # Calculating the material specific energy from Rosen's model using the updated temperature and density
    return new_e_material, new_T, new_E_rad 