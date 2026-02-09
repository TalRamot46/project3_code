# physical constants
c = 3e10  # speed of light [cm/s]
a_Kelvin = 7.5646e-15  # Radiation constant in erg cm^-3 K^-4
eV_to_erg = 1.60218e-12  # electron energy in CGS [erg/eV]
k_B = 1.380649e-16  # Boltzmann constant in CGS [erg/K]
K_per_Hev = 1.16045e7  # Conversion factor from keV to Kelvin
a_Hev = a_Kelvin * K_per_Hev**4  # Radiation constant in keV cm^-3 keV^-4

import numpy as np
from dataclasses import dataclass

from traitlets import Tuple
from project_3.rad_hydro_sim.problems import RadHydroCase

def calculate_temperature_from_specific_energy(e_material: np.ndarray, rho: np.ndarray) -> np.ndarray:
    T_Hev = (e_material * rho/ f)**(1/gamma) 
    return T_Hev

def calculate_beta_from_temperature_and_density(T: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return 4*a_Hev / (f * gamma) * T**(4-gamma) * rho**(mu - 1) # MAKE SURE THIS IS CORRECT BECAUSER OF a_Hev!

def calculate_sigma_from_temperature_and_density(T: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return 1.0 / (g * T**alpha * rho**(-lambda_ - 1))

def calculate_D_from_sigma(sigma: np.ndarray) -> np.ndarray:
    return c / (3 * sigma)

def calculate_A(beta: np.ndarray, sigma: np.ndarray, dt: float) -> np.ndarray:
    return chi * beta * sigma * dt

def calculate_abcd(sigma: np.ndarray, D: np.ndarray, A: np.ndarray, m_cells: np.ndarray, rho: np.ndarray, E: np.ndarray, T: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the coefficients a, b, c, d for the implicit update of material energy and radiation energy density.
    a = [a1, a2, ..., aN], a_i^n = a[i-1] at time step n
    """
    A = rho[:-2] * 2*D[:-2]*D[1:-1]/(D[:-2] + D[1:-1])
    B = rho[2:] * 2*D[1:-1]*D[2:]/(D[1:-1] + D[2:])
    D = chi*c*sigma[1:-1]/(1 + A[1:-1])
    a = -rho[1:-1] / (m_cells[1:-1]**2) * A
    b = rho[1:-1] / (m_cells[1:-1]**2) * (B - A) + 1/dt + D
    c = rho[1:-1] / (m_cells[1:-1]**2) * B
    d = D * a_Hev * T[1:-1]**4 + 1/dt * E[1:-1]
    return a, b, c, d

def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, use_scipy: bool = True) -> np.ndarray:
    """Solves the tridiagonal system Ax = d where A has sub-diagonal a, diagonal b, and super-diagonal c."""
    if use_scipy:
        from scipy.linalg import solve_banded
        # Create the banded matrix for solve_banded
        # The matrix A has the form:
        # [b[0], c[0], 0, ..., 0]
        # [a[0], b[1], c[1], ..., 0]
        # [0, a[1], b[2], ..., 0]
        # ...
        # [0, ..., a[n-2], b[n-1]]
        """ THIS IS WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEED TO FIX THE INDICES FOR a, b, c BECAUSE THEY ARE NOT THE SAME LENGTH! a and c are length n-1 while b is length n. ALSO NEED TO MAKE SURE THE BOUNDARY CONDITIONS ARE HANDLED CORRECTLY! """
        n = len(b)
        ab = np.zeros((3, n))
        ab[0, 1:] = c  # super-diagonal
        ab[1, :] = b   # diagonal
        ab[2, :-1] = a # sub-diagonal
        x = solve_banded((1, 1), ab, d)
        return x
    else: # Implement Thomas algorithm for tridiagonal systems
        n = len(b)
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        # Forward sweep
        for i in range(1, n-1):
            denom = b[i] - a[i-1] * c_prime[i-1]
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom

        d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])

        # Back substitution
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x
        

def hydro_step(e_star: np.ndarray, rho: np.ndarray, m_cells: np.ndarray, E: np.ndarray, dt: float, rad_hydro_case: RadHydroCase) -> np.ndarray:
    """
    Updates the material specific energy & radiation energy density based on the coupling between matter and radiation.
    
    Parameters:
        e_star: Material specific energy in erg/g
        rho: Material density in g/cm^3
        dt: Time step in seconds
        
    Returns:
        new_e_material: Updated material specific energy in erg/g
        e_rad: Radiation energy density in erg/cm^3
    """
    global alpha, gamma, mu, f, chi, lambda_, g
    alpha, gamma, mu, f, chi, lambda_, g = rad_hydro_case._get_params()

    # calculating the opacity & specific energy from Rosen's model
    T = calculate_temperature_from_specific_energy(e_star, rho)
    beta = calculate_beta_from_temperature_and_density(T, rho)
    sigma = calculate_sigma_from_temperature_and_density(T, rho)
    D = calculate_D_from_sigma(sigma)
    A = calculate_A(beta, sigma, dt)

    # calculating the coefficients for the implicit update
    a, b, c, d = calculate_abcd(sigma, D, A, m_cells, rho, E, T, dt)

    # solving the tridiagonal system for radiation energy density
    e_rad = solve_tridiagonal(a, b, c, d)
    new_U_R = A / (1 + A) * E + 1 / (1 + A) * a * T**4  # Placeholder, update with actual calculation
    new_T = (new_U_R / a_Hev)**(1/4)  # calculating the temperature from the updated effective radiation energy density
    new_e_material = f * gamma / (gamma - 1) * new_T**gamma * rho**(mu - 1)  # Calculating the material specific energy from Rosen's model using the updated temperature and density
    return new_e_material, e_rad