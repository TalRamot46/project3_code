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

def calculate_abcd(sigma: np.ndarray, D: np.ndarray, A: np.ndarray, m_cells: np.ndarray, rho: np.ndarray, E_rad
                   : np.ndarray, T_material: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the coefficients a, b, c, d for the implicit update of material energy and radiation energy density.
    a = [a1, a2, ..., aN], a_i^n = a[i-1] at time step n

    T_material is the material temperature, used for UR_star = a*T_material^4 (coupling drives E toward material equilibrium).
    """
    if HARMONIC_MEAN:
        D_face_left = rho[:-2] * harmonic_mean(D[:-2], D[1:-1])
        D_face_right = rho[2:] * harmonic_mean(D[1:-1], D[2:])
    else:
        D_face_left = rho[:-2] * arithmetic_mean(D[:-2], D[1:-1]) # Left face
        D_face_right = rho[2:] * arithmetic_mean(D[1:-1], D[2:]) # Right face
    F = chi*c*sigma[1:-1]/(1 + A[1:-1])
    coeff = rho[1:-1] / (m_cells[1:-1]**2)
    a = -coeff * D_face_left
    c_coeff = -coeff * D_face_right
    b = coeff * (D_face_right + D_face_left) + 1/dt + F
    UR_star = a_Kelvin * T_material[1:-1]**4
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

def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, use_scipy: bool = True) -> np.ndarray:
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
        

def radiation_step(state_star: RadHydroState, dt: float, rad_hydro_case: RadHydroCase) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # Material temperature from e_star (match working: use T_material for beta and sigma)
    T_material_star = calculate_temperature_from_specific_energy(e_star, rho, f_Kelvin, beta_Rosen, mu)
    beta = calculate_beta_from_temperature_and_density(T_material_star, rho)
    sigma = calculate_sigma_from_temperature_and_density(T_material_star, rho)
    D = calculate_D_from_sigma(sigma)
    A = calculate_A(beta, sigma, dt)

    # Build tridiagonal system; UR_star uses T_material (coupling drives E toward material equilibrium, match 1D Diffusion)
    a, b, c_coeff, d = calculate_abcd(sigma, D, A, m_cells, rho, E_rad, T_material_star, dt)

    t_drive = max(state_star.t, dt)
    T_left = rad_hydro_case.T0_Kelvin * (t_drive/(10**-9))**rad_hydro_case.tau
    E_left = a_Kelvin * T_left**4
    d[0] -= a[0] * E_left

    new_E_rad = solve_tridiagonal(a, b, c_coeff, d)
    new_E_rad[0] = E_left
    # Right BC: T_right_Kelvin=0 or None -> vacuum (E_right=0); >0 -> cold sink (match 1D Diffusion / run_diffusion_1d)
    T_right = rad_hydro_case.T_right_Kelvin or 0.0
    new_E_rad[-1] = 0.0 if T_right <= 0 else a_Kelvin * T_right**4
    new_T_rad = (new_E_rad / a_Kelvin) ** (1 / 4)

    # Coupled update (match working): new_UR = (A*E_new + UR_star)/(1+A), then derive T and e from Rosen EOS
    UR_star = a_Kelvin * state_star.T_material**4
    UR_star[0] = E_left
    UR_star[-1] = 1e-10 if T_right <= 0 else a_Kelvin * T_right**4
    new_UR = (A / (1 + A)) * new_E_rad + (1 / (1 + A)) * UR_star

    # Temperature from coupled result; material e from Rosen EOS (match working)
    new_T_material = (new_UR / a_Kelvin) ** (1 / 4)
    new_T_material[0] = T_left
    new_T_material[-1] = T_right if T_right > 0 else 0.0
    new_e_material = f_Kelvin * new_T_material**beta_Rosen * rho**(-mu)

    return new_T_material, new_e_material, new_T_rad, new_E_rad