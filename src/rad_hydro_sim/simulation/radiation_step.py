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

# Picard iteration controls for the nonlinear conduction solve (force_black="conduction").
PICARD_MAX_ITERS = 50
PICARD_TOL = 1e-8

import numpy as np
from dataclasses import dataclass
from typing import Tuple

def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray: return 2 * a * b / (a + b)
def arithmetic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray: return (a + b) / 2

from rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from hydro_sim.core.state import RadHydroState

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
    Build full-length (N = len(rho)) tridiagonal coefficients for the
    black-physics diffusion update (single temperature, e = E_rad = e_material):
        a_j e_{j-1}^{n+1} + b_j e_j^{n+1} + c_j e_{j+1}^{n+1} = d_j

    Mirrors ``calculate_abcd``'s face-weighted construction (same natural
    outflow behaviour at the right boundary from the edge-padded flux
    coefficients) but without the matter-radiation coupling term, since e is
    a single shared field here.
    """
    if HARMONIC_MEAN:
        D_face = harmonic_mean(D[:-1], D[1:])
    else:
        D_face = arithmetic_mean(D[:-1], D[1:])

    dx_cells = m_cells / rho
    dx_face = arithmetic_mean(dx_cells[:-1], dx_cells[1:])

    flux_coeff = D_face / dx_face  # defined at i=1,...,N-1
    flux_coeff = np.concatenate(([flux_coeff[0]], flux_coeff, [flux_coeff[-1]]))
                                  # defined at i=0,1,...,N-1,N
    lagrangian_coeff = 1.0 / dx_cells  # defined at j=1,...,N

    a = -lagrangian_coeff * flux_coeff[:-1]
    b = lagrangian_coeff * (flux_coeff[:-1] + flux_coeff[1:]) + 1.0 / dt
    c_coeff = -lagrangian_coeff * flux_coeff[1:]
    d = e_old / dt

    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c_coeff)) or np.any(np.isnan(d)):
        raise ValueError("NaN encountered in black-radiation tridiagonal coefficients.")
    if np.any(b <= 0):
        raise ValueError("Non-positive diagonal encountered in black-radiation tridiagonal system.")

    return a, b, c_coeff, d


def _matter_energy_density(T: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Paper's u = f T^beta rho^(1-mu) [erg/cm^3] (Krief 2021, Eq. 5 per volume)."""
    return f_Kelvin * T**beta_Rosen * rho**(1.0 - mu)


def calculate_conduction_abcd(
    D: np.ndarray,
    x_nodes: np.ndarray,
    rho: np.ndarray,
    u_old: np.ndarray,
    E_k: np.ndarray,
    T_k: np.ndarray,
    dt: float,
    geom,
    T_left: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal coefficients for the paper's nonlinear conduction equation.

    Krief (2021) Eq. (1) stores energy in the *matter* and uses radiation only
    as the flux potential:

        d(u)/dt = div( D grad(a T^4) ),   u = f T^beta rho^(1-mu)

    Solved implicitly in the flux potential E = a T^4. Because u(E) is strongly
    nonlinear (u ~ E^(beta/4), and across the heat front T spans several orders
    of magnitude), a single linearisation frozen at the old state is not
    energy-conservative and systematically retards a degenerate front. Instead
    this builds one Picard iterate about the current guess E_k:

        u(E^{k+1}) ~ u(E^k) + chi_v^k (E^{k+1} - E^k),
        chi_v = du/dE = f beta T^(beta-1) rho^(1-mu) / (4 a T^3),

    giving, after dividing through by chi_v^k,

        E^{k+1}/dt - (1/chi_v^k) div(D grad E^{k+1})
            = E^k/dt + (u_old - u(E^k)) / (chi_v^k dt).

    Iterating to convergence recovers the fully implicit, conservative update
    (the bracket vanishes as E^k -> E^{n+1}).

    This differs from ``calculate_black_abcd``, which stores the energy in the
    radiation field itself (d(aT^4)/dt = div(D grad aT^4)) -- a different PDE
    whenever a T^4 is not the dominant energy reservoir.

    The divergence is discretised in conservative area-weighted form,

        div(F)_j = [ A_{j+1/2} F_{j+1/2} - A_{j-1/2} F_{j-1/2} ] / V_j,

    with A = beta r^alpha and V = zeta (r_{j+1}^{alpha+1} - r_j^{alpha+1}) taken
    from ``geom``, so planar / cylindrical / spherical symmetry are all handled
    (for planar, A = 1 and V = dx, recovering the plain 1D operator).

    ``T_left=None`` imposes zero energy flux through the inner face, which is
    the symmetry condition at the origin (paper Eq. 21) and the only option
    available for omega >= omega_c, where the origin temperature diverges.
    """
    if HARMONIC_MEAN:
        D_face = harmonic_mean(D[:-1], D[1:])
    else:
        D_face = arithmetic_mean(D[:-1], D[1:])

    a_geom = geom.alpha
    r = np.asarray(x_nodes, dtype=float)
    r_cent = 0.5 * (r[:-1] + r[1:])

    V = geom.zeta * (r[1:] ** (a_geom + 1) - r[:-1] ** (a_geom + 1))
    A_face = geom.beta * r[1:-1] ** a_geom          # interior faces
    dr_face = r_cent[1:] - r_cent[:-1]              # centre-to-centre spacing

    flux_coeff = A_face * D_face / dr_face

    if T_left is None:
        # Symmetry / no-flux inner boundary.
        left_fc = 0.0
    else:
        # The Dirichlet value lives on the inner face of cell 0, a distance
        # r_cent[0] - r[0] from its centre -- not the centre-to-centre spacing
        # that edge-padding would copy in from face 1. Padding understates the
        # boundary gradient and starves the wave of the energy that sets its
        # front position.
        D_bc = arithmetic_mean(
            calculate_D_from_sigma(calculate_sigma_from_temperature_and_density(T_left, rho[0])),
            D[0],
        )
        left_fc = geom.beta * r[0] ** a_geom * D_bc / (r_cent[0] - r[0])

    flux_coeff = np.concatenate(([left_fc], flux_coeff, [flux_coeff[-1]]))

    chi_v = (f_Kelvin * beta_Rosen * T_k**(beta_Rosen - 1.0) * rho**(1.0 - mu)) / (4.0 * a_Kelvin * T_k**3)
    lagrangian_coeff = 1.0 / (V * chi_v)

    a = -lagrangian_coeff * flux_coeff[:-1]
    b = lagrangian_coeff * (flux_coeff[:-1] + flux_coeff[1:]) + 1.0 / dt
    c_coeff = -lagrangian_coeff * flux_coeff[1:]
    d = E_k / dt + (u_old - _matter_energy_density(T_k, rho)) / (chi_v * dt)

    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c_coeff)) or np.any(~np.isfinite(d)):
        raise ValueError("Non-finite value in conduction tridiagonal coefficients.")
    if np.any(b <= 0):
        raise ValueError("Non-positive diagonal encountered in conduction tridiagonal system.")

    return a, b, c_coeff, d


def conduction_radiation_step(
    state_star: RadHydroState,
    dt: float,
    rad_hydro_case: RadHydroCase,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nonlinear conduction update matching Krief (2021) Eq. (1).

    Returns (new_E, new_T) with E = a T^4 the flux potential.
    """
    global alpha, beta_Rosen, mu, f_Kelvin, chi, lambda_, g_Kelvin
    alpha, beta_Rosen, mu, f_Kelvin, chi, lambda_, g_Kelvin = rad_hydro_case._get_params()

    rho = state_star.rho
    x_nodes = state_star.x
    geom = rad_hydro_case.geom
    E_old = state_star.E_rad if state_star.E_rad is not None else a_Kelvin * state_star.T_material**4

    T_old = (E_old / a_Kelvin) ** 0.25
    u_old = _matter_energy_density(T_old, rho)

    # With no drive amplitude the inner boundary becomes the symmetry (zero
    # flux) condition of Eq. (21). That is the correct closure for a pure point
    # source, and the only usable one for omega >= omega_c, where the origin
    # temperature diverges (paper Table I).
    if rad_hydro_case.T0_Kelvin is None:
        T_left = None
        E_left = None
    else:
        t_drive = max(state_star.t, dt)
        T_left = float(rad_hydro_case.T0_Kelvin) * (t_drive / (10**-9)) ** rad_hydro_case.tau
        E_left = a_Kelvin * T_left**4

    # Picard iteration on the nonlinear u(E) relation. Coefficients D and
    # chi_v are re-evaluated at each iterate, so on convergence this is the
    # fully implicit, energy-conservative update rather than a single
    # linearisation frozen at the old state.
    E_k = E_old
    T_k = T_old
    for _ in range(PICARD_MAX_ITERS):
        sigma = calculate_sigma_from_temperature_and_density(T_k, rho)
        D = calculate_D_from_sigma(sigma)

        a, b, c_coeff, d = calculate_conduction_abcd(
            D, x_nodes, rho, u_old, E_k, T_k, dt, geom, T_left=T_left
        )
        if E_left is not None:
            d[0] -= a[0] * E_left

        E_new = solve_tridiagonal(a, b, c_coeff, d)
        E_new = np.maximum(E_new, 1e-300)  # keep T real for the next iterate
        T_new = (E_new / a_Kelvin) ** 0.25

        rel = np.max(np.abs(E_new - E_k) / (np.abs(E_k) + np.max(np.abs(E_k)) * 1e-12 + 1e-300))
        E_k, T_k = E_new, T_new
        if rel < PICARD_TOL:
            break

    return E_k, T_k


def calculate_flux(
    D: np.ndarray,
    m_cells: np.ndarray,
    rho: np.ndarray,
    E_rad: np.ndarray | None,
    E_bath: float,
):
    # calculation of flux at i=1,...,N-1
    if HARMONIC_MEAN:
        D_face = harmonic_mean(D[:-1], D[1:])
    else:
        D_face = arithmetic_mean(D[:-1], D[1:])
    
    # Exact physical spatial cell widths and center-to-center face distances:
    dx_cells = m_cells / rho
    dx_face = arithmetic_mean(dx_cells[:-1], dx_cells[1:])
    flux_coeff = D_face / dx_face # flux at i=1,...,N-1

    # Old implementation (assumed uniform density/mass profile; overestimates flux when omega != 0):
    # rho_face = arithmetic_mean(rho[:-1], rho[1:])
    # m_face = arithmetic_mean(m_cells[:-1], m_cells[1:])
    # flux_coeff = (D_face * rho_face) / m_face # flux at i=1,...,N-1

    flux = -flux_coeff * (E_rad[1:] - E_rad[:-1]) if E_rad is not None else np.zeros_like(rho) # flux at i=1,...,N-1

    # calculation of flux at i=0 (between ghost cell to the first cell)
    T_bath = E_bath ** (1/4.)
    sigma_bath = calculate_sigma_from_temperature_and_density(T_bath, rho[0])
    D_bath = calculate_D_from_sigma(sigma_bath)
    
    dx_bath = 0.5 * dx_cells[0]
    boundary_flux_coeff = D_bath / dx_bath
    # Old boundary flux implementation:
    # boundary_flux_coeff = (D_bath * rho[0]) / m_cells[0]

    boundary_flux = -boundary_flux_coeff * (E_rad[0] - E_bath)
    extended_flux = np.concatenate(([boundary_flux], flux))
    return extended_flux

def calculate_abcd(
    sigma: np.ndarray,
    D: np.ndarray,
    A: np.ndarray,
    m_cells: np.ndarray,
    rho: np.ndarray,
    E_rad: np.ndarray | None,
    T_material: np.ndarray,
    dt: float,
    bc_type: str,
    T_left: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Face-weighted coefficient builder (only supported scheme).

    Produces full-length a,b,c,d arrays (length N = len(rho)). Applies
    Marshak vacuum leakage on the right boundary and either a Marshak left
    modification or a Dirichlet subtraction depending on `bc_type`.
    """
    # Face-weighted implementation only
    if HARMONIC_MEAN:
        D_face = harmonic_mean(D[:-1], D[1:])
    else:
        D_face = arithmetic_mean(D[:-1], D[1:])

    # Exact physical spatial cell widths and center-to-center face distances:
    dx_cells = m_cells / rho
    dx_face = arithmetic_mean(dx_cells[:-1], dx_cells[1:])

    flux_coeff = D_face / dx_face # defined at i=1,...,N-1
    flux_coeff = np.concatenate(([flux_coeff[0]], flux_coeff, [flux_coeff[-1]]))
                                              # defined at i=0,1...N-1,N
    lagrangian_coeff = 1.0 / dx_cells # defined at j=1,...,N

    # Old implementation (assumed uniform density/mass profile; overestimates flux when omega != 0):
    # rho_face = arithmetic_mean(rho[:-1], rho[1:])
    # m_face = arithmetic_mean(m_cells[:-1], m_cells[1:])
    # flux_coeff = (D_face * rho_face) / m_face # defined at i=1,...,N-1
    # flux_coeff = np.concatenate(([flux_coeff[0]], flux_coeff, [flux_coeff[-1]]))
    # lagrangian_coeff = rho / m_cells # defined at j=1,...,N

    B = chi * c * sigma / (1 + A)
    a = -lagrangian_coeff * flux_coeff[:-1]
    b = lagrangian_coeff * (flux_coeff[:-1] + flux_coeff[1:]) + 1 / dt + B
    c_coeff = -lagrangian_coeff * flux_coeff[1:]

    UR_star = a_Kelvin * T_material**4
    d = B * UR_star + (1/dt) * E_rad


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
            # The Marshak boundary acts on the leftmost cell. Using index 0 keeps
            # the boundary coefficient consistent with the same control volume that
            # receives the imposed bath energy.
            rho_star_left = float(rho[0])
            dm_left = float(m_cells[0])
            if not np.isfinite(dm_left) or dm_left <= 0.0:
                cooling_left = 0.0
            else:
                cooling_left = c * rho_star_left / (2.0 * dm_left)
            b[0] = b[0] + a[0] + cooling_left
            a[0] = 0.0
            d[0] += cooling_left * E_bath
    elif bc_type == "Dirichlet":
        E_left = a_Kelvin * (T_left ** 4)
        d[0] -= a[0] * E_left
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

    # e_old is the radiation-convention energy density (E = a*T^4), matching
    # how it was created in initialize_problem and converted back at the end
    # of this function -- NOT the matter EOS (e = f*T^beta*rho^-mu).
    T_material_star = (e_old / a_Kelvin) ** 0.25
    sigma = calculate_sigma_from_temperature_and_density(T_material_star, rho)
    D = calculate_D_from_sigma(sigma)

    a, b, c_coeff, d = calculate_black_abcd(D, m_cells, rho, e_old, dt)

    t_drive = max(state_star.t, dt)
    T0_left = rad_hydro_case.T0_Kelvin if rad_hydro_case.T0_Kelvin is not None else 0.0
    T_left = T0_left * (t_drive / (10**-9)) ** rad_hydro_case.tau
    e_left = a_Kelvin * T_left**4

    bc_type = getattr(rad_hydro_case, "bc_type", "Dirichlet")
    if bc_type == "Marshak":
        # Free-streaming/effusive boundary flux, independent of the local
        # (possibly near-zero, cold-material) diffusivity -- matches
        # calculate_abcd's Marshak treatment for the non-black path.
        rho_left = float(rho[0])
        dm_left = float(m_cells[0])
        cooling_left = c * rho_left / (2.0 * dm_left) if (np.isfinite(dm_left) and dm_left > 0.0) else 0.0
        b[0] = b[0] + a[0] + cooling_left
        a[0] = 0.0
        d[0] += cooling_left * e_left
    else:
        # Dirichlet ghost-cell subtraction (matches calculate_abcd's approach;
        # the solved e[0] is influenced by, not force-set to, e_left).
        d[0] -= a[0] * e_left
    new_e = solve_tridiagonal(a, b, c_coeff, d)

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
        from menahem_new.subsonic_heat_wave_og import SubsonicHeatWave
        
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        new_F: Updated radiation flux in erg/cm^2/s
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
    # New naming: None | "gray" | "black" | "conduction"
    valid_modes = (None, "gray", "black", "conduction")
    if mode not in valid_modes:
        raise ValueError(f"Invalid force_black mode '{mode}'. Expected one of {valid_modes}.")

    if mode == "black":
        new_e, new_T = black_radiation_step(state_star, dt, rad_hydro_case)
        return new_T, new_e, new_T, new_e, np.zeros_like(new_e), 0.0

    if mode == "conduction":
        new_E, new_T = conduction_radiation_step(state_star, dt, rad_hydro_case)
        new_e_material = f_Kelvin * new_T**beta_Rosen * rho**(-mu)
        return new_T, new_e_material, new_T, new_E, np.zeros_like(new_E), 0.0

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

    new_F = calculate_flux(D, m_cells, rho, new_E_rad, E_bath=a_Kelvin * (T_left ** 4) if T_left is not None else 0.0)
    # important! The flux here is in i=0,...,N-1 convention, where the first entry is the flux at the left boundary, and we don't include the flux at the right boundary (we don't have a ghost cell on the right in this case).
    
    delta_m1 = arithmetic_mean(m_cells[0], m_cells[1])
    LHS =  delta_m1/rho[0]*((new_E_rad[0] - E_rad[0]) / dt + rho[0]/delta_m1 * new_F[1] - \
        chi*c*sigma[0] / (1 + A[0]) * (UR_star[0] - new_E_rad[0]))
    
    return new_T_material, new_e_material, new_T_rad, new_E_rad, new_F, LHS
