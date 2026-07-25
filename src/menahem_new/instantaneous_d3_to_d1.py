import numpy as np
import scipy.special
from menahem_new.supersonic_instantaneous_analytic import SupersonicInstantaneousAnalytic

def get_front_exponents_and_params(alpha, beta, lambdap, mu, omega):
    """
    Computes exponents and parameters for both d=1 and d=3 dimensions.
    """
    n = (4.0 + alpha - beta) / beta
    k = omega * (1.0 + lambdap)
    m = omega * (1.0 - mu)
    
    p_d1 = 2.0 - k - m + (1.0 - m) * n
    p_d3 = 2.0 - k - m + (3.0 - m) * n
    
    diff_2km = 2.0 - k - m
    
    if diff_2km > 0:
        l_d1 = (1.0 - m) / diff_2km
        l_d3 = (3.0 - m) / diff_2km
    else:
        l_d1 = -(1.0 / n) - (1.0 - m) / diff_2km
        l_d3 = -(1.0 / n) - (3.0 - m) / diff_2km
        
    return n, k, m, p_d1, p_d3, l_d1, l_d3


def calculate_xi0_d1_from_d3(xi_0_d3, alpha, beta, lambdap, mu, omega):
    """
    Calculates the 1D planar conduction front coordinate (xi_0_d1) 
    directly from a given 3D spherical coordinate (xi_0_d3) and 
    the problem's physical material parameters.
    """
    n, k, m, p_d1, p_d3, l_d1, l_d3 = get_front_exponents_and_params(
        alpha, beta, lambdap, mu, omega
    )
    
    second_arg = (1.0 / n) + 1.0
    beta_d3 = scipy.special.beta(l_d3, second_arg)
    beta_d1 = scipy.special.beta(l_d1, second_arg)
    
    inner_bracket = (p_d1 / (4.0 * np.pi * p_d3)) * (beta_d3 / beta_d1) * (xi_0_d3 ** p_d3)
    xi_0_d1 = inner_bracket ** (1.0 / p_d1)
    
    return xi_0_d1


def calculate_xi0_d3_from_d1(xi_0_d1, alpha, beta, lambdap, mu, omega):
    """
    Calculates the 3D spherical conduction front coordinate (xi_0_d3)
    directly from a given 1D planar coordinate (xi_0_d1) and
    the problem's physical material parameters.
    """
    n, k, m, p_d1, p_d3, l_d1, l_d3 = get_front_exponents_and_params(
        alpha, beta, lambdap, mu, omega
    )
    
    second_arg = (1.0 / n) + 1.0
    beta_d3 = scipy.special.beta(l_d3, second_arg)
    beta_d1 = scipy.special.beta(l_d1, second_arg)
    
    inner_bracket = (xi_0_d1 ** p_d1) * (4.0 * np.pi * p_d3 / p_d1) * (beta_d1 / beta_d3)
    xi_0_d3 = inner_bracket ** (1.0 / p_d3)
    
    return xi_0_d3


def calculate_r0_d1_from_d3(r_0_d3, xi_0_d3, alpha, beta, lambdap, mu, omega, Q, A, t):
    """
    Calculates the physical planar front tracking radius r_0(1, t) 
    at time t from a given spherical physical front radius r_0(3, t).
    """
    xi_0_d1 = calculate_xi0_d1_from_d3(xi_0_d3, alpha, beta, lambdap, mu, omega)
    n, k, m, p_d1, p_d3, l_d1, l_d3 = get_front_exponents_and_params(
        alpha, beta, lambdap, mu, omega
    )
    
    energy_scale = (Q**n) * A / t
    r_0_d1 = r_0_d3 * (xi_0_d1 / xi_0_d3) * (energy_scale ** ((1.0 / p_d1) - (1.0 / p_d3)))
    
    return r_0_d1


def run_mode_1(
    *,
    g: float,
    alpha: float,
    lambdap: float,
    f: float,
    beta: float,
    mu: float,
    rho0: float,
    omega: float,
    T0_HeV: float = 1.0,
    T0_Kelvin: float | None = None,
    Q: float | None = None,
):
    """
    Mode 1: Accepts the same parameters as SupersonicInstantaneousAnalytic.
    Solves for xi_0(d=1), and computes xi_0(d=3) from the cross-dimensional transformation.
    
    Returns:
    --------
    Tuple[float, float] : (xi_0_d1, xi_0_d3)
    """
    solver = SupersonicInstantaneousAnalytic(
        g=g,
        alpha=alpha,
        lambdap=lambdap,
        f=f,
        beta=beta,
        mu=mu,
        rho0=rho0,
        omega=omega,
        T0_HeV=T0_HeV,
        T0_Kelvin=T0_Kelvin,
        Q=Q,
    )
    xi_0_d1 = solver.xi_0
    xi_0_d3 = calculate_xi0_d3_from_d1(
        xi_0_d1, alpha, beta, lambdap, mu, omega
    )
    return xi_0_d1, xi_0_d3


def run_mode_2(
    r_d3: float,
    *,
    g: float,
    alpha: float,
    lambdap: float,
    f: float,
    beta: float,
    mu: float,
    rho0: float,
    omega: float,
    T0_HeV: float = 1.0,
    T0_Kelvin: float | None = None,
    Q: float | None = None,
):
    """
    Mode 2: Accepts the same parameters as SupersonicInstantaneousAnalytic and a front radius r(d=3).
    Computes the time (t) at which the front reaches r(d=3) in spherical geometry,
    and the corresponding front radius r(d=1) in planar geometry at that time.
    
    Returns:
    --------
    Tuple[float, float] : (t_sec, r_d1)
    """
    solver = SupersonicInstantaneousAnalytic(
        g=g,
        alpha=alpha,
        lambdap=lambdap,
        f=f,
        beta=beta,
        mu=mu,
        rho0=rho0,
        omega=omega,
        T0_HeV=T0_HeV,
        T0_Kelvin=T0_Kelvin,
        Q=Q,
        d=1,
    )
    xi_0_d1 = solver.xi_0
    
    n, k, m, p_d1, p_d3, l_d1, l_d3 = get_front_exponents_and_params(
        alpha, beta, lambdap, mu, omega
    )
    
    xi_0_d3 = calculate_xi0_d3_from_d1(
        xi_0_d1, alpha, beta, lambdap, mu, omega
    )
    
    # Calculate time t_sec from: r_d3 = xi_0_d3 * (Q^n * A * t)^(1 / p_d3)
    t_sec = ((r_d3 / xi_0_d3) ** p_d3) / ((solver.Q ** solver.n) * solver.A_Kelvin)
    
    # Calculate corresponding r(d=1) at time t_sec
    r_d1 = solver.heat_front_radius(t_sec)
    
    return t_sec, r_d1


# --- Example Usage / Verification ---
if __name__ == "__main__":
    hev_kelvin = 1.160451812e6   # K / eV

    # Example using standard solver parameters
    params = {
        'g': 1.0 / (hev_kelvin**2.0),
        'alpha': 2.0,
        'lambdap': 1.0,
        'f': 1.0 / (hev_kelvin**1.6),
        'beta': 1.6,
        'mu': 0.0,
        'rho0': 1.0,
        'omega': 0.2,
        'Q': 1.0,
    }
    
    print("=== Mode 1 ===")
    print("Getting xi_0(d=1) and xi_0(d=3) for the preset case")
    xi_0_d1, xi_0_d3 = run_mode_1(**params)
    print(f"Planar Conduction Front xi_0(d=1)  = {xi_0_d1:.6f}")
    print(f"Spherical Conduction Front xi_0(d=3) = {xi_0_d3:.6f}")
    
    print("\n=== Mode 2 ===")
    assumed_r_d3 = 0.1333  # cm
    print("Getting t_sec and r_d1 for the preset case with front at r(3) = 0.1333 cm")
    t_sec, r_d1 = run_mode_2(assumed_r_d3, **params)
    print(f"Given Spherical radius r(d=3) = {assumed_r_d3} cm")
    print(f"Calculated Time of Front      = {t_sec:.6e} seconds")
    print(f"Calculated Planar radius r(d=1)= {r_d1:.6f} cm")