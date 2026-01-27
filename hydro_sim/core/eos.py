import numpy as np

def pressure_ideal_gas(rho: np.ndarray, e: np.ndarray, gamma: float) -> np.ndarray:
    return (gamma - 1.0) * rho * e

def internal_energy_from_prho(p: np.ndarray, rho: np.ndarray, gamma: float) -> np.ndarray:
    return p / ((gamma - 1.0) * rho)

def sound_speed(rho: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    # c^2 = gamma p / rho
    return np.sqrt(gamma * p / rho)
