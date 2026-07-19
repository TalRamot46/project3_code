import numpy as np

def pressure_ideal_gas(rho: np.ndarray, e: np.ndarray, gamma: float) -> np.ndarray:
    return (gamma - 1.0) * rho * e

def internal_energy_from_prho(p: np.ndarray, rho: np.ndarray, gamma: float) -> np.ndarray:
    return p / ((gamma - 1.0) * rho)

def sound_speed(rho: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    # c^2 = gamma p / rho
    return np.sqrt(gamma * p / rho)

def apply_pressure_bc(p, bc_left, bc_right, t):
    p = p.copy()

    if isinstance(bc_left, dict):
        if bc_left["type"] == "pressure":
            p[0] = bc_left["p"](t)

    elif bc_left == "outflow":
        p[0] = p[1]

    elif bc_left == "none":
        pass

    else:
        raise ValueError(f"Unknown bc_left: {bc_left}")

    if bc_right == "outflow":
        p[-1] = p[-2]

    elif bc_right == "none":
        pass

    else:
        raise ValueError(f"Unknown bc_right: {bc_right}")

    return p
