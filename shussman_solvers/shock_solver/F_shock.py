import numpy as np

def F3(t: float, x: np.ndarray, tau: float, r: float) -> np.ndarray:
    V, P, u = x
    wm3 = 1.0 + 0.5 * tau
    wu3 = 0.5 * tau
    w3  = -wm3

    # x = [V, P, u]
    # We solve for A in the system:
    # A * [V', P', u']^T = [V, P, u]^T
    # where A is a 3x3 matrix derived from the ODEs.
    # We use Cramer's rule to solve for the derivatives.
    denom = (-V*(w3**2)*(t**2) + P + P*r) # the determinant of A 
                                          
    out = np.zeros(3, dtype=float)
    # apply Cramer's rule to solve for derivatives in terms of V, P, u
    out[0] = -(V*(P*tau - u*w3*wu3*t)) / (w3*t*denom)         # V'
    out[2] = -(V*(P*tau - u*w3*wu3*t)) / (denom)              # u'
    out[1] = -(P*(u*wu3 + r*u*wu3 - V*tau*w3*t)) / (denom)    # P'
    return out
