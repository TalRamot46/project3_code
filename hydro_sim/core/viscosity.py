import numpy as np

def artificial_viscosity(rho_new, u_half, sigma):
    """
    rho_new: cells
    u_half: nodes
    returns q_new: cells
    """
    du = u_half[1:] - u_half[:-1]          # cell-wise velocity jump
    compressing = du < 0                   # ui > ui+1
    q = np.zeros_like(rho_new)
    q[compressing] = sigma * rho_new[compressing] * (du[compressing]**2)
    return q
