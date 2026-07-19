import numpy as np

def artificial_viscosity(rho_new, u_half, sigma):
    """
    rho_new: cells
    u_half: nodes
    returns q_new: cells
    """
    du = u_half[1:] - u_half[:-1]          # u_i+1 - u_i for i=0..N-1
    compressing = du < 0                   # ui > ui+1
    q = np.zeros_like(rho_new)
    q[compressing] = sigma * rho_new[compressing] * (du[compressing]**2)
    return q
