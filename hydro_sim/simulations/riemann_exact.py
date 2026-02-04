import numpy as np
import math

def _pressure_function(p, rho, p0, gamma):
    """
    Toro-style f(p) and f'(p) pieces for rarefaction/shock.
    """
    if p <= p0:  # rarefaction
        a0 = math.sqrt(gamma * p0 / rho)
        pr = p / p0
        f = (2 * a0 / (gamma - 1.0)) * (pr**((gamma - 1.0)/(2*gamma)) - 1.0)
        fd = (1.0/(rho*a0)) * pr**(-(gamma + 1.0)/(2*gamma))
        return f, fd
    else:        # shock
        A = 2.0 / ((gamma + 1.0) * rho)
        B = (gamma - 1.0)/(gamma + 1.0) * p0
        f = (p - p0) * math.sqrt(A / (p + B))
        fd = math.sqrt(A / (p + B)) * (1.0 - 0.5*(p - p0)/(p + B))
        return f, fd

def solve_star_region(rhoL, uL, pL, rhoR, uR, pR, gamma, tol=1e-10, max_it=100):
    """
    Solve for p* and u* with Newton iteration (exact Riemann solver).
    """
    # initial guess (PVRS)
    aL = math.sqrt(gamma*pL/rhoL)
    aR = math.sqrt(gamma*pR/rhoR)
    p_pv = 0.5*(pL + pR) - 0.125*(uR - uL)*(rhoL+rhoR)*(aL+aR)
    p = max(1e-12, p_pv)

    for _ in range(max_it):
        fL, fLd = _pressure_function(p, rhoL, pL, gamma)
        fR, fRd = _pressure_function(p, rhoR, pR, gamma)
        g = fL + fR + (uR - uL)
        gp = fLd + fRd
        dp = -g / gp
        p_new = p + dp
        if p_new < 0:
            p_new = 1e-12
        if abs(dp) / (p_new + 1e-30) < tol:
            p = p_new
            break
        p = p_new

    fL, _ = _pressure_function(p, rhoL, pL, gamma)
    u = uL - fL
    return p, u

def sample_solution(x, t, left, right, gamma):
    """
    Return arrays rho,u,p,e at points x at time t for the exact solution.
    left/right = (rho,u,p)
    """
    rhoL,uL,pL = left
    rhoR,uR,pR = right

    if t == 0.0:
        rho = np.where(x < 0.0, rhoL, rhoR)
        u   = np.where(x < 0.0, uL, uR)
        p   = np.where(x < 0.0, pL, pR)
        e   = p / ((gamma-1.0)*rho)
        return rho,u,p,e

    p_star, u_star = solve_star_region(rhoL,uL,pL,rhoR,uR,pR,gamma)

    aL = math.sqrt(gamma*pL/rhoL)
    aR = math.sqrt(gamma*pR/rhoR)

    # Left wave
    if p_star <= pL:  # rarefaction
        a_star_L = aL * (p_star/pL)**((gamma-1.0)/(2*gamma))
        s_head_L = uL - aL
        s_tail_L = u_star - a_star_L
    else:             # shock
        sL = uL - aL * math.sqrt((gamma+1)/(2*gamma)*(p_star/pL) + (gamma-1)/(2*gamma))
        s_head_L = s_tail_L = sL

    # Right wave
    if p_star <= pR:  # rarefaction
        a_star_R = aR * (p_star/pR)**((gamma-1.0)/(2*gamma))
        s_head_R = uR + aR
        s_tail_R = u_star + a_star_R
    else:             # shock
        sR = uR + aR * math.sqrt((gamma+1)/(2*gamma)*(p_star/pR) + (gamma-1)/(2*gamma))
        s_head_R = s_tail_R = sR

    xi = x / t

    rho = np.zeros_like(x)
    u   = np.zeros_like(x)
    p   = np.zeros_like(x)

    # helper: star densities
    if p_star <= pL:
        rho_star_L = rhoL * (p_star/pL)**(1.0/gamma)
    else:
        rho_star_L = rhoL * ((p_star/pL + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1)*p_star/pL + 1.0))
    if p_star <= pR:
        rho_star_R = rhoR * (p_star/pR)**(1.0/gamma)
    else:
        rho_star_R = rhoR * ((p_star/pR + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1)*p_star/pR + 1.0))

    for i, s in enumerate(xi):
        if s < u_star:  # left side of contact
            if p_star <= pL:  # left rarefaction
                if s < s_head_L:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                elif s > s_tail_L:
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
                else:
                    # inside fan
                    ufan = (2/(gamma+1))*(aL + 0.5*(gamma-1)*uL + s)
                    afan = (2/(gamma+1))*(aL + 0.5*(gamma-1)*(uL - s))
                    rho[i] = rhoL * (afan/aL)**(2/(gamma-1))
                    p[i]   = pL   * (afan/aL)**(2*gamma/(gamma-1))
                    u[i]   = ufan
            else:  # left shock
                if s < s_head_L:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                else:
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
        else:  # right side of contact
            if p_star <= pR:  # right rarefaction
                if s > s_head_R:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                elif s < s_tail_R:
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
                else:
                    ufan = (2/(gamma+1))*(-aR + 0.5*(gamma-1)*uR + s)
                    afan = (2/(gamma+1))*(aR - 0.5*(gamma-1)*(uR - s))
                    rho[i] = rhoR * (afan/aR)**(2/(gamma-1))
                    p[i]   = pR   * (afan/aR)**(2*gamma/(gamma-1))
                    u[i]   = ufan
            else:  # right shock
                if s > s_head_R:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                else:
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star

    e = p / ((gamma - 1.0) * rho)
    return rho, u, p, e
