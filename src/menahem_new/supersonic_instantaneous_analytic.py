"""
Supersonic Instantaneous Point-Source Radiation Diffusion Analytic Solver.

Based on the paper:
Menahem Krief, "Analytic solutions of the nonlinear radiation diffusion equation
with an instantaneous point source in non-homogeneous media",
Physics of Fluids 33, 057105 (2021); doi: 10.1063/5.0050422.

This module implements the self-similar analytical solver for Eulerian 1D (d=1)
nonlinear heat conduction (n > 0) in non-homogeneous media:
    rho(r) = rho0 * r^(-omega)

EOS specific energy:
    u(T, rho) = f * T^beta * rho^(1-mu)
Rosseland mean opacity:
    1 / kappa_R(T, rho) = g * T^alpha * rho^(-lambda)

Self-similar transformation exponents (d=1):
    n = (4 + alpha - beta) / beta
    k = omega * (1 + lambda)
    m = omega * (1 - mu)
    p = 2 - k - m + (1 - m) * n

The boundary drive at origin is T(0, t) = T0 * t^tau, where:
    tau = -(1 - m) / (beta * p)

All calculations internal to the class use CGS units (cm, g, s, K, erg).
Convenience arguments accept T0 in HeV/Kelvin and t in ns/s.
"""

from __future__ import annotations

import math
import numpy as np
import scipy.special


class Units:
    sigma_sb = 5.670374419e-5   # erg / (cm^2 s K^4)
    clight = 2.99792458e10      # cm / s
    arad = 4.0 * sigma_sb / clight  # erg / (cm^3 K^4)
    ev_kelvin = 1.160451812e4   # K / eV
    hev_kelvin = 100.0 * ev_kelvin # K / hundred-eV (HeV)
    nsec = 1e-9                 # s / ns


class SupersonicInstantaneousAnalytic:
    """Analytical solver for instantaneous point source supersonic radiation wave in 1D planar geometry.

    Parameters
    ----------
    g : float
        Opacity coefficient (CGS units).
    alpha : float
        Opacity temperature power.
    lambdap : float
        Opacity density power (lambda).
    f : float
        EOS specific internal energy coefficient (CGS units).
    beta : float
        EOS temperature power.
    mu : float
        EOS density power.
    rho0 : float
        Density scale factor at r=1 cm (g/cm^3).
    omega : float
        Spatial density power-law exponent (rho(r) = rho0 * r^(-omega)).
    T0_HeV : float, optional
        Origin drive temperature coefficient at t = 1 ns in HeV (1 HeV = 100 eV).
        Default is 1.0 HeV.
    T0_Kelvin : float, optional
        Origin drive temperature coefficient at t = 1 ns in Kelvin.
        If provided, overrides T0_HeV.
    Q : float, optional
        Total initial energy Q (erg/cm^2 for d=1). If specified, Q is used directly
        and T0 is computed from Q.
    """

    def __init__(
        self,
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
        self.g = float(g)
        self.alpha = float(alpha)
        self.lambdap = float(lambdap)
        self.f = float(f)
        self.beta = float(beta)
        self.mu = float(mu)
        self.rho0 = float(rho0)
        self.omega = float(omega)

        # Dimension 1D planar
        self.d = 1
        self.Ad = self.areal_coeff(self.d)  # planar areal coefficient

        # Calculate similarity exponents
        self.n = (4.0 + self.alpha - self.beta) / self.beta
        if self.n <= 0:
            raise ValueError(f"This solver supports non-linear conduction (n > 0). Got n={self.n}")

        self.k = self.omega * (1.0 + self.lambdap)
        self.m = self.omega * (1.0 - self.mu)

        # Exponent p (Eq. 24 for d=1)
        self.p = 2.0 - self.k - self.m + (1.0 - self.m) * self.n

        if self.p <= 0:
            raise ValueError(f"Propagation condition p > 0 violated: p = {self.p}")
        if self.omega >= 1.0:
            raise ValueError(f"Mass divergence condition omega < d (1.0) violated: omega = {self.omega}")

        # Coefficient A (Eq. 15 for d=1)
        # A = (16 * sigma_sb * g) / (3 * beta * f^((4+alpha)/beta) * rho0^(1 + lambda + (1-mu)*(4+alpha)/beta))
        exp_f = (4.0 + self.alpha) / self.beta
        exp_rho = 1.0 + self.lambdap + (1.0 - self.mu) * exp_f
        self.A = (16.0 * Units.sigma_sb * self.g) / (3.0 * self.beta * (self.f ** exp_f) * (self.rho0 ** exp_rho))

        # Self-similar front coordinate xi_0 calculation (Eq. 44 & 45)
        diff_2km = 2.0 - self.k - self.m
        if math.isclose(diff_2km, 0.0):
            raise ValueError("2 - k - m = 0 is the marginal case (omega = omega_c), handled separately.")

        if diff_2km > 0:
            self.l_param = (1.0 - self.m) / diff_2km
        else:
            self.l_param = (1.0 / self.n) - (1.0 - self.m) / diff_2km

        beta_val = scipy.special.beta(self.l_param, (1.0 / self.n) + 1.0)
        numerator = self.p * (abs(diff_2km) ** (self.n + 1.0))
        denominator = self.n * (self.Ad ** self.n) * (beta_val ** self.n)

        self.xi_0 = (numerator / denominator) ** (1.0 / self.p)
    
        # f(xi -> 0) at origin (Eq. 42 evaluation at xi=0)
        self.f_0 = ( (self.n * (self.xi_0 ** diff_2km)) / (self.p * diff_2km) ) ** (1.0 / self.n)

        # Drive time exponent tau: T(0, t) = T0 * t^tau (where t is in seconds)
        self.tau = -(1.0 - self.m) / (self.beta * self.p)

        # Process T0 / Q inputs:
        # Determine drive amplitude Tb in CGS (Kelvin / s^tau)
        if Q is not None:
            self.Q = float(Q)
            # Compute T0 in Kelvin/s^tau from Q via Eq. (41)
            # w(0,t) = (Q^(2-k-m) / (A*t)^(1-m))^(1/p) * f_0
            # T(0,t) = (w(0,t) / (f * rho0^(1-mu)))^(1/beta)
            factor_w0 = (self.Q ** diff_2km / (self.A ** (1.0 - self.m))) ** (1.0 / self.p) * self.f_0
            self.Tb_cgs = (factor_w0 / (self.f * (self.rho0 ** (1.0 - self.mu)))) ** (1.0 / self.beta)
        else:
            if T0_Kelvin is not None:
                T0_k = float(T0_Kelvin)
            else:
                T0_k = float(T0_HeV) * Units.hev_kelvin

            # Convert drive amplitude defined at t_ns = 1 ns to t_sec = 1 s:
            # T(t_ns) = T0_k * t_ns^tau = T0_k * (t_sec / 1e-9)^tau = (T0_k * 1e9^tau) * t_sec^tau
            self.Tb_cgs = T0_k * (1.0e9 ** self.tau)

            # Invert Eq. (41) to determine Q (erg/cm^2) from Tb_cgs:
            # Tb_cgs^beta = (f_0 / (f * rho0^(1-mu))) * (Q^(2-k-m) / A^(1-m))^(1/p)
            w0_val = (self.Tb_cgs ** self.beta) * self.f * (self.rho0 ** (1.0 - self.mu))
            ratio_Q_A = (w0_val / self.f_0) ** self.p
            self.Q = (ratio_Q_A * (self.A ** (1.0 - self.m))) ** (1.0 / diff_2km)

    def areal_coeff(self, d: int) -> float:
        """Return areal coefficient for dimension d."""
        if d == 1: return 1
        elif d==2: return 2*np.pi
        elif d==3: return 4*np.pi
        else: raise ValueError(f"Unsupported dimension d={d}")

    def heat_front_radius(self, t_sec: float | np.ndarray) -> float | np.ndarray:
        """Calculate heat front position r_h(t) in cm at time t in seconds."""
        t = np.asarray(t_sec, dtype=float)
        r_h = self.xi_0 * ((self.Q ** self.n) * self.A * t) ** (1.0 / self.p)
        return r_h if r_h.ndim > 0 else float(r_h)

    def similarity_variable(self, r_cm: float | np.ndarray, t_sec: float | np.ndarray) -> float | np.ndarray:
        """Calculate dimensionless similarity variable xi = r / (Q^n * A * t)^(1/p)."""
        r = np.asarray(r_cm, dtype=float)
        t = np.asarray(t_sec, dtype=float)
        scale = ((self.Q ** self.n) * self.A * t) ** (1.0 / self.p)
        return r / scale

    def self_similar_profile(self, xi: float | np.ndarray) -> float | np.ndarray:
        """Evaluate self-similar function f(xi) (Eq. 42)."""
        xi_arr = np.asarray(xi, dtype=float)
        diff_2km = 2.0 - self.k - self.m
        f_val = np.zeros_like(xi_arr)

        mask = xi_arr < self.xi_0
        if np.any(mask):
            inside = (self.n * (self.xi_0 ** diff_2km - xi_arr[mask] ** diff_2km)) / (self.p * diff_2km)
            # Clip numerical noise near front
            inside = np.maximum(inside, 0.0)
            f_val[mask] = inside ** (1.0 / self.n)

        return f_val if f_val.ndim > 0 else float(f_val)

    def auxiliary_variable_w(self, r_cm: float | np.ndarray, t_sec: float | np.ndarray) -> float | np.ndarray:
        """Calculate auxiliary variable w(r, t) = r^m * u(r, t) (Eq. 23)."""
        r = np.asarray(r_cm, dtype=float)
        t = np.asarray(t_sec, dtype=float)
        xi = self.similarity_variable(r, t)
        f_xi = self.self_similar_profile(xi)

        diff_2km = 2.0 - self.k - self.m
        amplitude = ( (self.Q ** diff_2km) / ((self.A * t) ** (1.0 - self.m)) ) ** (1.0 / self.p)
        w_val = amplitude * f_xi
        return w_val if w_val.ndim > 0 else float(w_val)

    def temperature_profile(self, r_cm: float | np.ndarray, t_sec: float | np.ndarray) -> float | np.ndarray:
        """Calculate material temperature T(r, t) in Kelvin.

        Parameters
        ----------
        r_cm : float or ndarray
            Position in cm.
        t_sec : float or ndarray
            Time in seconds.
        """
        w_val = self.auxiliary_variable_w(r_cm, t_sec)
        # T(r, t) = [ w(r, t) / (f * rho0^(1-mu)) ]^(1/beta)
        denom = self.f * (self.rho0 ** (1.0 - self.mu))
        T_val = (w_val / denom) ** (1.0 / self.beta)
        return T_val

    def energy_density(self, r_cm: float | np.ndarray, t_sec: float | np.ndarray) -> float | np.ndarray:
        """Calculate specific internal energy per unit volume u(r, t) in erg/cm^3.

        u(r, t) = f * T^beta * rho(r)^(1-mu) = w(r, t) * r^(-m)
        """
        r = np.asarray(r_cm, dtype=float)
        w_val = self.auxiliary_variable_w(r, t_sec)
        # Avoid division by zero at r=0 if m > 0 by using T directly:
        T_val = self.temperature_profile(r, t_sec)
        rho_val = self.rho0 * np.maximum(r, 1e-30) ** (-self.omega)
        u_val = self.f * (T_val ** self.beta) * (rho_val ** (1.0 - self.mu))
        return u_val

    def flux(self, r_cm: float | np.ndarray, t_sec: float | np.ndarray) -> float | np.ndarray:
        """Calculate radiation flux F(r, t) in erg/(cm^2 s) (Eq. 33)."""
        r = np.asarray(r_cm, dtype=float)
        t = np.asarray(t_sec, dtype=float)
        w_val = self.auxiliary_variable_w(r, t)
        xi = self.similarity_variable(r, t)
        f_xi = self.self_similar_profile(xi)

        diff_2km = 2.0 - self.k - self.m
        F_val = np.zeros_like(r)
        mask = (xi < self.xi_0) & (f_xi > 0)
        if np.any(mask):
            r_m = np.maximum(r[mask], 1e-30)
            numerator = self.A * (r_m ** (self.k - 1.0)) * (w_val[mask] ** (self.n + 1.0)) * (xi[mask] ** diff_2km)
            denominator = self.p * f_xi[mask] ** self.n
            F_val[mask] = numerator / denominator

        return F_val if F_val.ndim > 0 else float(F_val)
