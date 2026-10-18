"""Power-law (Hammer-Rosen) material for gray IMC.

Energy and opacity, in whatever units the caller chooses for T, rho and energy:

    u_m(T, rho) = f T^beta rho^(1-mu)  [- a T^4 if subtract_radiation]
    k_t(T, rho) = rho^(1+lambda) / (g T^alpha)        total opacity [1/length]
    k_a = chi k_t,  k_s = (1 - chi) k_t                absorption / physical scattering

The default u_m is the material energy density of the SiO2 script and of
rad_hydro_sim.  With subtract_radiation=True it is the manuscript's
u_m = u - a T^4, which treats f T^beta rho^(1-mu) as matter plus equilibrium
radiation (md/plans/full_plan_imc_comparison.md, Context item 3).

The Fleck factor (alpha = 1) uses the absorption opacity:

    f = 1 / (1 + beta_fc c k_a dt),   beta_fc = du_r/du_m = 4 a T^3 / (du_m/dT).

fleck_mode="explicit" sets f = 1: emission frozen at T^n, no effective scattering.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawMaterial:
    f: float                     # EOS prefactor
    beta: float                  # EOS temperature exponent
    mu: float                    # EOS density exponent, u_m ~ rho^(1-mu)
    g: float                     # opacity prefactor, 1/k ~ g T^alpha rho^-(1+lambda)
    alpha: float                 # opacity temperature exponent
    lam: float                   # opacity density exponent, k ~ rho^(1+lambda)
    a: float                     # radiation constant, u_r = a T^4
    c: float                     # speed of light
    chi: float = 1.0             # absorption fraction of the total opacity
    subtract_radiation: bool = False
    T_floor: float = 0.0         # keeps T^-alpha finite

    def __post_init__(self):
        if not 0.0 < self.chi <= 1.0:
            raise ValueError(
                f"chi = {self.chi}: a transport equivalent needs 0 < chi <= 1 "
                "(k_s = (1 - chi) k_t must be non-negative)."
            )

    @classmethod
    def from_case(cls, case, subtract_radiation: bool = False, T_floor: float = 1.0):
        """CGS material (K, g/cm^3, erg/cm^3, cm/s) from a rad_hydro_sim RadHydroCase.

        Uses the same c and a as rad_hydro_sim.simulation.radiation_step, so the
        IMC and the diffusion solver share their constants.
        """
        from rad_hydro_sim.simulation.radiation_step import a_Kelvin, c

        return cls(
            f=case.f_Kelvin, beta=case.beta_Rosen, mu=case.mu,
            g=case.g_Kelvin, alpha=case.alpha, lam=case.lambda_,
            a=a_Kelvin, c=c, chi=case.chi,
            subtract_radiation=subtract_radiation, T_floor=T_floor,
        )

    def _T(self, T):
        return np.maximum(np.asarray(T, dtype=float), self.T_floor)

    def u_r(self, T):
        """Equilibrium radiation energy density a T^4."""
        return self.a * self._T(T) ** 4

    def u_m(self, T, rho):
        T = self._T(T)
        u = self.f * T**self.beta * rho ** (1.0 - self.mu)
        if self.subtract_radiation:
            u = u - self.a * T**4
        return u

    def du_m_dT(self, T, rho):
        T = self._T(T)
        d = self.f * self.beta * T ** (self.beta - 1.0) * rho ** (1.0 - self.mu)
        if self.subtract_radiation:
            d = d - 4.0 * self.a * T**3
        return d

    def temperature(self, u, rho, max_iter: int = 60, rtol: float = 1e-13):
        """Invert u_m(T, rho) = u.  Closed form, or Newton when a T^4 is subtracted."""
        u = np.asarray(u, dtype=float)
        rho = np.broadcast_to(np.asarray(rho, dtype=float), u.shape)
        scale = self.f * rho ** (1.0 - self.mu)
        T = (np.maximum(u, 0.0) / scale) ** (1.0 / self.beta)
        if self.subtract_radiation:
            # u_m(T) is increasing while du_m/dT > 0; the closed form is a lower bound.
            for _ in range(max_iter):
                res = self.u_m(T, rho) - u
                slope = self.du_m_dT(T, rho)
                if np.any(slope <= 0.0):
                    raise ValueError("du_m/dT <= 0: u_m = u - aT^4 is not invertible here.")
                step = res / slope
                T = np.maximum(T - step, 0.5 * T)
                if np.all(np.abs(step) <= rtol * np.maximum(T, 1e-300)):
                    break
            else:
                raise RuntimeError("Newton inversion of u_m(T) did not converge.")
        return np.maximum(T, self.T_floor)

    def opacity_total(self, T, rho):
        return rho ** (1.0 + self.lam) / (self.g * self._T(T) ** self.alpha)

    def opacities(self, T, rho):
        """(k_a, k_s): absorption and physical scattering opacities."""
        k_t = self.opacity_total(T, rho)
        return self.chi * k_t, (1.0 - self.chi) * k_t

    def beta_fc(self, T, rho):
        """Fleck & Cummings beta = du_r/du_m."""
        return 4.0 * self.a * self._T(T) ** 3 / self.du_m_dT(T, rho)

    def fleck(self, T, rho, dt, fleck_mode: str = "implicit"):
        if fleck_mode == "explicit":
            return np.ones_like(np.asarray(T, dtype=float))
        if fleck_mode != "implicit":
            raise ValueError("fleck_mode must be 'implicit' or 'explicit'.")
        k_a, _ = self.opacities(T, rho)
        return 1.0 / (1.0 + self.beta_fc(T, rho) * self.c * k_a * dt)
