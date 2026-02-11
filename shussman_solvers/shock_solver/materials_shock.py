# materials.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

HEV_IN_KELVIN = 1_160_500.0  # 1 HeV in Kelvin (as used in your MATLAB)

@dataclass(frozen=True, slots=True)
class Material:
    # Opacity / EOS exponents
    alpha: float
    beta: float
    lambda_: float   # "lambda" is reserved in Python
    mu: float

    # Prefactors (same meaning as MATLAB)
    f: float
    g: float
    sigma: float

    # EOS constant + reference specific volume
    r: float
    V0: float

    # Optional: metadata/name to help debugging/plot labels
    name: str = "Material"

    def rho0(self) -> float:
        """Reference density [g/cc] implied by V0."""
        return 1.0 / self.V0


def au_supersonic_variant_1(*, apply_sigma_conversion: bool = False) -> Material:
    """
    First MATLAB block (alpha=1.5, beta=1.6, rho0=19.32).
    Note: MATLAB keeps sigma in Watt/cm^2/K^4 and comments out the conversion.
    If your solver expects sigma in the converted HeV units, set apply_sigma_conversion=True.
    """
    alpha = 1.5
    beta = 1.6
    lambda_ = 0.2
    mu = 0.14

    f = 3.4e13 / (HEV_IN_KELVIN ** beta)                 # J/g/HeV^beta
    g = 1.0 / (7200.0 * (HEV_IN_KELVIN ** alpha))        # g/cm^2/HeV^alpha (per your MATLAB comment)
    sigma = 5.670373e-5                                  # W/cm^2/K^4 (as in MATLAB)
    if apply_sigma_conversion:
        # MATLAB comment (was commented out in first variant):
        # sigma = sigma*(1160400)*4*(10^-9);  % J/ns/cm^2/HeV^4
        sigma = sigma * 1_160_400.0 * 4.0 * 1e-9

    r = 0.25
    V0 = 1.0 / 19.32

    return Material(alpha, beta, lambda_, mu, f, g, sigma, r, V0, name="Au_v1")


def au_supersonic_variant_2() -> Material:
    """
    Second MATLAB block (alpha=3.1, beta=1.2, rho0=2.78),
    includes explicit sigma conversion in MATLAB.
    """
    alpha = 3.1
    beta = 1.2
    lambda_ = 0.3685
    mu = 0.0

    f = 3.6e11 * (100.0 ** beta) / (HEV_IN_KELVIN ** beta)
    g = 1.0 / (7714.0 * (HEV_IN_KELVIN ** alpha))

    sigma = 5.670373e-5
    # MATLAB:
    # mat.sigma=mat.sigma*(1160400)*4*(10^-9); %J/ns/cm^2/HeV^4
    sigma = sigma * 1_160_400.0 * 4.0 * 1e-9

    r = 0.66667
    V0 = 1.0 / 2.78

    return Material(alpha, beta, lambda_, mu, f, g, sigma, r, V0, name="Au_v2")
