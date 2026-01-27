from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Geometry:
    alpha: int
    beta: float
    zeta: float

def planar() -> Geometry:
    # PDF: planar => alpha=0, beta=1, zeta=1
    return Geometry(alpha=0, beta=1.0, zeta=1.0)

def cylindrical() -> Geometry:
    return Geometry(alpha=1, beta=2.0 * math.pi, zeta=math.pi)

def spherical() -> Geometry:
    return Geometry(alpha=2, beta=4.0 * math.pi, zeta=(4.0/3.0) * math.pi)
