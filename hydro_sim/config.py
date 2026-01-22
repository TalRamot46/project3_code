from dataclasses import dataclass

@dataclass
class SimConfig:
    gamma: float = 1.4
    sigma_visc: float = 1.0     # σ in q definition
    geometry: str = "planar"    # "planar"|"cylindrical"|"spherical"
    CFL: float = 0.5
    t_end: float = 0.25
    max_steps: int = 100000
    bc_left: str = "reflecting"
    bc_right: str = "reflecting"
