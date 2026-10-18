"""Nonuniform 1D slab mesh: cell edges, widths and (optionally) masses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    edges: np.ndarray            # cell edges x_0 < x_1 < ... < x_N [length]

    def __post_init__(self):
        edges = np.ascontiguousarray(self.edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("edges must be a strictly increasing 1D array with >= 2 entries.")
        object.__setattr__(self, "edges", edges)

    @classmethod
    def uniform(cls, length: float, n_cells: int, x0: float = 0.0) -> "Mesh":
        return cls(np.linspace(x0, x0 + length, n_cells + 1))

    @classmethod
    def from_masses(cls, m_cells, rho, x0: float = 0.0) -> "Mesh":
        """Mesh from Lagrangian cell masses per area and densities (dx = m / rho)."""
        dx = np.asarray(m_cells, dtype=float) / np.asarray(rho, dtype=float)
        return cls(x0 + np.concatenate(([0.0], np.cumsum(dx))))

    @property
    def n_cells(self) -> int:
        return self.edges.size - 1

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def masses(self, rho) -> np.ndarray:
        return np.asarray(rho, dtype=float) * self.widths
