"""Struct-of-arrays particle storage.

Each particle is an energy packet per unit transverse area: position x,
direction cosine mu, energy weight w, time t (birth time, or the start of
the step for census particles) and the index of the cell that contains it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Particles:
    x: np.ndarray
    mu: np.ndarray
    w: np.ndarray
    t: np.ndarray
    cell: np.ndarray

    @classmethod
    def empty(cls, n: int = 0) -> "Particles":
        return cls(
            np.empty(n), np.empty(n), np.empty(n), np.empty(n), np.empty(n, dtype=np.int64)
        )

    @property
    def size(self) -> int:
        return self.w.size

    def energy(self) -> float:
        return float(np.sum(self.w))

    def energy_per_cell(self, n_cells: int) -> np.ndarray:
        return np.bincount(self.cell, weights=self.w, minlength=n_cells)[:n_cells]

    @staticmethod
    def concatenate(parts) -> "Particles":
        parts = [p for p in parts if p.size > 0]
        if not parts:
            return Particles.empty()
        return Particles(
            np.concatenate([p.x for p in parts]),
            np.concatenate([p.mu for p in parts]),
            np.concatenate([p.w for p in parts]),
            np.concatenate([p.t for p in parts]),
            np.concatenate([p.cell for p in parts]).astype(np.int64),
        )

    def with_time(self, t0: float) -> "Particles":
        """Census particles start the next step at t0."""
        return Particles(self.x, self.mu, self.w, np.full(self.size, float(t0)), self.cell)
