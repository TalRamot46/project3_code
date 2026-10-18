"""Per-step tallies and the two energy ledgers.

Global ledger (F&C (4.29)): matter + radiation now = initial + bath in - leaks.

Per-cell ledger (Wollaber 2016, Eq. 93): for every cell j and every step,

    E_rad,j(t1) - E_rad,j(t0) = emitted_j - deposited_j
                                + (face_plus[j]   - face_minus[j])     # in at the left face
                                - (face_plus[j+1] - face_minus[j+1])   # out at the right face

Every packet's weight ends in exactly one place, so both ledgers close to
round-off in every realization (md/imc_mc_derivation.md, Sec. 5.2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepTallies:
    t0: float
    t1: float
    emitted: np.ndarray          # per cell, energy per area
    bath_energy: float           # entered through the left face
    deposited: np.ndarray        # per cell
    face_plus: np.ndarray        # per face, including the bath at face 0
    face_minus: np.ndarray
    track: np.ndarray            # per cell, integral of weight * time
    escaped_left: float
    escaped_right: float
    census_before: np.ndarray    # radiation energy per cell at t0 (after combing)
    census_after: np.ndarray     # at t1, before combing
    n_particles: int
    n_collisions: int
    n_capped: int

    @property
    def dt(self) -> float:
        return self.t1 - self.t0

    def mean_energy_density(self, widths) -> np.ndarray:
        """Step-averaged radiation energy density per cell (track-length estimator)."""
        return self.track / (self.dt * widths)

    def cell_balance_residual(self) -> np.ndarray:
        """Per-cell ledger residual (should be round-off)."""
        net_in = (self.face_plus[:-1] - self.face_minus[:-1]) - (
            self.face_plus[1:] - self.face_minus[1:]
        )
        return (self.census_after - self.census_before) - (
            self.emitted - self.deposited + net_in
        )

    def radiation_balance_residual(self) -> float:
        """Whole-domain radiation ledger for this step (should be round-off)."""
        return float(
            self.census_after.sum() - self.census_before.sum()
            - (self.emitted.sum() + self.bath_energy - self.deposited.sum()
               - self.escaped_left - self.escaped_right)
        )


@dataclass
class EnergyLedger:
    """Cumulative global ledger over a run."""
    initial: float
    bath_in: float = 0.0
    escaped_left: float = 0.0
    escaped_right: float = 0.0

    def add(self, tallies: StepTallies):
        self.bath_in += tallies.bath_energy
        self.escaped_left += tallies.escaped_left
        self.escaped_right += tallies.escaped_right

    def expected(self) -> float:
        return self.initial + self.bath_in - self.escaped_left - self.escaped_right

    def relative_error(self, current_total: float) -> float:
        expected = self.expected()
        return (current_total - expected) / max(abs(expected), 1e-300)
