"""Source particles for one IMC step: material emission, blackbody bath, initial radiation.

Energies are per unit transverse area.

Emission (F&C (4.13)): cell j emits E_j = f_j k_a,j c u_r,j dx_j dt, with
position uniform in the cell, mu uniform on [-1, 1] and time uniform in the
step (F&C (4.16b), (4.19), (4.20), (4.23)).

Bath (F&C (4.12)): an isotropic blackbody at the left face delivers
E_b = int sigma_SB T_bath^4 dt over the step, entering with density 2 mu on
(0, 1] (F&C (4.21)).  Birth times are sampled with density proportional to the
incident flux, not uniformly, so a drive that varies inside the step is
followed in time (md/imc_mc_derivation.md, Proposition 3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from imc.particles import Particles

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)


def gauss_integral(func, t0: float, t1: float) -> float:
    """Eight-point Gauss-Legendre integral of a vectorized func over [t0, t1]."""
    tm = 0.5 * (t0 + t1)
    th = 0.5 * (t1 - t0)
    return float(th * np.sum(_GL_WEIGHTS * func(tm + th * _GL_NODES)))


class TabulatedDrive:
    """Bath temperature T_D(t) from a two-column table, linearly interpolated.

    The table is (time, temperature); time_scale and temperature_scale convert
    them to the units of the simulation.  Before the first entry the drive is
    padded back to t = 0 at temperature T_pad.
    """

    def __init__(self, path, time_scale=1.0, temperature_scale=1.0, T_pad=0.0):
        data = np.loadtxt(Path(path))
        order = np.argsort(data[:, 0])            # tolerate unsorted input
        self.t = data[order, 0] * time_scale
        self.T = data[order, 1] * temperature_scale
        if self.t[0] > 0.0:
            self.t = np.insert(self.t, 0, 0.0)
            self.T = np.insert(self.T, 0, T_pad)

    def temperature(self, t):
        return np.interp(t, self.t, self.T)       # flat outside the table


class BathSource:
    """Isotropic blackbody bath at x = edges[0].

    flux(t) is the incident flux sigma_SB T_bath(t)^4 [energy/(area time)],
    vectorized.  energy_integral(t0, t1), if given, replaces the quadrature for
    the step energy; use it when the flux is singular at t = 0 (e.g. the
    paper's Test 1, where T_bath^4 ~ t^(-79/192)).
    """

    def __init__(self, flux, energy_integral=None, n_sub: int = 16):
        self.flux = flux
        self.energy_integral = energy_integral
        self.n_sub = n_sub

    def step_energy(self, t0: float, t1: float) -> float:
        if self.energy_integral is not None:
            return float(self.energy_integral(t0, t1))
        return gauss_integral(self.flux, t0, t1)

    def sample_times(self, t0: float, t1: float, n: int) -> np.ndarray:
        """Birth times with density ~ flux: pick a sub-interval by its energy, then uniform in it.

        The density is the flux averaged over n_sub equal sub-intervals, so it
        follows the drive to O(dt / n_sub).
        """
        sub = np.linspace(t0, t1, self.n_sub + 1)
        energies = np.array(
            [gauss_integral(self.flux, a, b) for a, b in zip(sub[:-1], sub[1:])]
        )
        total = energies.sum()
        if total <= 0.0:
            return t0 + (t1 - t0) * np.random.random(n)
        k = np.searchsorted(np.cumsum(energies) / total, np.random.random(n), side="right")
        k = np.minimum(k, self.n_sub - 1)
        return sub[k] + (sub[k + 1] - sub[k]) * np.random.random(n)


def allocate_emission(emission: np.ndarray, target: int) -> np.ndarray:
    """At least one packet per emitting cell, extras by largest remainder.

    Same scheme as the frozen SiO2 script (advisor's smarter allocation is M7a).
    """
    counts = np.zeros(emission.size, dtype=np.int64)
    emitting = emission > 0.0
    total = emission[emitting].sum()
    if total <= 0.0:
        return counts
    counts[emitting] = 1
    remaining = max(0, target - int(emitting.sum()))
    if remaining > 0:
        exact = remaining * emission / total
        add = np.floor(exact).astype(np.int64)
        counts += add
        leftover = remaining - int(add.sum())
        if leftover > 0:
            order = np.argsort(-(exact - add), kind="stable")
            counts[order[:leftover]] += 1
    return counts


@njit(cache=True)
def _emission_arrays(edges, emission, counts, t0, dt):
    n = 0
    for j in range(counts.size):
        n += counts[j]
    x = np.empty(n)
    mu = np.empty(n)
    w = np.empty(n)
    t = np.empty(n)
    cell = np.empty(n, dtype=np.int64)
    k = 0
    for j in range(counts.size):
        if counts[j] <= 0:
            continue
        weight = emission[j] / counts[j]
        left = edges[j]
        width = edges[j + 1] - edges[j]
        for _ in range(counts[j]):
            x[k] = left + width * np.random.random()
            mu[k] = 2.0 * np.random.random() - 1.0
            w[k] = weight
            t[k] = t0 + dt * np.random.random()
            cell[k] = j
            k += 1
    return x, mu, w, t, cell


def emission_particles(edges, emission, counts, t0, dt) -> Particles:
    return Particles(*_emission_arrays(edges, emission, counts, float(t0), float(dt)))


def bath_particles(edges, energy: float, n: int, times: np.ndarray) -> Particles:
    """n packets entering at the left face with density 2 mu on (0, 1]."""
    if n <= 0 or energy <= 0.0:
        return Particles.empty()
    mu = np.sqrt(1.0 - np.random.random(n))       # in (0, 1]
    return Particles(
        np.full(n, edges[0]), mu, np.full(n, energy / n), np.asarray(times, float),
        np.zeros(n, dtype=np.int64),
    )


def uniform_radiation_particles(edges, energy_density, per_cell: int, t0: float) -> Particles:
    """Isotropic radiation, uniform in each cell (initial condition)."""
    energy_density = np.broadcast_to(np.asarray(energy_density, float), (edges.size - 1,))
    widths = np.diff(edges)
    n_cells = widths.size
    cell = np.repeat(np.arange(n_cells, dtype=np.int64), per_cell)
    x = edges[cell] + widths[cell] * np.random.random(cell.size)
    mu = 2.0 * np.random.random(cell.size) - 1.0
    w = np.repeat(energy_density * widths / per_cell, per_cell)
    return Particles(x, mu, w, np.full(cell.size, float(t0)), cell)
