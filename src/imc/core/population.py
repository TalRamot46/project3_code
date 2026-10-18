"""Census population control.

comb() is a stratified comb (Wollaber 2016, Sec. 6.3.5, with one independent
offset per tooth): M teeth with spacing W/M over the stacked weights, each
tooth moved by its own uniform offset inside its stratum.  Every survivor gets
weight W/M, so energy is conserved exactly, and an input of weight w_j is
copied M w_j / W times on average, so the census is unbiased.  Unlike the
single-offset (systematic) comb of the frozen script, stratified resampling
also satisfies the variance condition (R1) of md/imc_mc_derivation.md, Sec. 7.

The advisor's per-cell allocation and per-cell comb are milestone M7.
"""
from __future__ import annotations

import numpy as np

from imc.core.particles import Particles


def comb(census: Particles, max_packets: int) -> Particles:
    """Comb the census down to max_packets equal-weight packets, if it is larger."""
    n = census.size
    if n <= max_packets:
        return census
    total = census.energy()
    if total <= 0.0:
        return Particles.empty()
    spacing = total / max_packets
    teeth = spacing * (np.arange(max_packets) + np.random.random(max_packets))
    idx = np.searchsorted(np.cumsum(census.w), teeth, side="right")
    idx = np.minimum(idx, n - 1)
    return Particles(
        census.x[idx], census.mu[idx], np.full(max_packets, spacing),
        census.t[idx], census.cell[idx],
    )
