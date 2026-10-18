"""One IMC time step on a fixed mesh: sources -> transport -> material update.

    1. From T^n: opacities k_a, k_s, Fleck factor f, u_r = a T^4, u_m(T^n).
    2. Sources: emission E_j = f k_a c u_r dx dt per cell, the bath, the census.
    3. Random walk to t^{n+1} (transport.transport_step).
    4. Material: u_m^{n+1} = u_m^n + (deposited_j - E_j) / dx_j  (F&C (1.18)),
       then T^{n+1} by inverting the EOS.
    5. Comb the census if it exceeds max_census.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from imc.core import population, sources, transport
from imc.core.material import PowerLawMaterial
from imc.core.mesh import Mesh
from imc.core.particles import Particles
from imc.core.tallies import EnergyLedger, StepTallies

BC = {"vacuum": transport.VACUUM, "reflecting": transport.REFLECTING}


@dataclass(frozen=True)
class IMCSettings:
    n_emission: int = 2500       # target emission packets per step
    n_bath: int = 4000           # bath packets per step
    max_census: int = 120000     # comb trigger
    max_events: int = 20000      # safety cap on events per history
    left_bc: str = "vacuum"      # the bath, if any, enters through the left face
    right_bc: str = "vacuum"
    fleck_mode: str = "implicit"  # "explicit" sets f = 1 (M2a)


@dataclass
class IMCState:
    T: np.ndarray                # material temperature per cell
    census: Particles            # radiation packets at time t
    t: float
    ledger: EnergyLedger = field(default=None)


def total_energy(state: IMCState, mesh: Mesh, rho, material: PowerLawMaterial) -> float:
    return float(np.sum(material.u_m(state.T, rho) * mesh.widths) + state.census.energy())


def initial_state(T0, mesh: Mesh, rho, material: PowerLawMaterial, census=None, t0=0.0) -> IMCState:
    census = Particles.empty() if census is None else census
    state = IMCState(np.array(T0, dtype=float), census, float(t0))
    state.ledger = EnergyLedger(initial=total_energy(state, mesh, rho, material))
    return state


def imc_step(
    state: IMCState, mesh: Mesh, rho, material: PowerLawMaterial, dt: float,
    bath: sources.BathSource | None, settings: IMCSettings,
) -> tuple[IMCState, StepTallies]:
    t0 = state.t
    t1 = t0 + dt
    widths = mesh.widths
    T = state.T

    k_a, k_s = material.opacities(T, rho)
    k_a = np.broadcast_to(k_a, widths.shape).astype(np.float64)
    k_s = np.broadcast_to(k_s, widths.shape).astype(np.float64)
    fleck = np.broadcast_to(material.fleck(T, rho, dt, settings.fleck_mode), widths.shape).astype(np.float64)
    u_m = material.u_m(T, rho)
    emitted = fleck * k_a * material.c * material.u_r(T) * widths * dt

    counts = sources.allocate_emission(emitted, settings.n_emission)
    emission = sources.emission_particles(mesh.edges, emitted, counts, t0, dt)

    bath_energy = 0.0
    bath_parts = Particles.empty()
    if bath is not None:
        bath_energy = bath.step_energy(t0, t1)
        if bath_energy > 0.0 and settings.n_bath > 0:
            times = bath.sample_times(t0, t1, settings.n_bath)
            bath_parts = sources.bath_particles(mesh.edges, bath_energy, settings.n_bath, times)

    census_before = state.census.energy_per_cell(mesh.n_cells)
    allp = Particles.concatenate([state.census, emission, bath_parts])

    (cx, cmu, cw, ccell, deposited, face_plus, face_minus, track,
     esc_l, esc_r, n_coll, n_capped) = transport.transport_step(
        allp.x, allp.mu, allp.w, allp.t, allp.cell,
        mesh.edges, k_a, k_s, fleck, material.c, t1,
        BC[settings.left_bc], BC[settings.right_bc], settings.max_events,
    )
    face_plus = face_plus.copy()
    face_plus[0] += bath_energy                  # the bath enters through face 0
    census = Particles(cx, cmu, cw, np.full(cx.size, t1), ccell)
    census_after = census.energy_per_cell(mesh.n_cells)

    u_new = u_m + (deposited - emitted) / widths
    if np.any(u_new <= 0.0) or np.any(~np.isfinite(u_new)):
        bad = int(np.argmin(u_new))
        raise RuntimeError(
            f"Non-positive material energy at t={t1:g} (cell {bad}): the emission "
            "exceeded the material energy. Reduce dt, or use fleck_mode='implicit'."
        )
    T_new = material.temperature(u_new, rho)

    tallies = StepTallies(
        t0=t0, t1=t1, emitted=emitted, bath_energy=bath_energy,
        deposited=deposited, face_plus=face_plus, face_minus=face_minus, track=track,
        escaped_left=esc_l, escaped_right=esc_r,
        census_before=census_before, census_after=census_after,
        n_particles=allp.size, n_collisions=n_coll, n_capped=n_capped,
    )
    new_state = IMCState(T_new, population.comb(census, settings.max_census), t1, state.ledger)
    new_state.ledger.add(tallies)
    return new_state, tallies


def seed_all(value: int):
    """Seed numpy's generator and numba's (they are separate)."""
    np.random.seed(value)
    transport.seed(value)
