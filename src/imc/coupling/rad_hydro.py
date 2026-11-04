"""IMC as a drop-in radiation solver for rad_hydro_sim (plan milestones M4, M5, M5b).

rad_hydro_sim calls `radiation_step(state, dt, case, T_left)` once per step;
`IMCSolver.step` has the same signature and return tuple

    (T_material, e_material, T_rad, E_rad, F, LHS),

so the integrator can switch solvers with SimulationConfig.radiation_solver.
The solver keeps what diffusion does not need: the census (radiation packets)
and the energy ledgers.  It is created by simulate_rad_hydro.

Conventions, matching the diffusion code and the manuscript's transport setup:
  * CGS units with rad_hydro_sim's c and a (PowerLawMaterial.from_case);
  * k_t = rho kappa_R, k_a = chi k_t, k_s = (1 - chi) k_t;
  * bath: isotropic blackbody at T_bath(t) entering at x = x_0, the Marshak
    drive of the diffusion code, integrated exactly over each step;
  * right face: vacuum (packets leave), the transport counterpart of the
    diffusion code's right_rad_bc="marshak_vacuum";
  * F: net energy current at faces 0..N-1 averaged over the step (diffusion's
    index convention, face 0 = the bath face).

Lagrangian coupling (M5, md/imc_code/imc_coupling.md Sec. 3): between two
radiation steps the hydro moves the mesh.  Each census packet keeps its cell,
its relative position in it (Mesh.carry) and its weight, so the radiation
energy per unit mass is unchanged by the motion: the transport counterpart of
the diffusion code's rad_energy_per_mass=True.  With rad_coupling="physical"
(M5b) the integrator also adds the radiation-pressure work to the census
(add_radiation_work).
"""
from __future__ import annotations

import numpy as np

from imc.core import sources, stepper
from imc.core.material import PowerLawMaterial
from imc.core.mesh import Mesh
from imc.core.particles import Particles
from imc.core.tallies import EnergyLedger


def power_law_bath(bath_solver, a: float, c: float, factor=None) -> sources.BathSource:
    """The self-similar Marshak bath T(t) = Tb t^tau (1 + B t^p)^(1/4), integrated exactly.

    T^4 = Tb^4 t^(4 tau) (1 + B t^p), so the step energy has a closed form even
    where the flux is singular at t = 0 (the paper's Test 1: p = -79/192).

    factor: optional calibration c(t) (bath_drive="calibrated"), T -> c(t) T.  It
    varies slowly, so the step energy is the closed form times c^4 at mid-step
    (error O(dt^2 c''), next to the exact power-law singularity).
    """
    Tb, tau = float(bath_solver.Tb), float(bath_solver.tau)
    B, p = float(bath_solver.bath_B), float(bath_solver.bath_power)
    k1, k2 = 4.0 * tau + 1.0, 4.0 * tau + p + 1.0
    if k1 <= 0.0 or k2 <= 0.0:
        raise ValueError("bath energy diverges at t = 0 for these exponents.")
    sigma_sb = 0.25 * a * c

    def flux(t):
        T = bath_solver.Tbath(time=np.maximum(t, 1e-300))
        return sigma_sb * (T if factor is None else T * factor(t)) ** 4

    def energy(t0, t1):
        prim = lambda t: t**k1 / k1 + B * t**k2 / k2          # noqa: E731
        e = sigma_sb * Tb**4 * (prim(t1) - prim(t0))
        return e if factor is None else e * float(factor(0.5 * (t0 + t1))) ** 4

    return sources.BathSource(flux, energy_integral=energy)


def tabulated_bath(path, a: float, c: float) -> sources.BathSource:
    """Bath from a (t [ns], T [eV]) table, as rad_hydro_sim's bath_table option."""
    from rad_hydro_sim.simulation.radiation_step import eV_to_erg, k_B

    drive = sources.TabulatedDrive(path, time_scale=1e-9, temperature_scale=eV_to_erg / k_B)
    sigma_sb = 0.25 * a * c

    def flux(t):
        return sigma_sb * drive.temperature(t) ** 4

    return sources.BathSource(flux)


class IMCSolver:
    """Persistent IMC radiation solver for one rad_hydro_sim run."""

    def __init__(self, case, config, state, bath_solver=None, census_per_cell: int = 4, bath_factor=None):
        if case.geom.alpha != 0:
            raise NotImplementedError("The IMC coupling is planar only.")
        if case.scenario not in ("radiation_only", "full_rad_hydro"):
            raise NotImplementedError(f"The IMC coupling has no radiation to solve in {case.scenario!r}.")
        if case.bc_type not in ("Marshak", "Marshak_Menahem"):
            raise NotImplementedError(
                "IMC needs an incident bath (bc_type='Marshak'); a Dirichlet surface "
                "temperature has no transport counterpart.")
        if case.scenario == "full_rad_hydro" and not getattr(config, "rad_energy_per_mass", False):
            raise ValueError(
                "IMC carries the census with the mass (radiation energy per unit mass is "
                "unchanged by the hydro step); set rad_energy_per_mass=True so that diffusion, "
                "which it is compared with, does the same.")
        self.material = PowerLawMaterial.from_case(
            case, subtract_radiation=getattr(config, "subtract_radiation_energy", False), T_floor=1.0)
        self.settings = stepper.IMCSettings(
            n_emission=config.imc_n_emission, n_bath=config.imc_n_bath,
            max_census=config.imc_max_census, max_events=10**6,
            left_bc="vacuum", right_bc="vacuum",
        )
        a, c = self.material.a, self.material.c
        if getattr(case, "bath_table", None):
            self.bath = tabulated_bath(case.bath_table, a, c)
        elif bath_solver is not None:
            self.bath = power_law_bath(bath_solver, a, c, factor=bath_factor)
        else:
            raise ValueError("bc_type='Marshak' needs the SubsonicHeatWave bath solver or case.bath_table.")

        stepper.seed_all(config.imc_seed)
        self.mesh = Mesh(state.x)
        census = sources.uniform_radiation_particles(self.mesh.edges, state.E_rad, census_per_cell, state.t)
        T0 = self.material.temperature(state.rho * state.e_material, state.rho)
        self.state = stepper.initial_state(T0, self.mesh, state.rho, self.material, census=census, t0=state.t)
        self.steps = 0
        self.particles = 0
        self.collisions = 0
        self.capped = 0
        self.max_cell_residual = 0.0
        self.max_ledger_error = 0.0
        self.work_total = 0.0          # sum of |radiation-pressure work| (M5b)
        self.work_to_matter = 0.0      # ... the part the census could not carry
        self.E_rad_track = np.asarray(state.E_rad, dtype=float).copy()

    def step(self, state, dt, case, T_left, diagnostics=None):
        """One radiation step; same signature and return tuple as radiation_step."""
        mesh = Mesh(state.x)
        census = self.state.census
        if not np.array_equal(mesh.edges, self.mesh.edges):
            # The hydro moved the mesh since the last step: carry the census with it.
            census = Particles(self.mesh.carry(census.x, census.cell, mesh),
                               census.mu, census.w, census.t, census.cell)
        self.mesh = mesh
        rho = state.rho
        T = self.material.temperature(rho * state.e_material, rho)
        imc_state = stepper.IMCState(T, census, state.t)
        # Per-step ledger: with hydro the matter energy also changes between
        # radiation steps (p dV work), so a run-long ledger would not close.
        imc_state.ledger = EnergyLedger(initial=stepper.total_energy(imc_state, mesh, rho, self.material))
        new, tl = stepper.imc_step(imc_state, mesh, rho, self.material, dt, self.bath, self.settings)
        self.state = new

        widths = mesh.widths
        # The state's E_rad is the end-of-step census, so the energy ledger stays
        # exact.  In optically thick cells almost every packet is absorbed within
        # the step and the census holds only the few emitted in its last instant:
        # an unbiased but very noisy snapshot.  The track-length estimate (time-
        # averaged over the step) is far smoother; it is kept for diagnostics.
        E_rad = tl.census_after / widths
        self.E_rad_track = tl.mean_energy_density(widths)
        T_rad = (np.maximum(E_rad, 0.0) / self.material.a) ** 0.25
        e_material = self.material.u_m(new.T, rho) / rho
        F = ((tl.face_plus - tl.face_minus) / dt)[:-1]

        self.steps += 1
        self.particles += tl.n_particles
        self.collisions += tl.n_collisions
        self.capped += tl.n_capped
        scale = max(tl.census_after.max(), tl.emitted.max(), tl.deposited.max(), 1e-300)
        self.max_cell_residual = max(self.max_cell_residual, np.abs(tl.cell_balance_residual()).max() / scale)
        err = new.ledger.relative_error(stepper.total_energy(new, mesh, rho, self.material))
        self.max_ledger_error = max(self.max_ledger_error, abs(err))
        if diagnostics is not None:
            diagnostics["F_left"] = (tl.bath_energy - tl.escaped_left) / dt
            diagnostics["F_right"] = tl.escaped_right / dt
        return new.T, e_material, T_rad, E_rad, F, 0.0

    def add_radiation_work(self, work: np.ndarray, max_fraction: float = 0.5):
        """Add the radiation-pressure work of the hydro step to the census (M5b).

        work[j] (energy per area) is -dt Pbar_rad,j (ubar_{j+1} - ubar_j), the
        discrete -P_rad dV of cell j.  The census of cell j is scaled by
        1 + work_j / C_j.  Where that would change the cell's census by more
        than max_fraction, or the cell has none (thick cells, where packets
        live much less than dt), large weights would be needed; the work then
        goes to the matter instead, which in such a cell is in equilibrium
        with the radiation within the step anyway.  Total energy is conserved
        either way.  Returns (E_rad, work charged to the matter per cell).
        """
        census = self.state.census
        n = self.mesh.n_cells
        C = census.energy_per_cell(n)
        ok = (C > 0.0) & (np.abs(work) <= max_fraction * C)
        factor = np.where(ok, 1.0 + work / np.where(C > 0.0, C, 1.0), 1.0)
        self.state.census = Particles(census.x, census.mu, census.w * factor[census.cell],
                                      census.t, census.cell)
        to_matter = np.where(ok, 0.0, work)
        self.work_total += float(np.abs(work).sum())
        self.work_to_matter += float(np.abs(to_matter).sum())
        return self.state.census.energy_per_cell(n) / self.mesh.widths, to_matter

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "mean_particles_per_step": self.particles / max(self.steps, 1),
            "collisions": self.collisions,
            "capped_histories": self.capped,
            "max_relative_cell_residual": self.max_cell_residual,
            "max_ledger_error": self.max_ledger_error,
            "radiation_work_total": self.work_total,
            "radiation_work_to_matter": self.work_to_matter,
        }
