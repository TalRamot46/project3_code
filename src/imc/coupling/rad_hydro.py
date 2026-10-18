"""IMC as a drop-in radiation solver for rad_hydro_sim (plan milestone M4).

rad_hydro_sim calls `radiation_step(state, dt, case, T_left)` once per step;
`IMCSolver.step` has the same signature and return tuple

    (T_material, e_material, T_rad, E_rad, F, LHS),

so the integrator can switch solvers with SimulationConfig.radiation_solver.
The solver keeps what diffusion does not need: the census (radiation packets)
and the energy ledger.  It is created by simulate_rad_hydro.

Conventions, matching the diffusion code and the manuscript's transport setup:
  * CGS units with rad_hydro_sim's c and a (PowerLawMaterial.from_case);
  * k_t = rho kappa_R, k_a = chi k_t, k_s = (1 - chi) k_t;
  * bath: isotropic blackbody at T_bath(t) entering at x = x_0, the Marshak
    drive of the diffusion code, integrated exactly over each step;
  * right face: vacuum (packets leave), the transport counterpart of the
    diffusion code's right_rad_bc="marshak_vacuum";
  * F: net energy current at faces 0..N-1 averaged over the step (diffusion's
    index convention, face 0 = the bath face).

Radiation-only for now (static mesh); the Lagrangian coupling is Phase C (M5).
"""
from __future__ import annotations

import numpy as np

from imc.core import sources, stepper
from imc.core.material import PowerLawMaterial
from imc.core.mesh import Mesh


def power_law_bath(bath_solver, a: float, c: float) -> sources.BathSource:
    """The self-similar Marshak bath T(t) = Tb t^tau (1 + B t^p)^(1/4), integrated exactly.

    T^4 = Tb^4 t^(4 tau) (1 + B t^p), so the step energy has a closed form even
    where the flux is singular at t = 0 (the paper's Test 1: p = -79/192).
    """
    Tb, tau = float(bath_solver.Tb), float(bath_solver.tau)
    B, p = float(bath_solver.bath_B), float(bath_solver.bath_power)
    k1, k2 = 4.0 * tau + 1.0, 4.0 * tau + p + 1.0
    if k1 <= 0.0 or k2 <= 0.0:
        raise ValueError("bath energy diverges at t = 0 for these exponents.")
    sigma_sb = 0.25 * a * c

    def flux(t):
        return sigma_sb * bath_solver.Tbath(time=np.maximum(t, 1e-300)) ** 4

    def energy(t0, t1):
        prim = lambda t: t**k1 / k1 + B * t**k2 / k2          # noqa: E731
        return sigma_sb * Tb**4 * (prim(t1) - prim(t0))

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

    def __init__(self, case, config, state, bath_solver=None, census_per_cell: int = 4):
        if case.geom.alpha != 0:
            raise NotImplementedError("The IMC coupling is planar only.")
        if case.scenario != "radiation_only":
            raise NotImplementedError("The IMC coupling is radiation_only for now (M4).")
        if case.bc_type not in ("Marshak", "Marshak_Menahem"):
            raise NotImplementedError(
                "IMC needs an incident bath (bc_type='Marshak'); a Dirichlet surface "
                "temperature has no transport counterpart.")
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
            self.bath = power_law_bath(bath_solver, a, c)
        else:
            raise ValueError("bc_type='Marshak' needs the SubsonicHeatWave bath solver or case.bath_table.")

        stepper.seed_all(config.imc_seed)
        mesh = Mesh(state.x)
        census = sources.uniform_radiation_particles(mesh.edges, state.E_rad, census_per_cell, state.t)
        T0 = self.material.temperature(state.rho * state.e_material, state.rho)
        self.state = stepper.initial_state(T0, mesh, state.rho, self.material, census=census, t0=state.t)
        self.steps = 0
        self.particles = 0
        self.collisions = 0
        self.capped = 0
        self.max_cell_residual = 0.0
        self.max_ledger_error = 0.0
        self.E_rad_track = np.asarray(state.E_rad, dtype=float).copy()

    def step(self, state, dt, case, T_left, diagnostics=None):
        """One radiation step; same signature and return tuple as radiation_step."""
        mesh = Mesh(state.x)
        rho = state.rho
        T = self.material.temperature(rho * state.e_material, rho)
        imc_state = stepper.IMCState(T, self.state.census, state.t, self.state.ledger)
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

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "mean_particles_per_step": self.particles / max(self.steps, 1),
            "collisions": self.collisions,
            "capped_histories": self.capped,
            "max_relative_cell_residual": self.max_cell_residual,
            "max_ledger_error": self.max_ledger_error,
        }
