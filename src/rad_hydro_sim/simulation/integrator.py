from hydro_sim.core.geometry import Geometry
from rad_hydro_sim.simulation.hydro_steps import (
    boundary_work,
    get_e_star_from_hydro,
    radiation_work,
    update_nodes_compatible,
    update_nodes_from_pressure,
)
from rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from rad_hydro_sim.simulation.radiation_step import (
    a_Kelvin,
    calculate_temperature_from_specific_energy,
    material_temperature,
    radiation_step,
)
from hydro_sim.problems.simulation_config import SimulationConfig
from hydro_sim.core.state import RadHydroState
import numpy as np
from typing import Tuple

S_PER_NS = 1e-9


def _radiation_options(config: SimulationConfig) -> dict:
    """Phase B keyword options for radiation_step (defaults = original scheme)."""
    return dict(
        material_update=getattr(config, "material_update", "u_r"),
        subtract_radiation_energy=getattr(config, "subtract_radiation_energy", False),
        right_rad_bc=getattr(config, "right_rad_bc", "ghost"),
    )


def _radiate(state, dt, case, config, T_left, imc_solver, diagnostics):
    """The radiation-matter step, by the diffusion solver or by IMC."""
    if getattr(config, "radiation_solver", "diffusion") == "imc":
        if imc_solver is None:
            raise ValueError("radiation_solver='imc' needs an IMCSolver (created by simulate_rad_hydro).")
        return imc_solver.step(state, dt, case, T_left, diagnostics=diagnostics)
    return radiation_step(state, dt, case, T_left, diagnostics=diagnostics, **_radiation_options(config))


def step_rad_hydro(
    state: RadHydroState,
    dt: float,
    case: RadHydroCase,
    config: SimulationConfig,
    T_left: float,
    *,
    imc_solver=None,
    diagnostics: dict | None = None,
) -> Tuple[RadHydroState, np.ndarray]:
    """
    Pseudo-code for Lagrangian step with radiation-hydro coupling:
    1. Perform half-step velocity update using current acceleration.
    2. Update all cell hydrodynamic variables (x, V, rho, e_star, p, m_cells) using half-step velocities.
    3. Take e_star, rho, and m_cells to perform radiation-matter coupling and get updated new_e_material and E_rad.
    4. Update pressure profile using new_e_material.

    Phase B options (SimulationConfig; defaults reproduce the original step):
    radiation_solver, material_update, subtract_radiation_energy,
    right_rad_bc, rad_energy_per_mass, hydro_energy.  If `diagnostics` is a
    dict it receives the step's boundary fluxes and boundary work for the
    energy budget (md/phase_b/phase_b_diffusion_changes.md).

    Phase C (md/phase_c/phase_c_changes.md): radiation_solver="imc" also in
    full_rad_hydro, and rad_coupling="physical" adds the radiation pressure
    P_rad = E_rad/3 to the momentum update and charges its p dV work to the
    radiation (needs hydro_energy="compatible" and rad_energy_per_mass).
    """
    hydro_energy = getattr(config, "hydro_energy", "standard")
    if hydro_energy not in ("standard", "compatible"):
        raise ValueError(f"hydro_energy must be 'standard' or 'compatible', got {hydro_energy!r}")
    subtract = getattr(config, "subtract_radiation_energy", False)
    if subtract and case.scenario == "hydro_only":
        raise NotImplementedError("subtract_radiation_energy needs radiation (radiation_only or full_rad_hydro).")
    rad_coupling = getattr(config, "rad_coupling", "matched")
    if rad_coupling not in ("matched", "physical"):
        raise ValueError(f"rad_coupling must be 'matched' or 'physical', got {rad_coupling!r}")
    physical = rad_coupling == "physical" and case.scenario == "full_rad_hydro"
    if physical and not (hydro_energy == "compatible" and getattr(config, "rad_energy_per_mass", False)):
        raise ValueError("rad_coupling='physical' needs hydro_energy='compatible' and rad_energy_per_mass=True: "
                         "the radiation's p dV work is the compatible scheme's, applied to E_rad per unit mass.")
    rad_diag = {} if diagnostics is not None else None

    # Subsonic ablation: p=0 at m=0 (ablated matter expands into vacuum).
    # Pressure drive: p_left = P0 * t^tau.
    if case.P0_Barye is not None:
        p_drive = case.P0_Barye * ((state.t / S_PER_NS) ** case.tau) if state.t > 0 else 0.0
        bc_left = p_drive
        bc_right = "outflow"
    else:
        bc_left = "vacuum"   # p=0 at m=0 (radiation/temperature drive)
        bc_right = "outflow"

    if case.scenario == "hydro_only" and hydro_energy == "compatible":
        # No retry/CFL shrinking here: the compatible close needs the same dt
        # in the predictor, the corrector and the clock.
        state_star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, dt, bc_left, bc_right)
        new_state = update_nodes_compatible(
            state, state_star, case, np.zeros_like(state_star.rho), dt, bc_left, bc_right, t_old=state.t)
        new_state.T_material = calculate_temperature_from_specific_energy(
            new_state.e_material, new_state.rho, case.f_Kelvin, case.beta_Rosen, case.mu)
        new_state.T_rad = state_star.T_rad if state_star.T_rad is not None else np.zeros_like(state_star.rho)
        new_state.E_rad = np.zeros_like(state_star.rho)
        LHS = np.zeros_like(state_star.rho)
    elif case.scenario == "hydro_only":
        # Retry mechanism for stability: if hydro update fails due to
        # negative/zero cell volumes, reduce dt and retry a few times.
        retries = 5
        # CFL-like safeguard: cap dt so nodes don't cross (limit displacement
        # to at most half a cell width in a single step).
        dx = np.min(np.diff(state.x)) if len(state.x) > 1 else 1.0
        max_speed = max(np.max(np.abs(state.u)), 1e-12)
        cfl_dt = 0.5 * dx / max_speed
        attempt_dt = min(dt, float(cfl_dt))
        while True:
            try:
                state_star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, attempt_dt, bc_left, bc_right)
                break
            except RuntimeError as e:
                if retries <= 0:
                    raise
                retries -= 1
                attempt_dt *= 0.5
        dt_used = attempt_dt
        new_e_material = state_star.e_material
        new_T_material = calculate_temperature_from_specific_energy(new_e_material, state_star.rho, case.f_Kelvin, case.beta_Rosen, case.mu)
        new_state = update_nodes_from_pressure(state_star, case, new_e_material, dt, bc_left, bc_right, t_old=state.t)
        new_E_rad = np.zeros_like(state_star.rho)
        new_T_rad = state_star.T_rad if state_star.T_rad is not None else np.zeros_like(state_star.rho)
        new_state.T_material = new_T_material
        new_state.T_rad = new_T_rad
        new_state.E_rad = new_E_rad
        LHS = np.zeros_like(state_star.rho)
    elif case.scenario == "radiation_only":
        try:
            new_T_material, new_e_material, new_T_rad, new_E_rad, new_F, LHS = _radiate(
                state, dt, case, config, T_left, imc_solver, rad_diag)
            new_state = state._replace(T_material=new_T_material, e_material=new_e_material, T_rad=new_T_rad, E_rad=new_E_rad)
        except ValueError as e:
            if "infs or NaNs" in str(e):
                # If the radiation solver fails due to NaNs/Infs, keep the state unchanged
                new_state = state._replace()
                LHS = np.zeros_like(state.rho)
            else:
                raise
    elif case.scenario == "full_rad_hydro":
        state_star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, dt, bc_left, bc_right)
        if getattr(config, "rad_energy_per_mass", False):
            # Manuscript d(v E_r)/dt: radiation rides with the mass through the
            # hydro step, so E_rad V is unchanged (original: E_rad unchanged).
            state_star.E_rad = state.E_rad * state.V / state_star.V
        new_T_material, new_e_material, new_T_rad, new_E_rad, new_F_rad, LHS = _radiate(
            state_star, dt, case, config, T_left, imc_solver, rad_diag)
        if hydro_energy == "compatible":
            P_rad_new = None
            if physical:
                # A7: P_rad = E_rad / 3.  IMC uses its track-length (step-averaged)
                # E_rad, far less noisy than the census snapshot.
                E_for_P = imc_solver.E_rad_track if imc_solver is not None else new_E_rad
                P_rad_new = np.maximum(E_for_P, 0.0) / 3.0
            new_state = update_nodes_compatible(
                state, state_star, case, new_e_material - state_star.e_material, dt,
                bc_left, bc_right, t_old=state.t, P_rad_new=P_rad_new)
            if physical:
                work = radiation_work(state, new_state, dt)
                if imc_solver is not None:
                    new_E_rad, to_matter = imc_solver.add_radiation_work(work)
                    new_state.e_material = new_state.e_material + to_matter / new_state.m_cells
                else:
                    new_E_rad = new_E_rad + work / new_state.V
                    if np.any(new_E_rad < 0.0):
                        raise RuntimeError("Radiation p dV work made E_rad negative; reduce dt.")
                new_T_rad = (np.maximum(new_E_rad, 0.0) / a_Kelvin) ** 0.25
            if imc_solver is not None:
                # material_temperature reads radiation_step's module-level material
                # parameters, which only the diffusion solve sets; the IMC's
                # material is bound to this case.
                new_T_material = imc_solver.material.temperature(
                    new_state.rho * new_state.e_material, new_state.rho)
            else:
                new_T_material = material_temperature(
                    new_state.rho * new_state.e_material, new_state.rho, subtract)
        else:
            new_state = update_nodes_from_pressure(state_star, case, new_e_material, dt, bc_left, bc_right, t_old=state.t)
        new_state.T_material = new_T_material
        new_state.T_rad = new_T_rad
        new_state.E_rad = new_E_rad
        new_state.F_rad = new_F_rad
    else:
        raise ValueError(f"Unknown scenario: {case.scenario}")

    if diagnostics is not None:
        diagnostics["F_left"] = rad_diag.get("F_left", 0.0) if rad_diag else 0.0
        diagnostics["F_right"] = rad_diag.get("F_right", 0.0) if rad_diag else 0.0
        diagnostics["W_boundary"] = (
            0.0 if case.scenario == "radiation_only" else boundary_work(state, new_state, dt))
    new_state.t += dt
    return new_state, LHS
