# ictt29/data_loader.py
import sys
sys.setrecursionlimit(40000)
import pickle
import traceback
import scipy.integrate
import os
import numpy as np
from pathlib import Path
from dataclasses import replace

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.menahem_new.ablation_solver_og import AblationSolver
from project3_code.menahem_new.subsonic_heat_wave_og import SubsonicHeatWave
from project3_code.menahem_new.piston_shock_og import PistonShock
from project3_code.rad_hydro_sim.plotting.RadHydroHistory import RadHydroHistory
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _ablation_kwargs_from_case,
    _heat_kwargs_from_case,
    _ns_amplitude_rescale,
)

USE_CACHE = True

# Custom Unpickler to solve cross-version NumPy namespace mapping (e.g. numpy._core to numpy.core)
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle newer data in older environments
        if 'numpy._core' in module:
            module = module.replace('numpy._core', 'numpy.core')
        # Handle older data in newer environments
        elif module == 'numpy.core' or module.startswith('numpy.core.'):
            # Safe check to see if modern _core exists
            try:
                import numpy._core
                module = module.replace('numpy.core', 'numpy._core')
            except ImportError:
                pass
        return super().find_class(module, name)


def get_sim_history(preset_name: str, case_label: str, N: int = None):
    """
    Load or run 1D Rad-Hydro simulation history.
    Saves to {case_label}_sim_history.npz.
    """
    case, config = get_preset(preset_name)
    if N is not None:
        config = replace(config, N=N)

    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    sim_cache_path = cache_dir / f"{case_label}_sim_history_N{config.N}.npz"
    if not sim_cache_path.exists() and config.N == 1000:
        legacy_path = cache_dir / f"{case_label}_sim_history.npz"
        if legacy_path.exists():
            sim_cache_path = legacy_path

    history = None
    if USE_CACHE and sim_cache_path.exists():
        print(f"Loading cached simulation history from {sim_cache_path}...")
        try:
            with np.load(sim_cache_path) as data:
                history = RadHydroHistory(
                    t=data["t"],
                    x=data["x"],
                    m=data["m"],
                    rho=data["rho"],
                    p=data["p"],
                    u=data["u"],
                    e=data["e"],
                    T=data["T"],
                    E_rad=data["E_rad"],
                    T_material=data["T_material"]
                )
            print("Simulation history loaded successfully from npz.")
        except Exception as e:
            print(f"Failed to load simulation cache: {e}. Traceback:")
            traceback.print_exc()
            print("Re-running simulation...")
            sim_cache_path.unlink(missing_ok=True)

    if history is None:
        print("Running simulation...")
        config = replace(config, show_plot=False, show_slider=False)
        _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)
        temp_sim_path = sim_cache_path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(
                temp_sim_path,
                t=history.t,
                x=history.x,
                m=history.m,
                rho=history.rho,
                p=history.p,
                u=history.u,
                e=history.e,
                T=history.T,
                E_rad=history.E_rad,
                T_material=history.T_material
            )
            os.replace(temp_sim_path, sim_cache_path)
            print(f"Simulation history saved successfully to {sim_cache_path}.")
        except Exception as e:
            print(f"Failed to save simulation cache safely: {e}. Traceback:")
            traceback.print_exc()
            if temp_sim_path.exists():
                temp_sim_path.unlink()

    return case, history


def get_sub_similarity_solver(case, case_label: str):
    """
    Load or solve SubsonicHeatWave similarity solver.
    Saves to {case_label}_sub_similarity_solver.pkl.
    """
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver_cache_path = cache_dir / f"{case_label}_sub_similarity_solver.pkl"

    solver = None
    if USE_CACHE and solver_cache_path.exists():
        print(f"Loading cached subsonic similarity solver from {solver_cache_path}...")
        try:
            with open(solver_cache_path, "rb") as f:
                solver = CustomUnpickler(f).load()
            if solver is not None:
                solver.ode_solver = scipy.integrate.ode(solver.fode).set_integrator(solver.ode_scheme)
            print("Subsonic similarity solver loaded successfully.")
        except Exception as e:
            print(f"Failed to load subsonic similarity solver cache: {e}. Traceback:")
            traceback.print_exc()
            print("Re-solving subsonic similarity ODEs...")

    if solver is None:
        print("Solving subsonic similarity ODEs (finding xsi_f via shooting method)...")
        heat_kwargs = _heat_kwargs_from_case(case)
        solver = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
        
        # Save cache by removing ode_solver temporarily to avoid pickling issues
        ode_solver = getattr(solver, "ode_solver", None)
        temp_solver_path = solver_cache_path.with_suffix(".tmp")
        try:
            if hasattr(solver, "ode_solver"):
                del solver.ode_solver
            with open(temp_solver_path, "wb") as f:
                pickle.dump(solver, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_solver_path, solver_cache_path)
            print(f"Subsonic similarity solver saved successfully to {solver_cache_path}.")
        except Exception as e:
            print(f"Failed to save subsonic solver cache safely: {e}. Traceback:")
            traceback.print_exc()
            if temp_solver_path.exists():
                temp_solver_path.unlink()
        finally:
            if ode_solver is not None:
                solver.ode_solver = ode_solver

    return solver


def get_shock_similarity_solver(case, case_label: str):
    """
    Load or solve PistonShock similarity solver.
    Saves to {case_label}_shock_similarity_solver.pkl.
    """
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver_cache_path = cache_dir / f"{case_label}_shock_similarity_solver.pkl"

    solver = None
    if USE_CACHE and solver_cache_path.exists():
        print(f"Loading cached shock similarity solver from {solver_cache_path}...")
        try:
            with open(solver_cache_path, "rb") as f:
                solver = CustomUnpickler(f).load()
            if solver is not None:
                solver.ode_solver = scipy.integrate.ode(solver.fode).set_integrator(solver.ode_scheme)
            print("Shock similarity solver loaded successfully.")
        except Exception as e:
            print(f"Failed to load shock similarity solver cache: {e}. Traceback:")
            traceback.print_exc()
            print("Re-solving shock similarity ODEs...")

    if solver is None:
        print("Solving shock similarity ODEs (finding xsi_s via root-finding)...")
        tau = float(case.tau or 0.0)
        if case.P0_Barye is not None:
            p0s = _ns_amplitude_rescale(float(case.P0_Barye), tau)
            tau_s = tau
        else:
            print("P0_Barye is None (Marshak case). Solving SubsonicHeatWave first to find piston pressure p0 and tau...")
            sub_solver = get_sub_similarity_solver(case, case_label)
            p0s = sub_solver.Pf * (sub_solver.A**sub_solver.a3) * (sub_solver.B**sub_solver.b3)
            tau_s = sub_solver.c3
        
        rho0 = case.rho0
        omega = float(getattr(case, "omega", 0.0))
        gamma_shock = float(case.r) + 1.0

        solver = PistonShock(
            rho0=rho0,
            omega=omega,
            p0=p0s,
            tau=tau_s,
            gamma=gamma_shock,
        )
        # Save cache by removing ode_solver temporarily to avoid pickling issues
        ode_solver = getattr(solver, "ode_solver", None)
        temp_solver_path = solver_cache_path.with_suffix(".tmp")
        try:
            if hasattr(solver, "ode_solver"):
                del solver.ode_solver
            with open(temp_solver_path, "wb") as f:
                pickle.dump(solver, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_solver_path, solver_cache_path)
            print(f"Shock similarity solver saved successfully to {solver_cache_path}.")
        except Exception as e:
            print(f"Failed to save shock solver cache safely: {e}. Traceback:")
            traceback.print_exc()
            if temp_solver_path.exists():
                temp_solver_path.unlink()
        finally:
            if ode_solver is not None:
                solver.ode_solver = ode_solver

    return solver


def get_ablation_solver(case, case_label: str):
    """
    Load or build patched AblationSolver.
    Saves to {case_label}_ablation_solver.pkl.
    """
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver_cache_path = cache_dir / f"{case_label}_ablation_solver.pkl"

    ablation_solver = None
    if USE_CACHE and solver_cache_path.exists():
        print(f"Loading cached AblationSolver from {solver_cache_path}...")
        try:
            with open(solver_cache_path, "rb") as f:
                ablation_solver = CustomUnpickler(f).load()
            if ablation_solver is not None:
                if hasattr(ablation_solver.heat_solver, "fode"):
                    ablation_solver.heat_solver.ode_solver = scipy.integrate.ode(ablation_solver.heat_solver.fode).set_integrator(ablation_solver.heat_solver.ode_scheme)
                if hasattr(ablation_solver.shock_solver, "fode"):
                    ablation_solver.shock_solver.ode_solver = scipy.integrate.ode(ablation_solver.shock_solver.fode).set_integrator(ablation_solver.shock_solver.ode_scheme)
            print("Loaded AblationSolver successfully from cache.")
        except Exception as e:
            print(f"Failed to load ablation solver cache: {e}. Traceback:")
            traceback.print_exc()
            print("Re-building solver...")

    if ablation_solver is None:
        print("Building AblationSolver...")
        kwargs = _ablation_kwargs_from_case(case)
        ablation_solver = AblationSolver(**kwargs)

        # Save cache by removing ode_solvers temporarily to avoid pickling issues
        heat_ode = getattr(ablation_solver.heat_solver, "ode_solver", None)
        shock_ode = getattr(ablation_solver.shock_solver, "ode_solver", None)
        temp_solver_path = solver_cache_path.with_suffix(".tmp")
        try:
            if hasattr(ablation_solver.heat_solver, "ode_solver"):
                del ablation_solver.heat_solver.ode_solver
            if hasattr(ablation_solver.shock_solver, "ode_solver"):
                del ablation_solver.shock_solver.ode_solver
            with open(temp_solver_path, "wb") as f:
                pickle.dump(ablation_solver, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_solver_path, solver_cache_path)
            print(f"AblationSolver saved successfully to {solver_cache_path}.")
        except Exception as e:
            print(f"Failed to save ablation solver cache safely: {e}. Traceback:")
            traceback.print_exc()
            if temp_solver_path.exists():
                temp_solver_path.unlink()
        finally:
            if heat_ode is not None:
                ablation_solver.heat_solver.ode_solver = heat_ode
            if shock_ode is not None:
                ablation_solver.shock_solver.ode_solver = shock_ode

    return ablation_solver
