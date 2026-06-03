# scratch/test_times.py
import sys
from pathlib import Path
import scipy.integrate
import numpy as np

if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None) or scipy.integrate.trapz

# Ensure project imports work
_REPO_ROOT = Path(r"c:\Users\TLP-001\Documents\GitHub\project3_code")
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MENAHEM_DIR = _REPO_ROOT / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.menahem_new.ablation_solver_og import AblationSolver

def analyze_preset(preset_name, label):
    print(f"\n=== Analyzing {preset_name} ({label}) ===")
    case, config = get_preset(preset_name)
    from dataclasses import replace
    config = replace(config, show_plot=False, show_slider=False)
    
    # Run simulation
    _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)
    t_max = max(history.t)
    print(f"Simulation t_max: {t_max*1e9:.4f} ns (t_sec_end = {case.t_sec_end:.4e})")
    
    # Setup ablation solver
    ablation_solver = AblationSolver(
        Tb=case.T0_Kelvin,
        tau=case.tau,
        g=case.g_Kelvin,
        alpha=case.alpha,
        lambdap=case.lambda_,
        f_heat=case.f_Kelvin,
        beta_heat=case.beta_Rosen,
        mu_heat=case.mu,
        gamma_heat=case.r + 1.0,
        rho0=case.rho0,
        omega=case.omega if hasattr(case, "omega") else 0.0,
        f_shock=case.f_Kelvin,
        beta_shock=case.beta_Rosen,
        mu_shock=case.mu,
        gamma_shock=case.r + 1.0
    )
    
    # Analyze times
    target_times = [0.5 * t_max, 0.75 * t_max, t_max]
    for t_target in target_times:
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]
        
        m_f = ablation_solver.heat_solver.ablated_mass(time=t_actual)
        m_s = ablation_solver.shock_solver.shocked_mass(time=t_actual)
        grid_m_max = m_sim[-1]
        
        print(f"t = {t_actual*1e9:.4f} ns:")
        print(f"  Ablated mass (heat front) m_f : {m_f:.6e} g/cm^2 ({m_f*1e3:.4f} mg/cm^2)")
        print(f"  Shock front m_s               : {m_s:.6e} g/cm^2 ({m_s*1e3:.4f} mg/cm^2)")
        print(f"  Grid Max Mass                 : {grid_m_max:.6e} g/cm^2 ({grid_m_max*1e3:.4f} mg/cm^2)")
        if m_f > grid_m_max:
            print("  --> WARNING: heat front m_f has EXITED the grid!")
        if m_s > grid_m_max:
            print("  --> WARNING: shock front m_s has EXITED the grid!")
        
        # Check density/pressure at shock front in simulation
        sim_rho = history.rho[idx_sim]
        sim_p = history.p[idx_sim]
        sim_u = history.u[idx_sim]
        
        # Find index in sim corresponding to m_f and m_s
        idx_mf = np.argmin(np.abs(m_sim - m_f))
        idx_ms = np.argmin(np.abs(m_sim - m_s))
        
        print(f"  At m_f (idx={idx_mf}): sim_rho={sim_rho[idx_mf]:.3f}, sim_p={sim_p[idx_mf]/1e12:.3f} MBar, sim_u={sim_u[idx_mf]/1e5:.3f} km/s")
        print(f"  At m_s (idx={idx_ms}): sim_rho={sim_rho[idx_ms]:.3f}, sim_p={sim_p[idx_ms]/1e12:.3f} MBar, sim_u={sim_u[idx_ms]/1e5:.3f} km/s")
        print(f"  Grid end (idx={len(m_sim)-1}): sim_rho={sim_rho[-1]:.3f}, sim_p={sim_p[-1]/1e12:.3f} MBar, sim_u={sim_u[-1]/1e5:.3f} km/s")

if __name__ == "__main__":
    analyze_preset("fig_9_comparison", "const_S")
    analyze_preset("fig_10_comparison", "const_P_shock")
