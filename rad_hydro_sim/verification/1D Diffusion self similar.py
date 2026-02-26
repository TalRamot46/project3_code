import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Use project-wide plot style (consistent fonts, units)
_REPO_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
from project_3.rad_hydro_sim.plotting import mpl_style  # noqa: F401 - apply project style
import tqdm
T_material_0_Kelvin = 300.0
T_bath_Kelvin = 1500000  # boundary temperature (K)
eV_joule = 1.60218e-19  # J/eV
erg_per_joule = 1.0e7   # erg/J
eV = eV_joule * erg_per_joule  # erg 
Hev=1.0e2 * eV  # erg

# eV = 1.60218e-12 erg
# Hev = 1.60218e-10 erg

k_B_joule = 1.38065e-23  # J/K
k_B = k_B_joule * erg_per_joule  # erg/K
KELVIN_PER_HEV = Hev / k_B  # in HeV

# k_B = 1.3805e-16 erg/K
# K_per_Hev = 1.1605e6 K/HeV
a_kelvin = 7.5646e-15    # radiation constant
a_hev = a_kelvin * (KELVIN_PER_HEV**4)  # radiation constant in HeV units

T_bath_hev = T_bath_Kelvin / KELVIN_PER_HEV 
T_bath = T_bath_Kelvin
T_material_0_hev = T_material_0_Kelvin / KELVIN_PER_HEV


# -----------------------------
# Parameters
# -----------------------------
# changing all simulation constants to cgs units.
c = 3*10**10        # speed of light (cm/s)
chi = 1000       # global multiplier χ  (new model)
kind_of_D_face = "arithmetic"  # "harmonic", "arithmetic", "geometric"
# self similarity model fudge factors (Kelvin-based Rosen coefficients)
f_Kelvin = 3.4 * 10**13 / (KELVIN_PER_HEV**1.6 * 19.32**0.14)         # specific energy coefficient (legacy name, Kelvin-based)
g_Kelvin = 1 / 7200 / (KELVIN_PER_HEV**1.6 * 19.32**0.14)              # opacity coefficient (legacy name, Kelvin-based)

alpha = 1.5       # opacity exponent
gamma = 1.6       # beta exponent
lambda_param = 0.2
mu = 0.14
rho = 19.32      # initial density (g/cm^3)

# Right boundary: T_right_Kelvin=0 -> vacuum (E_right=0); T_right_Kelvin>0 -> cold sink
T_right_Kelvin = 300.0  # default: 300 K cold sink; set 0 for vacuum (match Rad-Hydro)

# -----------------------------
# Grid and time step
# -----------------------------
L = 0.0003  # domain length (cm)
Nz = 1000
z = np.linspace(0.0, L, Nz)
dz = z[1] - z[0]

t_final_sec = 2e-9 
dt_sec = 5e-15
t_final_ns = t_final_sec * 10**9
dt_ns = dt_sec * 10**9

t_final = t_final_sec
dt = dt_sec 

Nt = int(t_final / dt) + 1

####################################
# realive timestep
####################################
def update_dt_relchange(dt, E, Eold, UR, URold, *, dtfac=0.05, growth_cap=1.1):
    """
    Adaptive dt based on max relative change in E and UR.

    dtfac: target relative change per step (~0.05 means ~5%)
    growth_cap: allow dt to increase by at most 10% per step (1.1)
    """
    # Protect from division by tiny numbers
    E_min = np.max(np.abs(E)) * 1e-3 + 1e-30
    dE = np.max(np.abs(E - Eold) / (np.abs(E) + E_min))

    U_min = np.max(np.abs(UR)) * 1e-3 + 1e-30
    dU = np.max(np.abs(UR - URold) / (np.abs(UR) + U_min))

    # Avoid blow-ups if change is ~0
    dE = max(dE, 1e-16)
    dU = max(dU, 1e-16)

    dttag1 = dt / dE * dtfac
    dttag2 = dt / dU * dtfac

    dt_new = min(dttag1, dttag2, growth_cap * dt)
    return dt_new, dE, dU


# -----------------------------
# Simulation unit system
# -----------------------------
CGS = "cgs"
HEV_NS = "hev|ns"
simulation_unit_system = CGS  # CGS or HEV_NS
if simulation_unit_system == CGS:
    T_material_0 = T_material_0_Kelvin
    T_bath = T_bath_Kelvin
    a = a_kelvin
    dt = dt_sec
    t_final = t_final_sec
    c = 3e10  # speed of light in cm/s
elif simulation_unit_system == HEV_NS:
    T_material_0 = T_material_0_hev
    T_bath = T_bath_hev
    a = a_hev
    dt = dt_ns
    t_final = t_final_ns
    c = 3e1  # speed of light in cm/ns

# -----------------------------
# Material model hooks (T in Kelvin!)
# -----------------------------
def sigma_of_T(T): 
    """Opacity σ(T). Placeholder: 1/σ(T) = g * T^α * ρ^(-λ-1)."""
    if simulation_unit_system == CGS:
        return 1.0 / (g_Kelvin * T ** alpha * rho**(-lambda_param - 1))
    elif simulation_unit_system == HEV_NS:
        return 1.0 / (g_Kelvin * T ** alpha * rho**(-lambda_param - 1))


def beta_of_T(T): 
    """β(T). Placeholder used in your code."""
    if simulation_unit_system == CGS:
        Cv_m = f_Kelvin * gamma * T ** (gamma - 1) * rho ** (-mu + 1)  # material specific heat
        Cv_R = 4.0 * a * T ** 3  # radiation specific heat
        return Cv_R/Cv_m
    elif simulation_unit_system == HEV_NS:
        return ((4.0 * a * rho ** (mu - 1)) / (f_Kelvin * gamma)) * T ** (4.0 - gamma)

def D_of_T(T):
    """Diffusion coefficient D(T) = c / (3 σ(T))."""
    # uses sigma_of_T so no need to convert T from Kelvin to HeV here.
    return c / (3.0 * sigma_of_T(T))


def U_m_of_T(UR):
    """Material internal energy U_m(T). Placeholder: U_m(T) = f*T^gamma*ρ^(-mu+1)."""
    if simulation_unit_system == CGS:
        T = (UR / a) ** 0.25
        return f_Kelvin * T ** gamma * rho ** (-mu + 1)


# -----------------------------
# Core implicit step
# -----------------------------
def implicit_step_self_similar_model(E, UR, *, tau=0.0, t=0.0, dt_local=None):
    """
    One backward-Euler step using:
      - variable D^n (harmonic face averages)
      - implicit coupling via A_i^n = beta_i^n*dt*chi*c*sigma_i^n
    Tridiagonal coefficients match your implementation.

    BCs:
      E_left  = a*(T_bath * t^tau)^4
      E_right = 0
    """
    if dt_local is None:
        dt_local = dt

    N = E.size
    n_int = N - 2  # solve for i=1..N-2

    # Boundary conditions
    # E_right: vacuum (0) when T_right_Kelvin=0; cold sink otherwise (match Rad-Hydro for verification)
    if simulation_unit_system == CGS:
        t_ns = t * 1e9  # convert to ns for BC calculation
        E_left = a * (T_bath * (t_ns**tau)) ** 4
        E_right = 0.0 if T_right_Kelvin <= 0 else a * (T_right_Kelvin ** 4)

    elif simulation_unit_system == HEV_NS:
        E_left = a * (T_bath * (t**tau)) ** 4
        E_right = 0.0 if T_right_Kelvin <= 0 else a * (T_right_Kelvin / KELVIN_PER_HEV) ** 4

    # Build D_i^n, beta_i^n, sigma_i^n from UR^n -> T^n
    Tn = (UR / a) ** 0.25
    Dn = D_of_T(Tn)
    betan = beta_of_T(Tn)
    sigman = sigma_of_T(Tn)

    # Harmonic face diffusion coefficients D_{i+1/2}^n (length N-1)
    #D_face = 2.0 * Dn[:-1] * Dn[1:] / (Dn[:-1] + Dn[1:] + 1e-20)
    if kind_of_D_face == "harmonic":
            D_face = 2.0 * Dn[:-1] * Dn[1:] / (Dn[:-1] + Dn[1:] + 1e-20)
    elif kind_of_D_face == "arithmetic":
        D_face = (Dn[:-1]+Dn[1:])/2
    elif kind_of_D_face == "geometric":
        D_face = np.sqrt(Dn[:-1] * Dn[1:])    # Convenience: A_i^n = beta_i^n*dt*chi*c*sigma_i^n
    A = betan * dt_local * chi * c * sigman
    coupling = chi * c * sigman / (1.0 + A)  # χ c σ / (1 + A)

    lower = np.zeros(n_int - 1)
    diag = np.zeros(n_int)
    upper = np.zeros(n_int - 1)
    rhs = np.zeros(n_int)

    for k in range(n_int):
        i = k + 1  # i=1..N-2
        D_imh = D_face[i - 1]  # i-1/2
        D_iph = D_face[i]      # i+1/2

        a_i = -D_imh / dz**2
        c_i = -D_iph / dz**2
        b_i = (1.0 / dt_local) + (D_imh + D_iph) / dz**2 + coupling[i]
        d_i = (E[i] / dt_local) + coupling[i] * UR[i]

        diag[k] = b_i
        rhs[k] = d_i
        if k > 0:
            lower[k - 1] = a_i
        if k < n_int - 1:
            upper[k] = c_i

    # BC contribution on first interior equation (i=1)
    D_1mh = D_face[0]
    a_1 = -D_1mh / dz**2
    rhs[0] -= a_1 * E_left

    # last BC term uses E_right=0 => no effect, kept for clarity
    # D_N2ph = D_face[-1]
    # c_N2 = -D_N2ph / dz**2
    # rhs[-1] -= c_N2 * E_right

    # Thomas algorithm
    for i in range(1, n_int):
        w = lower[i - 1] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        rhs[i] -= w * rhs[i - 1]

    E_inner = np.empty(n_int)
    E_inner[-1] = rhs[-1] / diag[-1]
    for i in range(n_int - 2, -1, -1):
        E_inner[i] = (rhs[i] - upper[i] * E_inner[i + 1]) / diag[i]

    E_new = E.copy()
    E_new[0] = E_left
    E_new[-1] = E_right
    E_new[1:-1] = E_inner

    UR_new = (A * E_new + UR) / (1.0 + A)
    return E_new, UR_new


# -----------------------------
# Simulation helpers
# -----------------------------
def init_state():
    """Initialize E, UR based on T_material_0."""
    E0 = a * T_material_0**4 * np.ones(Nz)
    UR0 = a * T_material_0**4 * np.ones(Nz)
    return E0, UR0


def run_time_loop(E, UR, tau, times_to_store, *, dtfac=0.05, dtmin=5e-15):
    """
    Run the PDE time loop; store Um, T at requested times.
    Returns stored_t, stored_Um, stored_T.
    """
    # add a progress bar using tqdm
    store_idx = 0

    stored_t, stored_Um, stored_T = [], [], []
    t = 0.0
    dt_local = dt  # start from your current dt
    pbar = tqdm.tqdm(total=t_final, desc="Simulating", unit="s", ncols=100)

    while t < t_final - 1e-30:
        # don't step past final time
        dt_local = min(dt_local, t_final - t)

        # force landing exactly on next store time (so you don't miss it)
        if store_idx < len(times_to_store):
            t_target = times_to_store[store_idx]
            if t < t_target <= t + dt_local:
                dt_local = t_target - t

        Eold = E.copy()
        URold = UR.copy()

        E, UR = implicit_step_self_similar_model(E, UR, tau=tau, t=t, dt_local=dt_local)

        t_next = t + dt_local

        Um = U_m_of_T(UR)
        T = (UR / a) ** 0.25
        # store if we hit store time
        if store_idx < len(times_to_store) and abs(t_next - times_to_store[store_idx]) < 0.5 * dt_local:
            if simulation_unit_system == CGS:
                stored_Um.append(np.array(Um).copy())
                stored_T.append(np.array(T).copy())
                stored_t.append(t_next * 1e9)  # ns
            else:
                stored_Um.append(np.array(Um).copy())
                stored_T.append(np.array(T).copy())
                stored_t.append(t_next)
            store_idx += 1

        # adapt dt for next step
        dt_new, dE, dU = update_dt_relchange(dt_local, E, Eold, UR, URold, dtfac=dtfac)
        if dtmin is not None:
            dt_new = max(dt_new, dtmin)

        pbar.update(t_next - t)
        t = t_next
        dt_local = dt_new
    pbar.close()
    return np.array(stored_t), np.array(stored_Um), np.array(stored_T)


def compute_front_half_and_energy(stored_Um, stored_T):
    """
    For each stored profile:
      - front position = first z where T < threshold*T_bath
      - total energy = ∫ Um dz
    """
    front_positions = []
    half_T_bath_positions = []
    total_energies = []

    for Ti, Ui in zip(stored_T, stored_Um):
        front_idx = np.argmax(np.abs(np.diff(Ti)))
        front_position = z[front_idx]
        front_positions.append(front_position)
        
        # Find first index where T drops below 0.8 * T_bath
        half_T_bath_idx = np.where(Ti < 0.8 * T_bath_hev)[0]
        if len(half_T_bath_idx) > 0:
            # Use the first occurrence
            idx = half_T_bath_idx[0]
            if idx > 0:
                # Linear interpolation between z[idx-1] and z[idx]
                half_T_bath_position = 0.5 * (z[idx] + z[idx - 1])
            else:
                half_T_bath_position = z[idx]
        else:
            # If no point is below 0.5 * T_bath, use the last point
            raise ValueError("No point found where T drops below 0.8 * T_bath.")
        
        half_T_bath_positions.append(half_T_bath_position)

        # hJ = 10^2 J
        # erg = 10^-7 J = 10^-9 hJ
        # 1 / cm^2 = 10^-2 / mm^2
        # => erg/cm^2 = 10^-11 hJ/mm^2
        # => integrate Um (erg/cm^3) over z (cm) gives erg/cm^2 = 10^-11 hJ/mm^2
        total_energy = np.trapezoid(Ui, z)
        total_energy_hJ_mm2 = total_energy * 1e-11  # convert erg/cm^2 to hJ/mm^2
        total_energies.append(total_energy_hJ_mm2)

    return np.array(front_positions), np.array(half_T_bath_positions), np.array(total_energies)

def fit_power_law(x, y):
    """
    Fit y ~ C * x^p using log10.
    Returns p, C.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    coeffs = np.polyfit(np.log(x), np.log(y), 1)  # slope, intercept
    
    p = coeffs[0]
    C = np.exp(coeffs[1])
    return p, C


def theoretical_front_position(t, rho_0, T_bath_hev, tau):
    if tau == 0.0:
        return 11.53e-4 * rho_0 ** (-1.03) * (T_bath_hev)**1.95 * t**0.5
    if tau == 0.1408:
        return 8.97e-4 * rho_0 ** (-1.03) * (T_bath_hev)**1.95 * t**0.775
    
def theoretical_total_energy(t, rho_0, T_bath_hev, tau):
    if tau == 0.0:
        return 0.29 * rho_0 ** (-0.17) * (T_bath_hev)**3.55 * t**0.5
    if tau == 0.1408:
        return 0.21 * rho_0 ** (-0.17) * (T_bath_hev)**3.55 * t
    
# extracting and plotting m0 vs tau from (t,mF) data of wavefronts for different taus.
def extract_constant_logfit(t, mF_or_energy_values, tau, rho_0, T_bath_hev, mu, lambda_param, alpha, gamma, mF_or_energy):
    """
    Log-space fit for m_F(t) = m0/e_0 * K * t^p.

    Returns
    -------
    const0 : float
    p_fit : float
    intercept : float   # b = ln(const0*K)
    """
    t = np.asarray(t, dtype=float)
    mF_or_energy_values = np.asarray(mF_or_energy_values, dtype=float)

    # keep only valid points (log requires positive)
    mask = (t > 0) & (mF_or_energy_values > 0) & np.isfinite(t) & np.isfinite(mF_or_energy_values)
    if mask.sum() < 2:
        raise ValueError("Need at least 2 valid points with t>0 and mF>0.")

    t = t[mask]
    mF_or_energy_values = mF_or_energy_values[mask]

    # known prefactor K
    if mF_or_energy == "mF":
        K = rho_0 ** ((mu - lambda_param) / 2.0) *  T_bath_hev ** ((4.0 - gamma + alpha) / 2.0)
        p_theory = (4.0 - gamma + alpha)*tau / 2.0 + 0.5
    elif mF_or_energy == "energy":
        K = rho_0 ** (-(mu + lambda_param) / 2.0) * T_bath_hev ** ((4.0 + gamma + alpha) / 2.0)
        p_theory = (4.0 + gamma + alpha)*tau / 2.0 + 0.5

    # y = C * t^p  =>  ln(y) = ln(C) + p * ln(t) = b + p * ln(t) | C = np.exp(b) in fit_power_law
    p_fit, C = fit_power_law(t, mF_or_energy_values)
    const0 = C / K
    return const0, p_fit, C


def plot_const0_vs_tau(
    data, *, rho_0, T_bath_hev, mu, lambda_param, alpha, gamma, mF_or_energy, show_plots=True
):
    taus = []
    const0s = []
    for tau, (t, mF_or_energy_values) in data.items():
        const0, _, _ = extract_constant_logfit(
            t, mF_or_energy_values, tau=tau,
            rho_0=rho_0, T_bath_hev=T_bath_hev, mu=mu, lambda_param=lambda_param, alpha=alpha, gamma=gamma,
            mF_or_energy=mF_or_energy
        )
        taus.append(tau)
        const0s.append(const0)

    taus = np.array(taus)
    const0s = np.array(const0s)
    const0_theory = {"mF": (11.53e-4, 8.97e-4), "energy": (0.29, 0.21)}[mF_or_energy]
    const0_name = {"mF": r"$m_0$", "energy": r"$e_0$"}[mF_or_energy]

    plt.figure(figsize=(6, 4))
    plt.scatter(taus, const0s, marker='x', color='black', label=f"Extracted {const0_name}")
    plt.scatter(taus, const0_theory, marker='x', color='green', label=f"Theoretical {const0_name}")
    plt.legend()
    plt.xlabel(r"$\tau$")
    plt.ylabel(const0_name)
    plt.title(r"Extracted " + const0_name + r" vs $\tau$")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("1D Diffusion self similar in gold/figures/const0_vs_tau", exist_ok=True)
    plt.savefig(f"1D Diffusion self similar in gold/figures/const0_vs_tau/{mF_or_energy}.png")
    if show_plots:
        plt.show()
    else:
        plt.close('all')

# -----------------------------
# Plotting helpers
# -----------------------------
# plot temperature profiles, front positions, total energies for all ts in stored_t - in 3 different functions.
# For each function, show the plot & save it to a file.
# Also fit power laws to front positions and total energies.
def plot_front_positions(stored_t, front_positions, tau):
    p_front, C_front = fit_power_law(stored_t, front_positions)
    print(f"[tau={tau}] Front position fits: x_f(t) ~ {C_front:.3e} * t^{p_front:.3f}")
    theoretical_points = [theoretical_front_position(ti, rho, T_bath_hev, tau) for ti in stored_t]
    stdev_percent = np.mean(np.abs((front_positions[10:] - theoretical_points[10:]) / theoretical_points[10:])) * 100
    print(f"[tau={tau}] Standard deviation from theoretical: {stdev_percent:.3e}")
    plt.figure(figsize=(12, 4))

    # fit data to theoretical
    plt.subplot(1, 2, 1)
    plt.plot(stored_t, front_positions, marker="x", color='black', linestyle="None", label="Simulated Front Position")
    plt.plot(stored_t, C_front * np.array(stored_t) ** p_front, label=f"Log Fit: t^{p_front:.3f}")
    plt.plot(stored_t, theoretical_points, label="Theoretical", linestyle="--")
    
    plt.xlabel("Time (ns)")
    plt.ylabel("Wave Front Position (cm)")
    plt.title(f"Wave Front Position vs Time (tau={tau}) (linear scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    plt.loglog(stored_t, front_positions, marker="x", color='black', linestyle="None", label="Simulated Front Position")
    plt.loglog(stored_t, C_front * np.array(stored_t) ** p_front, label=f"Fit: t^{p_front:.3f}")
    plt.loglog(stored_t, theoretical_points, label="Theoretical", linestyle="--")
    plt.xlabel("Time (ns)")
    plt.ylabel("Wave Front Position (cm)")
    plt.title(f"Wave Front Position vs Time (tau={tau}) (log-log scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # annotate the std on the plot, make it look good with a box
    plt.annotate(f"Std Dev from Theoretical: {stdev_percent:.2f} %", xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # make directory if not exists
    os.makedirs(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}", exist_ok=True)
    plt.savefig(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}/front_position.png")

def plot_half_T_bath_positions(stored_t, half_T_bath_positions, tau):
    p_front, C_front = fit_power_law(stored_t, half_T_bath_positions)
    print(f"[tau={tau}] Half position fits: x_f(t) ~ {C_front:.3e} * t^{p_front:.3f}")
    plt.figure(figsize=(12, 4))

    # fit data to theoretical
    plt.subplot(1, 2, 1)
    plt.plot(stored_t, half_T_bath_positions, marker="x", color='black', linestyle="None", label=r"Position where $T(z,t) = 0.5 T_{\mathrm{bath}}$")
    plt.plot(stored_t, C_front * np.array(stored_t) ** p_front, label=f"Log Fit: t^{p_front:.3f}")
    plt.plot(stored_t, [theoretical_front_position(ti, rho, T_bath_hev, tau) for ti in stored_t], label="Theoretical", linestyle="--")
    
    plt.xlabel("Time (ns)")
    plt.ylabel("Wave Half Position (cm)")
    plt.title(f"Wave Half Position vs Time (tau={tau}) (linear scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    plt.loglog(stored_t, half_T_bath_positions, marker="x", color='black', linestyle="None", label=r"Position where $T(z,t) = 0.5 T_{\mathrm{bath}}$")
    plt.loglog(stored_t, C_front * np.array(stored_t) ** p_front, label=f"Fit: t^{p_front:.3f}")
    plt.loglog(stored_t, [theoretical_front_position(ti, rho, T_bath_hev, tau) for ti in stored_t], label="Theoretical", linestyle="--")
    plt.xlabel("Time (ns)")
    plt.ylabel("Wave Half Position (cm)")
    plt.title(f"Wave Half Position vs Time (tau={tau}) (log-log scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    
    # make directory if not exists
    os.makedirs(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}", exist_ok=True)
    plt.savefig(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}/half_position.png")


def plot_energies(stored_t, total_energies, tau):
    p_energy, C_energy = fit_power_law(stored_t, total_energies)
    print(f"[tau={tau}] Total energy fits:   E(t)   ~ {C_energy:.3e} * t^{p_energy:.3f}")
    theoretical_points = [theoretical_total_energy(ti, rho, T_bath_hev, tau) for ti in stored_t]
    stdev_percent = np.mean(np.abs((total_energies - theoretical_points) / theoretical_points)) * 100
    print(f"[tau={tau}] Standard deviation from theoretical: {stdev_percent:.3e}")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(stored_t, total_energies, marker="x", color='black', linestyle="None", label="Simulated Material Energy")
    plt.plot(stored_t, C_energy * np.array(stored_t) ** p_energy, label=f"Fit: t^{p_energy:.3f}")
    plt.plot(stored_t, theoretical_points, label="Theoretical", linestyle="--")
    plt.xlabel("Time (ns)")
    plt.ylabel("Total Energy (hJ/mm^2)")
    plt.title(f"Total Energy vs Time (tau={tau}) (linear scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.loglog(stored_t, total_energies, marker="x", color='black', linestyle="None", label="Simulated Material Energy")
    plt.loglog(stored_t, C_energy * np.array(stored_t) ** p_energy, label=f"Fit: t^{p_energy:.3f}")
    plt.loglog(stored_t, [theoretical_total_energy(ti, rho, T_bath_hev, tau) for ti in stored_t], label="Theoretical", linestyle="--")
    plt.xlabel("Time (ns)")
    plt.ylabel("Total Energy (hJ/mm²)")
    plt.title(f"Total Energy vs Time (tau={tau}) (log-log scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # annotate the std on the plot, make it look good with a box
    plt.annotate(f"Std Dev from Theoretical: {stdev_percent:.2f} %", xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # make directory if not exists
    os.makedirs(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}", exist_ok=True)
    plt.savefig(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}/total_energy.png")


def plot_temperature_profiles(stored_t, stored_T, tau):
    colors = plt.cm.viridis(np.linspace(0, 1, len(stored_t)))
    plt.figure(figsize=(10, 6))
    for Ti, ti, color in zip(stored_T, stored_t, colors):
        plt.plot(z, Ti, label=f"t={ti:.1e} ns", color=color)
    plt.xlabel("z (cm)")
    plt.ylabel(r"$T(z,t)/T_{\mathrm{bath}}$")
    plt.grid(True)
    plt.legend()
    plt.title(f"Temperature profiles over time (tau={tau})")
    plt.tight_layout()
    # make directory if not exists
    os.makedirs(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}", exist_ok=True)
    plt.savefig(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}/temperature_profiles.png")



# -----------------------------
# One-call case runner (removes repeated code)
# -----------------------------
def run_case(*, tau, times_to_store, reset_initial_conditions=True):
    """
    Runs:
      1) init state
      2) time loop
      3) front/energy extraction
      4) plots
    """
    if reset_initial_conditions:
        E, UR = init_state()
    else:
        # If you ever want to continue from previous run, you can supply state externally
        E, UR = init_state()

    stored_t, stored_Um, stored_T = run_time_loop(E, UR, tau=tau, times_to_store=times_to_store)
    # store values in csv files
    # use pandas to save 2D arrays
    os.makedirs(f"data/data_tau_{tau}", exist_ok=True)
    import pandas as pd
    # sorted_T is a 2D array where each row is a stored_T at a time step. 
    # Create a csv file where each column is a time step and each row is the temperature profile (for positions) at that time step.
    # make sure that this is saved correctly.

    # make directory if not exists
    os.makedirs(f"1D Diffusion self similar in gold/data/data_tau_{tau}", exist_ok=True)
    df_sorted_T = pd.DataFrame(stored_T)
    df_sorted_T.to_csv(f"1D Diffusion self similar in gold/data/data_tau_{tau}/stored_T.csv", header=False, index=False)
    df_sorted_Um = pd.DataFrame(stored_Um)
    df_sorted_Um.to_csv(f"1D Diffusion self similar in gold/data/data_tau_{tau}/stored_Um.csv", header=False, index=False)
    df_stored_t = pd.DataFrame(stored_t)
    df_stored_t.to_csv(f"1D Diffusion self similar in gold/data/data_tau_{tau}/stored_time.csv", header=False, index=False)

    return {
        "tau": tau,
        "stored_t": np.array(stored_t),
        "stored_T": stored_T,
        "stored_Um": stored_Um,
    }

def plot_front_positions_and_energies(tau, show_plots=True):
    # read back the stored values from csv files using pandas
    import pandas as pd
    stored_T = pd.read_csv(f"1D Diffusion self similar in gold\data\data_tau_{tau}\stored_T.csv", header=None).to_numpy()
    stored_Um = pd.read_csv(f"1D Diffusion self similar in gold\data\data_tau_{tau}\stored_Um.csv", header=None).to_numpy()
    stored_t = pd.read_csv(f"1D Diffusion self similar in gold\data\data_tau_{tau}\stored_time.csv", header=None).to_numpy().flatten()
    front_positions, half_T_bath_positions, total_energies = compute_front_half_and_energy(stored_Um, stored_T)

    # make subfolder if not exists for saving plots of tau
    os.makedirs(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}", exist_ok=True)
    plot_temperature_profiles(stored_t, stored_T, tau)
    plot_front_positions(stored_t, front_positions, tau)
    plot_half_T_bath_positions(stored_t, half_T_bath_positions, tau)
    plot_energies(stored_t, total_energies, tau)
    if show_plots:
        plt.show()
    else:
        plt.close('all')

    return front_positions, total_energies


def run_fit_mF_and_energy(data, show_plots=True):
    plot_const0_vs_tau(
        {tau: (t, mF) for tau, (t, mF, _) in data.items()},
        rho_0=rho, T_bath_hev=T_bath_hev, mu=mu, lambda_param=lambda_param, alpha=alpha, gamma=gamma,
        mF_or_energy="mF",
        show_plots=show_plots
    )
    plot_const0_vs_tau(
        {tau: (t, energy) for tau, (t, _, energy) in data.items()},
        rho_0=rho, T_bath_hev=T_bath_hev, mu=mu, lambda_param=lambda_param, alpha=alpha, gamma=gamma,
        mF_or_energy="energy",
        show_plots=show_plots
    )

    
def simulate():
    times_to_store = np.logspace(np.log10(t_final * 0.04), np.log10(t_final), 20)
    taus = [0.0, 0.1408]
    data = {}
    show_plots = True

    results_tau0 = run_case(tau=taus[0], times_to_store=times_to_store, reset_initial_conditions=True)
    front_positions_tau0, total_energies_tau0 = plot_front_positions_and_energies(taus[0], show_plots=show_plots)
    data[taus[0]] = (results_tau0["stored_t"], front_positions_tau0, total_energies_tau0)
    results_tau_flux = run_case(tau=taus[1], times_to_store=times_to_store, reset_initial_conditions=True)
    front_positions_tau_flux, total_energies_tau_flux = plot_front_positions_and_energies(taus[1], show_plots=show_plots)
    data[taus[1]] = (results_tau_flux["stored_t"], front_positions_tau_flux, total_energies_tau_flux)
    run_fit_mF_and_energy(data, show_plots=show_plots)

def data_for_comparison():
    x_vals = np.array([0, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 15, 20])
    E_data = {
        0.01: np.array([
            1.00000, 0.40828, 0.06715, 0.00186, 0.00004, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000
        ]),
        0.10: np.array([
            1.00000, 0.80718, 0.55269, 0.24541, 0.08626, 0.02354, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000
        ]),
        1.00: np.array([
            1.00000, 0.91829, 0.80444, 0.63438, 0.49043, 0.37014, 0.04455, 0.00019, 0.00000, 0.00000, 0.00000, 0.00000
        ]),
        10.00: np.array([
            1.00000, 0.97428, 0.93620, 0.87285, 0.81040, 0.74844, 0.42723, 0.12130, 0.02372, 0.00334, 0.00003, 0.00000
        ]),
    }
    U_data = {
        0.01: np.array([
            0.00895, 0.00261, 0.00033, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000
        ]),
        0.10: np.array([
            0.09412, 0.06448, 0.03497, 0.01087, 0.00280, 0.00059, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000
        ]),
        1.00: np.array([
            0.63120, 0.55792, 0.46201, 0.33179, 0.23375, 0.16066, 0.01130, 0.00002, 0.00000, 0.00000, 0.00000, 0.00000
        ]),
        10.00: np.array([
            0.99995, 0.97259, 0.93211, 0.86487, 0.79877, 0.73343, 0.40145, 0.10437, 0.01850, 0.00236, 0.00002, 0.00000
        ]),
    }
    return x_vals, E_data, U_data

def compare_with_linear_results():
    # changing global parameters for comparison
    global tau, alpha, lambda_param, g_Kelvin, f_Kelvin, mu, gamma, L, dt, t_final, Nz, z, dz, T_bath_Kelvin, T_bath, chi,\
            T_material_0, eps, sigma, show_plots, Nt, z, dz
    show_plots = False
    x_vals, E_data, U_data = data_for_comparison()

    # setting self-similar model parameters for comparison with linear theory
    tau = 0.0
    alpha = 0
    lambda_param = -1
    g_Kelvin = 1
    f_Kelvin = a_hev # a_hev = a_kelvin / (K_per_Hev**4)
    mu = 1
    gamma = 4

    # Boundary and initial conditions
    c = 3.0e10        # speed (cm/s or arbitrary)
    sigma = 1      # opacity Σ (1/cm)
    eps = 1        # ε in your equation
    T_material_0 = 300      # initial matter "temperature" or energy U
    T_bath = 10000     # boundary temperature at left end
    L = 10          # length of the slab (cm)
    Nz = 5000     # number of spatial points
    t_final = (1.0e-9)/3  # final time (s)
    a = 7.5646e-15  # radiation constant
    
    L = 10  # domain length (cm)
    Nz = 5000     # number of spatial points
    z = np.linspace(0, L, Nz)
    dz = z[1] - z[0]
    t_final_sec = (1.0e-9)/3  # final time (s)
    dt_sec = 1.0e-12 # initial guess for time step (s)
    Nt = int(t_final / dt) + 1  # number of time steps (dimensional)
    chi = 1.0  # coupling coefficient
    t_final = t_final_sec
    dt = dt_sec 

    import pandas as pd
    times_to_store = np.array([0.01, 0.1, 1.0, 10.0]) / c  # in s
    run_case(tau=tau, times_to_store=times_to_store, reset_initial_conditions=True)
    stored_U = pd.read_csv(f"1D Diffusion self similar in gold\data\data_tau_{tau}\stored_Um.csv", header=None).to_numpy()
    stored_t = pd.read_csv(f"1D Diffusion self similar in gold\data\data_tau_{tau}\stored_time.csv", header=None).to_numpy().flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(stored_t)))
    plt.figure(figsize=(10, 6))
    for Ui, ti, color in zip(stored_U, times_to_store, colors):
        plt.scatter(x_vals/np.sqrt(3), U_data[ti*c], label=f"Linear Theory t={ti:.1e} s", marker='x', color=color)
        plt.plot(z, Ui / (a * T_bath**4), label=f"t={ti:.1e} s", color=color)
    plt.xlabel("z (cm)")
    plt.ylabel(r"$T(z,t)/T_{\mathrm{bath}}$")
    plt.xscale("log")
    plt.xlim(1e-2, L)
    plt.grid(True)
    plt.legend()
    plt.title(f"Temperature profiles over time (tau={tau})")
    plt.tight_layout()
    # make directory if not exists
    os.makedirs(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}", exist_ok=True)
    plt.savefig(f"1D Diffusion self similar in gold/figures/plots_tau_{tau}/comparison.png")

# -----------------------------
# Main
# -----------------------------
def main():
    simulate()

if __name__ == "__main__":
    main()
