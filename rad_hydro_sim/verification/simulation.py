from parameters import *
# -----------------------------
# Material model hooks (T in Kelvin!)
# -----------------------------
def sigma_of_T(T): 
    """Opacity σ(T). Placeholder: 1/σ(T) = g * T^α * ρ^(-λ-1)."""
    if simulation_unit_system == CGS:
        T_Hev = T / K_per_Hev  # convert to HeV for sigma_of_T
        #print(f"T_Hev inside sigma_of_T: {T_Hev}")
        return 1.0 / (g * T_Hev ** alpha * rho**(-lambda_param - 1))
    elif simulation_unit_system == HEV_NS:
        return 1.0 / (g * T ** alpha * rho**(-lambda_param - 1))


def beta_of_T(T): 
    """β(T). Placeholder used in your code."""
    if simulation_unit_system == CGS:
        Cv_m = f * beta * T ** (beta - 1) * rho ** (-mu + 1)  # material specific heat
        Cv_R = 4.0 * a * T ** 3  # radiation specific heat
        return Cv_R/Cv_m * K_per_Hev ** beta
    elif simulation_unit_system == HEV_NS:
        return ((4.0 * a * rho ** (mu - 1)) / (f * beta)) * T ** (4.0 - beta)

def D_of_T(T):
    """Diffusion coefficient D(T) = c / (3 σ(T))."""
    # uses sigma_of_T so no need to convert T from Kelvin to HeV here.
    return c / (3.0 * sigma_of_T(T))


def U_m_of_T(UR):
    """Material internal energy U_m(T). Placeholder: U_m(T) = f*T^beta*ρ^(-mu+1)."""
    if simulation_unit_system == CGS:
        T_Hev = (UR / a) ** 0.25 / K_per_Hev
        return f * T_Hev ** beta * rho ** (-mu + 1)
    elif simulation_unit_system == HEV_NS:
        T_Hev = (UR / a) ** 0.25
        return f * T_Hev ** beta * rho ** (-mu + 1)

# -----------------------------
# Core implicit step
# -----------------------------
# run with numba for speed
def implicit_step_self_similar_model(E, UR, *, t=0.0, dt_local=None, marshak_boundary=False):
    """
    One backward-Euler step using:
      - variable D^n (harmonic face averages)
      - implicit coupling via A_i^n = beta_i^n*dt*chi*c*sigma_i^n
    Tridiagonal coefficients match your implementation.

    BCs:
      E_left  = a*(T_D)^4 or marshak BC
      E_right = 0
    """
    if dt_local is None:
        dt_local = dt
    if marshak_boundary: 
        N = E.size
        n_int = N - 1  # solve for i=1..N-2
    else:
        N = E.size
        n_int = N - 2  # solve for i=1..N-2

    # Boundary conditions
    if simulation_unit_system == CGS:
        t_ns = t * 1e9  # convert to ns for BC calculation
        E_left = a * (K_per_Hev * get_TD(t_ns, t_array_TD, T_array_TD)) ** 4
        E_right = a * (300) ** 4
    elif simulation_unit_system == HEV_NS:
        E_left = a * (get_TD(t, t_array_TD, T_array_TD)) ** 4
        E_right = a * (300/K_per_Hev) ** 4

    T_left = (E_left / a) ** 0.25
    # Build D_i^n, beta_i^n, sigma_i^n from UR^n -> T^n
    # UR[-1] = 10**-5  # enforce UR_N=0 for stability
    Tn = (UR / a) ** 0.25
    Dn = D_of_T(Tn)
    betan = beta_of_T(Tn)
    sigman = sigma_of_T(Tn)
    TR = (E / a) ** 0.25

    #print(f"T_bath: {Tn[0]/K_per_Hev}",f"sigman[0]: {sigman[0]}")

    # Harmonic face diffusion coefficients D_{i+1/2}^n (length N-1)
    if kind_of_D_face == "harmonic":
        D_face = 2.0 * Dn[:-1] * Dn[1:] / (Dn[:-1] + Dn[1:] + 1e-20)
    elif kind_of_D_face == "arithmetic":
        D_face = (Dn[:-1]+Dn[1:])/2
    elif kind_of_D_face == "geometric":
        D_face = np.sqrt(Dn[:-1] * Dn[1:])
    # Convenience: A_i^n = beta_i^n*dt*chi*c*sigma_i^n
    A = betan * dt_local * chi * c * sigman
    coupling = chi * c * sigman / (1.0 + A)  # χ c σ / (1 + A)

    lower = np.zeros(n_int - 1)
    diag = np.zeros(n_int)
    upper = np.zeros(n_int - 1)
    rhs = np.zeros(n_int)

    if marshak_boundary:
        diag[0] = 1 + 2 * D_face[0] / (c * dz)
        upper[0] = - 2 * D_face[0] / (c * dz)
        rhs[0] = a * (T_left) ** 4

        for k in range(1, n_int):
            i = k  # i=1..N-2
            D_imh = D_face[i - 1]  # i-1/2
            D_iph = D_face[i]      # i+1/2

            a_i = -D_imh / dz**2
            c_i = -D_iph / dz**2
            b_i = (1.0 / dt_local) + (D_imh + D_iph) / dz**2 + coupling[i]
            d_i = (E[i] / dt_local) + coupling[i] * UR[i]

            diag[k] = b_i
            rhs[k] = d_i
            lower[k - 1] = a_i
            if k < n_int - 1:
                upper[k] = c_i

    else:
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
    D_N2ph = D_face[-1]
    c_N2 = -D_N2ph / dz**2
    rhs[-1] -= c_N2 * E_right

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
    E_new[-1] = E_right
    if not marshak_boundary:
        E_new[0] = E_left
        E_new[1:-1] = E_inner
    else:
        E_new[0:-1] = E_inner
    UR_new = (A * E_new + UR) / (1.0 + A)
    return E_new, UR_new

def run_time_loop(E, UR, times_to_store, *, dtfac=0.05, dtmin=5e-15, dtmax=2e-13, marshak_boundary=False):
    """
    Run the PDE time loop; store Um, T at requested times.
    Returns stored_t, stored_Um, stored_T.
    """
    # add a progress bar using tqdm
    store_idx = 0
    stored_t, stored_Um, stored_Tm, stored_TR = [], [], [], []
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
        E, UR = implicit_step_self_similar_model(E, UR, t=t, dt_local=dt_local, marshak_boundary=marshak_boundary)

        t_next = t + dt_local

        Um = U_m_of_T(UR)
        Tm = (UR / a) ** 0.25
        TR = (E / a) ** 0.25
        # store if we hit store time
        if store_idx < len(times_to_store) and abs(t_next - times_to_store[store_idx]) < 0.5 * dt_local:
            if simulation_unit_system == CGS:
                stored_Um.append(np.array(Um).copy())
                stored_Tm.append(np.array((Tm / K_per_Hev)).copy())
                stored_t.append(t_next * 1e9)  # ns
                stored_TR.append(np.array((TR / K_per_Hev)).copy())
            else:
                stored_Um.append(np.array(Um).copy())
                stored_Tm.append(np.array(Tm).copy())
                stored_TR.append(np.array(TR).copy())
                stored_t.append(t_next)
            store_idx += 1

        # adapt dt for next step
        dt_new, dE, dU = update_dt_relchange(dt_local, E, Eold, UR, URold, dtfac=dtfac, dtmax=dtmax)
        if dtmin is not None:
            dt_new = max(dt_new, dtmin)
        pbar.update(t_next - t)
        t = t_next
        dt_local = dt_new
    pbar.close()
    return np.array(stored_t), np.array(stored_Um), np.array(stored_Tm), np.array(stored_TR)

def compute_front_and_energy(stored_Um, stored_Tm):
    """
    For each stored profile:
      - front position = first z where T < threshold*T_bath
      - total energy = ∫ Um dz
    """
    front_positions = []
    total_energies = []

    for Ti, Ui in zip(stored_Tm, stored_Um):
        front_idx = np.argmax(np.abs(np.diff(Ti)))
        front_position = z[front_idx]
        front_positions.append(front_position)
        # hJ = 10^2 J
        # erg = 10^-7 J = 10^-9 hJ
        # 1 / cm^2 = 10^-2 / mm^2
        # => erg/cm^2 = 10^-11 hJ/mm^2
        # => integrate Um (erg/cm^3) over z (cm) gives erg/cm^2 = 10^-11 hJ/mm^2
        total_energy = np.trapezoid(Ui, z)
        total_energy_hJ_mm2 = total_energy * 1e-11  # convert erg/cm^2 to hJ/mm^2
        total_energies.append(total_energy_hJ_mm2)

    return np.array(front_positions), np.array(total_energies)