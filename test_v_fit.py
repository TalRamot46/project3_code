import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from rad_hydro_sim.problems.presets_utils import get_preset
from rad_hydro_sim.problems.presets_config import PRESET_FIG_8_CONSTANT_TEMPERATURE
import sys
sys.path.append('ictt29')
from menahem_new.ablation_solver import AblationSolver
from sub_fitting import _ablation_kwargs_from_case

case, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
kwargs = _ablation_kwargs_from_case(case)
solver = AblationSolver(**kwargs).find_xsi_f()

y_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
xsi_vec = y_grid * solver.xsi_f
profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)

V_val = profiles["V"]
valid_idx = (y_grid > 0.005) & np.isfinite(V_val)
y_valid = y_grid[valid_idx]
V_valid = V_val[valid_idx]

def power_law_front(y, c, d):
    return c * (1.0 - y)**d

try:
    popt_V, _ = curve_fit(power_law_front, y_valid, V_valid, p0=[1.0/0.17, -0.5], maxfev=10000)
    print("popt_V direct:", popt_V)
except Exception as e:
    print("Error fitting V directly:", e)

# Let's fit 1/V = rho instead and see what it gives:
rho_valid = 1.0 / V_valid
popt_rho, _ = curve_fit(power_law_front, y_valid, rho_valid, p0=[0.17, -0.5])
print("popt_rho:", popt_rho)

# Plot to see what happened to popt_V
plt.figure()
plt.plot(y_valid, V_valid, label='V_num')
plt.plot(y_valid, power_law_front(y_valid, *popt_V), label='V_fit (direct)')
plt.plot(y_valid, 1.0 / power_law_front(y_valid, *popt_rho), label='1 / rho_fit')
plt.yscale('log')
plt.legend()
plt.savefig('v_fit_test.png')
