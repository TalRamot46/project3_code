from project3_code.menahem_new.subsonic_heat_wave_og import SubsonicHeatWave
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

# Case parameters for Fig 9
g_Kelvin = 1.0 / (7200 * KELVIN_PER_HEV**1.5)
alpha = 1.5
lambda_ = 0.2
f_Kelvin = 3.4e13 / (KELVIN_PER_HEV**1.6)
beta_Rosen = 1.6
mu = 0.14
r = 0.25
gamma = r + 1.0

# Initialize for tau = 0.123
solver_123 = SubsonicHeatWave(
    Tb=1.0 * KELVIN_PER_HEV * (1./1e-9)**0.123,
    tau=0.123,
    g=g_Kelvin,
    alpha=alpha,
    lambdap=lambda_,
    f=f_Kelvin,
    beta=beta_Rosen,
    mu=mu,
    gamma=gamma
)
print("For tau = 0.123:")
print("cS:", solver_123.cS)
print("bath_power (cS - 4*tau):", solver_123.bath_power)

# Initialize for tau_exact = 111559/907297
tau_exact = 111559.0 / 907297.0
solver_exact = SubsonicHeatWave(
    Tb=1.0 * KELVIN_PER_HEV * (1./1e-9)**tau_exact,
    tau=tau_exact,
    g=g_Kelvin,
    alpha=alpha,
    lambdap=lambda_,
    f=f_Kelvin,
    beta=beta_Rosen,
    mu=mu,
    gamma=gamma
)
print("\nFor tau_exact = 111559/907297:")
print("cS:", solver_exact.cS)
print("bath_power (cS - 4*tau):", solver_exact.bath_power)
