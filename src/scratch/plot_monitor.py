import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Apply project style if available
try:
    from project3_code.rad_hydro_sim.plotting.mpl_style import apply_mpl_style
    apply_mpl_style()
except ImportError:
    pass

# Paths
results_dir = r"c:\Users\TLP-001\Documents\GitHub\project3_code\results\rad_hydro_sim\monitor"
fig8_path = os.path.join(results_dir, "Fig_8_comparison_T_0__1_HeV_tau__0_Au_early_time_marshak_monitor.csv")
fig9_path = os.path.join(results_dir, "Fig_9_comparison_T_0__1_HeV_tau__0.123_Au_early_times_marshak_monitor.csv")

# Conversion constants
KELVIN_PER_HEV = 1_160_500.0

def load_and_inspect(filepath):
    print(f"Loading {os.path.basename(filepath)}...")
    df = pd.read_csv(filepath)
    print("Columns:", df.columns.tolist())
    print("Time range:", df['time'].min(), "to", df['time'].max())
    print("T_bath_menahem range:", df['T_bath_menahem'].min(), "to", df['T_bath_menahem'].max())
    print("T_surface range:", df['T_surface'].min(), "to", df['T_surface'].max())
    return df

df8 = load_and_inspect(fig8_path)
df9 = load_and_inspect(fig9_path)

# Let's save the figures in the artifacts folder.
# We will create a two-panel figure.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Fig 8 plotting
# time to ns
t8_ns = df8['time'] * 1e9
T_bath_8_hev = df8['T_bath_menahem'] / KELVIN_PER_HEV
T_surf_8_hev = np.full_like(T_bath_8_hev, 1)


ax1.plot(t8_ns, T_bath_8_hev, label=r"$T_{\mathrm{bath}}$", color="#1f77b4", linewidth=2)
ax1.plot(t8_ns, T_surf_8_hev, label=r"$T_{\mathrm{surface}}$", color="#d62728", linewidth=2)
ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("Temperature [HeV]")
ax1.set_ylim(0,2)
# ax1.set_title(r"Fig 8 Comparison ($\tau = 0$, $Au$, early time)")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="best")

# Fig 9 plotting
# time to ns
t9_ns = df9['time'] * 1e9
T_bath_9_hev = df9['T_bath_menahem'] / KELVIN_PER_HEV
T_surf_9_hev = df9['T_surface'] / KELVIN_PER_HEV

ax2.plot(t9_ns, T_bath_9_hev, label=r"$T_{\mathrm{bath}}$", color="#1f77b4", linewidth=2)
ax2.plot(t9_ns, T_surf_9_hev, label=r"$T_{\mathrm{surface}}$", color="#d62728", linewidth=2)
ax2.set_xlabel("Time [ns]")
ax2.set_ylabel("Temperature [HeV]")
ax2.set_ylim(0,2)
# ax2.set_title(r"Fig 9 Comparison ($\tau = 0.123$, $Au$, early times)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="best")

plt.tight_layout()

# Save path
artifact_dir = r"C:\Users\TLP-001\.gemini\antigravity-ide\brain\cd888c64-0df1-4ad6-98da-ecbe8c4c1387"
os.makedirs(artifact_dir, exist_ok=True)
plot_path = os.path.join(artifact_dir, "monitor_comparison_plot.png")
plt.savefig(plot_path, dpi=200)
print(f"Saved plot to {plot_path}")
plt.show()