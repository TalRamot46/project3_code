# Profile figures: the six paper tests

Each figure shows density, pressure, velocity and temperature against Lagrangian mass at 1, 1.5 and 2 ns. The curves are:
- diffusion, light to dark blue by time;
- IMC, orange (the mean of 3 seeds);
- the semi-analytic `AblationSolver` (black).

> **2026-11-07:** these are the fine-mesh figures. Tests 1, 2 and 6 use 1000 cells, Tests 4 and 5 use 2500, and Test 3 uses its fine mesh. How the numbers moved from the default mesh is in [bath_calibration.md](bath_calibration.md), Addendum 2026-11-07.

The title of each figure names the test, the region, the bath and the axes. The baths are described in [bath_calibration.md](bath_calibration.md). The figure conventions, including the patch sliver left out of the analytic curve at the ablation front, are in [figures/README.md](figures/README.md).

| Test | Material and drive | Bath | Views |
|---|---|---|---|
| 1 | Au, constant T_s | paper | full domain, linear |
| 2 | Au, constant F_s | paper | full domain, linear |
| 3 | Au, ω = +0.5 | paper | full domain, heat region, shocked region; each linear, semi-log (log mass), semi-log (log fields), log-log |
| 4 | Au, ω = −0.5 | paper | as Test 3 |
| 5 | Cu | paper and flux-calibrated | full domain, linear |
| 6 | Al | paper and flux-calibrated | full domain, linear |

The regions and scales of Tests 3 and 4 are as follows.
- **Regions:** the *full domain*; the *heat region*, from the surface to 1.3 m_f (the ablated layer); the *shocked region*, from 0.85 m_f to 1.1 m_s, or from 1.05 m_f when the ablated layer is thin against the shocked one (Test 3). Its y limits come from the shocked matter alone, simulations and solver, so the plateaus and their differences fill the panels.
- **Scales:**
  - *linear*;
  - *semi-log (log mass)*: log mass, linear fields;
  - *semi-log (log fields)*: linear mass, log ρ, p and T;
  - *log-log*.

  On a log field axis the velocity, which changes sign, is drawn on a symmetric-log axis that is linear within ±1 km/s.

The y limits come from the simulations inside the window. Tests 3 and 4 have the same set of views under the other baths in `figures/calibrated_flux_bath/` (and `figures/calibrated_T_bath/` for Test 4), with the bath comparisons in `figures/overlay/`.

## Test 1: gold, constant surface temperature

![Test 1, paper bath, full domain, linear](figures/paper_bath/test1_full_profiles.png)

## Test 2: gold, constant surface flux

![Test 2, paper bath, full domain, linear](figures/paper_bath/test2_full_profiles.png)

## Test 3: gold, ω = +0.5

### Full domain

![Test 3, paper bath, full domain, linear](figures/paper_bath/test3_full/full_linear.png)

![Test 3, paper bath, full domain, semi-log (log mass)](figures/paper_bath/test3_full/full_semilogx.png)

![Test 3, paper bath, full domain, semi-log (log fields)](figures/paper_bath/test3_full/full_semilogy.png)

![Test 3, paper bath, full domain, log-log](figures/paper_bath/test3_full/full_loglog.png)

### Heat region

![Test 3, paper bath, heat region, linear](figures/paper_bath/test3_full/heat_linear.png)

![Test 3, paper bath, heat region, semi-log (log mass)](figures/paper_bath/test3_full/heat_semilogx.png)

![Test 3, paper bath, heat region, semi-log (log fields)](figures/paper_bath/test3_full/heat_semilogy.png)

![Test 3, paper bath, heat region, log-log](figures/paper_bath/test3_full/heat_loglog.png)

### Shocked region

![Test 3, paper bath, shocked region, linear](figures/paper_bath/test3_full/shock_linear.png)

![Test 3, paper bath, shocked region, semi-log (log mass)](figures/paper_bath/test3_full/shock_semilogx.png)

![Test 3, paper bath, shocked region, semi-log (log fields)](figures/paper_bath/test3_full/shock_semilogy.png)

![Test 3, paper bath, shocked region, log-log](figures/paper_bath/test3_full/shock_loglog.png)

## Test 4: gold, ω = −0.5

### Full domain

![Test 4, paper bath, full domain, linear](figures/paper_bath/test4_full/full_linear.png)

![Test 4, paper bath, full domain, semi-log (log mass)](figures/paper_bath/test4_full/full_semilogx.png)

![Test 4, paper bath, full domain, semi-log (log fields)](figures/paper_bath/test4_full/full_semilogy.png)

![Test 4, paper bath, full domain, log-log](figures/paper_bath/test4_full/full_loglog.png)

### Heat region

![Test 4, paper bath, heat region, linear](figures/paper_bath/test4_full/heat_linear.png)

![Test 4, paper bath, heat region, semi-log (log mass)](figures/paper_bath/test4_full/heat_semilogx.png)

![Test 4, paper bath, heat region, semi-log (log fields)](figures/paper_bath/test4_full/heat_semilogy.png)

![Test 4, paper bath, heat region, log-log](figures/paper_bath/test4_full/heat_loglog.png)

### Shocked region

![Test 4, paper bath, shocked region, linear](figures/paper_bath/test4_full/shock_linear.png)

![Test 4, paper bath, shocked region, semi-log (log mass)](figures/paper_bath/test4_full/shock_semilogx.png)

![Test 4, paper bath, shocked region, semi-log (log fields)](figures/paper_bath/test4_full/shock_semilogy.png)

![Test 4, paper bath, shocked region, log-log](figures/paper_bath/test4_full/shock_loglog.png)

## Test 5: copper

### Paper bath

![Test 5, paper bath, full domain, linear](figures/paper_bath/test5_full_profiles.png)

### Flux-calibrated bath

![Test 5, flux-calibrated bath, full domain, linear](figures/calibrated_flux_bath/test5_full_profiles.png)

## Test 6: aluminium

### Paper bath

![Test 6, paper bath, full domain, linear](figures/paper_bath/test6_full_profiles.png)

### Flux-calibrated bath

![Test 6, flux-calibrated bath, full domain, linear](figures/calibrated_flux_bath/test6_full_profiles.png)
