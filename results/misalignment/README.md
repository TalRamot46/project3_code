# misalignment: why the simulations and the semi-analytic solver differ

One folder per study; each has `runs/` (run data, untracked, `.npz`/`.npy`) and `figures/` (figures and tables, tracked).
The text is in `md/misalignment/` (start with its `README.md`); the figures are shown in context there.

| study | runs written by | text |
|---|---|---|
| `test3_on_the_solver/` | `test_misalignment.test3_on_the_solver` | `md/misalignment/03_startup.md`, `04_analytic_start.md` |
| `test3_hydro_only/` | `test_misalignment.test3_hydro_only` | `md/misalignment/05_shocked_region.md` § 1 |
| `test3_omega_scan/` | `test_misalignment.test3_omega_scan` (run data not on this machine) | `md/misalignment/03_startup.md` § 5 |
| `test3_transparency/` | `test3_transparency`, `test3_transparency_figures` | `md/misalignment/01_surface_equilibrium.md` § 2, `03_startup.md` § 3, `05_shocked_region.md` § 2 |
| `al_long/` | `test_misalignment.al_long`, `al_long_figures` (also `figures/slider/`) | `md/misalignment/01_surface_equilibrium.md` § 3, `02_patching.md` § 4 |
| `light_analytic_start/` | `light_analytic_start` (runs only; its figures are in `test3_transparency/figures/`) | `md/misalignment/01_surface_equilibrium.md` § 3 |

The earlier Test-3 analytic-start figures (fix A) are still drawn by `bath_calibration` into `results/phase_c/figures/paper_bath/test3_analytic_start/`.
