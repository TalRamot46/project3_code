# Test 3 on the solver: figures

Text: `md/phase_c/test3_on_the_solver.md`. All figures on one page: `md/phase_c/test3_figures.md`. Drawn by `uv run python -m rad_hydro_sim.verification.test3_on_the_solver --figures --table`.

| file | contents |
|---|---|
| `ratio_cold.png` | cold starts: density, pressure, velocity over the solver's and the entropy against the solver's, shocked region, 1, 1.5, 2 ns |
| `ratio_seeded.png` | the same for the analytic starts at 0.1 ns |
| `entropy.png` | entropy p/ρ^γ at 2 ns against the solver's, and each element's change between 1 and 2 ns; cold and seeded |
| `startup.png` | cold starts against time: pressure on the dense side of the front, heat front mass, shock mass over the solver's; surface temperature |
| `profiles_<run>.png` | one run: density, pressure, velocity, temperature in the shocked region at 1, 1.5, 2 ns |
| `full_<run>.png` | the same over the whole domain, against the full AblationSolver |
| `chi_convergence.png`, `chi_table.md` | the 2T cold start against the coupling factor χ = 1, 10, 100, 1000, with the 1T run and the solver |
| `table.md` | the ratios at 2 ns at fixed masses, the fronts, cells, steps, wall time |

Runs (`<run>`):

| name | start | radiation | mesh | target |
|---|---|---|---|---|
| `cold_1T_30K` | t = 0 | 1T conduction | new, graded surface | 30 K |
| `cold_1T_300K` | t = 0 | 1T conduction | new, graded surface | 300 K |
| `cold_2T_30K` | t = 0 | 2T diffusion | new, graded surface | 30 K |
| `cold_2T_30K_chi10`, `_chi100`, `_chi1000` | t = 0 | 2T diffusion, coupling × χ | new, graded surface | 30 K |
| `seeded_IMC_30K_smeared_seed1`, `_seed2` | solver at 0.1 ns, smeared shock | IMC | new, equal surface cells | 30 K |
| `seeded_2T_30K_smeared` | solver at 0.1 ns, smeared shock | 2T diffusion | new, equal surface cells | 30 K |
| `seeded_2T_30K` | solver at 0.1 ns, sharp shock on a face | 2T diffusion | new, equal surface cells | 30 K |
| `seeded_2T_300K` | solver at 0.1 ns, sharp shock on a face | 2T diffusion | new, equal surface cells | 300 K |

The χ runs appear only in `chi_convergence.png`; the IMC runs appear as the mean of their seeds in `ratio_seeded.png` and `entropy.png`, and seed 1 has its own profile figures.

Grey and orange curves in the comparison figures are the runs before this work (`test3_smooth`, `test3_smooth_analytic`: 2T diffusion and IMC, old mesh, 300 K).
