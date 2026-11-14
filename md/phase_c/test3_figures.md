# Test 3 against the solver: the figures in one page

Gold, ω = +0.5, T_s = 1 HeV. Every figure of the last three stages, in reading order. Text and numbers: [test3_on_the_solver.md](test3_on_the_solver.md) (this stage), [test3_hydro_only.md](test3_hydro_only.md), [test3_fix_A.md](test3_fix_A.md). Folder index: [figures/test3_on_the_solver/README.md](../../results/phase_c/figures/test3_on_the_solver/README.md).

In the comparison figures, grey and orange are the runs before this work (2T diffusion and IMC on the old mesh at 300 K); black is the solver. "Entropy" is p/ρ^γ.

## 1. Cold start from t = 0

**Shocked region at 1, 1.5 and 2 ns.** Density, pressure and velocity over the solver's; the entropy against the solver's curve.

![cold starts](../../results/phase_c/figures/test3_on_the_solver/ratio_cold.png)

**The run on the solver: 1T conduction, new mesh, 30 K.** Shocked region, then the whole domain.

![1T 30 K, shocked region](../../results/phase_c/figures/test3_on_the_solver/profiles_cold_1T_30K.png)

![1T 30 K, full domain](../../results/phase_c/figures/test3_on_the_solver/full_cold_1T_30K.png)

**The same at 300 K** (the target's temperature alone: density −6%).

![1T 300 K, shocked region](../../results/phase_c/figures/test3_on_the_solver/profiles_cold_1T_300K.png)

**2T diffusion on the new mesh, 30 K, χ = 1** (the two temperatures alone: density up to +16%).

![2T 30 K, shocked region](../../results/phase_c/figures/test3_on_the_solver/profiles_cold_2T_30K.png)

## 2. The 2T code against the coupling factor χ

Cold start, new mesh, 30 K, χ = 1, 10, 100, 1000, with the 1T run and the solver, at 2 ns. Numbers: [chi_table.md](../../results/phase_c/figures/test3_on_the_solver/chi_table.md).

![chi](../../results/phase_c/figures/test3_on_the_solver/chi_convergence.png)

## 3. The first picoseconds

Front pressure, heat front mass and shock mass over the solver's against time, and the surface temperature.

![startup](../../results/phase_c/figures/test3_on_the_solver/startup.png)

## 4. Analytic start at 0.1 ns

**Shocked region at 1, 1.5 and 2 ns.**

![analytic starts](../../results/phase_c/figures/test3_on_the_solver/ratio_seeded.png)

**2T diffusion, smeared shock, 30 K.** Shocked region, then the whole domain.

![2T smeared 30 K, shocked region](../../results/phase_c/figures/test3_on_the_solver/profiles_seeded_2T_30K_smeared.png)

![2T smeared 30 K, full domain](../../results/phase_c/figures/test3_on_the_solver/full_seeded_2T_30K_smeared.png)

**IMC (seed 1), smeared shock, 30 K.**

![IMC smeared 30 K, shocked region](../../results/phase_c/figures/test3_on_the_solver/profiles_seeded_IMC_30K_smeared_seed1.png)

**Sharp shock on a cell face, 30 K and 300 K** (the spike at 61–62 mg/cm²; at 300 K also the plateau beyond it).

![2T sharp 30 K](../../results/phase_c/figures/test3_on_the_solver/profiles_seeded_2T_30K.png)

![2T sharp 300 K](../../results/phase_c/figures/test3_on_the_solver/profiles_seeded_2T_300K.png)

## 5. Entropy

At 2 ns against the solver's, and each element's change between 1 and 2 ns.

![entropy](../../results/phase_c/figures/test3_on_the_solver/entropy.png)

## 6. The slider

```
uv run python -m rad_hydro_sim.verification.test3_on_the_solver --slider <run>
```

`<run>`: `cold_1T_30K`, `cold_1T_300K`, `cold_2T_30K`, `cold_2T_30K_chi10`, `cold_2T_30K_chi100`, `cold_2T_30K_chi1000`, `seeded_2T_30K_smeared`, `seeded_2T_30K`, `seeded_2T_300K`, `seeded_IMC_30K_smeared_seed1`, `seeded_IMC_30K_smeared_seed2`. Whole profile on fixed axes, 220 frames. Keys: `f` fits the y ranges to the mass window in view, `x` and `y` switch log / linear, arrows step one frame.

## 7. Earlier stage: the shocked region as a hydro-only problem

Hydro only (no radiation), driven by the heat front's pressure or the solver's piston.

![hydro only, cold start](../../results/phase_c/figures/test3_hydro_only/ratio_cold.png)

![hydro only, analytic start](../../results/phase_c/figures/test3_hydro_only/ratio_seeded.png)

![the drives](../../results/phase_c/figures/test3_hydro_only/drives.png)

Each hydro-only run's profiles: `results/phase_c/figures/test3_hydro_only/profiles_<start>_<drive>.png`.
