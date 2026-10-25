"""Gray 1D slab Implicit Monte Carlo (Fleck & Cummings 1971).

  core/      the solver: material, mesh, particles, sources, transport,
             population, tallies, stepper              (md/imc_code/imc_core.md)
  problems/  drivers built on core/: sio2 (the Back SiO2 slab), linear_step
             (one step vs S_N) and zero_d (vs Wollaber)  (md/imc_code/imc_problem_*.md)
  coupling/  IMC as rad_hydro_sim's radiation solver    (md/imc_code/imc_coupling.md)
  data/      input tables (back_pulse.txt: the SiO2 drive)
  archive/   the original standalone SiO2 script, frozen as a regression baseline

Derivation and proofs: md/theory/imc_mc_derivation.md.  Plan: md/plans/full_plan_imc_comparison.md.
"""
