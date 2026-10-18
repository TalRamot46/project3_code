"""Gray 1D slab Implicit Monte Carlo (Fleck & Cummings 1971).

  core/      the generic solver: material, mesh, particles, sources, transport,
             population, tallies, stepper            (md/imc_core.md)
  sio2/      the Back SiO2 problem: the frozen original script, its drive
             table and the same problem on core/      (md/imc_sio2.md)
  problems/  verification drivers: linear_step (vs S_N) and zero_d (vs Wollaber)

Derivation and proofs: md/imc_mc_derivation.md.  Plan: md/full_plan_imc_comparison.md.
"""
