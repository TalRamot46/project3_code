"""Gray 1D slab Implicit Monte Carlo (Fleck & Cummings 1971).

Package built from src/imc/sio2_1mm_gray_imc_check_og.py, which stays frozen
as a regression baseline.  The pieces:

  material.py   power-law (Hammer-Rosen) opacity and EOS, Fleck factor
  mesh.py       nonuniform slab cells
  particles.py  struct-of-arrays particle storage
  sources.py    emission, blackbody bath and initial radiation particles
  transport.py  the numba random-walk kernel for one time step
  population.py census population control (stratified comb)
  tallies.py    per-step tallies and energy ledgers
  stepper.py    one IMC time step: sources -> transport -> material update
  problems/     drivers: SiO2 slab (sio2.py) and the 0-D problem (zero_d.py)

Derivation and proofs: md/imc_mc_derivation.md.  Plan: md/full_plan_imc_comparison.md.
"""
