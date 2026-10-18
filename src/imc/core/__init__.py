"""Generic gray 1D slab IMC solver; see md/imc_core.md.

  material.py   power-law (Hammer-Rosen) opacity and EOS, Fleck factor
  mesh.py       nonuniform slab cells
  particles.py  struct-of-arrays particle storage
  sources.py    emission, blackbody bath and initial radiation particles
  transport.py  the numba random-walk kernel for one time step
  population.py census population control (stratified comb)
  tallies.py    per-step tallies and energy ledgers
  stepper.py    one IMC time step: sources -> transport -> material update
"""
