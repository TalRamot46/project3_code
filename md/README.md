# Documentation index

## `plans/`: what we are building and in which order
| File | Contents |
|---|---|
| [full_plan_imc_comparison.md](plans/full_plan_imc_comparison.md) | The IMC-vs-diffusion verification plan for the subsonic Marshak-wave paper: phases A–E, milestones M0–M11, the advisor addendum (A1–A9), and dated status blocks |
| [full_plan_sn_comparison.md](plans/full_plan_sn_comparison.md) | The S_N twin's plan (S0–S4…) |

## `theory/`: derivations and proofs
| File | Contents |
|---|---|
| [imc_mc_derivation.md](theory/imc_mc_derivation.md) | F&C's linearization from Eq. 1.1 to Eq. 2.1, proof that the random walk solves it (Theorems 1–4), guarantees for the code, and what remains open |
| [sn_derivation.md](theory/sn_derivation.md) | The S_N discretization of the same equation (`src/sn/sn.py`) |

## `imc_code/`: the IMC code in `src/imc/`
| File | Code it explains |
|---|---|
| [imc_core.md](imc_code/imc_core.md) | `src/imc/core/`: the generic solver, module by module, and the flow of one step |
| [imc_sio2.md](imc_code/imc_sio2.md) | `src/imc/sio2/`: the Back SiO₂ problem, the frozen original script, reference results |
| [imc_coupling.md](imc_code/imc_coupling.md) | `src/imc/coupling/`: IMC as rad_hydro_sim's radiation solver |
| [imc_problem_zero_d.md](imc_code/imc_problem_zero_d.md) | `src/imc/problems/zero_d.py`: the 0-D problem; Wollaber Figs. 1–2 reproduced |
| [imc_problem_linear_step.md](imc_code/imc_problem_linear_step.md) | `src/imc/problems/linear_step.py`: one linear step vs. S_N |

Figures: `imc_code/figures/`.

## `phase_b/`: Phase B of the plan (diffusion conventions, IMC in rad_hydro_sim)
| File | Contents |
|---|---|
| [phase_b_diffusion_changes.md](phase_b/phase_b_diffusion_changes.md) | **Every change made to the diffusion code**, why, how it was verified, and findings left unchanged |
| [phase_b_results.md](phase_b/phase_b_results.md) | Diffusion vs. IMC on SiO₂ and on the paper's gold tests |

Figures: `phase_b/figures/`.

## Where to start
- **New to the IMC:** `theory/imc_mc_derivation.md` §1, then `imc_code/imc_core.md`.
- **Reviewing the diffusion code changes:** `phase_b/phase_b_diffusion_changes.md`.
- **What's next:** the latest status block in `plans/full_plan_imc_comparison.md`.

When a later phase is added, give it its own folder (e.g. `phase_c/`), with a `figures/` subfolder next to its md files.
