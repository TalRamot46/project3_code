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
| [imc_problem_sio2.md](imc_code/imc_problem_sio2.md) | `src/imc/problems/sio2.py`: the Back SiO₂ problem, its drive (`src/imc/data/`), the archived original script (`src/imc/archive/`), reference results |
| [imc_coupling.md](imc_code/imc_coupling.md) | `src/imc/coupling/`: IMC as rad_hydro_sim's radiation solver |
| [imc_problem_zero_d.md](imc_code/imc_problem_zero_d.md) | `src/imc/problems/zero_d.py`: the 0-D problem; Wollaber Figs. 1–2 reproduced |
| [imc_problem_linear_step.md](imc_code/imc_problem_linear_step.md) | `src/imc/problems/linear_step.py`: one linear step vs. S_N |

Figures: `results/imc_code/figures/`.

## `phase_b/`: Phase B of the plan (diffusion conventions, IMC in rad_hydro_sim)
| File | Contents |
|---|---|
| [phase_b_diffusion_changes.md](phase_b/phase_b_diffusion_changes.md) | **Every change made to the diffusion code**, why, how it was verified, and findings left unchanged |
| [phase_b_results.md](phase_b/phase_b_results.md) | Diffusion vs. IMC on SiO₂ and on the paper's gold tests |

Figures: `results/phase_b/figures/`.

## `phase_c/`: Phase C of the plan (IMC in Lagrangian rad-hydro)
| File | Contents |
|---|---|
| [phase_c_changes.md](phase_c/phase_c_changes.md) | **Every Phase C code change** (census carried with the mesh, per-step ledger, radiation pressure M5b, the A9 v/c diagnostic, the paper presets), the decisions taken, and findings left unchanged |
| [phase_c_results.md](phase_c/phase_c_results.md) | IMC vs. diffusion vs. the semi-analytic solution on the paper's full rad-hydro Tests 1 and 2 |
| [phase_c_tests3to6_results.md](phase_c/phase_c_tests3to6_results.md) | Phase C½: the same comparison on all six paper tests (Au ω = 0, ±0.5; Cu; Al), which materials lose IMC–diffusion consistency, the bath options |
| [why_the_solver_differs.md](phase_c/why_the_solver_differs.md) | Why the semi-analytic curve misses Tests 4–6: surface non-equilibrium (Cu, Al) and the rigid-front patching (Test 4), with the code walk-through and suggested fixes |
| [bath_calibration.md](phase_c/bath_calibration.md) | The three bath drives (paper, temperature-calibrated, flux-calibrated) against the semi-analytic solution on Tests 1–6: what each matches, which is closest per case, and why |
| [profile_figures.md](phase_c/profile_figures.md) | The profile figures of the six paper tests in one place: Tests 1–2 (paper bath), Tests 3–4 in every region and scale (paper bath), Tests 5–6 under the paper and the flux-calibrated bath |
| [omega_sweep/](phase_c/omega_sweep/README.md) | Gold, ω = −0.8 … +0.6: heat front and shock (x, m) against time, diffusion and IMC with and without the flux calibration, against the solver |
| [legacy/](phase_c/legacy/) | Superseded attempts to improve the solver's patching: fix B (a numerical moving-front target, not an analytic benchmark) and the impulse-matched patch (analytic, not a consistent improvement); how both work and why they fail: [patching_fixes_explained.md](phase_c/legacy/patching_fixes_explained.md) |

Figures: `results/phase_c/figures/`, all full-profile figures by bath, plus per-case bath overlays; layout in [phase_c/figures/README.md](../results/phase_c/figures/README.md).

## Where to start
- **New to the IMC:** `theory/imc_mc_derivation.md` §1, then `imc_code/imc_core.md`.
- **Reviewing the diffusion code changes:** `phase_b/phase_b_diffusion_changes.md`, then `phase_c/phase_c_changes.md`.
- **What's next:** the latest status block in `plans/full_plan_imc_comparison.md`.

When a later phase is added, give it its own folder here for its text (e.g. `phase_c/`), and put its figures and tables in `results/<phase>/` (layout: `results/README.md`).

## Repositories, branches and results (2026-11-07)

Every test result, figures and tables included, is now in `results/` next to the run data (map: `results/README.md`); this folder keeps only the text.

| repository | remote | branches |
|---|---|---|
| `project3_code` (this one) | GitHub `TalRamot46/project3_code` | `main`, the stable line. `feat/patching-poc` is the current work: Phase C, the fine meshes, all of `md/phase_c/`. `docs/solver-differs-restructure` is the 4 Nov restructure of `why_the_solver_differs.md`, kept aside and not merged. `data/runs-backup` is on GitHub only and holds the run files behind the figures. |
| `subsonic_self_similar_paper` (the manuscript) | Overleaf's git bridge (`git.overleaf.com`); no GitHub copy | `main` only, since Overleaf syncs a single branch. Pull before editing (`git pull`), and push to publish to Overleaf (`git push`). |
| `project3_docs` | GitHub `TalRamot46/project3_docs` | `main`; the local copy is 1 commit ahead and not pushed. There is also a local branch `agents/subsonic-regime-derivation-eq443-462`. |

