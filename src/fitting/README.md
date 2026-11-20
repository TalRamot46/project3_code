# fitting: analytic fit formulas for the six paper tests

Was `src/ictt29/` (named after the ICTT-29 talk, June 2026). Split by role on 2026-11-19.

| folder | what | status |
|---|---|---|
| `test_docs/` | generator of the per-test formula documents, `project3_docs/fitting/tests/testN_*/` (see its README) | **current**, self-contained |
| `core/` | the fit families and loaders the older scripts share: `sub_fitting.py` (heat wave), `shock_fitting.py`, `data_loader.py` (cached solver/simulation, `results/ictt/cache/`), `sim_front_utils.py` | kept for `legacy_ictt29/` |
| `legacy_ictt29/` | the ICTT-29 scripts: `full_fitting*.py`, `plot_xt.py`, `energy_fitting.py`, `final_comparison.py`, `print_formulas.py`, `print_fitted_formulas.py` (Test 1 template filler, superseded by `test_docs/`), `build_table.py`, and LaTeX leftovers | legacy |

The legacy scripts add `../core` to `sys.path` themselves; run them from the repo root as before (their outputs still go to `results/ictt/`, relative to the working directory).
