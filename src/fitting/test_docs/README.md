# test_docs: generator of the per-test formula documents

Produces `project3_docs/fitting/tests/testN_*/testN_formulas.{tex,pdf}`, the figures in `figs/`
and `testN_fit_parameters.json` for the six paper tests.

```bash
cd project3_code/src/fitting/test_docs
python build_solvers.py 1 2 3 4 5 6   # AblationSolver per test (≈7 s each) -> results/fitting/cache/
python build_tight.py 1 2 5 6         # subsonic solution re-shot at rtol 1e-11 (≈45 s each; Tests 3, 4 reuse Test 1)
python run.py 1 2 3 4 5 6             # fits, checks, figures, LaTeX + pdflatex (≈5 s each)
```
Env overrides: `FITGEN_CACHE` (solver cache), `FITGEN_OUT` (output `tests/` folder).

| module | role |
|---|---|
| `exact.py` | all exponents as exact fractions from (α, β, λ, μ, r, τ, ω); checked against the solver |
| `tight.py` | tight-tolerance re-shooting of `SubsonicHeatWave` |
| `subfit.py` | subsonic fit family `A y^p (1-y)^e exp[Σ a_i y^i + Σ b_j (1-y)^(jδ)]` |
| `fits.py` | shock fit families (`piston`, `blend`) |
| `compute.py` | fits, dimensional scales, energies, fit formulas, verification vs. the solver |
| `figs.py`, `latex.py`, `run.py` | figures, LaTeX, driver |
