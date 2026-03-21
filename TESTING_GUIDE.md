# Testing Guide — project3_code

How to run tests effectively. You **do not** need `project_3/run_tests.py`; it's a convenience wrapper. All tests are run via **pytest**.

---

## Quick Reference


| Method               | Command                           | Notes                           |
| -------------------- | --------------------------------- | ------------------------------- |
| **Direct pytest**    | `python -m pytest tests/ -v`      | From **project3_code** root     |
| **Run-tests script** | `python project_3/run_tests.py`   | Same, but with `--fast` support |
| **Makefile**         | `make test`                       | From inside project3_code       |
| **After install**    | `run-tests` or `run-tests --fast` | If `pip install -e .` was run   |


---

## 1. Running tests directly (no `run_tests.py`)

You can run tests with **pytest** from the project root:

```bash
# From project3_code/ (your workspace root):
python -m pytest tests/ -v
```

### Path note

Tests import `project3_code.hydro_sim...`, so `project3_code` must be on `sys.path`:

- **Option A:** Run from **inside** `project3_code/` — `pyproject.toml` sets `pythonpath = [".."]`, so pytest adds the parent. If you only opened the `project3_code` folder, run from its parent:
  ```bash
  # From parent of project3_code:
  cd path/to/GitHub
  python -m pytest project3_code/tests -v
  ```
- **Option B:** Install in editable mode (recommended):
  ```bash
  pip install -e .
  ```
  Then `project3_code` is always importable.

---

## 2. Using `project_3/run_tests.py` (optional)

`run_tests.py` is a thin wrapper around pytest that:

1. Runs from the correct project root
2. Passes `-v` for verbose output
3. Supports `--fast` to skip slow tests

```bash
# Run all tests:
python project_3/run_tests.py

# Skip slow tests (verification simulations):
python project_3/run_tests.py --fast

# Pass any pytest arguments:
python project_3/run_tests.py -k diffusion
python project_3/run_tests.py tests/test_configs.py
python project_3/run_tests.py tests/test_verification_fit.py::test_radiation_diffusion_runs
```

---

## 3. Useful pytest options


| Option       | Example                     | Purpose                              |
| ------------ | --------------------------- | ------------------------------------ |
| `-v`         | `pytest tests/ -v`          | Verbose (show test names)            |
| `-k EXPR`    | `pytest -k "config or eos"` | Run tests whose names match `EXPR`   |
| `-m MARKER`  | `pytest -m "not slow"`      | Run only tests with/without a marker |
| `--tb=short` | `pytest --tb=short`         | Shorter tracebacks                   |
| `-x`         | `pytest -x`                 | Stop on first failure                |
| `--lf`       | `pytest --lf`               | Re-run only last failures            |


Examples:

```bash
# Fast test run (skip slow verification simulations):
python -m pytest tests/ -v -m "not slow"

# Only config and EOS tests:
python -m pytest tests/ -v -k "config or eos"

# Single test:
python -m pytest tests/test_configs.py::test_simulation_configs_not_empty -v
```

---

## 4. Recommended workflow

1. Install in editable mode: `pip install -e .` (once).
2. Quick checks during development:
  ```bash
   python -m pytest tests/ -v -m "not slow"
  ```
   or `python project_3/run_tests.py --fast`.
3. Full test run before commits:
  ```bash
   python -m pytest tests/ -v
  ```
   or `make test` from project3_code.

