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

## 2. Useful pytest options


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

### Debugging: show verification figures

The verification fit tests (`test_verification_fit.py`) can display comparison plots. Set `SHOW_PLOTS = True` at the top of the file, then run tests with `-s` so figures block until closed:

```bash
python -m pytest tests/test_verification_fit.py -v -s
```

---

## 3. Test markers and `-m "not slow"`

- `**slow**` — Full verification runs; use `-m "not slow"` to skip during development.

### Do slow tests "pass" when running with `-m "not slow"`?

**No.** When you run `pytest -m "not slow"`, slow-marked tests are **excluded** from the run. They are not executed at all — they neither pass nor fail. Pytest will report them as **deselected** (e.g. `N passed, M deselected`).

So a successful `--fast` / `-m "not slow"` run only confirms that non-slow tests passed. It does **not** verify that slow tests would pass. Always run the full suite (without `-m "not slow"`) before pushing or releasing to ensure slow verification tests also pass.

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

---

## 5. Debugging failing tests

### Prerequisites (fix "project3_code not found" and "numpy not recognized")

1. **Select the project's Python interpreter:** In Cursor/VS Code, use the interpreter picker (bottom status bar or Ctrl+Shift+P → "Python: Select Interpreter") and choose the `.venv` inside project3_code.
2. **Install dependencies in that interpreter:**
   ```bash
   # With .venv activated, from project3_code/:
   pip install -e ".[dev]"
   ```
   This installs numpy, pytest, and the project package so imports work.

**If using .venv:** run pip *from the venv* so packages go into it (otherwise pip may use user site-packages and the debugger won't find pytest):
```bash
.venv\Scripts\pip.exe install -e ".[dev]"
```
If your `.venv` doesn't have pip, recreate it: `python -m venv .venv --clear` then run the install.

### Using the Debugger (Cursor/VS Code built-in)

Use the built-in pytest integration and the play button above each test:

1. Open the failing test file (e.g. `tests/test_verification_fit.py`).
2. Set breakpoints where needed.
3. Click the **"Debug Test"** (play-with-bug icon) above a test, or use the Testing view in the sidebar.

For verification tests where you want to **see comparison plots**, set `SHOW_PLOTS = True` at the top of `test_verification_fit.py`, save, then run **Debug Test** as usual. Set it back to `False` when done.

Ensure the `.venv` interpreter is selected and that `pip install -e ".[dev]"` was run in that venv.

### Alternative: pytest's built-in debugger

No IDE setup needed — drop into a debugger on failure:

```bash
# Stop on first failure and open pdb:
python -m pytest tests/test_verification_fit.py -x --pdb

# Or only for a specific test:
python -m pytest tests/test_verification_fit.py::test_hydro_rad_hydro_vs_hydro_sim --pdb
```

When a test fails, you get a `(Pdb)` prompt. Useful commands: `p variable`, `pp variable`, `l` (list code), `n` (next line), `c` (continue), `q` (quit).

### Avoid running test files directly

Do **not** run `python tests/test_verification_fit.py` or use "Run Python File" on a test file. Test files are meant to be discovered by pytest, not executed as scripts. Use the built-in "Debug Test" button or `python -m pytest ...`.