"""
Call MATLAB profiles_for_report_super.m via MATLAB Engine for Python.
Compares with the Python implementation or runs the MATLAB solver directly.

Installation (run these first):
  pip uninstall matlab          # remove wrong PyPI package if present
  pip install matlabengine      # for MATLAB R2025b+ with Python 3.9–3.12
  # OR for MATLAB R2024a: use Python 3.11, then:
  #   cd "C:\\Program Files\\MATLAB\\R2024a\\extern\\engines\\python"
  #   python -m pip install .
"""
try:
    import matlab.engine  # type: ignore[import-untyped]
except ModuleNotFoundError as e:
    if "matlab" in str(e).lower():
        raise SystemExit(
            "MATLAB Engine for Python not found.\n"
            "1. Uninstall wrong package: pip uninstall matlab\n"
            "2. Install engine: pip install matlabengine  (MATLAB R2025b+, Python 3.9-3.12)\n"
            "   OR for R2024a: use Python 3.11, then install from MATLAB's extern/engines/python folder"
        ) from e
    raise
import numpy as np
from pathlib import Path

# Path to the matlab folder (F.m, solve_normalize.m, manager.m, Au.m, etc.)
_MATLAB_DIR = Path(__file__).resolve().parent / "matlab"

eng = matlab.engine.start_matlab()

# Make sure MATLAB sees the folder with F.m, solve_normalize.m, manager.m, etc.
eng.addpath(str(_MATLAB_DIR), nargout=0)

# Option 1: Run the full profiles_for_report_super.m script
eng.run("profiles_for_report_super", nargout=0)

# Retrieve results from MATLAB workspace
m_heat = np.array(eng.workspace["m_heat"])
x_heat = np.array(eng.workspace["x_heat"])
T_heat = np.array(eng.workspace["T_heat"])
t = np.array(eng.workspace["t"]).squeeze()
x = np.array(eng.workspace["x"])  # shape (N, 2): T and dT/dxi

print("From profiles_for_report_super.m:")
print("  m_heat shape:", m_heat.shape)
print("  xsi =", float(eng.workspace["xsi"]))
print("  T(0) =", x[-1, 0])

# Option 2: Call solve_normalize directly (for comparison / debugging)
# Uses alpha, beta, tau from the material (Au/Al in profiles_for_report_super: alpha=3.1, beta=1.2)
alpha, beta, tau, iternum, xsi0 = 3.1, 1.2, 0.141, 100, 1.0
t_norm, x_norm = eng.solve_normalize(alpha, beta, tau, iternum, xsi0, nargout=2)

t_np = np.array(t_norm).squeeze()
x_np = np.array(x_norm)  # shape (N, 2)
print("\nFrom solve_normalize(alpha=3.1, beta=1.2, tau=0.141):")
print("  T(0) =", x_np[-1, 0])

eng.quit()
