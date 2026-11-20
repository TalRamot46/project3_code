"""sys.path setup so that `project3_code.*` and the solver modules import, as in the legacy_ictt29 scripts."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent          # src/fitting/test_docs
_SRC = _HERE.parents[1]                           # src
for p in (_SRC / "menahem_new", _SRC, _SRC.parent, _SRC.parent.parent, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import numpy as np, scipy.integrate
if not hasattr(scipy.integrate, "simps"): scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(np, "trapz"): np.trapz = np.trapezoid
if not hasattr(np, "trapezoid"): np.trapezoid = np.trapz
