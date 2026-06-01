import pickle
from pathlib import Path
import sys

_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_MENAHEM_DIR = Path(__file__).resolve().parents[1] / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

import numpy as np
print("Numpy version:", np.__version__)

cache_path = Path("results/ictt/cache/heat_const_T_cache.pkl")
try:
    import numpy.core
    import numpy.core.numeric
    import numpy.core.multiarray
    sys.modules['numpy._core'] = sys.modules.get('numpy.core')
    sys.modules['numpy._core.numeric'] = sys.modules.get('numpy.core.numeric')
    sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray')

    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    print("Success! Keys:", data.keys())
except Exception as e:
    import traceback
    traceback.print_exc()
