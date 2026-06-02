# scratch/test_unpickle.py
import sys
import os
from pathlib import Path
import pickle
import threading

# Add parent path to import project modules if needed
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_REPO_ROOT = _REPO_PARENT / "project3_code"
_MENAHEM_DIR = _REPO_ROOT / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

import sys
import numpy
try:
    import numpy._core
    # Set the attributes on the numpy package so that __getattr__('core') isn't called
    numpy.core = numpy._core
    sys.modules['numpy.core'] = numpy._core
    sys.modules['numpy.core.numeric'] = numpy._core.numeric
    sys.modules['numpy.core.multiarray'] = numpy._core.multiarray
except ImportError:
    # If numpy._core doesn't exist (NumPy 1.x), map new numpy._core lookups to numpy.core
    sys.modules['numpy._core'] = sys.modules.get('numpy.core')
    sys.modules['numpy._core.numeric'] = sys.modules.get('numpy.core.numeric')
    sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray')

cache_path = Path(_REPO_ROOT / "results/ictt/cache/full_fitting_cache.pkl")
print("Cache exists:", cache_path.exists())

def test_pure_python():
    print("Testing pure Python Unpickler...")
    try:
        with open(cache_path, "rb") as f:
            data = pickle._Unpickler(f).load()
        print("Pure Python Unpickler SUCCESS!")
        print("Keys:", list(data.keys()))
        return True
    except Exception as e:
        print("Pure Python Unpickler FAILED:", e)
        import traceback
        traceback.print_exc()
        return False

def test_thread_stack():
    print("Testing thread with larger stack size...")
    res = {}
    def run():
        try:
            with open(cache_path, "rb") as f:
                res["data"] = pickle.load(f)
            print("Thread stack SUCCESS!")
        except Exception as e:
            res["error"] = e
            print("Thread stack FAILED:", e)
            import traceback
            traceback.print_exc()

    # Set thread stack size to 8MB
    threading.stack_size(8 * 1024 * 1024)
    t = threading.Thread(target=run)
    t.start()
    t.join()
    return "data" in res

if __name__ == "__main__":
    if not test_pure_python():
        test_thread_stack()
