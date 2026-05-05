$ErrorActionPreference = "Stop"

Set-Location "C:\Users\TLP-001\Documents\GitHub\project3_code"

# Stabilize common nondeterminism sources in numerical backends.
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

$py = "C:\Users\TLP-001\AppData\Local\Microsoft\WindowsApps\python.exe"

& $py -c "import sys,numpy,scipy,matplotlib; print(sys.executable); print(numpy.__version__, scipy.__version__, matplotlib.__version__)"
& $py "menahem_new\ablation_solver.py"

# run with:
# powershell -ExecutionPolicy Bypass -File .\run_menahem_like_codex.ps1