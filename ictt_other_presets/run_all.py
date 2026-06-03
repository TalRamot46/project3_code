# ictt_other_presets/run_all.py
import sys
import subprocess
from pathlib import Path

def run_script(script_name: str):
    print("\n" + "=" * 80)
    print(f"RUNNING SCRIPT: {script_name}")
    print("=" * 80)
    
    script_path = Path(__file__).resolve().parent / script_name
    
    # Run the script using the same Python interpreter
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\nERROR: Script {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\nSUCCESS: Script {script_name} completed successfully.")

if __name__ == "__main__":
    scripts = [
        "sub_fitting.py",
        "shock_fitting.py",
        "full_fitting.py",
        "full_fitting_eulerian.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\n" + "=" * 80)
    print("ALL FITTING PIPELINES COMPLETED SUCCESSFULLY FOR FIG 9 AND FIG 10 PRESETS!")
    print("=" * 80)
