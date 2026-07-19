# scratch/copy_caches.py
import shutil
from pathlib import Path

# Source and destination directories
src_other = Path("results/ictt_other_presets/cache")
src_fig8 = Path("results/ictt/cache")
dest = Path("results/ictt/cache")
dest.mkdir(parents=True, exist_ok=True)

# Mapping of (source_path, dest_path)
mappings = [
    # Fig 8 (const_T)
    (src_fig8 / "heat_const_T_cache.pkl", dest / "const_T_cache.pkl"),
    (src_fig8 / "shock_const_T_cache.pkl", dest / "const_T_cache_shock.pkl"),
    (src_fig8 / "full_fitting_cache.pkl", dest / "const_T_full_fitting_cache.pkl"),
    (src_fig8 / "heat_const_T_sub_similarity_solver.pkl", dest / "const_T_sub_similarity_solver.pkl"),
    (src_fig8 / "shock_const_T_shock_similarity_solver.pkl", dest / "const_T_shock_similarity_solver.pkl"),

    # Fig 9 (const_S)
    (src_other / "flux_const_cache.pkl", dest / "const_S_cache.pkl"),
    (src_other / "flux_const_cache_shock.pkl", dest / "const_S_cache_shock.pkl"),
    (src_other / "flux_const_full_fitting_cache.pkl", dest / "const_S_full_fitting_cache.pkl"),
    (src_other / "flux_const_shock_similarity_solver.pkl", dest / "const_S_shock_similarity_solver.pkl"),
    (src_other / "flux_const_sub_similarity_solver.pkl", dest / "const_S_sub_similarity_solver.pkl"),

    # Fig 10 (const_P_shock)
    (src_other / "ablation_p_const_cache.pkl", dest / "const_P_shock_cache.pkl"),
    (src_other / "ablation_p_const_cache_shock.pkl", dest / "const_P_shock_cache_shock.pkl"),
    (src_other / "ablation_p_const_full_fitting_cache.pkl", dest / "const_P_shock_full_fitting_cache.pkl"),
    (src_other / "ablation_p_const_shock_similarity_solver.pkl", dest / "const_P_shock_shock_similarity_solver.pkl"),
    (src_other / "ablation_p_const_sub_similarity_solver.pkl", dest / "const_P_shock_sub_similarity_solver.pkl"),
]

for src, dst in mappings:
    if src.exists():
        print(f"Copying {src} -> {dst}")
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Error copying {src}: {e}")
    else:
        print(f"Source file not found: {src}")

print("Cache replication completed.")
