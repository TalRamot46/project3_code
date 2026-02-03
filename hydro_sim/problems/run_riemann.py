# problems/run_riemann.py
import numpy as np
import sys
from pathlib import Path
import argparse
import numba as nb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riemann_problem import RIEMANN_TEST_CASES
from riemann_sim import simulate_riemann
from riemann_plots import plot_riemann_comparison



def main():
    # -- stage 1 : getting simulation parameters -- 
    USE_PARSER = False
    if USE_PARSER:
        ap = argparse.ArgumentParser()
        ap.add_argument("--test", type=int, default=1, choices=[1,2,3,4])
        ap.add_argument("--N", type=int, default=1000)
        ap.add_argument("--gamma", type=float, default=1.4)
        ap.add_argument("--CFL", type=float, default=0.5)
        ap.add_argument("--sigma", type=float, default=None, help="Override viscosity sigma (otherwise uses case default)")
        ap.add_argument("--save", type=str, default=None, help="Save figure path (png)")
        args = ap.parse_args()
    else:
        class Args:
            test = 1
            N = 1000
            gamma = 1.4
            CFL = 0.5
            sigma = None
            save = None
        args = Args()

    # -- stage 2 : set up problem parameters --
    case = RIEMANN_TEST_CASES[args.test]
    sigma = args.sigma if args.sigma is not None else case.sigma_visc

    # -- stage 3 : run simulation --
    x_cells, num, ex, meta = simulate_riemann(
        args.test, Ncells=args.N, gamma=args.gamma, CFL=args.CFL, sigma_visc=sigma
    )

    # -- stage 3 : plot results --
    plot_riemann_comparison(
        x_cells=x_cells,
        rho_num=num["rho"], p_num=num["p"], u_num=num["u"], e_num=num["e"],
        rho_ex=ex["rho"],  p_ex=ex["p"],  u_ex=ex["u"],  e_ex=ex["e"],
        test_id=meta["test_id"], t_end=meta["t_end"], Ncells=meta["Ncells"], gamma=meta["gamma"],
        x_min=meta["x_min"], x_max=meta["x_max"],
        title_extra=meta["title_extra"],
        savepath=args.save,
        show=True
    )

if __name__ == "__main__":
    main()
    # run in the terminal with:
    # python -m project_3.hydro_sim.problems.run_riemann --test 1 --N 1000