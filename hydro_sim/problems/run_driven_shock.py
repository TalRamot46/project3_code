# problems/run_driven_shock.py
import numpy as np
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.geometry import planar
from driven_shock_problem import PowerLawPressureDrive
from driven_shock_sim import simulate_driven_shock
from driven_shock_plots import plot_driven_shock_profiles, plot_driven_shock_slider, save_driven_shock_gif


def main():
    # -- stage 1 : getting simulation parameters --
    USE_PARSER = False
    if USE_PARSER:
        ap = argparse.ArgumentParser(description="Driven shock (power-law pressure) simulation")

        # Spatial / temporal resolution
        ap.add_argument("--N", type=int, default=1000, help="Number of cells")
        ap.add_argument("--t_end", type=float, default=1.0, help="Final simulation time")
        ap.add_argument("--CFL", type=float, default=0.5, help="CFL number")
        ap.add_argument("--sigma", type=float, default=1.0, help="Artificial viscosity coefficient")

        # Material / EOS
        ap.add_argument("--gamma", type=float, default=1.4, help="Adiabatic index")

        # Initial material state
        ap.add_argument("--rho0", type=float, default=1.0, help="Initial density")
        ap.add_argument("--p0", type=float, default=1e-6, help="Initial pressure (near vacuum)")
        ap.add_argument("--u0", type=float, default=0.0, help="Initial velocity")

        # Domain
        ap.add_argument("--x_min", type=float, default=0.0)
        ap.add_argument("--x_max", type=float, default=1.0)

        # Driving parameters
        ap.add_argument("--P0", type=float, default=1.0, help="Pressure scale")
        ap.add_argument("--tau", type=float, default=1.0, help="Power-law exponent: p ~ t^tau")

        # Output
        ap.add_argument("--save", type=str, default=None, help="Save figure path (png)")
        ap.add_argument("--no-show", action="store_true", help="Do not show plots")

        args = ap.parse_args()
    else:
        class Args:
            N = 1000
            t_end = 1e-6
            CFL = 0.2
            sigma =1.0
            gamma = 0.25
            rho0 = 19.32
            p0 = 1e-9
            u0 = 0.0
            x_min = 0.0
            x_max = 1 / (19.32) * 1e-5
            P0 = 1e-6
            tau = 0
            save = None
            no_show = False
        args = Args()

    # -- stage 2 : set up problem parameters --
    case = PowerLawPressureDrive(
        name="Power-law driven shock",
        x_min=args.x_min,
        x_max=args.x_max,
        t_end=args.t_end,
        rho0=args.rho0,
        p0=args.p0,
        u0=args.u0,
        gamma=args.gamma,
        tau=args.tau,
        P0=args.P0,
    )

    # -- stage 3 : run simulation --
    x_cells, state, meta, hist = simulate_driven_shock(
        case,
        Ncells=args.N,
        CFL=args.CFL,
        sigma_visc=args.sigma,
    )

    # -- stage 4 : plot results --
    plot_driven_shock_profiles(
        x_cells=x_cells,
        rho=state.rho,
        p=state.p,
        u=0.5 * (state.u[:-1] + state.u[1:]),
        e=state.e,
        case=case,
        t=state.t,
        savepath=args.save,
        show=not args.no_show,
    )
    # slider view
    plot_driven_shock_slider(hist, case, show=not args.no_show)

    # optional gif
    save_driven_shock_gif(hist, case, gif_path="driven_shock.gif", fps=20, stride=2)


if __name__ == "__main__":
    main()

    # Example run:
    # python -m project_3.hydro_sim.problems.run_driven_shock \
    #   --N 2000 --t_end 0.5 --P0 1.0 --tau 1.5 --gamma 1.4
