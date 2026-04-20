import numpy as np
import sys

from matplotlib import pyplot as plt

from piston_shock import PistonShock
from subsonic_heat_wave import SubsonicHeatWave, Units

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('AblationSolver')

class AblationSolver():
    """An object to computes the solution an ablation problem
    including the ablative heat wave and shock regions

    The temperature boundary condition is 
    T(0,t) = Tb * t ** tau

    The medium consists of an ideal gas with the following properties:
    A polytropic eos with an adiabatic index gamma:
    p = (gamma-1) * rho * sie

    The sie (energy per units mass) is given by:

    sie(T, rho) = f * (T ** beta) * (rho ** -mu)
    
    each of those EOS properties is defined with independent 
    parameters (gamma, f, beta , mu) for the ablation and shock regions.

    The Rosseland oapcity (used only in the ablation region) is given as:
    1/kappa_Rosseland(T, rho) = g * (T ** alpha) * (rho ** -lambda)

    """
    def __init__(self, *,
                 
                 # heat solution depends on these paramters
                 Tb,          # boundary temperature coeff
                 tau,         # boundary temperature temporal power
                 g,           # opacity coeff
                 alpha,       # opacity temmperature power
                 lambdap,     # opacity density power
                 f_heat,      # energy coeff
                 beta_heat,   # energy temmperature power
                 mu_heat,     # energy density power
                 gamma_heat,  # adiabatic index for the heat (ablation) region
                 
                 # shock solution depends on these paramters
                 rho0,        # spatial density coeff
                 omega,       # spatial density power
                 f_shock,      # energy coeff
                 beta_shock,   # energy temmperature power
                 mu_shock,     # energy density power
                 gamma_shock, # adiabatic index
                 ):

        logger.info(f"creating an Ablation solver...")

        # --- set the solver for the ablative heat wave region
        self.heat_solver = SubsonicHeatWave(
            Tb=Tb,
            tau=tau,
            g=g,
            alpha=alpha,
            lambdap=lambdap,
            f=f_heat,
            beta=beta_heat,
            mu=mu_heat,
            gamma=gamma_heat,
        ).find_xsi_f()#.plot_profiles()

        # --- set the solver for the shock region
        # pressure on heat front
        p0s = self.heat_solver.Pf * (self.heat_solver.A**self.heat_solver.a3) * (self.heat_solver.B**self.heat_solver.b3)
        tau_s = self.heat_solver.c3
        self.shock_solver = PistonShock(
            rho0=rho0,
            omega=omega,
            p0=p0s,
            tau=tau_s,
            gamma=gamma_shock,
        )

        # these EOS parameters in the shock
        # region are not a part of the shock solver since 
        # it does not have the notion of temperature -
        # so instead it will be used by this AblationSolver object
        self.f_shock = f_shock
        self.beta_shock = beta_shock
        self.mu_shock = mu_shock

        logger.info(f"f_shock={self.f_shock:g}")
        logger.info(f"beta_shock={self.beta_shock:g}")
        logger.info(f"mu_shock={self.mu_shock:g}")

    def solve(self, *, mass, time):
        """
        calculates the hydrodynamic profiles on the given
        Largrangian mass coordinates at the given time
        """
        # logger.info(f"calculating hydro profiles at t={time:g}")
        assert type(mass) == np.ndarray
        assert len(mass) > 0
        assert all(m>0. for m in mass)
        assert np.min(mass[1:]-mass[:1]) > 0. #Lagrangian coordinate (acumulated cell mass) is monotone

        ablated_mass = self.heat_solver.ablated_mass(time=time)

        # divide Lagrangian coordinates into ablated (heat solution) and unablated (shock solution) mass
        mass_heat = mass[mass <= ablated_mass]
        mass_shock = mass[mass > ablated_mass]

        # adding the ablated mass point explicitly to shock masses - since we need it for the heat wave position
        # will be removed later
        assert len(mass_shock) > 2
        mass_shock = np.array([ablated_mass, *mass_shock])
        assert mass_shock[1] - mass_shock[0] > 1e-10 * mass_shock[1]

        # solve shock profiles
        result_shock = self.shock_solver.solve(mass=mass_shock, time=time)
        result_shock["temperature"] = ((result_shock["pressure"] * result_shock["density"]**(self.mu_shock-1.))/(self.shock_solver.r * self.f_shock))**(1./self.beta_shock)
        
        # the heat wave position is the position of an ablated mass in the shocked region
        heat_position = result_shock["position"][0]
        
        # solve heat profiles
        result_heat = self.heat_solver.solve(mass=mass_heat, time=time)
        
        # recalibrate positions of all ablated positions by the current heat_position
        result_heat["position"] += heat_position

        # set zero flux in shock region
        result_shock["radiation_energy_flux"] = np.zeros_like(result_shock["sie"])

        # patch heat + shock profiles
        result = dict()
        for key in ["density", "pressure", "velocity", "temperature", "sie", "radiation_energy_flux", "position"]:
            result_shock[key] = result_shock[key][1:] # remove the shock solution at the ablated mass (that was added before in order the get the heat position)
            result[key] = np.array(list(result_heat[key]) + list(result_shock[key]))
            assert len(result[key]) == len(mass), f"{len(result[key])} {len(mass)}"
        
        # other scalar quantities
        result["heat_position"] = heat_position
        
        result["ablated_mass"]      = result_heat["ablated_mass"]
        result["boundary_position"] = result_heat["boundary_position"]
        result["boundary_velocity"] = result_heat["boundary_velocity"]
        result["ablated_mass"]      = result_heat["ablated_mass"]
        
        result["piston_position"] = result_shock["piston_position"]
        result["shock_position"]  = result_shock["shock_position"]
        result["shock_velocity"]  = result_shock["shock_velocity"]

        return result
 
def test_profiles_omega():
    time_rise = Units.nsec
    tau = 0.123
    # tau = 0.
    Tb = Units.hev_kelvin * (1./time_rise)**tau
    rho0 = 19.3
    # omega = 0.
    omega = 0.2

    # MORDI Au
    # alpha=1.5
    # beta=1.6
    # f=3.4*1e13/(Units.hev_kelvin**beta)
    # g=1/(7200*Units.hev_kelvin**alpha)
    # mat=dict(
    #     tau=tau,
    #     alpha=alpha,
    #     beta=beta,
    #     lambdap=0.2,
    #     mu=0.14,
    #     gamma=1.25,
    #     Tb=Tb,
    #     g=g,
    #     f=f,
    # )

    mat=dict(
        tau=tau,
        g=5.682464574187187e-6,
        alpha=0.23183890413119132,
        lambdap=0.13829986198312327,
        f_heat=3168.6938116243796,
        beta_heat=1.6525149133503083,
        mu_heat=0.1206770123528319,
        gamma_heat=1.262299832903183,
        Tb=Tb,
        rho0=19.3,
        omega=omega,
    )
    
    for key in ["f_heat", "beta_heat", "mu_heat", "gamma_heat"]:
        mat[key.replace("_heat", "_shock")] = mat[key]
    
    import pickle
    filename = "solver.pickle"
    from os import path
    if path.isfile(filename):
        with open(filename, 'rb') as handle:
            solver = pickle.load(handle)
            for key, value in mat.items():
                keyp=key.replace("_heat", "").replace("_shock", "")
                print(key, value, [getattr(solver, key, None), getattr(solver.heat_solver, key, None), getattr(solver.shock_solver, key, None)])
                assert value in {getattr(solver, keyp, None), getattr(solver.heat_solver, keyp, None), getattr(solver.shock_solver, keyp, None)}
    else:
        solver = AblationSolver(
            **mat,
        )
        with open(filename, 'wb') as handle:
            pickle.dump(solver, handle, protocol=pickle.HIGHEST_PROTOCOL)

    L = 1e-3
    ############
    
    times = np.linspace(0., 2.061e-9*3., 500)
    m_ab = [solver.heat_solver.ablated_mass(time=t) for t in times]
    # x_boundary = np.array([solver.boundary_position(time=t) for t in times])

    plt.plot(times, m_ab)
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("ablated mass [g]", fontsize=12)
    plt.xlabel("time [sec]", fontsize=12)
    plt.suptitle(solver.heat_solver.title, fontsize=12)
    plt.show()

    # plt.plot(times, -x_boundary)
    # plt.grid()
    # # plt.yscale("log")
    # # plt.xscale("log")
    # # plt.legend()
    # plt.autoscale(enable=True, axis='both', tight=True)
    # plt.xlabel("time [sec]", fontsize=12)
    # plt.ylabel("$-x_{{boundary}}$ [cm]", fontsize=12)
    # plt.suptitle(solver.title, fontsize=12)
    # plt.show()

    # num_cells = 200
    # coordinate = np.array(list(sorted(set(
    #     list(np.linspace(0., L, num_cells+1)) \
    # ))))

    num_cells = 100
    coordinate = np.array(list(sorted(set(
        list(np.linspace(0., L/1000, num_cells*6)) +\
        list(np.linspace(L/1000, L/20, num_cells*6)) +\
        list(np.linspace(L/20, L/3., num_cells)) +\
        list(np.linspace(L/3., L, num_cells+1))
    ))))

    dx = coordinate[1:] - coordinate[:-1]
    rcell = 0.5*(coordinate[1:] + coordinate[:-1])

    density = rho0 / (1.-omega) * (coordinate[1:]**(1.-omega) - coordinate[:-1]**(1.-omega))/(coordinate[1:] - coordinate[:-1])

    # exact integral of mass in each cell gives this density
    mass_cells = density * dx
    mass_from_cells = np.cumsum(mass_cells)
    mass = np.array([1e-30, 1e-7*mass_from_cells[0]]+ list(mass_from_cells))

    ##### plot hydro profiles
    for it, time in enumerate(np.array([0.25,0.5,0.75])*times[-1]):
        solution = solver.solve(mass=mass, time=time)
        
        for use_r in [True, False]:
            for fg in ["density", "velocity", "pressure", "temperature"]:
                plt.figure(fg+str(use_r))
                if use_r: 
                    plt.plot(solution["position"], solution[fg], label=f"t={time:g}", marker="o")
                    plt.axvline(x=solution["shock_position"], lw=2, c="k", ls="--")
                    plt.axvline(x=solution["heat_position"], lw=2, c="b", ls="--")
                    plt.axvline(x=solution["boundary_position"], lw=2, c="r", ls="--")
                    if fg == "density" and it==0:
                        plt.plot(rcell, density, ls="--", c="r", lw=1.5, label=f"inital density")
                else:
                    plt.plot(mass, solution[fg], label=f"t={time:g}")
                    plt.axvline(x=solution["ablated_mass"], lw=2, c="k", ls="--")
                    if fg == "density" and it==0:
                        plt.plot(mass_from_cells, density, ls="--", c="r", lw=2, label=f"inital density")
    for use_r in [True, False]:
        for fg in ["density", "velocity", "pressure", "temperature"]:
            plt.figure(fg+str(use_r))
            plt.legend()
            plt.grid()
            plt.autoscale(enable=True, axis='both', tight=True)
            if use_r: plt.xlabel("x [cm]")
            else: plt.xlabel("mass [g/cm^2]")
            plt.ylabel(fg)

    plt.show()

    ########## plot position as a function of time
    results = [solver.solve(mass=mass, time=time) for time in times[1:]]
    position_times = np.array([r["position"] for r in results]).T
    shock_position = np.array([r["shock_position"] for r in results])
    piston_position = np.array([r["piston_position"] for r in results])
    heat_position = np.array([r["heat_position"] for r in results])
    boundary_position = np.array([r["boundary_position"] for r in results])
    plt.figure("position")
    for pos in position_times:
        plt.plot(times[1:], pos, c="k",    lw=0.5)#, marker="o", markersize=1.)
    plt.plot(times[1:], shock_position,    lw=1.5, ls="--", c="r", label="shock")
    plt.plot(times[1:], heat_position,     lw=1.5, ls="--", c="fuchsia", label="heat")
    plt.plot(times[1:], piston_position,   lw=1.5, ls="--", c="b", label="piston")
    plt.plot(times[1:], boundary_position, lw=1.5, ls="--", c="g", label="boundary")

    plt.legend()
    plt.xlabel("time [sec]")
    plt.ylabel("position [cm]")
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.suptitle(solver.heat_solver.title, fontsize=12)
    plt.savefig("rt.png", bbox_inches='tight')
    plt.savefig("rt.pdf", bbox_inches='tight')

    plt.xlim([0., times[-1]])
    plt.ylim([-0.1*L, L])
    plt.suptitle(solver.heat_solver.title, fontsize=12)
    plt.savefig("rt_zoom.pdf", bbox_inches='tight')
    plt.savefig("rt_zoom.png", bbox_inches='tight')
    plt.show()
    quit()

def test_profiles():
    time_rise = Units.nsec

    # tau = 0.
    # m_max=0.003

    tau = 0.123
    m_max=0.0016

    Tb = Units.hev_kelvin * (1./time_rise)**tau
    omega = 0.
    assert omega==0.

    # MORDI Au
    times_profiles = np.array([0.05, 0.1, 0.15])*Units.nsec
    alpha=1.5
    beta=1.6
    f=3.4*1e13/(Units.hev_kelvin**beta)
    g=1/(7200.*Units.hev_kelvin**alpha)
    mat=dict(
        rho0=19.32,
        tau=tau,
        alpha=alpha,
        beta_heat=beta,
        lambdap=0.2,
        mu_heat=0.14,
        gamma_heat=1.25,
        Tb=Tb,
        g=g,
        f_heat=f,
        omega=omega,
    )

    # mat=dict(
    #     tau=tau,
    #     g=5.682464574187187e-6,
    #     alpha=0.23183890413119132,
    #     lambdap=0.13829986198312327,
    #     f_heat=3168.6938116243796,
    #     beta_heat=1.6525149133503083,
    #     mu_heat=0.1206770123528319,
    #     gamma_heat=1.262299832903183,
    #     Tb=Tb,
    #     rho0=19.32,
    #     omega=omega,
    # )
    
    for key in ["f_heat", "beta_heat", "mu_heat", "gamma_heat"]:
        mat[key.replace("_heat", "_shock")] = mat[key]
    
    import pickle
    filename = "solver.pickle"
    from os import path
    if path.isfile(filename):
        with open(filename, 'rb') as handle:
            solver = pickle.load(handle)
            for key, value in mat.items():
                keyp=key.replace("_heat", "").replace("_shock", "")
                print(key, value, [getattr(solver, key, None), getattr(solver.heat_solver, key, None), getattr(solver.shock_solver, key, None)])
                assert value in {getattr(solver, keyp, None), getattr(solver.heat_solver, keyp, None), getattr(solver.shock_solver, keyp, None)}
    else:
        solver = AblationSolver(
            **mat,
        )
        with open(filename, 'wb') as handle:
            pickle.dump(solver, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plot mass
    times = np.linspace(0., 2.061e-9, 200)
    m_ab = np.array([solver.heat_solver.ablated_mass(time=t) for t in times])
    # x_boundary = np.array([solver.boundary_position(time=t) for t in times])

    plt.plot(times, m_ab)
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("ablated mass [g]", fontsize=12)
    plt.xlabel("time [sec]", fontsize=12)
    plt.suptitle(solver.heat_solver.title, fontsize=12)
    plt.show()

    # plt.plot(times, -x_boundary)
    # plt.grid()
    # # plt.yscale("log")
    # # plt.xscale("log")
    # # plt.legend()
    # plt.autoscale(enable=True, axis='both', tight=True)
    # plt.xlabel("time [sec]", fontsize=12)
    # plt.ylabel("$-x_{{boundary}}$ [cm]", fontsize=12)
    # plt.suptitle(solver.title, fontsize=12)
    # plt.show()

    # # num_cells = 1000
    # # coordinate = np.array(list(sorted(set(
    # #     list(np.linspace(0., L/100, num_cells+1)) + 
    # #     list(np.linspace(L/100, L/20, num_cells+1)) + \
    # #     list(np.linspace(L/20, L/5, num_cells+1)) + \
    # #     list(np.linspace(L/5, L, num_cells+1)) \
    # # ))))

    L = 1e-3
    num_cells = 5000
    coordinate = np.array(list(sorted(set(
        list(np.linspace(0., L, num_cells+1)) \
    ))))

    dx = coordinate[1:] - coordinate[:-1]
    rcell = 0.5*(coordinate[1:] + coordinate[:-1])

    density = mat["rho0"] / (1.-omega) * (coordinate[1:]**(1.-omega) - coordinate[:-1]**(1.-omega))/(coordinate[1:] - coordinate[:-1])

    # exact integral of mass in each cell gives this density
    mass_cells = density * dx
    mass = np.cumsum(mass_cells)
    mass = np.array([1e-30, 1e-7*mass[0]]+ list(mass))

    ##### plot hydro profiles
    for time in times_profiles:
        solution = solver.solve(mass=mass, time=time)
        solution["temperature"] /= Units.hev_kelvin
        for use_r in [True, False]:
            for fg in ["density", "velocity", "pressure", "temperature"]:
                plt.figure(fg+str(use_r))
                if use_r: 
                    plt.plot(solution["position"], solution[fg], label=f"t={time:g}", marker="o")
                    plt.axvline(x=solution["shock_position"], lw=2, c="k", ls="--")
                    plt.axvline(x=solution["heat_position"], lw=2, c="b", ls="--")
                    plt.axvline(x=solution["boundary_position"], lw=2, c="r", ls="--")
                else:    
                    plt.plot(mass, solution[fg], label=f"t={time:g}")
                    plt.axvline(x=solution["ablated_mass"], lw=2, c="k", ls="--")
    for use_r in [True, False]:
        for fg in ["density", "velocity", "pressure", "temperature"]:
            plt.figure(fg+str(use_r))
            plt.legend()
            plt.grid()
            plt.autoscale(enable=True, axis='both', tight=True)
            if use_r: plt.xlabel("x [cm]")
            else: 
                plt.xlabel("mass [g/cm^2]")
                plt.xlim(xmax=m_max)
            plt.ylabel(fg)

    plt.show()

    ########## plot position as a function of time
    results = [solver.solve(mass=mass, time=time) for time in times[1:]]
    position_times = np.array([r["position"] for r in results]).T
    shock_position = np.array([r["shock_position"] for r in results])
    piston_position = np.array([r["piston_position"] for r in results])
    heat_position = np.array([r["heat_position"] for r in results])
    boundary_position = np.array([r["boundary_position"] for r in results])
    plt.figure("position")
    for pos in position_times:
        plt.plot(times[1:], pos, c="k",    lw=0.5)#, marker="o", markersize=1.)
    plt.plot(times[1:], shock_position,    lw=2.5, ls="--", c="r", label="shock")
    plt.plot(times[1:], heat_position,     lw=2.5, ls="--", c="fuchsia", label="heat")
    plt.plot(times[1:], piston_position,   lw=1.5, ls="--", c="b", label="piston")
    plt.plot(times[1:], boundary_position, lw=1.5, ls="--", c="k", label="boundary")

    plt.legend()
    plt.xlabel("time [sec]")
    plt.ylabel("position [cm]")
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.suptitle(solver.heat_solver.title, fontsize=12)
    plt.savefig("rt.png", bbox_inches='tight')
    plt.savefig("rt.pdf", bbox_inches='tight')

    plt.xlim([0., times[-1]])
    plt.ylim([-0.1*L, L])
    plt.suptitle(solver.heat_solver.title, fontsize=12)
    plt.savefig("rt_zoom.pdf", bbox_inches='tight')
    plt.savefig("rt_zoom.png", bbox_inches='tight')
    plt.show()

    # plt.plot(np.log(times[1:]), np.log(heat_position),    lw=2.5, ls="--", c="r", label="shock/heat")
    # plt.plot(np.log(times[1:]), np.log(shock_position),    lw=2.5, ls="--", c="r", label="shock/heat")
    # # plt.plot(times[1:], shock_position/heat_position,    lw=2.5, ls="--", c="r", label="shock/heat")
    # # plt.plot(times[1:], shock_position/piston_position,     lw=2.5, ls="--", c="fuchsia", label="shock/piston")
    # plt.show()

    quit()

if __name__ == "__main__":

    test_profiles()
    # test_profiles_omega()