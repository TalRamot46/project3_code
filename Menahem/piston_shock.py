import numpy as np
import sys
import scipy.integrate
import scipy.optimize
from matplotlib import pyplot as plt

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('PistonShock')

class NumericalParameters:
    xsi_origin = 1e-50
    xsi_s_initial_guess = 1.
    eps_fatal_xsi_s_root = 1e-5 #the maximal allowed tolerance of the root finding for xsi_s

class PistonShock():
    """An object to computes the solution to the piston shock wave problem 
    In a medium with an initial power law density
    rho(r, t=0) = rho0 * r ** -omega
    This is a planar problem with Boundary temperature power law: 
    P(m=0, t) = p0 * t ** tau
    where p0 and tau are given input values.
    """

    def __init__(self, *,
                 rho0,      # spatial density coeff
                 omega,     # spatial density power
                 p0,        # boundary pressure coeff
                 tau,       # boundary pressure temporal power
                 gamma,     # adiabatic index
                 ode_scheme="dopri5"):

        logger.info(f"creating a PistonShock calculator...")

        self.rho0 = rho0
        self.omega = omega
        self.p0 = p0
        self.tau = tau
        self.gamma = gamma
        self.ode_scheme = ode_scheme

        logger.info(f"rho0={self.rho0:g}")
        logger.info(f"omega={self.omega:g}")
        logger.info(f"p0={self.p0:g}")
        logger.info(f"tau={self.tau:g}")
        logger.info(f"gamma={self.gamma:g}")
        logger.info(f"ode_scheme={self.ode_scheme!r}")
        
        assert self.rho0 > 0.
        assert self.p0 > 0.
        assert self.omega < 1., f"must have omega < 1"
        assert self.gamma > 1., f"must have gamma > 1"
        assert self.tau > -2., f"must have tau > -2 for shock to propagate outwards"
        assert ode_scheme in {"dopri5", "dop853", "lsoda", "zvode", "vode"}
        
        self.v0 = 1. / self.rho0
        self.r = gamma - 1.
        self.rep = f"tau={self.tau:g} w={self.omega:g}"

        # coefficients for exact calculation of V integral
        self.q1 = 1.-self.omega
        self.q2 = (2.-self.omega) / (self.tau + 2.)

        self.ode_solver = scipy.integrate.ode(self.fode).set_integrator(ode_scheme)
        
        # Solve ODE via R(xsi)=1/V or via V(xsi)
        self.use_rho_ode = False
        # self.use_rho_ode = True
        # self.use_rho_ode = self.omega < 0.

        self.xsi_s = None

        # calculate the exact self similar shock coordinate xsi_s (by requiring that integration of the ODE from the xsi_s to the origin that that P(0)=1)
        self.xsi_s = self.find_xsi_s_fsolve()
        # self.xsi_s = self.find_xsi_s_bisect()

        self.title = f"$\\omega={self.omega:.4g}" \
                     f", \\ \\ \\tau={self.tau:.4g}" \
                     f", \\ \\ \\xi_{{s}}={self.xsi_s:.5g}" \
                     f", \\ \\ \\gamma={self.gamma:.4g}$"
        
        # values at the origin
        self.R0_or_V0, self.U0, self.P0 = self.integrate_inward(xsi_s=self.xsi_s)

        # values at the shock front
        self.Rs_or_Vs, self.Us, self.Ps = self.get_shock_values(xsi_s=self.xsi_s)

        if self.use_rho_ode:
            self.R0, self.Rs = self.R0_or_V0, self.Rs_or_Vs
            self.V0, self.Vs = 1./self.R0, 1./self.Rs
        else:
            self.V0, self.Vs = self.R0_or_V0, self.Rs_or_Vs
            self.R0, self.Rs = 1./self.V0, 1./self.Vs 

        V0xsi0 = self.V0*NumericalParameters.xsi_origin

        logger.info(f"{self.rep}: solutions at the origin: R0={self.R0:g}, V0={self.V0:g}, U0={self.U0:g}, P0={self.P0:g}")
        logger.info(f"{self.rep}: solutions at the shock:  Rs={self.Rs:g}, Vs={self.Vs:g}, Us={self.Us:g}, Ps={self.Ps:g}")
        logger.info(f"{self.rep}: V(xsi_origin)={self.V0} xsi_origin={NumericalParameters.xsi_origin} [V(xsi)*xsi]={V0xsi0}")

        if self.V0 <= 0.:
            logger.fatal(f"{self.rep}: integration failed, negative value for V0={self.V0}")
        if self.U0 <= 0.:
            logger.fatal(f"{self.rep}: integration failed, negative value for U0={self.U0}")
        if self.P0 <= 0.:
            logger.fatal(f"{self.rep}: integration failed, negative value for P0={self.P0}")
        if V0xsi0 > 1e-7:
            logger.fatal(f"{self.rep}: integration failed, V0xsi0={V0xsi0} too large")
        if self.V0 <= 0. or self.U0 <= 0. or self.P0 <= 0. or V0xsi0 > 1e-7:
            self.plot_profiles()
            sys.exit(1)
        
        # total integral of V(xsi) on [0, xsi_s]
        self.V_integral_total = self.get_V_integral(xsi=self.xsi_s, V=self.Vs, U=self.Us)
        logger.info(f"{self.rep}: integral of V(xsi) on [0, xsi_s] = {self.V_integral_total}")

        self.set_energy_integrals()

    def initial_specific_volume(self, *, mass):
        """
        returns the initial unperturbed specific volume v(m, 0) a given Lagrangian mass coordinate
        """
        return (self.v0*((1.-self.omega)*mass)**self.omega)**(1./(1.-self.omega))
        
    def initial_position(self, *, mass):
        """
        returns the initial eulerian position of a given Lagrangian mass coordinate
        """
        return (self.v0*(1.-self.omega)*mass)**(1./(1.-self.omega))

    def _position_temporal_factor(self, *, time):
        return (self.v0*self.p0*time**(2.+self.tau))**(1./(2.-self.omega))

    def _velocity_temporal_factor(self, *, time):
        return self._position_temporal_factor(time=time) * (2.+self.tau)/(time*(2.-self.omega))

    def _energy_temporal_factor(self, *, time):
        return (self.v0*self.p0**(3.-self.omega)*time**(self.tau*(3.-self.omega)+2.))**(1./(2.-self.omega))

    def piston_position(self, *, time):
        """
        returns position of the piston (the system boundary on which the pressure b.c. is acting)
        at the given time
        """
        return self._position_temporal_factor(time=time) * self.U0 * (2.-self.omega)/(self.tau+2.)

    def piston_velocity(self, *, time):
        """
        returns velocity of the piston (the system boundary on which the pressure b.c. is acting)
        at the given time
        """
        return self._velocity_temporal_factor(time=time) * self.U0 * (2.-self.omega)/(self.tau+2.)
    
    def mass_to_width(self, *, time, V_integral):
        """
        returns the spatial distance from the system boundary (the piston),
        of a given Lagrangian coordinage which has a given total V_integral.
        """
        return self._position_temporal_factor(time=time) * V_integral

    def shocked_width(self, *, time):
        """
        returns the spatial width of the shocked region (from the system boundary (the piston)),
        at the given time
        """
        return self.mass_to_width(time=time, V_integral=self.V_integral_total)

    def shock_position(self, *, time):
        """
        returns the shock wave position at the given time
        """
        return self.piston_position(time=time) + self.shocked_width(time=time)

    def shock_position2(self, *, time):
        """
        another method to calculate the shock wave position at the given time
        it is algebriacally equal to the method shock_position
        """
        return self.initial_position(mass=self.shocked_mass(time=time))

        # # Substitution of the expression for shocked_mass in the initial position formula,
        # # gives the following direct expression, see lyx
        # return self._position_temporal_factor(time=time) * ((1.-self.omega)*self.xsi_s)**(1./(1.-self.omega))

    def shock_velocity(self, *, time):
        """
        returns shock velocity at the given time
        """
        return self._velocity_temporal_factor(time=time) * ((1.-self.omega)*self.xsi_s)**(1./(1.-self.omega))

    def shock_time(self, *, shock_position):
        """
        returns time for the shock wave to reach a given position x
        """
        return (1. / (self.v0*self.p0) * (shock_position / ((1.-self.omega)*self.xsi_s)**(1./(1.-self.omega))) ** (2. - self.omega)) ** (1./(2.+self.tau))

    def shocked_mass(self, *, time):
        """
        returns the mass of the shocked region at the given time
        """
        return self.xsi_s / self.xsi_over_m(time=time) 

    def kinetic_energy(self, *, time):
        """
        returns the total kinetic energy in the system at the given time
        """
        return self._energy_temporal_factor(time=time) * self.energy_kinetic_intgeral

    def internal_energy(self, *, time):
        """
        returns the total internal energy in the system at the given time
        """
        return self._energy_temporal_factor(time=time) * self.energy_internal_intgeral

    def total_energy(self, *, time):
        """
        returns the total energy in the system at the given time
        """
        return self._energy_temporal_factor(time=time) * self.energy_total_intgeral

    def xsi_over_m(self,*, time):
        return (self.v0*(self.p0*time**(self.tau+2.))**(self.omega-1.))**(1./(2.-self.omega))

    def solve(self, *, mass, time):
        """
        calculates the hydrodynamic profiles on the given
        Largrangian mass coordinates at the given time
        """
        # logger.info(f"{self.rep}: calculating hydro profiles at t={time:g}")
        assert time > 0.
        assert type(mass) == np.ndarray
        assert len(mass) > 0
        assert all(m>0. for m in mass)
        if len(mass) > 1:
            assert np.min(mass[1:]-mass[:1]) > 0. #Lagrangian coordinate (acumulated cell mass) is monotone

        # self similar coordinate
        xsi_vec = mass * self.xsi_over_m(time=time)

        assert xsi_vec[-1] > 100 * NumericalParameters.xsi_origin

        # self similar profiles
        V, U, P = self.get_self_similar_profiles(xsi_vec=xsi_vec)

        position = np.array([
             # shocked region
            self._position_temporal_factor(time=time)*(self.q1*xsi*Vi+self.q2*Ui)\
            if xsi <= self.xsi_s else \
            # unshocked region
            self.initial_position(mass=m)
            for xsi, m, Vi, Ui in zip(xsi_vec, mass, V, U)
        ])
        
        # make sure lines do no cross
        assert sorted(position) == list(position), print(position)

        # hydodynamic profiles from self similar profiles
        v = V * (self.v0*self.v0*(self.p0*time**(self.tau+2.))**self.omega)**(1./(2.-self.omega))
        u = U * (self.v0*self.p0*time**(self.omega+self.tau))**(1./(2.-self.omega))
        p = P * self.p0 * time**self.tau

        # fill the unshocked region with the initial density
        for i in reversed(range(len(xsi_vec))):
            if xsi_vec[i] > self.xsi_s:
                assert v[i] == 0.
                v[i] = self.initial_specific_volume(mass=mass[i])
            else:
                if i < len(xsi_vec)-1:
                    logger.debug(f"reached shock at cell={i+1} xsi=[{xsi_vec[i]:g}, {xsi_vec[i+1]:g}] xsi_s={self.xsi_s:g}")
                else:
                    logger.debug("shock is out of the system")
                break
        
        rho = 1. / v

        return dict(
            piston_position=self.piston_position(time=time), 
            piston_velocity=self.piston_velocity(time=time), 
            shock_position=self.shock_position(time=time), 
            shock_velocity=self.shock_velocity(time=time), 
            position=position, 
            density=rho, 
            velocity=u, 
            pressure=p,
            sie=p*v/self.r, #specific internal energy
        )
        
    def get_shock_values(self,*, xsi_s):
        """
        get the values of the self similar profiles at the shock front
        """
        assert xsi_s > 0.

        # Hugoniot
        w, r, tau = self.omega, self.r, self.tau
        Vs = r / (r + 2.) * ((1.-w)*xsi_s)**(w / (1. - w))
        Us = 2. / r * (2. + tau) * (1. - w) / (2. - w) * xsi_s * Vs
        Ps = r / Vs * 0.5*(Us*Us)
        
        assert Vs > 0.
        assert Us > 0.
        assert Ps > 0.

        Rs_or_Vs = 1./Vs if self.use_rho_ode else Vs

        return Rs_or_Vs, Us, Ps

    def fode(self, xsi, x):
        """
        the self similar ODE, written as a first order 
        form ODE in the dx/dxsi=F(x, xsi), where x=(V, U, P) or (rho, U, P)
        """
        assert xsi >= NumericalParameters.xsi_origin, xsi
        w, tau, r = self.omega, self.tau, self.r

        U, P = x[1], x[2]
        in2w = 1. / (2. - w)
        a = (tau + 2.) * (1. - w) * in2w
        a2 = a*a
        a3 = a2*a
        b = (1. + r) * a * P
        c1 = (tau + 2.) * w * in2w
        c2 = U * (w + tau) * in2w
        c3 = P * (tau + (1. + r) * w * (tau + 2.) * in2w)

        # ODE for (R, U, P)
        if self.use_rho_ode:
            R = x[0]
            V = 1. / R
            X = xsi * V
            delta1 = -(c1*a2*xsi - c2*a*R - c3/X)
        
        # ODE for (V, U, P)
        else:
            V = x[0]
            X = xsi * V
            delta1 = V*(c1*a2*X - c2*a  - c3/xsi)

        delta = a3*xsi*X - b
        delta2 =  (c3*a -c1*b)*V   + c2*a2*X
        delta3 =  (c3*a2-c1*a*b)*X + c2*b

        if abs(delta) < 1e-30:
            logger.fatal(f"{self.rep}: discrimenant {delta} to close to zero")
            quit()
        
        return [delta1/delta, delta2/delta, delta3/delta]

    def integrate_inward(self, *, xsi_s):
        """
        An auxiliary function used find the exact value of xsi_s via the shooting method.
        It integrates the self similar ODE from a given xsi_s (not necessarily the exact one),
        to the the origin.
        returns the values of V,U,P at the origin.
        For the correct xsi_s, the value of P should be equal to 1.0,
        Therefore, the root of integrate_inward(xsi_s)[2]-1=0 is the exact xsi_s of the problem.
        """
        if isinstance(xsi_s, np.ndarray):
            # scipy root-finders may pass a single-value ndarray.
            assert xsi_s.size == 1
            xsi_s = float(xsi_s[0])
        assert xsi_s > 0., xsi_s

        # Hugoniot - values at the shock
        Rs_or_Vs, Us, Ps = self.get_shock_values(xsi_s=xsi_s)

        # integrating from to shock to the origin
        self.ode_solver.set_initial_value([Rs_or_Vs, Us, Ps], xsi_s)

        ## integrate directly to origin (the ode solver will figure out the stepsize)
        ## may not be accurate enough - we need extra resolution near origin 
        # R0_or_V0, U0, P0 = self.ode_solver.integrate(NumericalParameters.xsi_origin)

        # linear + logarithmic grid near origin
        xsi_vec = list(np.geomspace(NumericalParameters.xsi_origin, xsi_s/10., 100)) + \
                  list(np.linspace(xsi_s/2., xsi_s, 100))

        # integrate from the first point before xsi_s (in the array xsi_vec) towards the origin
        for xsi in xsi_vec[::-1][1:]:
            value = self.ode_solver.integrate(xsi)

        assert xsi == NumericalParameters.xsi_origin

        R0_or_V0, U0, P0 = value
        logger.info(f"{self.rep}: integrate inward with xsi_s={xsi_s}: U0={U0} P0={P0}, f0-1={P0-1}")
        
        return R0_or_V0, U0, P0

    def fxsi_s(self, xsi_s):
        """
        xsi_s is the root of this function
        returns P(0)-1, where P was integrated from the given xsi_s to the origin
        """
        if isinstance(xsi_s, np.ndarray):
            # fsolve wraps scalar unknowns as shape-(1,) arrays.
            assert xsi_s.size == 1
            xsi_s = float(xsi_s[0])
        return self.integrate_inward(xsi_s=xsi_s)[2]-1.

    def find_xsi_s_fsolve(self):
        """
        use scipy.optimize.fsolve to find xsi_s
        """
        xsi_s_root = scipy.optimize.fsolve(func=self.fxsi_s, x0=NumericalParameters.xsi_s_initial_guess)
        print(xsi_s_root, 0.9*self.fxsi_s(xsi_s_root), self.fxsi_s(xsi_s_root), 1.1*self.fxsi_s(xsi_s_root), np.isclose(self.fxsi_s(xsi_s_root), 0., rtol=1e-8, atol=1e-8))
        assert np.isclose(self.fxsi_s(xsi_s_root), 0., rtol=NumericalParameters.eps_fatal_xsi_s_root, atol=NumericalParameters.eps_fatal_xsi_s_root), f"find_xsi_s_fsolve failed omega={self.omega:g} err={self.fxsi_s(xsi_s_root)}"
        logger.info(f"{self.rep}: Final calculated xsi_s={self.xsi_s}")
        return xsi_s_root[0]

    def find_xsi_s_bisect(self):
        """
        use scipy.optimize.bisect to find xsi_s
        """
        xsi_s_root = scipy.optimize.bisect(f=self.fxsi_s, a=1e-5, b=100., maxiter=1000)
        print(xsi_s_root, 0.9*self.fxsi_s(xsi_s_root), self.fxsi_s(xsi_s_root), 1.1*self.fxsi_s(xsi_s_root), np.isclose(self.fxsi_s(xsi_s_root), 0., rtol=1e-8, atol=1e-8))
        assert np.isclose(self.fxsi_s(xsi_s_root), 0., rtol=NumericalParameters.eps_fatal_xsi_s_root, atol=NumericalParameters.eps_fatal_xsi_s_root), f"find_xsi_s_bisect failed omega={self.omega:g} err={self.fxsi_s(xsi_s_root)}"
        logger.info(f"{self.rep}: Final calculated xsi_s={self.xsi_s}")
        return xsi_s_root

    def plot_fxsi_s(self):
        """
        plots the function for which xsi_s is a root
        """
        xsi_s_vec = np.linspace(0.001, 3., 50)
        f0 = np.vectorize(self.fxsi_s)(xsi_s_vec)
        plt.plot(xsi_s_vec, f0, "b")
        # plt.axhline(y=0., ls="--", lw=2, c="k")
        if self.xsi_s != None:
            plt.axvline(x=self.xsi_s, ls="--", lw=1.5, c="k")

        plt.suptitle(self.title, fontsize=12)
        plt.grid()
        plt.ylabel("$F(\\xi_{{s}})$", fontsize=12)
        plt.xlabel("$\\xi_{{s}}$", fontsize=12)
        plt.autoscale(enable=True, axis='both', tight=True)
        # plt.savefig("fxsi_s.png", bbox_inches='tight')
        # plt.savefig("fxsi_s.pdf", bbox_inches='tight')
        plt.show()

        return self

    def get_self_similar_profiles(self, *, xsi_vec):
        """
        calculate the self similar profiles V, U, P evaluated 
        on the values of the given vector of xsi
        for the unshocked region xsi>xsi_s, we fill with 0
        """
        N = len(xsi_vec)
        assert N > 1
        assert sorted(list(xsi_vec)) == list(xsi_vec)
        assert all(xsi>=NumericalParameters.xsi_origin for xsi in xsi_vec)
        
        V, U, P = np.zeros(N), np.zeros(N), np.zeros(N)
        initial_values = self.Rs_or_Vs, self.Us, self.Ps
        self.ode_solver.set_initial_value(initial_values, self.xsi_s)
        for i in reversed(range(N)):
            if self.xsi_s * (1-1e-9) < xsi_vec[i] <= self.xsi_s:
                # on the shock front (in the shocked side) 
                value = initial_values
            elif xsi_vec[i] < self.xsi_s:
                # shocked region
                value = self.ode_solver.integrate(xsi_vec[i])
                assert value[0] > 0.
                assert value[1] > 0.
                assert value[2] > 0.
            else:
                # unshocked region
                value = 0., 0., 0.
            V[i], U[i], P[i] = value
            if self.use_rho_ode and V[i] > 0.: V[i] = 1./V[i]
        
        return V, U, P

    def get_V_integral(self, *, xsi, V, U):
        """
        calculate the integral of V in the range [0, xsi],
        given the values V, U evaluated the the end point xsi
        The integral is obtained from an exact analytic expression
        """
        assert xsi > 0.
        if xsi > self.xsi_s: 
            assert V == 0.
            assert U == 0.
            return 0.
        assert xsi >= NumericalParameters.xsi_origin

        if V <=0. or U <= 0.:
            logger.fatal(f"{self.rep}: V integral failed, negative value xsi={xsi} V={V} U={U}")
            sys.exit(1)

        integ = self.q1*xsi*V+self.q2*(U-self.U0)
        if integ < 0.:
            # numerical errors can give a small negative result for very small xsi
            assert integ >= -1e-8 and xsi < 1e-9, f"{integ} {xsi}"
            integ = 0.
        return integ

    def set_energy_integrals(self):
        assert self.xsi_s != None

        logger.info(f"{self.rep}: setting energy integrals...")

        xsi_end = self.xsi_s
        xsi_vec = np.array(sorted(list(set(
            list(np.linspace( NumericalParameters.xsi_origin, xsi_end, 10000)) + \
            list(np.geomspace(NumericalParameters.xsi_origin, xsi_end/10, 5000))
        ))))
            
        V, U, P = self.get_self_similar_profiles(xsi_vec=xsi_vec)
        integrand_kin = 0.5*U*U
        integrand_in = P*V/self.r
        integrand_tot = integrand_kin + integrand_in
        ekin_integral_simps = scipy.integrate.simpson(y=integrand_kin, x=xsi_vec)
        ein_integral_simps = scipy.integrate.simpson(y=integrand_in, x=xsi_vec)
        etot_integral_simps = scipy.integrate.simpson(y=integrand_tot, x=xsi_vec)
        
        logger.info(f"{self.rep}: ekin_integral simps={ekin_integral_simps}")
        logger.info(f"{self.rep}: ekin_integral trapz={np.trapezoid(y=integrand_kin, x=xsi_vec)}")

        logger.info(f"{self.rep}: ein_integral simps={ein_integral_simps}")
        logger.info(f"{self.rep}: ein_integral trapz={np.trapezoid(y=integrand_in, x=xsi_vec)}")

        logger.info(f"{self.rep}: etot_integral simps={etot_integral_simps}")
        logger.info(f"{self.rep}: etot_integral trapz={np.trapezoid(y=integrand_tot, x=xsi_vec)}")

        self.energy_time_power = (self.tau * (3.-self.omega) + 2.) / (2.-self.omega)
        logger.info(f"{self.rep}: energy time power={self.energy_time_power:g}")

        self.energy_total_intgeral = self.U0/self.energy_time_power

        logger.info(f"{self.rep}: etot_integral anal={self.energy_total_intgeral}")

        assert abs(self.energy_total_intgeral/etot_integral_simps-1) < 1e-5
        self.energy_kinetic_intgeral = ekin_integral_simps
        self.energy_internal_intgeral = ein_integral_simps

        logger.info(f"{self.rep}: kinetic/internal ratio={self.energy_kinetic_intgeral/self.energy_internal_intgeral:g}")

        return self

    def test_V_integral(self):
        assert self.xsi_s != None

        xsi_end = self.xsi_s
        xsi_vec = np.array(sorted(list(set(
            list(np.linspace( NumericalParameters.xsi_origin, xsi_end, 10000)) + \
            list(np.geomspace(NumericalParameters.xsi_origin, xsi_end/10, 500))
        ))))
            
        
        V, U, _ = self.get_self_similar_profiles(xsi_vec=xsi_vec)
        
        V_integral_total_simps = scipy.integrate.simpson(y=V, x=xsi_vec)
        
        logger.info(f"{self.rep}: V_integral_total simps={V_integral_total_simps}")
        logger.info(f"{self.rep}: V_integral_total trapz={np.trapezoid(y=V, x=xsi_vec)}")

        # xsi_integral_vec = np.array(xsi_vec)
        # V_integral_cum = scipy.integrate.cumtrapz(y=V, x=xsi_vec)
        # logger.info(f"{self.rep}: V_integral_total trapz={V_integral_cum[-1]}")


        # print(xsi_vec[0]*V[0])
        # print(xsi_vec)
        # print(V)

        vint = self.q1*(xsi_vec[-1]*V[-1]) + self.q2 * (U[-1]-U[0])
        logger.info(f"{self.rep}: V0 {V[0]} U0 {U[0]}")
        logger.info(f"{self.rep}: vint {vint}")

        logger.info(f"{self.rep}: V0 {self.V0} U0 {self.U0}")
        logger.info(f"{self.rep}: vvvint={self.V_integral_total}")

        err = abs(self.V_integral_total/V_integral_total_simps-1.)
        logger.info(f"{self.rep}: V_integral_total error vs simpson={err:g}")
        assert err < 1e-3, err
        
        return self

    def test_etot_integral(self):
        assert self.xsi_s != None

        xsi_end = self.xsi_s
        xsi_vec = np.array(sorted(list(set(
            list(np.linspace( NumericalParameters.xsi_origin, xsi_end, 10000)) + \
            list(np.geomspace(NumericalParameters.xsi_origin, xsi_end/10, 5000))
        ))))
            
        
        V, U, P = self.get_self_similar_profiles(xsi_vec=xsi_vec)
        integrand = P*V/self.r+0.5*U*U
        etot_integral_total_simps = scipy.integrate.simpson(y=integrand, x=xsi_vec)
        
        logger.info(f"{self.rep}: etot_integral_total simps={etot_integral_total_simps}")
        logger.info(f"{self.rep}: etot_integral_total trapz={np.trapezoid(y=integrand, x=xsi_vec)}")

        denom = self.tau * (3.-self.omega) + 2.
        eint = (1-self.omega)*(2.+self.tau)*xsi_vec[-1]*integrand[-1] -(2.-self.omega) * (P[-1]*U[-1]-U[0])
        eint /= denom
        logger.info(f"{self.rep}: V0 {V[0]} U0 {U[0]}")
        logger.info(f"{self.rep}: eint {eint}")
        logger.info(f"{self.rep}: eints {(2.-self.omega)*self.U0/denom}")

        # cum plot
        # etot_integral_cum_trapz = scipy.integrate.cumtrapz(y=integrand, x=xsi_vec)

        # etot_integral_cum = (1-self.omega)*(2.+self.tau)*xsi_vec*integrand -(2.-self.omega) * (P*U-U[0])
        # etot_integral_cum /= denom

        # plt.plot(xsi_vec, etot_integral_cum)
        # plt.plot(0.5*(xsi_vec[1:]+xsi_vec[:-1]), etot_integral_cum_trapz, ls="--")
        # plt.show()

        return self

    def plot_profiles(self):
        """
        plots the self similar profiles on [0, xsi_s]
        """
        # xsi_vec = np.linspace(NumericalParameters.xsi_origin, self.xsi_s, 1000)
        xsi_vec = np.array(sorted(list(set(
            list(np.linspace( NumericalParameters.xsi_origin, self.xsi_s, 10000)) + \
            list(np.geomspace(NumericalParameters.xsi_origin, self.xsi_s/10, 100))
        ))))

        V, U, P = self.get_self_similar_profiles(xsi_vec=xsi_vec)

        V_integral = [self.get_V_integral(xsi=xsi, V=Vi, U=Ui) for xsi, Vi, Ui in zip(xsi_vec, V, U)]
        V_integral_cum = scipy.integrate.cumulative_trapezoid(y=V, x=xsi_vec)

        plt.figure()
        # plt.plot(xsi_vec, 1./V, label=f"$R(\\xi)$")
        plt.plot(xsi_vec, V, label=f"$V(\\xi)$")
        # plt.plot(xsi_vec, V*xsi_vec, label=f"$\\xi V(\\xi)$")
        # plt.plot(xsi_vec, V_integral, label=f"$\\int V(\\xi)$")
        # plt.plot(0.5*(xsi_vec[1:]+xsi_vec[:-1]), V_integral_cum, ls="--", label=f"$\\int V(\\xi)$ trapz")
        plt.plot(xsi_vec, U, label=f"$U(\\xi)$")
        plt.plot(xsi_vec, P, label=f"$P(\\xi)$")

        # R~xsi**k for xsi->0 (in cases that R->0 or infty at xsi->0)
        R_pow_asymp = self.tau*(2.-self.omega)/((self.r+1.)*(self.tau+2.)*(1.-self.omega)) + self.omega/(1.-self.omega)
        plt.plot(xsi_vec, xsi_vec**-R_pow_asymp, label=f"$R \\propto \\xi^{{k}}$,  k={-R_pow_asymp:g}")
        
        # plt.xscale("log")
        # plt.yscale("log")
        plt.suptitle(self.title)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("$\\xi$", fontsize=12)
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.show()

        return self

def test_xsi_s():
    r = 0.25 

    # # plots xsi_s as a function of tau for various omegas
    # tau_vec = np.linspace(-0.5, 2., 69)
    # wvec = [-0.5, -0.25, 0., 0.25, 0.5]
    
    # solutions = [[
    #     PistonShock(
    #         rho0=1.,
    #         omega=w,
    #         p0=1.,
    #         tau=tau,
    #         gamma=1.+r,
    #     )
    #     for tau in tau_vec]
    # for w in wvec]

    # xsi_s = np.array([[solutions[iw][it].xsi_s for it in range(len(tau_vec))] for iw in range(len(wvec))])
    # U0 = np.array([[solutions[iw][it].U0 for it in range(len(tau_vec))] for iw in range(len(wvec))])

    # plt.figure("xsis")
    # for iw, w in enumerate(wvec):
    #     plt.plot(tau_vec, xsi_s[iw, :], lw=1.5, label=f"$\\omega={w:g}$")

    # plt.grid()
    # plt.legend(fontsize=12)
    # plt.xlabel("$\\tau$", fontsize=13)
    # plt.ylabel("$\\xi_{{s}}$", fontsize=13)
    # plt.autoscale(enable=True, axis='both', tight=True)
    # plt.savefig("xsis_tau.png", bbox_inches='tight')
    # plt.savefig("xsis_tau.pdf", bbox_inches='tight')
    # plt.show()

    # plt.figure("U0")
    # for iw, w in enumerate(wvec):
    #     plt.plot(tau_vec, U0[iw, :], lw=1.5, label=f"$\\omega={w:g}$")

    # plt.grid()
    # plt.legend(fontsize=12)
    # plt.xlabel("$\\tau$", fontsize=13)
    # plt.ylabel("$U(0)$", fontsize=13)
    # plt.autoscale(enable=True, axis='both', tight=True)
    # plt.savefig("U0_tau.png", bbox_inches='tight')
    # plt.savefig("U0_tau.pdf", bbox_inches='tight')
    # plt.show()

    # mesh plot tau-omega 
    tau_vec = np.linspace(-0.5, 2., 70)
    wvec = np.linspace(-0.5, 0.5, 50)

    import pickle
    filename = "solvers.pickle"
    from os import path
    if path.isfile(filename):
        with open(filename, 'rb') as handle:
            solvers = pickle.load(handle)
    else:
        solvers = [[
            PistonShock(
                rho0=1.,
                omega=w,
                p0=1.,
                tau=tau,
                gamma=1.+r,
            )
            for tau in tau_vec]
        for w in wvec]
        with open(filename, 'wb') as handle:
            pickle.dump(solvers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    xsi_s = np.array([[solvers[iw][it].xsi_s for it in range(len(tau_vec))] for iw in range(len(wvec))])
    U0 = np.array([[solvers[iw][it].U0 for it in range(len(tau_vec))] for iw in range(len(wvec))])
    eratio = np.array([[solvers[iw][it].energy_kinetic_intgeral/solvers[iw][it].energy_internal_intgeral for it in range(len(tau_vec))] for iw in range(len(wvec))])

    plt.figure("xsis_mesh")
    plt.pcolormesh(tau_vec, wvec, xsi_s, cmap='jet', shading='gouraud')
    plt.colorbar()
    CS = plt.contour(tau_vec, wvec, xsi_s, colors='k', lw=0.3, levels=list(np.linspace(0.4, 1., 7))+list(np.linspace(1.1, 2., 6)))
    plt.clabel(CS, inline=True, fontsize=7)
    plt.xlabel("$\\tau$", fontsize=13)
    plt.ylabel("$\\omega$", fontsize=13)
    plt.title("$\\xi_{{s}}(\\tau, \\omega)$", fontsize=14)
    plt.savefig("xsis_mesh.png", bbox_inches='tight')
    plt.savefig("xsis_mesh.pdf", bbox_inches='tight')
    plt.show()

    plt.figure("u0_mesh")
    plt.pcolormesh(tau_vec, wvec, U0, cmap='jet', shading='gouraud')
    plt.colorbar()
    CS = plt.contour(tau_vec, wvec, U0, colors='k', lw=0.3, levels=np.linspace(0.4, 1., 16))
    plt.clabel(CS, inline=True, fontsize=7)
    plt.xlabel("$\\tau$", fontsize=13)
    plt.ylabel("$\\omega$", fontsize=13)
    plt.title("$U(0)$", fontsize=14)
    plt.savefig("U0_mesh.png", bbox_inches='tight')
    plt.savefig("U0_mesh.pdf", bbox_inches='tight')
    plt.show()

    plt.figure("eratio_mesh")
    plt.pcolormesh(tau_vec, wvec, eratio, cmap='jet', shading='gouraud')
    plt.colorbar()
    CS = plt.contour(tau_vec, wvec, eratio, colors='k', lw=0.3, levels=np.linspace(0.4, 3.2, 15))
    plt.clabel(CS, inline=True, fontsize=7)
    plt.xlabel("$\\tau$", fontsize=13)
    plt.ylabel("$\\omega$", fontsize=13)
    plt.title("$E_{{k}}/E_{{in}}$", fontsize=14)
    plt.savefig("eratio_mesh.png", bbox_inches='tight')
    plt.savefig("eratio_mesh.pdf", bbox_inches='tight')
    plt.show()

    plt.figure("xspow")
    tau_vec = np.linspace(-2., 2., 100)
    wvec = np.linspace(-1., 1., 100)
    xspow = [[(tau+2.)/(2.-w) for tau in tau_vec] for w in wvec]
    plt.pcolormesh(tau_vec, wvec, xspow, cmap='jet', shading='gouraud')
    plt.colorbar()
    CS = plt.contour(tau_vec, wvec, xspow, colors='k', lw=0.3, levels=15)
    plt.clabel(CS, inline=True, fontsize=7)
    plt.xlabel("$\\tau$", fontsize=13)
    plt.ylabel("$\\omega$", fontsize=13)
    plt.title("shock position exponent", fontsize=12)
    plt.savefig("xspow_mesh.png", bbox_inches='tight')
    plt.savefig("xspow_mesh.pdf", bbox_inches='tight')
    plt.show()

    plt.figure("mspow")
    tau_vec = np.linspace(-2., 2., 100)
    wvec = np.linspace(-1., 1., 100)
    mspow = [[(tau+2.)*(1.-w)/(2.-w) for tau in tau_vec] for w in wvec]
    plt.pcolormesh(tau_vec, wvec, mspow, cmap='jet', shading='gouraud')
    plt.colorbar()
    CS = plt.contour(tau_vec, wvec, mspow, colors='k', lw=0.3, levels=15)
    plt.clabel(CS, inline=True, fontsize=7)
    plt.xlabel("$\\tau$", fontsize=13)
    plt.ylabel("$\\omega$", fontsize=13)
    plt.title("shocked mass exponent", fontsize=12)
    plt.savefig("mspow_mesh.png", bbox_inches='tight')
    plt.savefig("mspow_mesh.pdf", bbox_inches='tight')
    plt.show()

    plt.figure("powasymp")
    tau_vec = np.linspace(-1., 1.5, 100)
    wvec = np.linspace(-1., 0.6, 100)
    V_pow_asymp = [[tau*(2.-w)/((r+1.)*(tau+2.)*(1.-w)) + w/(1.-w) for tau in tau_vec] for w in wvec]
    plt.pcolormesh(tau_vec, wvec, V_pow_asymp, cmap='jet', shading='gouraud')
    plt.colorbar()
    CS = plt.contour(tau_vec, wvec, V_pow_asymp, colors='k', lw=0.3, levels=15)
    plt.clabel(CS, inline=True, fontsize=7)
    plt.xlabel("$\\tau$", fontsize=13)
    plt.ylabel("$\\omega$", fontsize=13)
    plt.title("$q(\\tau, \\omega)$", fontsize=14)
    plt.savefig("V_pow_asymp_mesh.png", bbox_inches='tight')
    plt.savefig("V_pow_asymp_mesh.pdf", bbox_inches='tight')
    plt.show()

def test_self_similar_profiles():
    r = 0.25 

    for w in [-0.5, 0., 0.5]:
        for tau in [-0.5, 0., 0.5]:

            solver = PistonShock(
                rho0=1.,
                omega=w,
                p0=1.,
                tau=tau,
                gamma=1.+r,
            )

            xsi_vec = np.array(sorted(list(set(
                list(np.linspace( NumericalParameters.xsi_origin, solver.xsi_s, 1000)) + \
                list(np.geomspace(NumericalParameters.xsi_origin, solver.xsi_s/10, 100))
            ))))
    
            V, U, P = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
            plt.plot(xsi_vec, V, c="r", lw=2, label=f"$V(\\xi)$")
            plt.plot(xsi_vec, U, c="b", lw=2, label=f"$U(\\xi)$")
            plt.plot(xsi_vec, P, c="k", lw=2, label=f"$P(\\xi)$")
            # plt.xscale("log")
            plt.suptitle(solver.title)
            plt.grid()
            plt.legend(fontsize=13)
            plt.xlabel("$\\xi$", fontsize=13)
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.savefig(f"profiles_ss_tau_{tau:g}_w_{w:g}.png", bbox_inches='tight')
            plt.savefig(f"profiles_ss_tau_{tau:g}_w_{w:g}.pdf", bbox_inches='tight')
            plt.show()

def test_shock_position():

    Mbar = 1e12
    nsec = 1e-9

    # ############
    # # r = 0.25
    # r = 2.1
    # rho0 = 19.3
    # tau = -0.447
    # omega = 0.
    # p0 = 2.71*Mbar * (1./nsec)**tau
    # L = 1e-3
    # ############

    # ############
    # r = 0.25  
    r = 2.1
    rho0 = 1.
    # tau = 0.
    # tau = -0.447
    # tau = 0.447
    tau=0.5
    omega = -0.9
    # omega = -0.474
    p0 = 1.
    L = 1.
    # ############

    # ############
    # r = 8.25
    # rho0 = 187.4
    # tau = -0.5
    # omega = -0.474
    # p0 = 500.1
    # L = 1.
    # ############
    
    solver = PistonShock(
        rho0=rho0,
        omega=omega,
        p0=p0,
        tau=tau,
        gamma=1.+r,
    )

    # solver.plot_profiles()
    x_shock = np.linspace(0., L, 1000)
    times = [solver.shock_time(shock_position=x) for x in x_shock]
    
    x_piston = [solver.piston_position(time=t) for t in times]
    
    print(times[-1])
    # times = np.linspace(0., 0.5, 1000)
    # r_shock = [solver.shock_position(time=t) for t in times]

    plt.plot(times, x_shock, label="shock")
    plt.plot(times, x_piston, label="piston")
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("position [cm]")
    plt.show()

def test_profiles():
    # Mbar = 1e12
    # nsec = 1e-9
    # # # ############
    # r = 0.25
    # # r = 2.1
    # rho0 = 19.3

    # # tau = -0.447
    # # tau = 0.
    # # omega = 0.5

    # tau = -0.45
    # omega = 0.5

    # # PROBLEMS NEGATIVE piston VELOCITY BUGGGGGGGGGGGGGGGGGGGG
    # # tau = -0.7
    # # omega = 0.

    # # tau = 0
    # # omega = -1.5

    # # tau = 0.5
    # # omega = 0.

    # # tau = 0.8
    # # omega = 0.5

    # p0 = 2.71*Mbar * (1./nsec)**tau
    # L = 1e-3

    # # # ablation example
    # # rho0=19.3
    # # omega=0
    # # p0=4.34113e+10
    # # tau=-0.207444
    # # r=1.2623-1.
    # ###########
    # ############


    # cgs example
    r=0.25
    # tau, omega = -0.5, 0.5
    # tau, omega = 0., 0.
    # tau, omega = 0., 0.5
    # tau, omega = 0.5, 0.5

    # zero density near origin
    # tau, omega = 0.5, -0.380952*1.1
    # finite density near origin
    tau, omega = 0.5, -0.380952
    # infinite density near origin
    # tau, omega = 0.5, -0.380952*0.9

    # tau, omega = -0.5, -0.5
    # tau, omega = 0.5, 0.
    # tau, omega = 0., -0.5
    # tau, omega = -0.5, 0.
    # tau, omega = 0.5, -0.5
    # tau, omega = 0.8, 0.7
    # tau, omega = 0., -2

    p0 = 1
    rho0 = 1

    solver = PistonShock(
        rho0=rho0,
        omega=omega,
        p0=p0,
        tau=tau,
        gamma=1.+r,
    )
    L=1.

    ##### plot self similar profiles
    # solver.plot_profiles()#.plot_fxsi_s()
    # solver.test_V_integral()
    solver.test_etot_integral()
    #### plot shock position
    x_shock = np.linspace(0., L*1., 1000)
    times = np.array([solver.shock_time(shock_position=x) for x in x_shock])
    v_shock = np.array([solver.shock_velocity(time=t) for t in times])
    x_piston = np.array([solver.piston_position(time=t) for t in times])
    v_piston = np.array([solver.piston_velocity(time=t) for t in times])

    x_shock2 = np.array([solver.shock_position2(time=t) for t in times])
    print(times[-1])

    etot = np.array([solver.total_energy(time=t) for t in times])
    ekin = np.array([solver.kinetic_energy(time=t) for t in times])
    eint = np.array([solver.internal_energy(time=t) for t in times])

    plt.figure()
    plt.plot(times, x_shock, label="shock")
    plt.plot(times, x_piston, label="piston")
    plt.grid()
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("position [cm]")
    # plt.show()
    
    plt.figure()
    plt.plot(times, np.abs(x_shock2/x_shock-1.), label="shock position err two methods", ls="--")
    plt.grid()
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("err")
    # plt.show()

    plt.figure()
    plt.plot(times, v_shock, label="shock")
    plt.plot(times, v_piston, label="piston")
    plt.grid()
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("velocity [cm/s]")
    # plt.show()

    plt.figure()
    plt.plot(times, etot, label="total energy")
    plt.plot(times, ekin, label="kinetic energy")
    plt.plot(times, eint, label="internal energy")
    plt.grid()
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("energy [erg]")
    # plt.show()

    num_cells = 200
    coordinate = np.linspace(0., L, num_cells+1)
    dx = coordinate[1:] - coordinate[:-1]
    rcell = 0.5*(coordinate[1:] + coordinate[:-1])

    # exact integral of mass in each cell gives this density
    density = rho0 / (1.-omega) * (coordinate[1:]**(1.-omega) - coordinate[:-1]**(1.-omega))/(coordinate[1:] - coordinate[:-1])

    mass_cells = density * dx
    mass = np.cumsum(mass_cells)

    coordinate_tag = [solver.initial_position(mass=m) for m in mass]
    for c1,c2 in zip(coordinate[1:], coordinate_tag):
        assert abs(c1/c2-1) < 1e-13

    mass = np.array([1e-4*mass[0], *list(mass)])

    ##### plot hydro profiles
    labels = dict(
        density="$\\rho(x,t) \\ \\ \\mathrm{{[g/cm]}}$",
        pressure="$p(x,t) \\ \\ \\mathrm{{[dyn/cm^{{2}}]}}$",
        velocity="$u(x,t) \\ \\ \\mathrm{{[cm/s]}}$",
    )
    labels_m = dict(
        density="$\\rho(m,t) \\ \\ \\mathrm{{[g/cm]}}$",
        pressure="$p(m,t) \\ \\ \\mathrm{{[dyn/cm^{{2}}]}}$",
        velocity="$u(m,t) \\ \\ \\mathrm{{[cm/s]}}$",
    )
    for time in np.array([0.4,0.5,0.75,0.9])*times[-1]:
        solution = solver.solve(mass=mass, time=time)
        
        for use_r in [True, False]:
            for fg in ["density", "velocity", "pressure"]:
                plt.figure(fg+str(use_r))
                if use_r: 
                    plt.plot(solution["position"], solution[fg], label=f"t={time:.2f}")#, marker="o")
                    plt.axvline(x=solution["piston_position"], lw=1.2, c="k", ls="--")
                    plt.axvline(x=solution["shock_position"], lw=1.2, c="b", ls="--")
                else:    
                    plt.plot(mass, solution[fg], label=f"t={time:.2f}")
    for use_r in [True, False]:
        for fg in ["density", "velocity", "pressure"]:
            plt.figure(fg+str(use_r))
            plt.legend(fontsize=12)
            plt.grid()
            plt.autoscale(enable=True, axis='both', tight=True)
            if use_r: 
                plt.xlabel("$x \\ \\mathrm{{[cm]}}$", fontsize=12)
                plt.ylabel(labels[fg], fontsize=12)
            else: 
                plt.xlabel("$m \\ \\mathrm{{[g]}}$", fontsize=12)
                plt.ylabel(labels_m[fg], fontsize=12)
            plt.title(f"$\\omega={omega:g}, \\ \\tau={tau:g}$")

            plt.savefig(f"{fg}_euler_{use_r}_{tau}_{omega}.png", bbox_inches='tight')
            plt.savefig(f"{fg}_euler_{use_r}_{tau}_{omega}.pdf", bbox_inches='tight')
    
    # plt.show()

    ########## plot position as a function of time
    position_times = np.array([np.array(solver.solve(mass=mass, time=time)["position"]) for time in times[1:]]).T

    plt.figure("position")
    for pos in position_times:
        plt.plot(times[1:], pos, c="k", lw=0.5)#, marker="o", markersize=1.)
    plt.plot(times, x_shock,    lw=2, ls="--", c="r", label="shock")#,  marker="o", markersize=5.)
    plt.plot(times, x_piston, lw=2, ls="--", c="b", label="piston")
    plt.legend(fontsize=13)
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]", fontsize=13)
    plt.ylabel("position [cm]", fontsize=13)
    plt.suptitle(solver.title)
    plt.savefig(f"rt_{tau}_{omega}.png", bbox_inches='tight')
    plt.savefig(f"rt_{tau}_{omega}.pdf", bbox_inches='tight')
    plt.show()
    quit()

if __name__ == "__main__":

    test_profiles()
    # test_shock_position()
    # test_xsi_s()
    # test_self_similar_profiles()