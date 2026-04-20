import numpy as np
import sys
import scipy.integrate
import scipy.optimize
from matplotlib import pyplot as plt


import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('SubsonicHeatWave')

class Units:
    sigma_sb = 5.670374419e-5
    clight = 2.99792458e10
    arad = 4. * sigma_sb / clight 
    ev_kelvin = 1.160451812e4
    hev_kelvin = 100. * ev_kelvin
    nsec = 1e-9

class NumericalParameters:
    Pf_initial_guess = 0.5 #fsolve initial guess for Pf root finding
    
    P0_xtol = 1e-8 # tolerance for root finding of Pf
    T0_xtol = 1e-8 # tolerance for bracketing xsi_f

    P0_err_fatal = 1e-2 #fatal if resulting P0 is larger than this value (for the correct xsi_f)
    T0_err_fatal = 1e-3 #fatal if |T0-1| is larger than this value (for the correct xsi_f)

    # xsi_f range for bracketing 
    xsi_f_low = 0.01
    xsi_f_high = 1.

    ode_scheme = "dopri5"  #equivalent to ode45 used by Shussman&Heizler
    # ode_scheme = "dop853"
    # ode_scheme = "lsoda"

    eps_xsi_f = 1e-6 #numerical integration will start from xsi_star=xsi_f*(1-eps_xsi_f). Asymptotic apprxoimation will be used for xsi_star<xsi<xsi_f

class SubsonicHeatWave():
    """An object to computes the solution an ablation problem

    The temperature boundary condition is 
    T(0,t) = Tb * t ** tau

    The medium consists of an ideal gas with the following properties:
    A polytropic eos with an adiabatic index gamma:
    p = (gamma-1) * rho * sie

    The sie (energy per units mass) is given by:

    sie(T, rho) = f * (T ** beta) * (rho ** -mu)
    
    The Rosseland oapcity is given as:
    1/kappa_Rosseland(T, rho) = g * (T ** alpha) * (rho ** -lambda)

    """
    def __init__(self, *,
                 Tb,        # boundary temperature coeff
                 tau,       # boundary temperature temporal power
                 g,         # opacity coeff
                 alpha,     # opacity temmperature power
                 lambdap,   # opacity density power
                 f,         # energy coeff
                 beta,      # energy temmperature power
                 mu,        # energy density power
                 gamma,     # adiabatic index
                 ode_scheme=NumericalParameters.ode_scheme,
                 ):

        logger.info(f"creating a SubsonicHeatWave calculator...")

        self.Tb = Tb
        self.tau = tau
        self.g = g
        self.alpha = alpha
        self.lambdap = lambdap
        self.f = f
        self.beta = beta
        self.mu = mu
        self.gamma = gamma
        self.ode_scheme = ode_scheme

        logger.info(f"Tb={self.Tb:g}")
        logger.info(f"tau={self.tau:g}")
        logger.info(f"g={self.g:g}")
        logger.info(f"alpha={self.alpha:g}")
        logger.info(f"lambda={self.lambdap:g}")
        logger.info(f"f={self.f:g}")
        logger.info(f"beta={self.beta:g}")
        logger.info(f"mu={self.mu:g}")
        logger.info(f"gamma={self.gamma:g}")
        logger.info(f"ode_scheme={self.ode_scheme!r}")
        
        assert self.beta > 0.
        assert self.Tb > 0.
        assert self.gamma > 1., f"must have gamma > 1"
        # assert self.tau > -2., f"must have tau > -2 for shock to propagate outwards"
        assert ode_scheme in {"dopri5", "dop853", "lsoda", "zvode", "vode"}
        
        self.r = gamma - 1.
        self.rep = f"tau={self.tau:g} alpha={self.alpha:g} beta={self.beta:g} lamda={self.lambdap:g} mu={self.mu:g}"

        self.title = f"$ \\tau={self.tau:.4g}" \
                     f", \\alpha={self.alpha:.4g}" \
                     f", \\beta={self.beta:.4g}" \
                     f", \\lambda={self.lambdap:.4g}" \
                     f", \\mu={self.mu:.4g}" \
                     f", \\gamma={self.gamma:.4g}$"

        self.n =(4.+alpha)/beta
        self.k = 1.-mu
        self.q = self.k*self.n+self.lambdap-1.
        self.tau_garnier = 1. / (4.+alpha-2.*beta)

        # dimensional constants
        self.A = self.Tb * (self.r * f) ** (1. / beta)
        self.B = 16. * Units.sigma_sb * g / (3. * beta * (self.r * f) ** self.n)

        denom = 1./(4.+2*lambdap-4.*mu)

        # xsi powers
        self.a = 2.*beta-beta*lambdap-8.-2.*alpha+mu*(4.+alpha)
        self.b = mu -2.
        self.c = 3.*mu-2.*lambdap-2.+tau*(2.*(beta-alpha-4.)-beta*lambdap+mu*(4.+alpha))
        self.a *= denom
        self.b *= denom
        self.c *= denom

        # v powers
        self.a1 = 4+alpha-2*beta
        self.b1 = 1.
        self.c1 = tau*(4.+alpha-2.*beta)-1.
        self.a1 *= -2.*denom
        self.b1 *= -2.*denom
        self.c1 *= -2.*denom

        # u powers
        self.a2 = mu*(4.+alpha)-beta*(2.+lambdap)
        self.b2 = mu
        self.c2 = tau*(mu*(4.+alpha)-beta*(2.+lambdap))-mu
        self.a2 *= -denom
        self.b2 *= -denom
        self.c2 *= -denom

        # p powers
        self.a3 = mu*(4.+alpha)-beta*lambdap-4.-alpha
        self.b3 = mu-1.
        self.c3 = tau*(mu*(4.+alpha)-beta*lambdap-4.-alpha)+1-mu
        self.a3 *= -2.*denom
        self.b3 *= -2.*denom
        self.c3 *= -2.*denom

        assert self.n > 0.

        self.ode_solver = scipy.integrate.ode(self.fode).set_integrator(ode_scheme)
        
        for key in {"xsi_f", "Pf", "V0", "Vp0", "P0", "Pp0", "U0", "T0", "xsi_min"}:
            setattr(self, key, None)

        logger.info(f"ablated mass temporal power {-self.c:g}")
        logger.info(f"position temporal power {self.c2+1:g}")
        logger.info(f"pressure temporal power {self.c3:g}")
        
        # ablated mass must increase over time
        assert self.c < 0.
        
        # ----- check relations among profiles powers
        # mass equation
        assert np.isclose(self.a2, self.a1-self.a,       rtol=1e-15, atol=1e-12)
        assert np.isclose(self.b2, self.b1-self.b,       rtol=1e-15, atol=1e-12)
        assert np.isclose(self.c2, self.c1-self.c-1.,    rtol=1e-15, atol=1e-12)
        # momentum equation
        assert np.isclose(self.a3, self.a1-2.*self.a,    rtol=1e-15, atol=1e-12)
        assert np.isclose(self.b3, self.b1-2.*self.b,    rtol=1e-15, atol=1e-12)
        assert np.isclose(self.c3, self.c1-2.*self.c-2., rtol=1e-15, atol=1e-12)
        # energy equation
        assert np.isclose(self.a3, -(   2.*self.a+self.a1*self.q)/(self.n-1.), rtol=1e-15, atol=1e-12)
        assert np.isclose(self.b3, -(1.+2.*self.b+self.b1*self.q)/(self.n-1.), rtol=1e-15, atol=1e-12)
        assert np.isclose(self.c3, -(1.+2.*self.c+self.c1*self.q)/(self.n-1.), rtol=1e-15, atol=1e-12)

    def get_T(self, P, V):
         return (P*V**(1.-self.mu))**(1./self.beta)

    def get_xsi_grid(self, *, xsi_f, fac=1.):
        """
        fine grid near front + linear + logarithmic grid near origin
        It is used only for plotting purpuses
        """
        xsi_f_star = xsi_f*(1.-NumericalParameters.eps_xsi_f)
        xsi_f_almost = xsi_f*(1.-5e-16)
        assert xsi_f_star < xsi_f_almost
        # return np.array(sorted(set(
        #     list(np.geomspace(1e-50,          xsi_f_star/100., int(50 *fac))) + \
        #     list(np.linspace(xsi_f_star/100., 0.9*xsi_f_star,  int(400*fac))) + \
        #     list(np.linspace(0.9*xsi_f_star,  0.95*xsi_f_star, int(50 *fac))) + \
        #     list(np.linspace(0.95*xsi_f_star, xsi_f_star,      int(100*fac))) + \
        #     list(np.linspace(xsi_f_star,      xsi_f_almost,    int(100*fac)))
        # )))

        return np.array(sorted(set(
            list(np.linspace(1e-10, xsi_f_almost,      int(500*fac)))
        )))

    def _position_temporal_factor(self, *, time):
        return (self.A**self.a2) * (self.B**self.b2) * (time**(self.c2+1.)) / (self.c2+1.)

    def _energy_temporal_factor(self, *, time):
        return (self.A**(2.*self.a2-self.a)) * (self.B**(2.*self.b2-self.b)) * (time**(2.*self.c2-self.c))

    def boundary_position(self, *, time):
        """
        returns position of the system boundary
        at the given time
        """
        return self._position_temporal_factor(time=time) * self.U0

    def boundary_velocity(self, *, time):
        """
        returns velocity of the system boundary
        at the given time
        """
        return self.boundary_position(time=time) * (self.c2+1.) / time

    def ablated_mass(self, *, time):
        """
        returns the total ablated mass at the given time
        """
        return self.xsi_f / self.xsi_over_m(time=time) 

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
        return (self.A**self.a) * (self.B**self.b) * (time**self.c)

    def get_position_self_similar_factor(self, *, xsi, V, U):
        """
        the position of any lagrangian element is some temporal factor
        multiplied by the result of this function
        """
        if xsi >= self.xsi_f:
            return 0.
        
        fac = (U - self.c * xsi * V)
        if fac > 0.:
            # due to numerical errors the factor can be small but positive
            # this happens since numerically, V is not exactly zero at xsi_f
            assert fac < 1e-3, fac
            return 0.
        return fac

    def solve(self, *, mass, time):
        """
        calculates the hydrodynamic profiles on the given
        Largrangian mass coordinates at the given time
        """
        # logger.info(f"{self.rep}: calculating hydro profiles at t={time:g}")
        assert type(mass) == np.ndarray
        assert self.xsi_f != None
        assert time > 0.
        assert all(m>0. for m in mass)
        if len(mass) > 1:
            assert np.min(mass[1:]-mass[:1]) > 0. #Lagrangian coordinate (acumulated cell mass) is monotone

        # self similar coordinate on the given Lagrangian coordinate
        xsi_vec = mass * self.xsi_over_m(time=time)

        # self similar profiles
        result = self.get_self_similar_profiles(xsi_vec=xsi_vec)
        V, P, U, S = result["V"], result["P"], result["U"], result["S"]

        boundary_position = self.boundary_position(time=time)

        # the eulerian positions of the given Lagrangian coordinates
        position = self._position_temporal_factor(time=time) * np.array([self.get_position_self_similar_factor(xsi=xsi_i, V=Vi, U=Ui) for xsi_i, Vi, Ui in zip(xsi_vec, V, U)])
        
        assert all(x <= 0. or np.isnan(x) for x in position)

        # hydodynamic profiles from self similar profiles
        v = V * (self.A**self.a1) * (self.B**self.b1) * (time**self.c1)
        u = U * (self.A**self.a2) * (self.B**self.b2) * (time**self.c2)
        p = P * (self.A**self.a3) * (self.B**self.b3) * (time**self.c3)
        
        # radiation energy flux
        aS = self.a+(self.q+1.)*self.a1+self.n*self.a3
        bS = self.b+(self.q+1.)*self.b1+self.n*self.b3 + 1.
        cS = self.c+(self.q+1.)*self.c1+self.n*self.c3
        radiation_energy_flux = S * (self.A**aS) * (self.B**bS) * (time**cS)
        
        return dict(
            boundary_position=boundary_position, 
            boundary_velocity=self.boundary_velocity(time=time), 
            ablated_mass=self.ablated_mass(time=time), 
            position=position, 
            density=1./v, 
            velocity=u, 
            pressure=p,
            temperature=((p*v**(1-self.mu))/(self.r*self.f))**(1./self.beta),
            sie=p*v/self.r, #specific internal energy
            radiation_energy_flux=radiation_energy_flux,
        )
 
    def get_self_similar_profiles(self, *, xsi_vec, xsi_f=None, Pf=None):
        """
        calculate the self similar profiles evaluated 
        on the values of the given vector of xsi
        outside the ablation region xsi>xsi_f, we fill with nan
        """
        N = len(xsi_vec)
        assert sorted(list(xsi_vec)) == list(xsi_vec)
        assert all(xsi >=0. for xsi in xsi_vec)
        if N > 1:
            assert np.all(np.diff(xsi_vec) > 0.)

        # if not given xsi_f/Pf, this solver object must have calculated the
        # correct xsi_f and Pf - use those in this case
        if xsi_f == None: xsi_f = self.xsi_f
        if Pf == None: Pf = self.Pf
        assert xsi_f != None and Pf != None

        # set the initial values at the given xsi_f
        xsi_f_star, initial_values = self.get_initial_values(xsi_f=xsi_f, Pf=Pf)
        self.ode_solver.set_initial_value(y=initial_values, t=xsi_f_star)

        V, Vp, P, Pp, U = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

        succ_all, succ_most = True, True
        xsi_min = None
        imin = 0
        # integrate from the first point before xsi_f (in the array xsi_vec) towards the origin
        for i in reversed(range(N)):
            if xsi_f_star <= xsi_vec[i] < xsi_f:
                # on the ablation front
                result = self.get_asymptotic_solution_near_front(xsi=xsi_vec[i], xsi_f=xsi_f, Pf=Pf)
            elif xsi_vec[i] < xsi_f_star:
                # ablation region
                result = self.ode_solver.integrate(xsi_vec[i])
                succ_all = self.ode_solver.successful()
            else:
                # out of ablation region (m>m_f)
                result = [np.nan] * 5
            
            if succ_all:
                V[i], Vp[i], P[i], Pp[i], U[i] = result
                xsi_min = xsi_vec[i]
                imin = i
            else:
                imin = min(i+1, N-1)
                xsi_min = xsi_vec[imin]
                succ_most = xsi_min < xsi_f * 1e-2
                break
        
        T = self.get_T(V=V, P=P)
        S = -P**(self.n-1.) * V**(self.q) * (V*Pp + self.k*P*Vp)

        return dict(
            V=V, 
            Vp=Vp, 
            P=P, 
            Pp=Pp, 
            U=U, 
            T=T,
            S=S,
            imin=imin,
            xsi_min=xsi_min, #minimal xsi for which the ODE soluion did not crash
            succ_all=succ_all,
            succ_most=succ_most,
        )

    def fode(self, xsi, x):
        """
        the self similar ODE, written as a first order 
        form ODE in the form dx/dxsi=F(x, xsi), where
        x = (V, V', P, P', u)
        this function returns the derivatives:
        x' = (V', V'', P', P'', U') 
        """
        c, c1, c2, c3, n, k, q, r = self.c, self.c1, self.c2, self.c3, self.n, self.k, self.q, self.r
        V, Vp, P, Pp, U = x

        cxsi = c*xsi
        Up = cxsi*Vp + c1*V

        com = (c1+c)*cxsi*Vp + (c2+c)*Up
        cxsi2 = cxsi*cxsi

        Vpp = (V**(-q)*P**(1.-n)) *(cxsi*((1.+1./r)*P*Vp+1./r*V*Pp) + P*V*((1.+1./r)*c1+c3/r)) \
              - k*q/V*P*Vp*Vp-(n-1.)*V/P*Pp*Pp-(q+k*n+1.)*Pp*Vp \
              + V * com
        Vpp /= k*P - cxsi2*V

        Ppp = -cxsi2*Vpp - com

        return [Vp, Vpp, Pp, Ppp, Up]

    def get_asymptotic_solution_near_front(self, *, xsi, xsi_f=None, Pf=None):
        """
        given xsi_f, Pf calculates the asymptotic solution near the heat front.
        """
        if xsi_f == None: xsi_f = self.xsi_f
        if Pf == None: Pf = self.Pf
        assert xsi_f != None and Pf != None

        # asymptotic solution of the form V(xsi)=Vstar*(xsi_f-xsi)**delta
        delta = 1./self.q
        Vstar = (-(1.+1./self.r)*self.q*self.c/self.k * xsi_f * Pf ** (1.-self.n))**delta
        
        if np.isnan(Vstar):
            return [np.nan]*5

        assert Vstar > 0., Vstar

        dxsi = xsi_f - xsi
        assert dxsi > 0.
        cxsif = self.c*xsi_f
        cxsif2 = cxsif*cxsif
        V = Vstar*dxsi**delta
        U = V*cxsif
        P = Pf-V*cxsif2
        Vp = -delta*V/dxsi
        Pp = -Vp*cxsif2

        return [V, Vp, P, Pp, U]

    def get_initial_values(self, *, xsi_f=None, Pf=None):
        """
        calculates the initial condition to the numerical ODE solver
        at the point xsi_f_star which is close to xsi_f.
        the values of these initial condition are calculated using 
        the asymptotic solution near the heat front
        """
        if xsi_f == None: xsi_f = self.xsi_f
        xsi_f_star = xsi_f*(1.-NumericalParameters.eps_xsi_f)
        result_asymptotic = self.get_asymptotic_solution_near_front(xsi=xsi_f_star, xsi_f=xsi_f, Pf=Pf)
        return xsi_f_star, result_asymptotic

    def integrate_inward(self, *, Pf, xsi_f):
        """
        An auxiliary function used find the exact value of Pf via the shooting method.
        It integrates the self similar ODE from a given xsi=xsi_f (not necessarily the exact one),
        to the the origin, with initial value of Pf.
        For the correct Pf, the value of the pressure at the origin, P(0) should be equal to 0,
        """
        assert xsi_f > 0., xsi_f

        # set the initial values at xsi_f_star which is close to
        # the given xsi_f, using the asymptotic solution near xsi_f
        xsi_f_star, initial_values = self.get_initial_values(xsi_f=xsi_f, Pf=Pf)
        self.ode_solver.set_initial_value(y=initial_values, t=xsi_f_star)

        #### USE a single integration step to 0 (the ode solver will do the substepping)
        result = self.ode_solver.integrate(0.)
        succ = self.ode_solver.successful()
        xsi = 0.
        if succ:
            V, Vp, P, Pp, U = result
        else:
            V, Vp, P, Pp, U = np.nan, np.nan, -1., np.nan, np.nan

        ##### USE an array of xsi instead of letting the ODE solver to figure out the stepsize
        # succ = True
        # xsi_vec = self.get_xsi_grid(xsi_f=xsi_f)
        # # integrate from the first point before xsi_f (in the array xsi_vec) towards the origin
        # for xsi in xsi_vec[:-1][::-1]:
        #     assert xsi < xsi_f
        #     result = self.ode_solver.integrate(xsi)
        #     succ = self.ode_solver.successful()
        #     if succ:
        #         V, Vp, P, Pp, U = result
        #     else:
        #         if xsi > xsi_f * 1e-2:
        #             V, Vp, P, Pp, U = np.nan, np.nan, -1., np.nan, np.nan
        #         break
        
        # msg = f"V0={V:g} U0={U:g} P0={P:g} T0={(P*V**(1.-self.mu))**(1./self.beta)}" if succ else f"FAILED at xsi={xsi:g}"
        # logger.info(f"{self.rep}: integrate inward xsi_f={xsi_f} Pf={Pf}: {msg}")

        return V, Vp, P, Pp, U, xsi

    def fPf(self, Pf, xsi_f):
        """
        for any xsi_f, Pf is the root of this function
        returns P(0), where P was integrated from the given xsi_f to the origin
        """
        return self.integrate_inward(Pf=Pf,xsi_f=xsi_f)[2]

    def find_P0(self, *, xsi_f):
        """
        use scipy.optimize.fsolve to find the Pf such that P(0)=0
        where the ODE was integrated from the given xsi_f to the origin
        """
        logger.info(f"{self.rep}: xsi_f={xsi_f}: Solving Pf root such that P(0)=0")
        
        root, info, ier, msg = scipy.optimize.fsolve(
            func=self.fPf, 
            x0=NumericalParameters.Pf_initial_guess, 
            args=(xsi_f), 
            xtol=NumericalParameters.P0_xtol,
            full_output=True,
        )

        if ier != 1:
            logger.warning(f"{self.rep}: xsi_f={xsi_f}: fsolve(Pf) did not converge (ier={ier}): {msg}")

        Pf_root = root[0]
        V0, Vp0, P0, Pp0, U0, xsi_min = self.integrate_inward(Pf=Pf_root, xsi_f=xsi_f)
        if np.isfinite(P0) and np.isfinite(V0) and P0 > 0. and V0 > 0.:
            T0 = self.get_T(P=P0, V=V0)
        else:
            T0 = np.nan

        result = dict(
            Pf=Pf_root, 
            V0=V0, 
            Vp0=Vp0, 
            P0=P0, 
            Pp0=Pp0, 
            U0=U0, 
            T0=T0,
            xsi_min=xsi_min, #minimal xsi for which the ODE soluion did not crash
        )
        logger.info(f"{self.rep}: xsi_f={xsi_f}: Root found Pf={Pf_root}: P0={P0:g} T0={T0} xsi_min={result['xsi_min']:g}")

        return result

    def fxsi_f(self, xsi_f):
        """
        xsi_f is the root of this function
        returns T(0)-1, where the ODE was integrated from the given xsi_f to the origin
        using the correct value of Pf (that is, such that P(0)=0)
        """
        if type(xsi_f) == np.ndarray:
            # fsolve solver wraps xsi_f in an ndarray
            assert len(xsi_f) == 1
            xsi_f_f = xsi_f[0]
        else:
            xsi_f_f = xsi_f
        T0 = self.find_P0(xsi_f=xsi_f_f)["T0"]
        if not np.isfinite(T0):
            logger.warning(f"{self.rep}: xsi_f={xsi_f_f}: invalid T0={T0}, skipping this point while bracketing")
            return np.nan
        res = T0 - 1.
        return res

    def find_xsi_f(self):
        """
        use scipy.optimize.bisect to bracket xsi_f, by requiring that T(0)=1
        This will set this solver to be ready to calculate the solution
        since now we will have the correct xsi_f and Pf
        """
        logger.info(f"{self.rep}: Begin bracketing xsi_f such that T(0)=1...")

        assert self.xsi_f == None
        assert self.Pf == None

        # Some xsi_f guesses may produce non-physical states (e.g. P0 <= 0), which
        # makes T0 undefined and therefore unsuitable for bisect. Scan for a finite
        # sign-changing sub-interval inside the user-provided [low, high] range.
        xsi_scan = np.linspace(NumericalParameters.xsi_f_low, NumericalParameters.xsi_f_high, 25)
        f_scan = [self.fxsi_f(xsi) for xsi in xsi_scan]

        bracket = None
        for i in range(len(xsi_scan) - 1):
            fa = f_scan[i]
            fb = f_scan[i + 1]
            if not (np.isfinite(fa) and np.isfinite(fb)):
                continue
            if fa == 0.:
                bracket = (xsi_scan[i], xsi_scan[i])
                break
            if fb == 0.:
                bracket = (xsi_scan[i + 1], xsi_scan[i + 1])
                break
            if fa * fb < 0.:
                bracket = (xsi_scan[i], xsi_scan[i + 1])
                break

        if bracket is None:
            finite_pts = [(x, f) for x, f in zip(xsi_scan, f_scan) if np.isfinite(f)]
            raise RuntimeError(
                f"{self.rep}: failed to find a finite sign-changing bracket for xsi_f in "
                f"[{NumericalParameters.xsi_f_low:g}, {NumericalParameters.xsi_f_high:g}]. "
                f"finite evaluations={finite_pts}"
            )

        if bracket[0] == bracket[1]:
            xsi_f_root = bracket[0]
        else:
            a, b = bracket
            logger.info(f"{self.rep}: using xsi_f bracket [{a:g}, {b:g}] for T(0)=1 root")
            xsi_f_root = scipy.optimize.bisect(
                f=self.fxsi_f, 
                a=a,
                b=b,
                xtol=NumericalParameters.T0_xtol,
            )

        result = self.find_P0(xsi_f=xsi_f_root)
        logger.info(f"{self.rep}: Root found xsi_f={xsi_f_root}")

        result = dict(
            xsi_f=xsi_f_root,
            **result,
        )

        # set all results as this object's attributes
        for key, value in result.items():
            assert hasattr(self, key)
            assert getattr(self, key) == None
            setattr(self, key, value)
        
        logger.info(f"{self.rep}: at ablation front: Pf={self.Pf:g}")
        logger.info(f"{self.rep}: at the origin: R0={1./self.V0:g} V0={self.V0} P0={self.P0:g} U0={self.U0} T0={self.T0}")

        # sainity checks
        fatal = False
        if self.V0 < 0.:
            fatal = True
            logger.fatal(f"{self.rep}: solver failed, negative value for V0={self.V0}")
        if self.U0 >= 0.:
            fatal = True
            logger.fatal(f"{self.rep}: solver failed, positive value for U0={self.U0}")
        if self.P0 < 0.:
            fatal = True
            logger.fatal(f"{self.rep}: solver failed, negative value for P0={self.P0}")
        if self.P0 > NumericalParameters.P0_err_fatal:
            fatal = True
            logger.fatal(f"{self.rep}: solver failed P0={self.P0} is too large")
        if abs(self.T0-1.) > NumericalParameters.T0_err_fatal:
            fatal = True
            logger.fatal(f"{self.rep}: solver failed T0={self.T0} is too far from 1")
        if fatal:
            self.plot_profiles()
            sys.exit(1)

        self.set_energy_integrals()
        return self

    def plot_fPf(self, *, xsi_f):
        """
        given a xsi_f, plots the function for which Pf is a root, such that P(0)=0
        also plots some P(xsi) profiles to demonstrate 
        """
        Pf_vec = np.linspace(0.0001, 4., 100)
        fPf = np.vectorize(self.fPf)(Pf_vec, xsi_f)
        plt.plot(Pf_vec, fPf, "--bo", label=f"$\\xi_{{f}}={xsi_f:g}$")

        Pf_root = self.find_P0(xsi_f=xsi_f)["Pf"]
        plt.axvline(x=Pf_root, ls="--", label=f"root $P_{{f}}={Pf_root:g}$",lw=2, c="k")

        plt.title(self.title, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.ylabel("$F_{{\\xi_{{s}}}}(P_{{f}})$", fontsize=12)
        plt.xlabel("$P_{{f}}$", fontsize=12)
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.show()
        
        # plot profiles
        Pf_vec = np.linspace(Pf_root*0.5, Pf_root*1.5, 15)
        xsi_vec = self.get_xsi_grid(xsi_f=xsi_f)
        plt.figure("")
        for Pf in Pf_vec:
            result = self.get_self_similar_profiles(xsi_vec=xsi_vec, xsi_f=xsi_f, Pf=Pf)
            P, T, imin, xsi_min = result["P"], result["T"], result["imin"], result["xsi_min"]
            plt.plot(xsi_vec, P, label=f"$P(\\xi), P_{{f}}={Pf:.4g}, P(0)={P[imin]:g}, T(0)={T[imin]:g}, \\xi_{{min}}={xsi_min:g}$")

        plt.title(self.title+f", $\\xi_{{f}}={xsi_f:g}$", fontsize=12)
        plt.grid()
        plt.legend(fontsize=10)
        plt.xlabel("$\\xi$", fontsize=12)
        plt.ylabel("$P(\\xi)$", fontsize=12)
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.show()
        
        return self

    def plot_fxsi_f(self):
        """
        plots the function for which xsi_f is a root, such that T(0)=1
        also plots some P(xsi), T(xsi) profiles to demonstrate 
        """
        Nx = 20
        xsi_f_vec = np.linspace(0.1, 1.2, Nx)
        fxsi_f_vec = np.nan*np.ones(Nx)
        T0_vec = np.nan*np.ones(Nx)
        P0_vec = np.nan*np.ones(Nx)
        Pf_vec = np.nan*np.ones(Nx)
        xsi_min_vec = np.nan*np.ones(Nx)

        for i, xsi_f in enumerate(xsi_f_vec):
            result0 = self.find_P0(xsi_f=xsi_f)
            T0_vec[i] = result0["T0"]
            fxsi_f_vec[i] = T0_vec[i] - 1.
            P0_vec[i] = result0["P0"]
            Pf_vec[i] = result0["Pf"]
            xsi_min_vec[i] = result0["xsi_min"]

        plt.figure("fxsi_f")
        plt.plot(xsi_f_vec, fxsi_f_vec, "--bo")
        plt.axhline(y=0., ls="--", c="k", lw=2.)
        plt.xlabel(f"$\\xi_{{f}}$", fontsize=12)
        plt.ylabel("$F(\\xi_{{s}})=T_{{\\xi_{{s}}}}(0)-1$", fontsize=12)
        plt.title(self.title, fontsize=12)
        plt.grid()
        plt.autoscale(enable=True, axis='both', tight=True)

        plt.figure("Pfx")
        plt.plot(xsi_f_vec, Pf_vec, "--bo")
        plt.xlabel(f"$\\xi_{{f}}$", fontsize=12)
        plt.ylabel(f"$P_{{f}}$", fontsize=12)
        plt.title(self.title, fontsize=12)
        plt.grid()
        plt.autoscale(enable=True, axis='both', tight=True)

        plt.figure("T0x")
        plt.plot(xsi_f_vec, T0_vec, "--bo")
        plt.axhline(y=1., ls="--", c="k", lw=2.)
        plt.xlabel(f"$\\xi_{{f}}$", fontsize=12)
        plt.ylabel(f"$T(0)$", fontsize=12)
        plt.title(self.title, fontsize=12)
        plt.grid()
        plt.autoscale(enable=True, axis='both', tight=True)

        plt.figure("P0x")
        plt.plot(xsi_f_vec, P0_vec, "--bo")
        plt.xlabel(f"$\\xi_{{f}}$", fontsize=12)
        plt.ylabel(f"$P(0)$", fontsize=12)
        plt.title(self.title, fontsize=12)
        plt.grid()
        plt.autoscale(enable=True, axis='both', tight=True)

        plt.figure("xsiminx")
        plt.plot(xsi_f_vec, xsi_min_vec, "--bo")
        plt.xlabel(f"$\\xi_{{f}}$", fontsize=12)
        plt.ylabel(f"$\\xi_{{min}}$", fontsize=12)
        plt.title(self.title, fontsize=12)
        plt.grid()
        plt.autoscale(enable=True, axis='both', tight=True)
        
        plt.show()

        for xsi_f, Pf in zip(xsi_f_vec, Pf_vec):
            xsi_vec = self.get_xsi_grid(xsi_f=xsi_f)
            result = self.get_self_similar_profiles(xsi_vec=xsi_vec, xsi_f=xsi_f, Pf=Pf)
            P, T, imin, xsi_min = result["P"], result["T"], result["imin"], result["xsi_min"]

            label = f"$\\xi_{{f}}={xsi_f:.4g}, P(0)={P[imin]:g}, T(0)={T[imin]:g}, \\xi_{{min}}={xsi_min:g}$"
            plt.figure("PXSI")
            plt.plot(xsi_vec, P, label=label)

            plt.figure("TXSI")
            plt.plot(xsi_vec, T, label=label)

        plt.figure("PXSI")
        plt.title(self.title)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("$\\xi$", fontsize=12)
        plt.ylabel("$P(\\xi)$", fontsize=12)
        plt.autoscale(enable=True, axis='both', tight=True)

        plt.figure("TXSI")
        plt.title(self.title)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("$\\xi$", fontsize=12)
        plt.ylabel("$T(\\xi)$", fontsize=12)
        plt.autoscale(enable=True, axis='both', tight=True)
        
        plt.show()

        return self

    def get_V_integral(self, *, xsi, V, U, xsi0, V0, U0):
        """
        calculate the integral of V in the range [0, xsi],
        given the values V, U evaluated the the end point xsi
        The integral is obtained from an exact analytic expression
        """
        assert xsi >= 0.
        if xsi > self.xsi_f: 
            assert V == np.nan
            assert U == np.nan
            return np.nan

        if V <0. or U > 0.:
            logger.fatal(f"{self.rep}: V integral failed xsi={xsi} wrong sign value V={V} U={U}")
            sys.exit(1)

        integ = ((U-U0) - self.c*(xsi*V-xsi0*V0))/ (self.c2 + 1.)
        assert integ >= 0.
        # if integ < 0.:
        #     # numerical errors can give a small negative result for very small xsi
        #     assert integ >= -1e-8 and xsi < 1e-10, f"{integ} {xsi}"
        #     integ = 0.
        return integ

    def set_energy_integrals(self):
        """
        calculates numerically the energy integrals
        of the similarity profiles
        """
        assert self.xsi_f != None

        logger.info(f"{self.rep}: setting energy integrals...")

        xsi_vec = self.get_xsi_grid(xsi_f=self.xsi_f, fac=2.)
            
        result= self.get_self_similar_profiles(xsi_vec=xsi_vec)
        V, U, P, S = result["V"], result["U"], result["P"], result["S"]
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

        self.energy_time_power = 2.*self.c2-self.c
        logger.info(f"{self.rep}: energy time power={self.energy_time_power:g}")

        self.energy_kinetic_intgeral = ekin_integral_simps
        self.energy_internal_intgeral = ein_integral_simps
        self.energy_total_intgeral = etot_integral_simps
        logger.info(f"{self.rep}: kinetic/internal ratio={self.energy_kinetic_intgeral/self.energy_internal_intgeral:g}")

        # analytical formula for the total energy
        integral = (-S -P*U -self.c*xsi_vec*integrand_tot)/self.energy_time_power
        self.energy_total_intgeral_analytic = integral[-1] - integral[0]

        logger.info(f"{self.rep}: etot_integral anal={self.energy_total_intgeral_analytic}")
        logger.info(f"{self.rep}: error anal energy integral={abs(self.energy_total_intgeral_analytic/self.energy_total_intgeral-1.):g}")

        return self

    def test_V_integral(self):
        assert self.xsi_f != None
        assert self.Pf != None

        xsi_vec = self.get_xsi_grid(xsi_f=self.xsi_f)
            
        result = self.get_self_similar_profiles(xsi_vec=xsi_vec)
        V, U = result["V"], result["U"]
        V_integral_total_simps = scipy.integrate.simpson(y=V, x=xsi_vec)
        
        logger.info(f"{self.rep}: V_integral_total simps={V_integral_total_simps}")
        logger.info(f"{self.rep}: V_integral_total trapz={np.trapezoid(y=V, x=xsi_vec)}")

        # xsi_integral_vec = np.array(xsi_vec)
        # V_integral_cum = scipy.integrate.cumtrapz(y=V, x=xsi_vec)
        # logger.info(f"{self.rep}: V_integral_total trapz={V_integral_cum[-1]}")


        # print(xsi_vec[0]*V[0])
        # print(xsi_vec)
        # print(V)

        vint = self.get_V_integral(xsi=xsi_vec[-1], V=V[-1], U=U[-1])
        logger.info(f"{self.rep}: V0 {V[0]} U0 {U[0]}")
        logger.info(f"{self.rep}: vint {vint}")

        vint = ((U[-1]-U[0]) - self.c*(xsi_vec[-1]*V[-1]-xsi_vec[0]*V[0]))/ (self.c2 + 1.)

        # print((U[-1]-U[0]) - self.c*(xsi_vec[-1]*V[-1]-xsi_vec[0]*V[0]))/ (self.c2 + 1.))
        logger.info(f"{self.rep}: vint {vint}")

        quit()
        # logger.info(f"{self.rep}: V0 {self.V0} U0 {self.U0}")
        # logger.info(f"{self.rep}: vvvint={self.V_integral_total}")

        # err = abs(self.V_integral_total/V_integral_total_simps-1.)
        # logger.info(f"{self.rep}: V_integral_total error vs simpson={err:g}")
        # assert err < 1e-3, err

    def test_etot_integral(self):
        assert self.xsi_f != None
        assert self.Pf != None

        xsi_vec = self.get_xsi_grid(xsi_f=self.xsi_f, fac=20.)
        # xsi_vec = np.linspace(self.xsi_f/1000000000., self.xsi_f, 10000)
        
        result= self.get_self_similar_profiles(xsi_vec=xsi_vec)
        V, U, P, S = result["V"], result["U"], result["P"], result["S"]

        integrand = P*V/self.r+0.5*U*U
        etot_integral_total_simps = scipy.integrate.simpson(y=integrand, x=xsi_vec)
        
        logger.info(f"{self.rep}: etot_integral_total simps={etot_integral_total_simps}")
        logger.info(f"{self.rep}: etot_integral_total trapz={np.trapezoid(y=integrand, x=xsi_vec)}")

        # # cum plot
        etot_integral_cum_trapz = scipy.integrate.cumulative_trapezoid(y=integrand, x=xsi_vec)

        denom = 2. * self.c2-self.c
        sterm = -S
        sterm /= denom

        PU_term = -P*U
        PU_term /= denom
        
        e_term = -self.c * xsi_vec*integrand
        e_term /= denom

        integral = sterm + PU_term + e_term
        etot_integral_cum = integral - integral[0]

        logger.info(f"{self.rep}: eint {etot_integral_cum[-1]}")
        
        print(sterm[0], sterm[-1], sterm[-1]-sterm[0])
        print(PU_term[0], PU_term[-1])
        print(e_term[0], e_term[-1])
        print(denom)
        plt.plot(xsi_vec, etot_integral_cum, label="cum")
        plt.plot(0.5*(xsi_vec[1:]+xsi_vec[:-1]), etot_integral_cum_trapz, ls="--", label="trapz")
        plt.plot(xsi_vec, integrand, "g", label="E")
        plt.plot(xsi_vec, -sterm, "k", label="S")
        plt.plot(xsi_vec, PU_term,"b", label="PU")
        plt.plot(xsi_vec, e_term,"r", label="E*xsi")
        plt.legend()
        plt.grid()
        plt.show()

        return self

    def plot_profiles(self, *, xsi_f=None, Pf=None):
        """
        plots the self similar profiles on [0, xsi_f]
        """
        if xsi_f == None: xsi_f = self.xsi_f
        if Pf == None: Pf = self.Pf
        assert xsi_f != None and Pf != None

        xsi_vec = self.get_xsi_grid(xsi_f=xsi_f, fac=5.)
        result = self.get_self_similar_profiles(xsi_vec=xsi_vec, xsi_f=xsi_f, Pf=Pf)
        V, P, U, T, S, imin = result["V"], result["P"], result["U"], result["T"], result["S"], result["imin"]

        # plt.plot(xsi_vec, 1./V, c="r",       lw=2, label=f"$R(\\xi)$",)
        plt.plot(xsi_vec, V,                 lw=2, label=f"$V(\\xi)$")
        plt.plot(xsi_vec, -U,   c="fuchsia", lw=2, label=f"$-U(\\xi)$")
        plt.plot(xsi_vec, P,    c="lime",    lw=2, label=f"$P(\\xi)$ p0={P[imin]:g}")
        plt.plot(xsi_vec, T,    c="b",       lw=2, label=f"$T(\\xi)$ T0={T[imin]:g}")
        # plt.plot(xsi_vec, S,    c="k",       lw=2, label=f"$S(\\xi)$")

        # # plot asymptotic solution near xsi_f
        result_as = list(zip(*[self.get_asymptotic_solution_near_front(xsi=xsi) for xsi in xsi_vec]))
        Vas, Pas, Uas = np.array(result_as[0]), np.array(result_as[2]), np.array(result_as[4])
        Tas = self.get_T(P=Pas, V=Vas)
        plt.plot(xsi_vec, Vas,    lw=2, ls="--", label=f"$V(\\xi)$ asymp")
        plt.plot(xsi_vec, -Uas,   lw=2, ls="--", label=f"$-U(\\xi)$ asymp")
        plt.plot(xsi_vec, Pas,    lw=2, ls="--", label=f"$P(\\xi)$ asymp")
        plt.plot(xsi_vec, Tas,    lw=2, ls="--", label=f"$T(\\xi)$ asymp")

        # plt.xscale("log")
        # plt.yscale("log")
        plt.title(self.title+f", $\\xi_{{f}}={xsi_f:.4g}, P_{{f}}={Pf:.4g}$", fontsize=10)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("$\\xi$", fontsize=12)
        # plt.autoscale(enable=True, axis='both', tight=True)
        plt.xlim([0.,xsi_f])
        plt.ylim([0.,1.])
        plt.show()


        V_integral = [self.get_V_integral(xsi=xsi, V=Vi, U=Ui, xsi0=xsi_vec[0], V0=V[0], U0=U[0]) for xsi, Vi, Ui in zip(xsi_vec, V, U)]
        V_integral_cum = scipy.integrate.cumulative_trapezoid(y=V, x=xsi_vec)
        V_integral2 = [(Ui -U[0] - self.c * xsi * Vi)/ (self.c2+1.) for xsi, Vi, Ui in zip(xsi_vec, V, U)]
        plt.plot(xsi_vec, V*xsi_vec, ls="--", label=f"$\\xi V(\\xi)$")
        plt.plot(xsi_vec, V_integral, ls="--", label=f"$\\int V(\\xi)$")
        plt.plot(xsi_vec, V_integral2, ls="--", label=f"$\\int V(\\xi)$ 2")
        plt.plot(0.5*(xsi_vec[1:]+xsi_vec[:-1]), V_integral_cum, ls="--", label=f"$\\int V(\\xi)$ trapz")
        plt.title(self.title+f", $\\xi_{{f}}={xsi_f:.4g}, P_{{f}}={Pf:.4g}$", fontsize=12)
        plt.grid()
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=12)
        plt.xlabel("$\\xi$", fontsize=12)
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.show()

        return self

def test_xsi_f():
    """
    plots xsi_f as a function of tau
    """
    # tau_vec = [-0.1,0.,0.1]#np.linspace(-0.48, 0.7, 4)
    tau_vec = np.linspace(-0.5, 0.5, 13)[::-1]
    mat=dict(
        alpha=1.5,
        beta=1.6,
        lambdap=0.2,
        mu=0.14,
        gamma=1.25,
        Tb=1.,
        g=1.,
        f=1.,
    )

    ############

    xsi_f = []
    Pf = []
    T0 = []
    P0 = []
    xsi_min = []

    for tau in tau_vec:
        solver = SubsonicHeatWave(
            **mat,
            tau=tau,
        ).find_xsi_f()

        xsi_f.append(solver.xsi_f)
        Pf.append(solver.Pf)
        P0.append(solver.P0)
        T0.append(solver.T0)
        xsi_min.append(solver.xsi_min)

    print(xsi_min)
    print(T0)
    print(P0)

    plt.plot(tau_vec, xsi_f, "--ro", label=f"$\\xi_{{f}}$")
    plt.plot(tau_vec, Pf, "--bo", label=f"$P_{{f}}$")
    plt.title(solver.title, fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.xlabel("$\\tau$", fontsize=12)
    # plt.ylabel("$\\xi_{{s}}$", fontsize=12)
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.savefig("xsif_tau.png", bbox_inches='tight')
    plt.savefig("xsif_tau.pdf", bbox_inches='tight')
    plt.show()

def test_ablated_mass():

    time_rise = Units.nsec
    # tau = 0.123
    tau = 0.
    Tb = Units.hev_kelvin * (1./time_rise)**tau

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
        alpha=0.23183890413119132,
        beta=1.6525149133503083,
        lambdap=0.13829986198312327,
        mu=0.1206770123528319,
        gamma=1.262299832903183,

        Tb=Tb,
        g=5.682464574187187e-6,
        f=3168.6938116243796,
    )
    
    solver = SubsonicHeatWave(
        **mat,
    )

    solver.find_xsi_f()
    solver.plot_profiles()

    times = np.linspace(0., 1e-9, 100)
    m_ab = [solver.ablated_mass(time=t) for t in times]
    x_boundary = [-solver.boundary_position(time=t) for t in times]

    plt.plot(times, m_ab)
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("ablated mass [g]", fontsize=12)
    plt.xlabel("time [sec]", fontsize=12)
    plt.title(solver.title, fontsize=12)
    plt.show()

    plt.plot(times, x_boundary)
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]", fontsize=12)
    plt.ylabel("$-x_{{boundary}}$ [cm]", fontsize=12)
    plt.title(solver.title, fontsize=12)
    plt.show()

def test_profiles():
    time_rise = Units.nsec
    tau = 0.123
    # tau = 0.
    Tb = Units.hev_kelvin * (1./time_rise)**tau
    rho0 = 19.3

    # MORDI Au
    alpha=1.5
    beta=1.6
    f=3.4*1e13/(Units.hev_kelvin**beta)
    g=1/(7200*Units.hev_kelvin**alpha)
    mat=dict(
        tau=tau,
        alpha=alpha,
        beta=beta,
        lambdap=0.2,
        mu=0.14,
        gamma=1.25,
        Tb=Tb,
        g=g,
        f=f,
    )

    # mat=dict(
    #     tau=tau,
    #     alpha=0.23183890413119132,
    #     beta=1.6525149133503083,
    #     lambdap=0.13829986198312327,
    #     mu=0.1206770123528319,
    #     gamma=1.262784,

    #     Tb=Tb,
    #     g=5.682464574187187e-6,
    #     f=3168.6938116243796,
    # )
    
    solver = SubsonicHeatWave(
        **mat,
    )

    solver.find_xsi_f()
    
    solver.test_etot_integral()
    solver.plot_profiles()
    # solver.test_V_integral()

    L = 1e-3
    ############
    
    # times = np.linspace(0., 2.061e-9, 1000)
    times = np.linspace(0., 1.6808e-9, 1000)
    m_ab = [solver.ablated_mass(time=t) for t in times]
    x_boundary = np.array([solver.boundary_position(time=t) for t in times])

    etot = np.array([solver.total_energy(time=t) for t in times])
    ekin = np.array([solver.kinetic_energy(time=t) for t in times])
    eint = np.array([solver.internal_energy(time=t) for t in times])

    plt.plot(times, m_ab)
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("ablated mass [g]", fontsize=12)
    plt.xlabel("time [sec]", fontsize=12)
    plt.title(solver.title, fontsize=12)
    plt.show()

    plt.plot(times, -x_boundary)
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]", fontsize=12)
    plt.ylabel("$-x_{{boundary}}$ [cm]", fontsize=12)
    plt.title(solver.title, fontsize=12)
    plt.show()

    plt.figure()
    plt.plot(times, etot, label="total energy")
    plt.plot(times, ekin, label="kinetic energy")
    plt.plot(times, eint, label="internal energy")
    plt.grid()
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("energy [erg]")
    plt.show()

    # num_cells = 1000
    # coordinate = np.array(list(sorted(set(
    #     list(np.linspace(0., L/100, num_cells+1)) + 
    #     list(np.linspace(L/100, L/20, num_cells+1)) + \
    #     list(np.linspace(L/20, L/5, num_cells+1)) + \
    #     list(np.linspace(L/5, L, num_cells+1)) \
    # ))))

    num_cells = 1000
    coordinate = np.array(list(sorted(set(
        list(np.linspace(0., L, num_cells+1)) \
    ))))

    dx = coordinate[1:] - coordinate[:-1]
    rcell = 0.5*(coordinate[1:] + coordinate[:-1])

    # exact integral of mass in each cell gives this density
    mass_cells = rho0 * dx
    mass = np.cumsum(mass_cells)
    mass = np.array([1e-30, 1e-7*mass[0]]+ list(mass))

    ##### plot hydro profiles
    for time in np.array([0.5,1., 1.5])*Units.nsec:
        solution = solver.solve(mass=mass, time=time)
        
        for use_r in [True, False]:
            for fg in ["density", "velocity", "pressure", "temperature", "radiation_energy_flux"]:
                plt.figure(fg+str(use_r))
                if use_r: 
                    plt.plot(solution["position"], solution[fg], label=f"t={time:g}", marker="o")
                    plt.axvline(x=solution["boundary_position"], lw=2, c="k", ls="--")
                else:    
                    plt.plot(mass, solution[fg], label=f"t={time:g}")
                    plt.axvline(x=solution["ablated_mass"], lw=2, c="k", ls="--")
    for use_r in [True, False]:
        for fg in ["density", "velocity", "pressure", "temperature", "radiation_energy_flux"]:
            plt.figure(fg+str(use_r))
            plt.legend()
            plt.grid()
            plt.autoscale(enable=True, axis='both', tight=True)
            if use_r: plt.xlabel("x [cm]")
            else: plt.xlabel("mass [g/cm^2]")
            plt.ylabel(fg)

    plt.show()

    ########## plot position as a function of time
    position_times = np.array([np.array(solver.solve(mass=mass, time=time)["position"]) for time in times[1:]]).T

    plt.figure("position")
    for pos in position_times:
        plt.plot(times[1:], pos, c="k", lw=0.5)#, marker="o", markersize=1.)
    plt.axhline(y=0,            lw=1.5, ls="--", c="r", label="ablation front")
    plt.plot(times, x_boundary, lw=1.5, ls="--", c="b", label="boundary")
    plt.legend()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.xlabel("time [sec]")
    plt.ylabel("position [cm]")
    plt.show()
    quit()

if __name__ == "__main__":

    # test_xsi_f()
    test_profiles()
    # test_ablated_mass()