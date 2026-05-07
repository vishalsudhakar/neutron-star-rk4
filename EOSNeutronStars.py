'''
This library includes a 4th order Runge-Kutta method to numerically solve the
Relativistic Hydrodynamical Equations given an Equation of State.

Following Oppenheimer & Volkoff (1939) and Chandrasekhar (1957), the equations
are re-written in terms of dimensionless quantities.

The Equations of State (EOS) used here were derived by Wiringa et al. (1988)
and the fit functions were given by Kutschera & Kotlorz (1993). Two additional
tabulated EOS — APR and RG — are supported via cubic spline or linear
interpolation provided by the interpolator module.

New EOS can be substituted for the ones presented here; the numerical solver
functions are self-consistent provided the EOS is a function of baryon density
and the initial conditions are specified. For the EOS presented here, use:

    h  = any value < 1  (smaller → higher precision, longer runtime)
    R0 = 0
    t0 = 0.5            (for UV14-TN1 use t0 = 0.829)
    u0 = 0

Dependencies:
    numpy
    matplotlib
    interpolator  (local module)

Feel free to contact me regarding any questions at vsudhakar7@gatech.edu
'''

import numpy as np
import matplotlib.pyplot as pt

from interpolator import (
    E_linear_interpolated,
    E_cubic_interpolated,
    dE_dn_linear_interpolated,
    dE_dn_cubic_interpolated,
    d2E_dn2_linear_interpolated,
    d2E_dn2_cubic_interpolated,
)

# Fundamental constants
h_c             = 1240                          # MeV fm
hbar_c          = h_c / (2 * np.pi)            # MeV fm
m_n             = 939.565                       # MeV / c^2
c_energydensity = (np.pi**2 * m_n**4) / h_c**3 # Characteristic energy density [MeV fm^-3]


# Equations of State
class EOS:
    '''
    Defines the Equations of State (EOS) and their first and second derivatives
    with respect to baryon density.

    Supported EOS:
        i = 1  ->  AV14+UVII  (analytic fit)
        i = 2  ->  UV14-UVII  (analytic fit)
        i = 3  ->  UV14-TN1   (analytic fit)
        i = 4  ->  APR        (tabulated, interpolated)
        i = 5  ->  RG         (tabulated, interpolated)

    Parameters:
    interp_type : str — interpolation method for tabulated EOS: 'cubic' (default) or 'linear'
    '''

    def __init__(self, interp_type='cubic'):
        self.interpolation = interp_type

    def _interpolate(self, n, eos_name, quantity):
        '''
        Internal dispatcher for the interpolated EOS functions.

        Parameters:
        n        : float — baryon density [fm^-3]
        eos_name : str   — 'APR' or 'RG'
        quantity : str   — which quantity to return: 'E', 'dE_dn', or 'd2E_dn2'

        Returns:
        float — interpolated value of the requested quantity at density n
        '''
        funcs = {
            'cubic': {
                'E':       E_cubic_interpolated,
                'dE_dn':   dE_dn_cubic_interpolated,
                'd2E_dn2': d2E_dn2_cubic_interpolated,
            },
            'linear': {
                'E':       E_linear_interpolated,
                'dE_dn':   dE_dn_linear_interpolated,
                'd2E_dn2': d2E_dn2_linear_interpolated,
            },
        }
        return funcs[self.interpolation][quantity](n, eos_name)

    def E(self, n, i):
        '''
        Returns the energy per baryon E(n) for the chosen EOS.

        Parameters:
        n : float — baryon density [fm^-3]
        i : int   — EOS index (1–5, see class docstring)

        Returns:
        float — energy per baryon [MeV]
        '''
        if i == 1:
            func = 2.6511 + 76.744*n - 183.611*n**2 + 459.906*n**3 - 122.832*n**4
        elif i == 2:
            func = 7.57891 - 1.23275*n + 227.384*n**2 - 146.596*n**3 + 324.823*n**4 - 120.355*n**5
        elif i == 3:
            func = 6.33041 - 28.1793*n + 288.397*n**2 - 65.2281*n**3
        elif i == 4:
            func = self._interpolate(n, 'APR', 'E')
        elif i == 5:
            func = self._interpolate(n, 'RG', 'E')
        else:
            raise ValueError('EOS index i must be in [1, 5]')

        return func   # MeV

    def dE_dn(self, n, i):
        '''
        Returns the first derivative of E(n) with respect to baryon density.

        Parameters:
        n : float — baryon density [fm^-3]
        i : int   — EOS index (1–5, see class docstring)

        Returns:
        float — dE/dn [MeV fm^3]
        '''
        if i == 1:
            func = 76.744 - 2*183.611*n + 3*459.906*n**2 - 4*122.832*n**3
        elif i == 2:
            func = -1.23275 + 2*227.384*n - 3*146.596*n**2 + 4*324.823*n**3 - 5*120.355*n**4
        elif i == 3:
            func = -28.1793 + 2*288.397*n - 3*65.2281*n**2
        elif i == 4:
            func = self._interpolate(n, 'APR', 'dE_dn')
        elif i == 5:
            func = self._interpolate(n, 'RG', 'dE_dn')
        else:
            raise ValueError('EOS index i must be in [1, 5]')

        return func   # MeV fm^3

    def d2E_dn2(self, n, i):
        '''
        Returns the second derivative of E(n) with respect to baryon density.

        Parameters:
        n : float — baryon density [fm^-3]
        i : int   — EOS index (1–5, see class docstring)

        Returns:
        float — d²E/dn² [MeV fm^6]
        '''
        if i == 1:
            func = -2*183.611 + 6*459.906*n - 12*122.832*n**2
        elif i == 2:
            func = 2*227.384 - 6*146.596*n + 12*324.823*n**2 - 20*120.355*n**3
        elif i == 3:
            func = 2*288.397 - 6*65.2281*n
        elif i == 4:
            func = self._interpolate(n, 'APR', 'd2E_dn2')
        elif i == 5:
            func = self._interpolate(n, 'RG', 'd2E_dn2')
        else:
            raise ValueError('EOS index i must be in [1, 5]')

        return func   # MeV fm^6


# Relativistic Hydrodynamical Equations
class HydrodynamicalEquations:
    '''
    Encapsulates the Tolman–Oppenheimer–Volkoff (TOV) equations and the
    associated variable transformations written in dimensionless form
    following Oppenheimer & Volkoff (1939) and Chandrasekhar (1957).
    '''

    def __init__(self):
        self.eos = EOS()

    def n(self, t):
        '''
        Returns the baryon density as a function of the dimensionless parameter t.

        Parameters:
        t : float — dimensionless baryon density parameter

        Returns:
        float — baryon density n [fm^-3]
        '''
        term1 = np.sinh(t / 4)**3
        term2 = 3 * np.pi**2 * (hbar_c / m_n)**3
        return term1 / term2

    def pressure(self, n, i):
        '''
        Returns the dimensionless pressure for a given baryon density and EOS.

        Parameters:
        n : float — baryon density [fm^-3]
        i : int   — EOS index (1–5)

        Returns:
        float — dimensionless pressure P / c_energydensity
        '''
        return (n**2 * self.eos.dE_dn(n, i)) / c_energydensity

    def energydensity(self, n, i):
        '''
        Returns the dimensionless energy density for a given baryon density and EOS.

        Parameters:
        n : float — baryon density [fm^-3]
        i : int   — EOS index (1–5)

        Returns:
        float — dimensionless energy density ε / c_energydensity
        '''
        return (n * (self.eos.E(n, i) + m_n)) / c_energydensity

    def dt_dP(self, t, i):
        '''
        Returns dt/dP — the inverse of dP/dt — required to compute dt/dR.

        Parameters:
        t : float — dimensionless baryon density parameter
        i : int   — EOS index (1–5)

        Returns:
        float — dt/dP, or 0 if dP/dt = 0 (with a warning printed)
        '''
        nv    = self.n(t)
        term1 = 2*nv * self.eos.dE_dn(nv, i) + nv**2 * self.eos.d2E_dn2(nv, i)
        term2 = 3 * np.sinh(t / 4)**2 * np.cosh(t / 4) / (12 * np.pi**2 * (hbar_c / m_n)**3)
        value = (term1 * term2) / c_energydensity

        if value != 0:
            return value**-1
        else:
            print("dp_dt = 0")
            return 0

    def dt_dR(self, R, t, u, i):
        '''
        Returns dt/dR — the rate of change of the dimensionless baryon density
        parameter with respect to dimensionless radius, for use away from the core.

        Parameters:
        R : float — dimensionless radius
        t : float — dimensionless baryon density parameter
        u : float — dimensionless mass
        i : int   — EOS index (1–5)

        Returns:
        float — dt/dR
        '''
        nv    = self.n(t)
        P     = self.pressure(nv, i)
        eps   = self.energydensity(nv, i)

        term1 = -4 * np.pi * R
        term2 = self.dt_dP(t, i)
        term3 = (P + eps) / (1 - 2*u / R)
        term4 = P + u / (4 * np.pi * R**2)

        return term1 * term2 * term3 * term4

    def dt_dR1(self, R, t, i):
        '''
        Returns dt/dR evaluated at the core of the neutron star (R → 0) to
        avoid the singularity in the full TOV expression.

        Parameters:
        R : float — dimensionless radius
        t : float — dimensionless baryon density parameter
        i : int   — EOS index (1–5)

        Returns:
        float — dt/dR at the core
        '''
        nv  = self.n(t)
        P   = self.pressure(nv, i)
        eps = self.energydensity(nv, i)

        term1 = -4 * np.pi * R
        term2 = self.dt_dP(t, i)
        term3 = P + eps
        term4 = P

        return term1 * term2 * term3 * term4

    def du_dR(self, R, t, i):
        '''
        Returns du/dR — the rate of change of the dimensionless mass with
        respect to dimensionless radius.

        Parameters:
        R : float — dimensionless radius
        t : float — dimensionless baryon density parameter
        i : int   — EOS index (1–5)

        Returns:
        float — du/dR
        '''
        return 4 * np.pi * self.energydensity(self.n(t), i) * R**2

    def runga(self, h, R0, t0, u0, i):
        '''
        Performs a 4th-order Runge-Kutta integration of the TOV equations,
        stepping outward in radius by h until the baryon density or pressure
        drops to zero. Returns the mass and radius of the neutron star for
        the given central baryon density.

        Parameters:
        h  : float — radial step size ΔR
        R0 : float — initial (dimensionless) radius
        t0 : float — initial dimensionless baryon density parameter
        u0 : float — initial dimensionless mass
        i  : int   — EOS index (1–5)

        Returns:
        tuple — (n(t0) [fm^-3], radius [km], mass [M_sun])
        '''
        un = u0
        tn = t0
        R  = R0
        Ri = R
        ui = un

        nv = self.n(tn)
        pv = self.pressure(nv, i)

        while nv >= 0 and pv >= 0:
            ui = un
            ti = tn
            Ri = R

            k1_u = h * self.du_dR(R, ti, i)
            k1_t = h * (self.dt_dR1(R, ti, i) if R == 0 else self.dt_dR(R, ti, ui, i))

            nv = self.n(ti + k1_t / 2)
            pv = self.pressure(nv, i)
            if nv < 0 or pv < 0:
                break

            k2_u = h * self.du_dR(R + h/2, ti + k1_t/2, i)
            k2_t = h * self.dt_dR(R + h/2, ti + k1_t/2, ui + k1_u/2, i)

            nv = self.n(ti + k2_t / 2)
            pv = self.pressure(nv, i)
            if nv < 0 or pv < 0:
                break

            k3_u = h * self.du_dR(R + h/2, ti + k2_t/2, i)
            k3_t = h * self.dt_dR(R + h/2, ti + k2_t/2, ui + k2_u/2, i)

            nv = self.n(ti + k3_t)
            pv = self.pressure(nv, i)
            if nv < 0 or pv < 0:
                break

            k4_u = h * self.du_dR(R + h,   ti + k3_t,   i)
            k4_t = h * self.dt_dR(R + h,   ti + k3_t,   ui + k3_u, i)

            tn = ti + (k1_t + 2*k2_t + 2*k3_t + k4_t) / 6
            un = ui + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
            R  = Ri + h

            nv = self.n(tn)
            pv = self.pressure(nv, i)

        return (self.n(t0), Ri * 13.69, ui * 9.29)


# Run & plot
class run(EOS, HydrodynamicalEquations):
    '''
    Orchestrates the mass-radius computation over a range of central baryon
    densities and exposes plotting utilities for the results.

    Parameters:
    interp_type : str — interpolation method for tabulated EOS: 'cubic' (default) or 'linear'
    '''

    def __init__(self, interp_type='cubic'):
        super().__init__(interp_type)
        self.eos = EOS(interp_type)

        self.cd            = np.array([])
        self.r             = np.array([])
        self.m             = np.array([])
        self.complete_data = ()

        pt.rcParams['font.weight']        = 'normal'
        pt.rcParams['axes.labelweight']   = 'normal'
        pt.rcParams['axes.linewidth']     = 1.5
        pt.rcParams['lines.linewidth']    = 1.5

    def loop(self, h, R0, t0, u0, i):
        '''
        Iterates over a range of central baryon densities, calling runga() for
        each and collecting the resulting (baryon density, radius, mass) tuples.

        Parameters:
        h  : float — radial step size ΔR
        R0 : float — initial dimensionless radius
        t0 : float — initial dimensionless baryon density parameter
        u0 : float — initial dimensionless mass
        i  : int   — EOS index (1–5)

        Returns:
        tuple — (cd, r, mass) each a list of values across baryon densities
        '''
        t  = t0
        cd, r, mass = [], [], []

        dP_dt = (self.dt_dP(t, i))**-1

        while dP_dt >= 0:
            each_data = self.runga(h, R0, t, u0, i)
            cd.append(each_data[0])
            r.append(each_data[1])
            mass.append(each_data[2])

            t += 0.01
            print(f"Calculating for baryon density t = {t}")

            dP_dt = (self.dt_dP(t, i))**-1

            if t > 3.05:
                break

        return (cd, r, mass)

    def calculate(self, h, R0, t0, u0, i):
        '''
        Runs the full mass-radius calculation and stores the results internally.

        Parameters:
        h  : float — radial step size ΔR
        R0 : float — initial dimensionless radius
        t0 : float — initial dimensionless baryon density parameter
        u0 : float — initial dimensionless mass
        i  : int   — EOS index (1–5)

        Returns:
        None — results stored in self.cd, self.r, self.m, and self.complete_data
        '''
        data = self.loop(h, R0, t0, u0, i)

        self.cd            = np.array(data[0])
        self.r             = np.array(data[1])
        self.m             = np.array(data[2])
        self.complete_data = (self.cd, self.r, self.m)

    def plot_mass_baryon_density(self, label='', color=(173/255, 52/255, 62/255), savefig=False):
        '''
        Plots mass vs. central baryon density.

        Parameters:
        label   : str   — legend label for the curve
        color   : tuple — RGB colour for the curve (default: dark red)
        savefig : bool  — if True, saves the figure to 'mass_n0.pdf'

        Returns:
        None
        '''
        pt.plot(self.cd, self.m, label=label, color=color)
        pt.ylabel(r'Mass ($M_{\odot}$)', fontsize=13)
        pt.xlabel(r'$n_0$ (fm$^{-3}$)',  fontsize=13)
        pt.xlim([0, 2.5])
        pt.ylim([0, 3])
        pt.legend(loc=4, frameon=False, fontsize=13)
        pt.minorticks_on()
        pt.tick_params(which='minor', direction='in', width=2)
        pt.tick_params(which='major', direction='in', width=2)
        if savefig:
            pt.savefig('mass_n0.pdf')
        pt.show()

    def plot_radius_baryon_density(self, label='', color=(173/255, 52/255, 62/255), savefig=False):
        '''
        Plots radius vs. central baryon density.

        Parameters:
        label   : str   — legend label for the curve
        color   : tuple — RGB colour for the curve (default: dark red)
        savefig : bool  — if True, saves the figure to 'radius_n0.pdf'

        Returns:
        None
        '''
        pt.plot(self.cd, self.r, label=label, color=color)
        pt.ylabel('Radius (km)',            fontsize=13)
        pt.xlabel(r'$n_0$ (fm$^{-3}$)',    fontsize=13)
        pt.xlim([0, 2.5])
        pt.ylim([0, 15])
        pt.legend(loc=4, frameon=False, fontsize=13)
        pt.minorticks_on()
        pt.tick_params(which='minor', direction='in', width=2)
        pt.tick_params(which='major', direction='in', width=2)
        if savefig:
            pt.savefig('radius_n0.pdf')
        pt.show()

    def plot_mass_radius(self, label='', color=(173/255, 52/255, 62/255), savefig=False):
        '''
        Plots mass vs. radius and overlays observational constraints from
        PSR J0030+0451 and PSR J0740+6620.

        Parameters:
        label   : str   — legend label for the curve
        color   : tuple — RGB colour for the curve (default: dark red)
        savefig : bool  — if True, saves the figure to 'mass_radius.pdf'

        Returns:
        None
        '''
        pt.plot(self.r, self.m, label=label, color=color)
        pt.ylabel(r'Mass ($M_{\odot}$)', fontsize=13)
        pt.xlabel('Radius (km)',          fontsize=13)
        pt.xlim([6, 20])
        pt.ylim([0, 3])

        # Observational data points with uncertainties
        point_x = [13.02, 13.7]
        point_y = [1.44,  2.08]
        error_x = [[1.06, 1.24], [1.5,  2.6 ]]
        error_y = [[0.14, 0.07], [0.15, 0.07]]

        pt.errorbar(point_x, point_y, xerr=error_x, yerr=error_y, fmt='o', color='black')
        pt.annotate('PSR J0030+0451', (13.02, 1.44), textcoords='offset points', xytext=(10, 10), fontsize=11)
        pt.annotate('PSR J0740+6620', (13.7,  2.08), textcoords='offset points', xytext=(10, 10), fontsize=11)

        pt.legend(loc=4, frameon=False, fontsize=11)
        pt.minorticks_on()
        pt.tick_params(which='minor', direction='in', width=2)
        pt.tick_params(which='major', direction='in', width=2)

        if savefig:
            pt.savefig('mass_radius.pdf')
        pt.show()
