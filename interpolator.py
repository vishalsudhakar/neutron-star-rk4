'''
This library provides interpolation utilities for tabulated Equations of State (EOS)
used in the relativistic hydrodynamics solver.

Two tabulated EOS are supported:
    - APR  : Akmal, Pandharipande & Ravenhall (1998)
    - RG   : Relativistic mean-field model (Glendenning)

Data is loaded from:
    data/eos_APR.table
    data/eos_RG.table

Discretized first and second derivatives of E(n) are constructed from the tabulated
values using a second-order finite difference scheme (non-uniform grid). Interpolation
is then performed either linearly or via cubic spline (scipy.interpolate.CubicSpline).

The public-facing interpolated functions — E, dE/dn, and d²E/dn² — are consumed by
the EOS class in the main solver and follow the same calling convention as the
analytic EOS fit functions.

Dependencies:
    numpy
    pandas
    scipy

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Column indices within the EOS table files
_E_col = 4   # Energy per baryon [MeV]
_n_col = 1   # Baryon density    [fm^-3]

# APR EoS — rows below start_APR are below the regime of interest
_start_APR = 916
_data_APR  = np.transpose(np.array(pd.read_csv('/Users/vishalsudhakar/Documents/classes/Spring 2026/computational_astrophysics/final-project/neutron-stars/tabulated-eos/eos_APR.table', header=None)))
_E_APR_raw = _data_APR[_E_col][_start_APR:]   # MeV
_n_APR_raw = _data_APR[_n_col][_start_APR:]   # fm^-3

# RG EoS
_start_RG  = 823
_data_RG   = np.transpose(np.array(pd.read_csv('/Users/vishalsudhakar/Documents/classes/Spring 2026/computational_astrophysics/final-project/neutron-stars/tabulated-eos/eos_RG.table', header=None)))
_E_RG_raw  = _data_RG[_E_col][_start_RG:]     # MeV
_n_RG_raw  = _data_RG[_n_col][_start_RG:]     # fm^-3


# ---------------------------------------------------------------------------
# Finite-difference derivative construction
# ---------------------------------------------------------------------------

class DiscretizedDerivatives:
    '''
    Constructs discretized first and second derivatives of E(n) from tabulated
    data using a second-order finite difference scheme on a non-uniform grid.

    The scheme follows the standard weighted finite difference approach:

        dE/dn  |_i  =  [ Δn_{i-1}² (E_{i+1} - E_i) + Δn_{i+1}² (E_i - E_{i-1}) ]
                        / [ Δn_{i+1} Δn_{i-1}² - Δn_{i-1} Δn_{i+1}² ]

        d²E/dn² |_i =  2 [ Δn_{i-1} (E_{i+1} - E_i) + Δn_{i+1} (E_i - E_{i-1}) ]
                        / [ Δn_{i+1}² Δn_{i-1} - Δn_{i-1}² Δn_{i+1} ]

    where Δn_{i±1} = n_{i±1} - n_i.

    Edge points are excluded from the returned arrays because boundary information
    is insufficient to evaluate the stencil there.
    '''

    @staticmethod
    def first_derivative(E_, n_):
        '''
        Returns the discretized first derivative dE/dn at each interior grid point.

        Parameters:
        E_  : array_like — tabulated energy per baryon values [MeV]
        n_  : array_like — tabulated baryon density values [fm^-3],
              must be monotonically increasing

        Returns:
        dE_dn : ndarray of shape (N-2,) — first derivative at interior points [MeV fm^3]
        '''
        N     = np.size(n_)
        dE_dn = np.zeros(N)

        for i in range(1, N - 1):
            Dn_im1 = n_[i - 1] - n_[i]   # Δn_{i-1} = n_{i-1} - n_i
            Dn_ip1 = n_[i + 1] - n_[i]   # Δn_{i+1} = n_{i+1} - n_i

            numerator   = Dn_im1**2 * (E_[i + 1] - E_[i]) + Dn_ip1**2 * (E_[i] - E_[i - 1])
            denominator = Dn_ip1 * Dn_im1**2 - Dn_im1 * Dn_ip1**2

            dE_dn[i] = numerator / denominator

        return dE_dn[1:N - 1]   # exclude edge points

    @staticmethod
    def second_derivative(E_, n_):
        '''
        Returns the discretized second derivative d²E/dn² at each interior grid point.

        Parameters:
        E_  : array_like — tabulated energy per baryon values [MeV]
        n_  : array_like — tabulated baryon density values [fm^-3],
              must be monotonically increasing

        Returns:
        d2E_dn2 : ndarray of shape (N-2,) — second derivative at interior points [MeV fm^6]
        '''
        N       = np.size(n_)
        d2E_dn2 = np.zeros(N)

        for i in range(1, N - 1):
            Dn_im1 = n_[i - 1] - n_[i]   # Δn_{i-1} = n_{i-1} - n_i
            Dn_ip1 = n_[i + 1] - n_[i]   # Δn_{i+1} = n_{i+1} - n_i

            numerator   = Dn_im1 * (E_[i + 1] - E_[i]) + Dn_ip1 * (E_[i] - E_[i - 1])
            denominator = Dn_ip1**2 * Dn_im1 - Dn_im1**2 * Dn_ip1

            d2E_dn2[i] = 2 * numerator / denominator

        return d2E_dn2[1:N - 1]   # exclude edge points


# ---------------------------------------------------------------------------
# Pre-compute derivatives and trim edge points from raw tables
# ---------------------------------------------------------------------------

_dd = DiscretizedDerivatives

dE_dn_APR   = _dd.first_derivative(_E_APR_raw, _n_APR_raw)
dE_dn_RG    = _dd.first_derivative(_E_RG_raw,  _n_RG_raw)

d2E_dn2_APR = _dd.second_derivative(_E_APR_raw, _n_APR_raw)
d2E_dn2_RG  = _dd.second_derivative(_E_RG_raw,  _n_RG_raw)

# Trim the raw tables to match the interior-only derivative arrays
n_APR = _n_APR_raw[1:-1]
E_APR = _E_APR_raw[1:-1]

n_RG  = _n_RG_raw[1:-1]
E_RG  = _E_RG_raw[1:-1]


# ---------------------------------------------------------------------------
# Interpolation methods
# ---------------------------------------------------------------------------

class Interpolator:
    '''
    Provides linear and cubic spline interpolation for arbitrary tabulated data (x_i, y_i).

    The class is generalised so that it can be used for E(n), dE/dn, and d²E/dn²
    without modification. When the target value lies outside the data range the
    edge value is returned (constant extrapolation).
    '''

    @staticmethod
    def linear(x_target, y_i, x_i):
        '''
        Linearly interpolates between adjacent tabulated points (x_i, y_i).

        Parameters:
        x_target : float — the independent variable value at which to interpolate
        y_i      : array_like — dependent (y) data values
        x_i      : array_like — independent (x) data values,
                   must be monotonically increasing

        Returns:
        float — interpolated value of y at x_target
        '''
        if x_target < np.min(x_i):
            return y_i[np.argmin(x_i)]
        if x_target > np.max(x_i):
            return y_i[np.argmax(x_i)]

        # Index of the lower bound of the interval containing x_target
        i = np.nonzero(x_i - x_target >= 0)[0][0] - 1
        m = (y_i[i + 1] - y_i[i]) / (x_i[i + 1] - x_i[i])
        return m * (x_target - x_i[i]) + y_i[i]

    @staticmethod
    def cubic(x_target, y_i, x_i):
        '''
        Cubic spline interpolation using scipy.interpolate.CubicSpline with
        'natural' boundary conditions (second derivatives vanish at the endpoints).

        Parameters:
        x_target : float — the independent variable value at which to interpolate
        y_i      : array_like — dependent (y) data values
        x_i      : array_like — independent (x) data values,
                   must be monotonically increasing

        Returns:
        float — interpolated value of y at x_target
        '''
        if x_target < np.min(x_i):
            return y_i[np.argmin(x_i)]
        if x_target > np.max(x_i):
            return y_i[np.argmax(x_i)]

        cs = CubicSpline(x=x_i, y=y_i, bc_type='natural')
        return cs(x_target)

# ---------------------------------------------------------------------------
# Pre-built cubic spline objects (constructed once at import time for efficiency)
# ---------------------------------------------------------------------------

_bc_type = 'not-a-knot'

_CS_APR = CubicSpline(x=n_APR, y=E_APR, bc_type=_bc_type)
_CS_RG  = CubicSpline(x=n_RG,  y=E_RG,  bc_type=_bc_type)


# ---------------------------------------------------------------------------
# Public interpolated EOS functions
# These functions are imported and called by EOS in the main solver.
# ---------------------------------------------------------------------------

def E_linear_interpolated(n_target, eos):
    '''
    Returns E(n) at n_target via linear interpolation of the tabulated EOS.

    Parameters:
    n_target : float — baryon density [fm^-3]
    eos      : str   — equation of state identifier: 'APR' or 'RG'

    Returns:
    float — energy per baryon E(n_target) [MeV]
    '''
    if eos == 'APR':
        return Interpolator.linear(n_target, E_APR, n_APR)
    elif eos == 'RG':
        return Interpolator.linear(n_target, E_RG,  n_RG)
    else:
        raise ValueError("Invalid EOS: options are 'APR' or 'RG'")


def dE_dn_linear_interpolated(n_target, eos):
    '''
    Returns dE/dn at n_target via linear interpolation of the discretized derivative.

    Parameters:
    n_target : float — baryon density [fm^-3]
    eos      : str   — equation of state identifier: 'APR' or 'RG'

    Returns:
    float — first derivative dE/dn at n_target [MeV fm^3]
    '''
    if eos == 'APR':
        return Interpolator.linear(n_target, dE_dn_APR, n_APR)
    elif eos == 'RG':
        return Interpolator.linear(n_target, dE_dn_RG,  n_RG)
    else:
        raise ValueError("Invalid EOS: options are 'APR' or 'RG'")


def d2E_dn2_linear_interpolated(n_target, eos):
    '''
    Returns d²E/dn² at n_target via linear interpolation of the discretized second derivative.

    Parameters:
    n_target : float — baryon density [fm^-3]
    eos      : str   — equation of state identifier: 'APR' or 'RG'

    Returns:
    float — second derivative d²E/dn² at n_target [MeV fm^6]
    '''
    if eos == 'APR':
        return Interpolator.linear(n_target, d2E_dn2_APR, n_APR)
    elif eos == 'RG':
        return Interpolator.linear(n_target, d2E_dn2_RG,  n_RG)
    else:
        raise ValueError("Invalid EOS: options are 'APR' or 'RG'")


def E_cubic_interpolated(n_target, eos):
    '''
    Returns E(n) at n_target via cubic spline interpolation of the tabulated EOS.

    Parameters:
    n_target : float — baryon density [fm^-3]
    eos      : str   — equation of state identifier: 'APR' or 'RG'

    Returns:
    float — energy per baryon E(n_target) [MeV]
    '''
    if eos == 'APR':
        interpolator = _CS_APR
    elif eos == 'RG':
        interpolator = _CS_RG
    else:
        raise ValueError("Invalid EOS: options are 'APR' or 'RG'")

    return interpolator(n_target)


def dE_dn_cubic_interpolated(n_target, eos):
    '''
    Returns dE/dn at n_target via the first derivative of the cubic spline of E(n).

    Parameters:
    n_target : float — baryon density [fm^-3]
    eos      : str   — equation of state identifier: 'APR' or 'RG'

    Returns:
    float — first derivative dE/dn at n_target [MeV fm^3]
    '''
    if eos == 'APR':
        interpolator = _CS_APR
    elif eos == 'RG':
        interpolator = _CS_RG
    else:
        raise ValueError("Invalid EOS: options are 'APR' or 'RG'")

    return interpolator.derivative(nu=1)(n_target)


def d2E_dn2_cubic_interpolated(n_target, eos):
    '''
    Returns d²E/dn² at n_target via the second derivative of the cubic spline of E(n).

    Parameters:
    n_target : float — baryon density [fm^-3]
    eos      : str   — equation of state identifier: 'APR' or 'RG'

    Returns:
    float — second derivative d²E/dn² at n_target [MeV fm^6]
    '''
    if eos == 'APR':
        interpolator = _CS_APR
    elif eos == 'RG':
        interpolator = _CS_RG
    else:
        raise ValueError("Invalid EOS: options are 'APR' or 'RG'")

    return interpolator.derivative(nu=2)(n_target)


# ---------------------------------------------------------------------------
# Plotting utility
# ---------------------------------------------------------------------------
class InterpolationPlotter:
    '''
    Utility class for visualising the interpolated EOS functions E(n), dE/dn,
    and d²E/dn² for both APR and RG models.
    '''

    @staticmethod
    def plot(interpolation='cubic', resolution=30):
        '''
        Produces a 2×3 grid of scatter plots showing the interpolated values of
        E(n), dE/dn, and d²E/dn² for the APR (top row) and RG (bottom row) EOS.

        Parameters:
        interpolation : str — interpolation method: 'linear' or 'cubic'
        resolution    : int — number of evaluation points along the density axis

        Returns:
        None — displays the figure via plt.show()
        '''
        fig, axs = plt.subplots(2, 3, sharex=True)

        eos_configs = [
            ('APR', n_APR, 'lightcoral', 0),
            ('RG',  n_RG,  'lightskyblue', 1),
        ]

        func_map = {
            'linear': (E_linear_interpolated, dE_dn_linear_interpolated, d2E_dn2_linear_interpolated),
            'cubic':  (E_cubic_interpolated,  dE_dn_cubic_interpolated,  d2E_dn2_cubic_interpolated),
        }

        if interpolation not in func_map:
            raise ValueError("interpolation must be 'linear' or 'cubic'")

        E_fn, dE_fn, d2E_fn = func_map[interpolation]

        for eos_name, n_data, color, row in eos_configs:
            n_vals   = np.geomspace(n_data[0], n_data[-1], resolution)
            E_vals   = [E_fn(n,   eos_name) for n in n_vals]
            dE_vals  = [dE_fn(n,  eos_name) for n in n_vals]
            d2E_vals = [d2E_fn(n, eos_name) for n in n_vals]

            axs[row, 0].scatter(n_vals, E_vals,   c=color, label=rf'$E(n)$ ({eos_name})')
            axs[row, 1].scatter(n_vals, dE_vals,  c=color, label=rf"$E'(n)$ ({eos_name})")
            axs[row, 2].scatter(n_vals, d2E_vals, c=color, label=rf"$E''(n)$ ({eos_name})")

        for ax in axs.flatten():
            ax.legend()

        # Log scale for derivative panels where values span several orders of magnitude
        for col in [1, 2]:
            axs[0, col].set_yscale('log')
            axs[1, col].set_yscale('log')

        plt.tight_layout()
        plt.show()
