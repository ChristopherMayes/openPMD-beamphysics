"""
Resistive wall impedance implementation.

This module provides functions to compute the longitudinal impedance Z(k) and
wakefield W(z) for resistive wall beam pipes using direct numerical integration.
It implements the flat (parallel plate) geometry model with AC conductivity
effects, based on the formalism in SLAC-PUB-10707.

Functions
---------
ac_conductivity
    Frequency-dependent AC conductivity with relaxation time
surface_impedance
    Surface impedance for a conducting wall
longitudinal_impedance
    Longitudinal impedance Z(k) from surface impedance integration
wakefield_from_impedance
    Wakefield W(z) via cosine/sine transform of Z(k)

References
----------
Bane & Stupakov, SLAC-PUB-10707 (2004)
https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf
"""

import numpy as np
from scipy.integrate import quad, quad_vec

from ..units import c_light, Z0


def sinhc(x):
    """
    Numerically stable sinh(x)/x function.

    Uses Taylor series expansion for small x to avoid numerical instability.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s)

    Returns
    -------
    y : float or np.ndarray
        sinh(x)/x, defined as 1 at x = 0
    """
    x = np.asarray(x)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    y = np.empty_like(x, dtype=np.float64)

    # Use Taylor series for small x
    small = np.abs(x) < 1e-5
    x_small = x[small]
    x_large = x[~small]

    y[small] = 1 + x_small**2 / 6 + x_small**4 / 120
    y[~small] = np.sinh(x_large) / x_large

    if scalar_input:
        return float(y[0])
    return y


def ac_conductivity(k, sigma0, ctau):
    """
    Frequency-dependent AC conductivity with relaxation time.

    Implements the Drude model for AC conductivity:
        σ(k) = σ₀ / (1 - i k c τ)

    Parameters
    ----------
    k : float or np.ndarray
        Longitudinal wave number [1/m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c*τ [m]

    Returns
    -------
    sigma : complex or np.ndarray of complex
        AC conductivity [S/m]
    """
    return sigma0 / (1 - 1j * k * ctau)


def surface_impedance(k, sigma0, ctau):
    """
    Surface impedance for a conducting wall.

    Parameters
    ----------
    k : float or np.ndarray
        Longitudinal wave number [1/m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c*τ [m]

    Returns
    -------
    zeta : complex or np.ndarray of complex
        Surface impedance [dimensionless]
    """
    sigma = ac_conductivity(k, sigma0, ctau)
    return (1 - 1j) * np.sqrt(k * c_light / (2 * sigma * Z0 * c_light))


def _impedance_integrand(k, x, a, sigma0, ctau):
    """
    Impedance integrand for flat geometry.

    Parameters
    ----------
    k : float
        Longitudinal wave number [1/m]
    x : float or np.ndarray
        Integration variable (dimensionless)
    a : float
        Half-gap height [m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c*τ [m]

    Returns
    -------
    result : complex or np.ndarray of complex
        Integrand value(s) [Ohm/m]
    """
    x = np.asarray(x)

    # Prefactor
    prefactor = Z0 / (2 * np.pi * a)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = surface_impedance(k, sigma0, ctau)
        cosh_x = np.cosh(x)
        shc_x = sinhc(x)

        denom = cosh_x * (cosh_x / z - 1j * k * a * shc_x)

        # Assign result only where denominator is finite
        result = np.where(np.isfinite(denom), prefactor / denom, 0.0 + 0.0j)

    return result.astype(np.complex128)


def longitudinal_impedance(k, a, sigma0, ctau):
    """
    Compute longitudinal impedance Z(k) for flat (parallel plate) geometry.

    Integrates the surface impedance kernel over all transverse modes.

    Parameters
    ----------
    k : float or np.ndarray
        Longitudinal wave number [1/m]
    a : float
        Half-gap height [m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c*τ [m]

    Returns
    -------
    Zk : complex or np.ndarray of complex
        Longitudinal impedance [Ohm/m]

    Examples
    --------
    >>> import numpy as np
    >>> ks = np.linspace(0, 1e5, 100)
    >>> Zk = longitudinal_impedance(ks, a=4.5e-3, sigma0=2.4e7, ctau=2.4e-6)
    """

    @np.vectorize
    def _Zk_scalar(k_val):
        if k_val == 0:
            return 0.0 + 0.0j

        def integrand(x):
            return _impedance_integrand(k_val, x, a, sigma0, ctau)

        return quad_vec(integrand, 0, np.inf)[0]

    return _Zk_scalar(k)


def wakefield_from_impedance(z, Zk_func, k_max=1e7, epsabs=1e-9, epsrel=1e-6):
    """
    Compute wakefield W(z) from Re[Z(k)] using a cosine transform.

    Uses direct quadrature:
        W(z) = (2/π) ∫₀^k_max Re(Z(k)) cos(kz) dk

    Parameters
    ----------
    z : float or np.ndarray
        Longitudinal position [m]. Positive z is behind the source.
    Zk_func : callable
        Function returning complex impedance Z(k) [Ohm/m] for wave number k [1/m]
    k_max : float, optional
        Upper limit of k integration [1/m]. Default is 1e7.
    epsabs : float, optional
        Absolute tolerance for integration. Default is 1e-9.
    epsrel : float, optional
        Relative tolerance for integration. Default is 1e-6.

    Returns
    -------
    Wz : float or np.ndarray
        Wakefield [V/C/m]

    Notes
    -----
    The wakefield is zero for z < 0 (ahead of the source particle).

    Examples
    --------
    >>> import numpy as np
    >>> from functools import partial
    >>> Zk = partial(longitudinal_impedance, a=4.5e-3, sigma0=2.4e7, ctau=2.4e-6)
    >>> zs = np.linspace(0, 100e-6, 50)
    >>> Wz = [wakefield_from_impedance(z, Zk) for z in zs]
    """

    def _wakefield_scalar(z_val):
        if z_val < 0:
            return 0.0

        def integrand(k):
            return np.real(Zk_func(k)) * np.cos(k * z_val)

        result, _ = quad(integrand, 0, k_max, epsabs=epsabs, epsrel=epsrel, limit=200)
        return (2 / np.pi) * result

    z = np.asarray(z)
    if z.ndim == 0:
        return _wakefield_scalar(float(z))

    return np.array([_wakefield_scalar(zi) for zi in z])


def characteristic_length(a, sigma0):
    """
    Characteristic length scale s₀ for resistive wall wakefield.

    From SLAC-PUB-10707 Eq. 5:
        s₀ = (2a² / (Z₀ σ₀))^(1/3)

    Parameters
    ----------
    a : float
        Half-gap height (flat) or radius (round) [m]
    sigma0 : float
        DC conductivity [S/m]

    Returns
    -------
    s0 : float
        Characteristic length [m]
    """
    return (2 * a**2 / (Z0 * sigma0)) ** (1 / 3)


class FlatResistiveWallImpedance:
    """
    Resistive wall impedance model for flat (parallel plate) geometry.

    This class encapsulates the physical parameters and provides methods
    to compute impedance Z(k) and wakefield W(z).

    Parameters
    ----------
    half_gap : float
        Half-gap height between plates [m]
    conductivity : float
        DC electrical conductivity [S/m]
    relaxation_time : float
        Drude-model relaxation time [s]

    Attributes
    ----------
    a : float
        Half-gap height [m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c*τ [m]
    s0 : float
        Characteristic length scale [m]

    Examples
    --------
    >>> imp = FlatResistiveWallImpedance(
    ...     half_gap=4.5e-3,
    ...     conductivity=2.4e7,
    ...     relaxation_time=8e-15
    ... )
    >>> ks = np.linspace(0, 1e5, 100)
    >>> Zk = imp.impedance(ks)
    >>> zs = np.linspace(0, 100e-6, 20)
    >>> Wz = imp.wakefield(zs)
    """

    def __init__(self, half_gap, conductivity, relaxation_time):
        self.a = half_gap
        self.sigma0 = conductivity
        self.ctau = c_light * relaxation_time
        self.s0 = characteristic_length(half_gap, conductivity)

    def impedance(self, k):
        """
        Compute longitudinal impedance Z(k).

        Parameters
        ----------
        k : float or np.ndarray
            Longitudinal wave number [1/m]

        Returns
        -------
        Zk : complex or np.ndarray of complex
            Longitudinal impedance [Ohm/m]
        """
        return longitudinal_impedance(k, self.a, self.sigma0, self.ctau)

    def wakefield(self, z, k_max=1e7, epsabs=1e-9, epsrel=1e-6):
        """
        Compute wakefield W(z) from impedance.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position [m]
        k_max : float, optional
            Upper limit of k integration [1/m]. Default is 1e7.
        epsabs : float, optional
            Absolute tolerance for integration. Default is 1e-9.
        epsrel : float, optional
            Relative tolerance for integration. Default is 1e-6.

        Returns
        -------
        Wz : float or np.ndarray
            Wakefield [V/C/m]
        """
        return wakefield_from_impedance(
            z, self.impedance, k_max=k_max, epsabs=epsabs, epsrel=epsrel
        )

    def plot_impedance(self, k_max=1e6, n_points=200, ax=None):
        """
        Plot real and imaginary parts of impedance vs wave number.

        Parameters
        ----------
        k_max : float, optional
            Maximum wave number [1/m]. Default is 1e6.
        n_points : int, optional
            Number of points to plot. Default is 200.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        ks = np.linspace(0, k_max, n_points)
        Zk = self.impedance(ks)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(ks, np.real(Zk), label="Re[Z(k)]")
        ax.plot(ks, np.imag(Zk), label="Im[Z(k)]")
        ax.set_xlabel(r"$k$ (1/m)")
        ax.set_ylabel(r"$Z(k)$ (Ω/m)")
        ax.legend()
        ax.set_title(f"Flat geometry: a={self.a*1e3:.2f} mm, σ₀={self.sigma0:.2e} S/m")

        return ax

    def plot_wakefield(self, z_max=200e-6, n_points=50, k_max=1e6, ax=None):
        """
        Plot wakefield vs longitudinal position.

        Parameters
        ----------
        z_max : float, optional
            Maximum z position [m]. Default is 200 µm.
        n_points : int, optional
            Number of points to plot. Default is 50.
        k_max : float, optional
            Upper limit of k integration [1/m]. Default is 1e6.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        import matplotlib.pyplot as plt

        zs = np.linspace(0, z_max, n_points)
        Wz = self.wakefield(zs, k_max=k_max)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(zs * 1e6, Wz * 1e-12)
        ax.set_xlabel(r"$z$ (µm)")
        ax.set_ylabel(r"$W(z)$ (V/pC/m)")
        ax.set_title(f"Flat geometry: a={self.a*1e3:.2f} mm")

        return ax

    def __repr__(self):
        return (
            f"FlatResistiveWallImpedance("
            f"half_gap={self.a}, "
            f"conductivity={self.sigma0}, "
            f"relaxation_time={self.ctau/c_light})"
        )
