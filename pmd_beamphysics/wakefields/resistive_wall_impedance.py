"""
Resistive wall impedance calculations for flat and round geometries.

This module implements the longitudinal impedance Z(k) and wakefield W(z)
for resistive wall beam pipes using direct numerical integration. It supports
both flat (parallel plate) and round (circular pipe) geometries with AC
conductivity effects based on the Drude model.

Theory
------
The longitudinal impedance is computed from the surface impedance using
electromagnetic field matching at the beam pipe boundary. The formulas
follow from solving Maxwell's equations with boundary conditions imposed
by the conducting wall.

For a conducting wall, the surface impedance is defined as:

    ζ = E_∥ / H_∥ = (1-i) √(k c / (2 σ Z₀ c))

where σ is the (possibly frequency-dependent) conductivity.

The AC conductivity follows the Drude model:

    σ(k) = σ₀ / (1 - i k c τ)

where τ is the relaxation time accounting for the anomalous skin effect
at high frequencies.

Functions
---------
sinhc
    Numerically stable sinh(x)/x function
ac_conductivity
    Frequency-dependent AC conductivity with relaxation time
surface_impedance
    Surface impedance for a conducting wall
longitudinal_impedance_flat
    Longitudinal impedance Z(k) for flat geometry
longitudinal_impedance_round
    Longitudinal impedance Z(k) for round geometry
wakefield_from_impedance
    Wakefield W(z) via cosine transform of Re[Z(k)]
characteristic_length
    Characteristic length scale s₀

Classes
-------
FlatResistiveWallImpedance
    Impedance model for flat (parallel plate) geometry
RoundResistiveWallImpedance
    Impedance model for round (circular pipe) geometry

References
----------
.. [1] N. Mounet and E. Métral, "Electromagnetic field created by a
   macroparticle in an infinitely long and axisymmetric multilayer beam
   pipe," Phys. Rev. ST Accel. Beams 18, 034402 (2015).
   https://doi.org/10.1103/PhysRevSTAB.18.034402

.. [2] K. Bane and G. Stupakov, "Resistive wall wakefield in the LCLS
   undulator beam pipe," SLAC-PUB-10707 (2004).
   https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf

.. [3] A. Chao, "Physics of Collective Beam Instabilities in High Energy
   Accelerators," Wiley, 1993, Chapter 2.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, quad_vec

from ..units import c_light, Z0

__all__ = [
    "sinhc",
    "ac_conductivity",
    "surface_impedance",
    "longitudinal_impedance_flat",
    "longitudinal_impedance_round",
    "wakefield_from_impedance",
    "characteristic_length",
    "ResistiveWallImpedance",
    # Legacy aliases for backwards compatibility
    "FlatResistiveWallImpedance",
    "RoundResistiveWallImpedance",
]


def sinhc(x: float | np.ndarray) -> float | np.ndarray:
    """
    Numerically stable sinh(x)/x function.

    Computes:

    .. math::

        \\text{sinhc}(x) = \\frac{\\sinh(x)}{x}

    Uses Taylor series expansion for small x to avoid numerical instability:

    .. math::

        \\text{sinhc}(x) \\approx 1 + \\frac{x^2}{6} + \\frac{x^4}{120} + O(x^6)

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


def ac_conductivity(
    k: float | np.ndarray,
    sigma0: float,
    ctau: float,
) -> complex | np.ndarray:
    """
    Frequency-dependent AC conductivity with relaxation time.

    Implements the Drude model for AC conductivity [2]_:

    .. math::

        \\sigma(k) = \\frac{\\sigma_0}{1 - i k c \\tau}

    This accounts for the anomalous skin effect at high frequencies where
    the mean free path of electrons becomes comparable to the skin depth.

    Parameters
    ----------
    k : float or np.ndarray
        Longitudinal wave number [1/m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c·τ [m], where τ is the Drude relaxation time

    Returns
    -------
    sigma : complex or np.ndarray of complex
        AC conductivity [S/m]

    Notes
    -----
    At k=0, returns σ₀ (DC conductivity).

    For copper at room temperature: σ₀ ≈ 5.96×10⁷ S/m, τ ≈ 2.7×10⁻¹⁴ s.

    References
    ----------
    .. [2] K. Bane and G. Stupakov, SLAC-PUB-10707 (2004), Eq. (4).
    """
    return sigma0 / (1 - 1j * k * ctau)


def surface_impedance(
    k: float | np.ndarray,
    sigma0: float,
    ctau: float,
) -> complex | np.ndarray:
    """
    Surface impedance for a conducting wall with AC conductivity.

    The surface impedance relates tangential electric and magnetic fields
    at the wall surface [1]_:

    .. math::

        \\zeta(k) = \\frac{E_\\parallel}{H_\\parallel}
                  = (1 - i) \\sqrt{\\frac{k c}{2 \\sigma(k) Z_0 c}}

    where σ(k) is the frequency-dependent AC conductivity.

    Parameters
    ----------
    k : float or np.ndarray
        Longitudinal wave number [1/m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c·τ [m]

    Returns
    -------
    zeta : complex or np.ndarray of complex
        Surface impedance [dimensionless]

    References
    ----------
    .. [1] N. Mounet and E. Métral, Phys. Rev. ST Accel. Beams 18, 034402
       (2015), Eq. (17) in the thick-wall limit.
    """
    sigma = ac_conductivity(k, sigma0, ctau)
    return (1 - 1j) * np.sqrt(k * c_light / (2 * sigma * Z0 * c_light))


def _impedance_integrand_flat(
    k: float,
    x: float | np.ndarray,
    a: float,
    sigma0: float,
    ctau: float,
) -> complex | np.ndarray:
    """
    Impedance integrand for flat (parallel plate) geometry.

    The longitudinal impedance for flat geometry is given by [2]_:

    .. math::

        Z_\\parallel(k) = \\frac{Z_0}{2\\pi a} \\int_0^\\infty
            \\frac{dx}{\\cosh(x)[\\cosh(x)/\\zeta - ik a \\cdot \\text{sinhc}(x)]}

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
        Relaxation distance c·τ [m]

    Returns
    -------
    result : complex or np.ndarray of complex
        Integrand value(s) [Ohm/m]

    References
    ----------
    .. [2] K. Bane and G. Stupakov, SLAC-PUB-10707 (2004), Eq. (52).
    """
    x = np.asarray(x)

    # Prefactor: Z0 / (2π a)
    prefactor = Z0 / (2 * np.pi * a)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        zeta = surface_impedance(k, sigma0, ctau)
        cosh_x = np.cosh(x)
        shc_x = sinhc(x)

        # Denominator: cosh(x) * [cosh(x)/ζ - i k a sinhc(x)]
        denom = cosh_x * (cosh_x / zeta - 1j * k * a * shc_x)

        # Assign result only where denominator is finite
        result = np.where(np.isfinite(denom), prefactor / denom, 0.0 + 0.0j)

    return result.astype(np.complex128)


# Note: Round geometry uses a closed-form expression, no integrand function needed.
# See longitudinal_impedance_round() for the implementation.


def longitudinal_impedance_flat(
    k: float | np.ndarray,
    a: float,
    sigma0: float,
    ctau: float,
) -> complex | np.ndarray:
    """
    Compute longitudinal impedance Z(k) for flat (parallel plate) geometry.

    Integrates the surface impedance kernel over all transverse modes [2]_:

    .. math::

        Z_\\parallel(k) = \\frac{Z_0}{2\\pi a} \\int_0^\\infty
            \\frac{dx}{\\cosh(x)[\\cosh(x)/\\zeta(k) - ik a \\cdot \\text{sinhc}(x)]}

    where ζ(k) is the surface impedance including AC conductivity effects.

    Parameters
    ----------
    k : float or np.ndarray
        Longitudinal wave number [1/m]
    a : float
        Half-gap height between parallel plates [m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c·τ [m]

    Returns
    -------
    Zk : complex or np.ndarray of complex
        Longitudinal impedance [Ohm/m]

    References
    ----------
    .. [2] K. Bane and G. Stupakov, SLAC-PUB-10707 (2004), Eq. (52).

    Examples
    --------
    >>> import numpy as np
    >>> ks = np.linspace(0, 1e5, 100)
    >>> Zk = longitudinal_impedance_flat(ks, a=4.5e-3, sigma0=2.4e7, ctau=2.4e-6)
    """

    @np.vectorize
    def _Zk_scalar(k_val):
        if k_val == 0:
            return 0.0 + 0.0j

        def integrand(x):
            return _impedance_integrand_flat(k_val, x, a, sigma0, ctau)

        return quad_vec(integrand, 0, np.inf)[0]

    return _Zk_scalar(k)


def longitudinal_impedance_round(
    k: float | np.ndarray,
    a: float,
    sigma0: float,
    ctau: float,
) -> complex | np.ndarray:
    """
    Compute longitudinal impedance Z(k) for round (circular pipe) geometry.

    Uses the closed-form expression for a single-layer resistive wall [2]_:

    .. math::

        Z_\\parallel(k) = \\frac{Z_0}{2\\pi a}
            \\left(\\frac{1}{\\zeta(k)} - \\frac{ika}{2}\\right)^{-1}

    where ζ(k) is the surface impedance including AC conductivity effects.

    Parameters
    ----------
    k : float or np.ndarray
        Longitudinal wave number [1/m]
    a : float
        Pipe radius [m]
    sigma0 : float
        DC conductivity [S/m]
    ctau : float
        Relaxation distance c·τ [m]

    Returns
    -------
    Zk : complex or np.ndarray of complex
        Longitudinal impedance [Ohm/m]

    References
    ----------
    .. [2] K. Bane and G. Stupakov, SLAC-PUB-10707 (2004), Eq. (2).
       See also Phys. Rev. ST Accel. Beams 18, 034402 (2015), Eq. (7).

    Examples
    --------
    >>> import numpy as np
    >>> ks = np.linspace(0, 1e5, 100)
    >>> Zk = longitudinal_impedance_round(ks, a=4.5e-3, sigma0=2.4e7, ctau=2.4e-6)
    """
    k = np.asarray(k)
    scalar_input = k.ndim == 0
    k = np.atleast_1d(k)

    # Prefactor: Z0 / (2π a)
    prefactor = Z0 / (2 * np.pi * a)

    # Surface impedance
    zeta = surface_impedance(k, sigma0, ctau)

    # Closed-form impedance: Z = (Z0 / 2πa) / (1/ζ - ika/2)
    # Handle k=0 case where impedance is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        Zk = prefactor / (1.0 / zeta - 1j * k * a / 2)
        Zk = np.where(k == 0, 0.0 + 0.0j, Zk)

    if scalar_input:
        return complex(Zk[0])
    return Zk


def wakefield_from_impedance(
    z: float | np.ndarray,
    Zk_func: callable,
    k_max: float = 1e7,
    epsabs: float = 1e-9,
    epsrel: float = 1e-6,
) -> float | np.ndarray:
    """
    Compute wakefield W(z) from Re[Z(k)] using a cosine transform.

    The longitudinal wakefield is related to the impedance by [3]_:

    .. math::

        W(z) = \\frac{2c}{\\pi} \\int_0^{k_{\\max}} \\text{Re}[Z(k)] \\cos(kz) \\, dk

    This uses the fact that the impedance satisfies Z(-k) = Z*(k) for
    a causal system.

    Parameters
    ----------
    z : float or np.ndarray
        Longitudinal position [m]. Negative z is behind the source
        (trailing particle), matching the convention in ResistiveWallWakefield.
    Zk_func : callable
        Function returning complex impedance Z(k) [Ohm/m] for wave number
        k [1/m]
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
    The wakefield is zero for z > 0 (ahead of the source particle)
    due to causality. This matches the sign convention used by
    ResistiveWallWakefield.

    References
    ----------
    .. [3] A. Chao, "Physics of Collective Beam Instabilities in High
       Energy Accelerators," Wiley, 1993, Chapter 2, Eq. (2.24).

    Examples
    --------
    >>> import numpy as np
    >>> from functools import partial
    >>> Zk = partial(longitudinal_impedance_flat, a=4.5e-3, sigma0=2.4e7, ctau=2.4e-6)
    >>> zs = np.linspace(0, 100e-6, 50)
    >>> Wz = [wakefield_from_impedance(z, Zk) for z in zs]
    """

    def _wakefield_scalar(z_val):
        if z_val > 0:
            return 0.0

        def integrand(k):
            return np.real(Zk_func(k)) * np.cos(k * z_val)

        result, _ = quad(integrand, 0, k_max, epsabs=epsabs, epsrel=epsrel, limit=200)
        return (2 * c_light / np.pi) * result

    z = np.asarray(z)
    if z.ndim == 0:
        return _wakefield_scalar(float(z))

    return np.array([_wakefield_scalar(zi) for zi in z])


def characteristic_length(a: float, sigma0: float) -> float:
    """
    Characteristic length scale s₀ for resistive wall wakefield.

    From SLAC-PUB-10707 Eq. (5) [2]_:

    .. math::

        s_0 = \\left( \\frac{2 a^2}{Z_0 \\sigma_0} \\right)^{1/3}

    This length scale determines the transition between the short-range
    and long-range wakefield behavior. For z << s₀, the wakefield rises
    as z^(-1/2); for z >> s₀, it oscillates and decays.

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

    References
    ----------
    .. [2] K. Bane and G. Stupakov, SLAC-PUB-10707 (2004), Eq. (5).
    """
    return (2 * a**2 / (Z0 * sigma0)) ** (1 / 3)


class ResistiveWallImpedance:
    """
    Resistive wall impedance model for flat or round geometry.

    This class encapsulates the physical parameters and provides methods
    to compute impedance Z(k) and wakefield W(z) for a beam in a conducting
    beam pipe.

    Supported geometries:

    - **flat**: Parallel plate geometry. The impedance is computed by
      integrating over transverse modes (SLAC-PUB-10707 Eq. 52).
    - **round**: Circular pipe geometry. The impedance uses a closed-form
      expression (SLAC-PUB-10707 Eq. 2).

    .. note::

        This class uses analytical formulas for Z(k) based on AC resistivity.
        For short bunches, the ``ResistiveWallWakefield`` class (which uses
        pseudomode fits to Bane-Stupakov numerical calculations) may give
        more accurate results, differing by ~10-20% depending on bunch length.

    Parameters
    ----------
    radius : float
        Pipe radius (round) or half-gap height (flat) [m]. Must be positive.
    conductivity : float
        DC electrical conductivity [S/m]. Must be positive.
    relaxation_time : float
        Drude-model relaxation time [s]. Must be non-negative.
    geometry : str, optional
        Geometry type: 'round' or 'flat'. Default is 'round'.

    Attributes
    ----------
    radius : float
        Pipe radius or half-gap height [m]
    conductivity : float
        DC conductivity [S/m]
    relaxation_time : float
        Relaxation time [s]
    geometry : str
        Geometry type ('round' or 'flat')
    ctau : float
        Relaxation distance c·τ [m]
    s0 : float
        Characteristic length scale [m]

    References
    ----------
    .. [2] K. Bane and G. Stupakov, SLAC-PUB-10707 (2004).

    Examples
    --------
    >>> imp = ResistiveWallImpedance(
    ...     radius=4.5e-3,
    ...     conductivity=2.4e7,
    ...     relaxation_time=8e-15,
    ...     geometry='round'
    ... )
    >>> ks = np.linspace(0, 1e5, 100)
    >>> Zk = imp.impedance(ks)
    >>> zs = np.linspace(0, 100e-6, 20)
    >>> Wz = imp.wakefield(zs)
    """

    def __init__(
        self,
        radius: float,
        conductivity: float,
        relaxation_time: float,
        geometry: str = "round",
    ) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if conductivity <= 0:
            raise ValueError("conductivity must be positive")
        if relaxation_time < 0:
            raise ValueError("relaxation_time must be non-negative")
        if geometry not in ("round", "flat"):
            raise ValueError(f"geometry must be 'round' or 'flat', got {geometry!r}")

        self.radius = radius
        self.conductivity = conductivity
        self.relaxation_time = relaxation_time
        self.geometry = geometry
        self.ctau = c_light * relaxation_time
        self.s0 = characteristic_length(radius, conductivity)

    def impedance(self, k: float | np.ndarray) -> complex | np.ndarray:
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
        if self.geometry == "round":
            return longitudinal_impedance_round(
                k, self.radius, self.conductivity, self.ctau
            )
        else:  # flat
            return longitudinal_impedance_flat(
                k, self.radius, self.conductivity, self.ctau
            )

    def wakefield(
        self,
        z: float | np.ndarray,
        k_max: float = 1e7,
        epsabs: float = 1e-9,
        epsrel: float = 1e-6,
    ) -> float | np.ndarray:
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

    def plot_impedance(
        self,
        k_max: float = 1e6,
        n_points: int = 200,
        ax=None,
    ):
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
        ax.set_title(
            f"{self.geometry.capitalize()} geometry: a={self.radius*1e3:.2f} mm, "
            f"σ₀={self.conductivity:.2e} S/m"
        )

        return ax

    def plot_wakefield(
        self,
        z_max: float = 200e-6,
        n_points: int = 50,
        k_max: float = 1e6,
        ax=None,
    ):
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

        # Use negative z values (wake is behind the source)
        zs = np.linspace(-z_max, 0, n_points)
        Wz = self.wakefield(zs, k_max=k_max)

        if ax is None:
            fig, ax = plt.subplots()

        # Plot -z so that "behind" appears to the right (positive axis)
        ax.plot(-zs * 1e6, Wz * 1e-12)
        ax.set_xlabel(r"$-z$ (µm)")
        ax.set_ylabel(r"$W(z)$ (V/pC/m)")
        ax.set_title(
            f"{self.geometry.capitalize()} geometry: a={self.radius*1e3:.2f} mm"
        )

        return ax

    def convolve_density(
        self,
        density: np.ndarray,
        dz: float,
        offset: float = 0,
        include_self_kick: bool = True,
    ) -> np.ndarray:
        """
        Compute integrated wakefield by convolving density with impedance using FFT.

        This method efficiently computes the wake potential by working in
        frequency domain using the real-valued FFT (rfft) for efficiency:

        1. Compute rfft of density to get λ̃(k) for k ≥ 0
        2. Multiply by impedance Z(k) and c
        3. Inverse rfft to get the integrated wake in position space

        The longitudinal impedance Z(k) is related to the wakefield W(z) by:

        .. math::

            Z(k) = \\frac{1}{c} \\int_0^{\\infty} W(z) e^{-ikz} dz

        So the integrated wake from a charge distribution λ(z) is:

        .. math::

            V(z) = \\int \\lambda(z') W(z - z') dz'
                 = c \\cdot \\mathcal{F}^{-1}[ \\tilde{\\lambda}(k) \\cdot Z(k) ]

        .. note::

            This method uses the analytical impedance formula Z(k) directly.
            The impedance model gives results that differ by ~10-20% from the
            pseudomode-based ResistiveWallWakefield.convolve_density, which uses
            fitted parameters from Bane-Stupakov's numerical calculations.
            For most practical applications with short bunches, the pseudomode
            approximation may be more accurate. The analytical impedance is
            useful when access to Z(k) directly is needed.

        Parameters
        ----------
        density : np.ndarray
            Charge density array [C/m]. Positive values represent positive charge.
        dz : float
            Grid spacing [m].
        offset : float, optional
            Offset for the z coordinate [m]. Default is 0.
            For example, an offset of -1.23 computes the wake at z=-1.23 m
            relative to the density grid.
        include_self_kick : bool, optional
            Whether to include the additional ½ W(0⁻) self-kick contribution.
            Default is True. The FFT convolution naturally includes only half
            of the self-kick at each grid point; setting this to True adds the
            other half. For a delta-function source, this makes the impedance
            method match the pseudomode at the source position. For extended
            bunches, the effect is proportional to the local density.

        Returns
        -------
        integrated_wake : np.ndarray
            Integrated longitudinal wakefield [V/m]. Same length as input density.

        Notes
        -----
        The FFT-based method is O(N log N) compared to O(N²) for direct convolution.

        The impedance Z(k) has units of [Ohm/m] = [V/A/m]. The density λ has
        units of [C/m]. The product c·Z·λ integrated over k gives [V/m].

        Examples
        --------
        >>> imp = ResistiveWallImpedance(
        ...     radius=2.5e-3,
        ...     conductivity=5.96e7,
        ...     relaxation_time=27e-15,
        ...     geometry='round'
        ... )
        >>> n = 1000
        >>> dz = 1e-6  # 1 micron spacing
        >>> z = np.arange(n) * dz
        >>> # Gaussian density profile
        >>> sigma_z = 50e-6
        >>> density = 1e-9 / (sigma_z * np.sqrt(2*np.pi)) * np.exp(-0.5*(z - 0.5*n*dz)**2/sigma_z**2)
        >>> wake = imp.convolve_density(density, dz)
        """
        n = len(density)

        # Zero-pad to 2n for linear (non-circular) convolution
        n_padded = 2 * n
        density_padded = np.zeros(n_padded)
        density_padded[:n] = density

        # Compute wavenumber array for full FFT frequencies
        # fftfreq returns [0, 1, ..., n/2-1, -n/2, ..., -1] / (n*dz) in cycles/m
        # Multiply by 2*pi to get angular wavenumber k [1/m]
        k = 2 * np.pi * np.fft.fftfreq(n_padded, d=dz)

        # Evaluate impedance at |k| (impedance is defined for k >= 0)
        k_abs = np.abs(k)
        Zk = self.impedance(k_abs)

        # Apply Hermitian symmetry: Z(-k) = Z*(k) for real-valued output
        Zk = np.where(k < 0, np.conj(Zk), Zk)

        # Apply phase shift for offset if needed
        if offset != 0:
            phase_shift = np.exp(-1j * k * offset)
            Zk = Zk * phase_shift

        # Forward FFT with continuous normalization
        density_fft = np.fft.fft(density_padded) * dz

        # Multiply in frequency domain (convolution theorem)
        wake_fft = density_fft * Zk * c_light

        # Inverse FFT with continuous normalization
        wake_full = np.fft.ifft(wake_fft) / dz

        # Extract the first n points (the causal part of the convolution)
        integrated_wake = np.real(wake_full[:n])

        # Add the extra ½ self-kick if requested
        # The FFT convolution naturally gives ½ of the self-kick contribution.
        # To match the convention of ResistiveWallWakefield.convolve_density,
        # we add another ½ W(0⁻) * λ(z) * dz term.
        # Note: wakefield uses z < 0 convention (behind source), so use -dz/2
        if include_self_kick:
            # Evaluate wakefield at small negative z to approximate W(0⁻)
            W0 = self.wakefield(-dz / 2)
            # The self-kick adds ½ W(0⁻) * λ(z) * dz to each point
            # Units: [V/C/m] * [C/m] * [m] = [V/m] ✓
            integrated_wake = integrated_wake + 0.5 * W0 * density * dz

        return integrated_wake

    def __repr__(self) -> str:
        return (
            f"ResistiveWallImpedance("
            f"radius={self.radius}, "
            f"conductivity={self.conductivity}, "
            f"relaxation_time={self.relaxation_time}, "
            f"geometry={self.geometry!r}, "
            f"s0={self.s0:.3e})"
        )


def FlatResistiveWallImpedance(
    half_gap: float,
    conductivity: float,
    relaxation_time: float,
) -> ResistiveWallImpedance:
    """
    Create a resistive wall impedance model for flat (parallel plate) geometry.

    This is a convenience function that creates a `ResistiveWallImpedance`
    with ``geometry='flat'``.

    Parameters
    ----------
    half_gap : float
        Half-gap height between plates [m]. Must be positive.
    conductivity : float
        DC electrical conductivity [S/m]. Must be positive.
    relaxation_time : float
        Drude-model relaxation time [s]. Must be non-negative.

    Returns
    -------
    ResistiveWallImpedance
        Impedance model with flat geometry.

    Examples
    --------
    >>> imp = FlatResistiveWallImpedance(
    ...     half_gap=4.5e-3,
    ...     conductivity=2.4e7,
    ...     relaxation_time=8e-15
    ... )
    """
    return ResistiveWallImpedance(
        radius=half_gap,
        conductivity=conductivity,
        relaxation_time=relaxation_time,
        geometry="flat",
    )


def RoundResistiveWallImpedance(
    radius: float,
    conductivity: float,
    relaxation_time: float,
) -> ResistiveWallImpedance:
    """
    Create a resistive wall impedance model for round (circular pipe) geometry.

    This is a convenience function that creates a `ResistiveWallImpedance`
    with ``geometry='round'``.

    Parameters
    ----------
    radius : float
        Pipe radius [m]. Must be positive.
    conductivity : float
        DC electrical conductivity [S/m]. Must be positive.
    relaxation_time : float
        Drude-model relaxation time [s]. Must be non-negative.

    Returns
    -------
    ResistiveWallImpedance
        Impedance model with round geometry.

    Examples
    --------
    >>> imp = RoundResistiveWallImpedance(
    ...     radius=4.5e-3,
    ...     conductivity=2.4e7,
    ...     relaxation_time=8e-15
    ... )
    """
    return ResistiveWallImpedance(
        radius=radius,
        conductivity=conductivity,
        relaxation_time=relaxation_time,
        geometry="round",
    )
