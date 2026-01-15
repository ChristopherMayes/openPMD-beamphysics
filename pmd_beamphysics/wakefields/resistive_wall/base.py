"""
Base classes and low-level functions for resistive wall wakefields.

This module provides the ResistiveWallWakefieldBase class and low-level
functions for computing resistive wall impedance and wakefield.

Classes
-------
ResistiveWallWakefieldBase
    Abstract base class with shared properties for resistive wall models

Functions
---------
sinhc
    Numerically stable sinh(x)/x
ac_conductivity
    Drude-model AC conductivity
surface_impedance
    Surface impedance for conducting wall
longitudinal_impedance_round
    Longitudinal impedance Z(k) for round pipe
longitudinal_impedance_flat
    Longitudinal impedance Z(k) for flat geometry
wakefield_from_impedance
    Wakefield W(z) via cosine transform (quadrature)
wakefield_from_impedance_fft
    Wakefield W(z) via FFT (fast)
characteristic_length
    Characteristic length s₀
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.integrate import quad, quad_vec

from ...units import c_light, Z0
from ..base import WakefieldBase


class Geometry(str, Enum):
    """Beam pipe geometry for resistive wall wakefield calculations."""

    ROUND = "round"
    FLAT = "flat"


__all__ = [
    # Enum
    "Geometry",
    # Class
    "ResistiveWallWakefieldBase",
    # Low-level functions
    "sinhc",
    "ac_conductivity",
    "surface_impedance",
    "longitudinal_impedance_round",
    "longitudinal_impedance_flat",
    "wakefield_from_impedance",
    "wakefield_from_impedance_fft",
    "characteristic_length",
    "s0f",
    "Gammaf",
    "krs0_round",
    "krs0_flat",
    "Qr_round",
    "Qr_flat",
]


# =============================================================================
# Low-level impedance/wakefield functions
# =============================================================================


def sinhc(x: float | np.ndarray) -> float | np.ndarray:
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
    Frequency-dependent AC conductivity with relaxation time (Drude model).

    $$\\sigma(k) = \\frac{\\sigma_0}{1 - i k c \\tau}$$

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
    sigma : complex or np.ndarray of complex
        AC conductivity [S/m]
    """
    return sigma0 / (1 - 1j * k * ctau)


def surface_impedance(
    k: float | np.ndarray,
    sigma0: float,
    ctau: float,
) -> complex | np.ndarray:
    """
    Surface impedance for a conducting wall with AC conductivity.

    $$\\zeta(k) = (1 - i) \\sqrt{\\frac{k c}{2 \\sigma(k) Z_0 c}}$$

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
    """Impedance integrand for flat (parallel plate) geometry."""
    x = np.asarray(x)
    prefactor = Z0 / (2 * np.pi * a)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        zeta = surface_impedance(k, sigma0, ctau)
        cosh_x = np.cosh(x)
        shc_x = sinhc(x)
        denom = cosh_x * (cosh_x / zeta - 1j * k * a * shc_x)
        result = np.where(np.isfinite(denom), prefactor / denom, 0.0 + 0.0j)

    return result.astype(np.complex128)


def longitudinal_impedance_flat(
    k: float | np.ndarray,
    a: float,
    sigma0: float,
    ctau: float,
) -> complex | np.ndarray:
    """
    Compute longitudinal impedance Z(k) for flat (parallel plate) geometry.

    Uses numerical integration over transverse modes (SLAC-PUB-10707 Eq. 52).

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

    Uses closed-form expression (SLAC-PUB-10707 Eq. 2).

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
    """
    k = np.asarray(k)
    scalar_input = k.ndim == 0
    k = np.atleast_1d(k)

    prefactor = Z0 / (2 * np.pi * a)
    zeta = surface_impedance(k, sigma0, ctau)

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
    Compute wakefield W(z) from Re[Z(k)] using a cosine transform (quadrature).

    For z ≤ 0 (trailing particles):

    $$W(z) = \\frac{2c}{\\pi} \\int_0^{k_{\\max}} \\text{Re}[Z(k)] \\cos(kz) \\, dk$$

    Returns 0 for z > 0 (causality).

    Parameters
    ----------
    z : float or np.ndarray
        Longitudinal position [m]. Negative z is behind the source.
    Zk_func : callable
        Function returning complex impedance Z(k) [Ohm/m]
    k_max : float, optional
        Upper limit of k integration [1/m]. Default is 1e7.
    epsabs : float, optional
        Absolute tolerance. Default is 1e-9.
    epsrel : float, optional
        Relative tolerance. Default is 1e-6.

    Returns
    -------
    Wz : float or np.ndarray
        Wakefield [V/C/m]
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


def wakefield_from_impedance_fft(
    z: np.ndarray,
    Zk_func: callable,
    k_max: float = 1e7,
    n_fft: int = 8192,
) -> np.ndarray:
    """
    Compute wakefield W(z) from Z(k) using FFT and interpolation.

    Much faster than `wakefield_from_impedance` for arrays.

    Parameters
    ----------
    z : np.ndarray
        Longitudinal positions [m]. Negative z is behind the source.
    Zk_func : callable
        Function returning complex impedance Z(k) [Ohm/m]
    k_max : float, optional
        Upper limit of k integration [1/m]. Default is 1e7.
    n_fft : int, optional
        Number of FFT points. Default is 8192.

    Returns
    -------
    Wz : np.ndarray
        Wakefield [V/C/m]
    """
    from scipy.fft import irfft
    from scipy.interpolate import interp1d

    z = np.asarray(z)

    dk = k_max / (n_fft - 1)
    k_grid = np.linspace(0, k_max, n_fft)

    Zk_grid = Zk_func(k_grid)
    ReZ = np.real(Zk_grid)

    n_full = 2 * (n_fft - 1)
    dz = 2 * np.pi / (n_full * dk)
    z_grid = np.arange(n_full) * dz

    W_grid = irfft(ReZ, n=n_full) * n_full * dk * (c_light / np.pi)

    interp = interp1d(z_grid, W_grid, kind="cubic", bounds_error=False, fill_value=0.0)

    result = interp(-z)
    result = np.where(z > 0, 0.0, result)

    return result


def characteristic_length(a: float, sigma0: float) -> float:
    """
    Characteristic length scale s₀ for resistive wall wakefield.

    From SLAC-PUB-10707 Eq. (5):

    $$s_0 = \\left( \\frac{2 a^2}{Z_0 \\sigma_0} \\right)^{1/3}$$

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


# Alias for backwards compatibility
s0f = characteristic_length


# =============================================================================
# Polynomial fits for pseudomode parameters
# =============================================================================

# AC wake Fitting Formula Polynomial coefficients
# found by fitting digitized plots in SLAC-PUB-10707 Fig. 14
_krs0_round_poly = np.poly1d(
    [
        -0.03432181,
        0.30787769,
        -1.1017812,
        1.98421548,
        -1.79215353,
        0.42197608,
        1.81248274,
    ]
)

_krs0_flat_poly = np.poly1d(
    [
        -0.00820164,
        0.08196954,
        -0.33410072,
        0.70455758,
        -0.76928545,
        0.21623914,
        1.2976896,
    ]
)

_Qr_round_poly = np.poly1d(
    [
        0.04435072,
        -0.39683778,
        1.41363674,
        -2.52751228,
        2.19131144,
        1.64830007,
        1.14151479,
    ]
)

_Qr_flat_poly = np.poly1d(
    [
        0.02054322,
        -0.1863843,
        0.67249006,
        -1.19170403,
        0.86545076,
        0.96306531,
        1.04673215,
    ]
)


def krs0_round(G: float) -> float:
    """
    k_r*s_0 from SLAC-PUB-10707 Fig. 14 for round geometry.

    This is from a polynomial fit of the digitized data.

    Parameters
    ----------
    G : float
        Dimensionless relaxation time Γ

    Returns
    -------
    float
        Dimensionless product k_r*s_0
    """
    return _krs0_round_poly(G)


def krs0_flat(G: float) -> float:
    """
    k_r*s_0 from SLAC-PUB-10707 Fig. 14 for flat geometry.

    This is from a polynomial fit of the digitized data.

    Parameters
    ----------
    G : float
        Dimensionless relaxation time Γ

    Returns
    -------
    float
        Dimensionless product k_r*s_0
    """
    return _krs0_flat_poly(G)


def Qr_round(G: float) -> float:
    """
    Q_r from SLAC-PUB-10707 Fig. 14 for round geometry.

    This is from a polynomial fit of the digitized data.

    Parameters
    ----------
    G : float
        Dimensionless relaxation time Γ

    Returns
    -------
    float
        Dimensionless quality factor Q_r
    """
    return _Qr_round_poly(G)


def Qr_flat(G: float) -> float:
    """
    Q_r from SLAC-PUB-10707 Fig. 14 for flat geometry.

    This is from a polynomial fit of the digitized data.

    Parameters
    ----------
    G : float
        Dimensionless relaxation time Γ

    Returns
    -------
    float
        Dimensionless quality factor Q_r
    """
    return _Qr_flat_poly(G)


def Gammaf(relaxation_time: float, radius: float, conductivity: float) -> float:
    """
    Dimensionless relaxation time Γ = cτ/s₀.

    Parameters
    ----------
    relaxation_time : float
        Drude relaxation time τ [s]
    radius : float
        Pipe radius (round) or half-gap (flat) [m]
    conductivity : float
        DC conductivity [S/m]

    Returns
    -------
    float
        Dimensionless relaxation time Γ
    """
    return c_light * relaxation_time / s0f(radius, conductivity)


# =============================================================================
# Base class for resistive wall wakefields
# =============================================================================


@dataclass
class ResistiveWallWakefieldBase(WakefieldBase):
    """
    Base class for resistive wall wakefield models based on SLAC-PUB-10707.

    This abstract base class provides shared properties and methods for
    resistive wall wakefield calculations. Use the concrete subclasses:

    - :class:`ResistiveWallWakefield`: Accurate impedance-based model
    - :class:`ResistiveWallPseudomode`: Fast pseudomode-based model

    Parameters
    ----------
    radius : float
        Radius of the beam pipe [m]. For flat geometry, this is half the gap.
    conductivity : float
        Electrical conductivity of the wall material [S/m].
    relaxation_time : float
        Drude-model relaxation time of the conductor [s].
    geometry : str, optional
        Geometry of the beam pipe: 'round' or 'flat'. Default is 'round'.

    Attributes
    ----------
    s0 : float
        Characteristic length scale [m]

    References
    ----------
    Bane & Stupakov, SLAC-PUB-10707 (2004)
    https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf
    """

    radius: float
    conductivity: float
    relaxation_time: float
    geometry: Geometry = Geometry.ROUND

    # Internal material database (SI units)
    MATERIALS = {
        "copper-slac-pub-10707": {"conductivity": 6.5e7, "relaxation_time": 27e-15},
        "copper-genesis4": {"conductivity": 5.813e7, "relaxation_time": 27e-15},
        "aluminum-genesis4": {"conductivity": 3.571e7, "relaxation_time": 8e-15},
        "aluminum-slac-pub-10707": {"conductivity": 4.2e7, "relaxation_time": 8e-15},
        "aluminum-alloy-6061-t6-20C": {"conductivity": 2.5e7, "relaxation_time": 8e-15},
        "aluminum-alloy-6063-t6-20C": {"conductivity": 3.0e7, "relaxation_time": 8e-15},
    }

    def __post_init__(self):
        if not isinstance(self.radius, (int, float)) or self.radius <= 0:
            raise ValueError(f"radius must be a positive number, got {self.radius}")

        if self.conductivity <= 0:
            raise ValueError(f"conductivity must be positive, got {self.conductivity}")

        if self.relaxation_time < 0:
            raise ValueError(
                f"relaxation_time must be non-negative, got {self.relaxation_time}"
            )

        # Convert string to enum if needed (for backwards compatibility)
        if isinstance(self.geometry, str):
            try:
                object.__setattr__(self, "geometry", Geometry(self.geometry))
            except ValueError:
                raise ValueError(
                    f"Unsupported geometry: {self.geometry!r}. "
                    f"Must be Geometry.ROUND or Geometry.FLAT"
                )

    @classmethod
    def from_material(
        cls,
        material: str,
        radius: float,
        geometry: Geometry | str = Geometry.ROUND,
    ):
        """
        Create a wakefield from a known material preset.

        Parameters
        ----------
        material : str
            Material name. Must be in list(cls.MATERIALS)
        radius : float
            Pipe radius [m]
        geometry : str
            Geometry type: 'round' or 'flat'
        """
        if material not in cls.MATERIALS:
            raise ValueError(
                f"Unknown material {material!r}. Available: {list(cls.MATERIALS)}"
            )

        props = cls.MATERIALS[material]
        return cls(
            radius=radius,
            conductivity=props["conductivity"],
            relaxation_time=props["relaxation_time"],
            geometry=geometry,
        )

    def material_from_properties(self, tol=0.01):
        """
        Attempt to identify the material based on conductivity and relaxation time.

        Parameters
        ----------
        tol : float, optional
            Relative tolerance for matching, default is 1% (0.01)

        Returns
        -------
        material : str or None
            Name of the matched material, or None if no match is found.
        """
        from math import isclose

        for name, props in self.MATERIALS.items():
            cond_match = isclose(self.conductivity, props["conductivity"], rel_tol=tol)
            tau_match = isclose(
                self.relaxation_time, props["relaxation_time"], rel_tol=tol
            )
            if cond_match and tau_match:
                return name
        return None

    @property
    def s0(self):
        """
        Characteristic length scale s₀ of the resistive wall wakefield [m].

        From SLAC-PUB-10707 Eq. (5):

        $$s_0 = \\left( \\frac{2 a^2}{Z_0 \\sigma_0} \\right)^{1/3}$$

        where a is the pipe radius (round) or half-gap (flat), Z₀ is the
        impedance of free space, and σ₀ is the DC conductivity.
        """
        return s0f(self.radius, self.conductivity)

    @property
    def W0(self):
        """
        Characteristic wake amplitude W₀ at z=0 [V/C/m].

        From SLAC-PUB-10707:
        - Round: W₀ = c·Z₀ / (π·a²)
        - Flat:  W₀ = c·Z₀·π / (16·a²) = W₀_round · (π²/16)

        The dimensionless scaled wake is Ŵ(z/s₀) = W(z) / W₀.
        """
        W0_round = c_light * Z0 / (np.pi * self.radius**2)
        if self.geometry == Geometry.ROUND:
            return W0_round
        else:  # FLAT
            return W0_round * np.pi**2 / 16

    def __call__(self, z: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the wakefield at position z (convenience method)."""
        return self.wake(z)
