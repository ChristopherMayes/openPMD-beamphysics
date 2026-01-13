"""
Resistive wall wakefield implementation.

This module provides analytical models for short-range resistive wall wakefields
in accelerator beam pipes, based on the approach described in SLAC-PUB-10707.
It supports both round and flat geometries and includes effects of AC conductivity
through material relaxation times.

Two models are available:

- **ResistiveWallWakefield**: Accurate impedance-based model using FFT
  convolution. Recommended for most applications.

- **ResistiveWallPseudomode**: Fast pseudomode-based model using polynomial fits.
  Good for quick calculations, ~10-20% difference from full impedance.

Low-level functions are also exported for direct impedance/wakefield evaluation.

Classes
-------
ResistiveWallWakefieldBase
    Abstract base class with shared properties
ResistiveWallWakefield
    Accurate impedance-based wakefield model
ResistiveWallPseudomode
    Fast pseudomode-based wakefield model

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

References
----------
.. [1] K. Bane and G. Stupakov, "Resistive wall wakefield in the LCLS
   undulator beam pipe," SLAC-PUB-10707 (2004).
   https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf

.. [2] D. Sagan, "Bmad Manual," Section 24.6: Short-Range Wakefields.
   https://www.classe.cornell.edu/bmad/manual.html

.. [3] A. Chao, "Physics of Collective Beam Instabilities in High Energy
   Accelerators," Wiley, 1993, Chapter 2.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.integrate import quad, quad_vec

from ..units import c_light, epsilon_0, Z0
from .base import WakefieldBase, Pseudomode, PseudomodeWakefield, ImpedanceWakefield


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
    from scipy.interpolate import interp1d
    from scipy.fft import irfft

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
    k_r*s_0 from SLAC-PUB-10707 Fig. 14 for round geometry
    This is from a polynomial fit of the digitized data

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


def pseudomode(A: float, d: float, k: float, phi: float) -> PseudomodeWakefield:
    """
    Create a single-mode pseudomode wakefield.

    .. deprecated::
        Use :class:`PseudomodeWakefield` directly instead. This function
        is maintained for backwards compatibility.

    Models the longitudinal wakefield as a damped sinusoid:
        W(z) = A * exp(d * z) * sin(k * z + φ)

    Parameters
    ----------
    A : float
        Amplitude coefficient [V/C/m].
    d : float
        Exponential decay rate [1/m].
    k : float
        Oscillation wavenumber [1/m].
    phi : float
        Phase offset [rad].

    Returns
    -------
    PseudomodeWakefield
        A single-mode pseudomode wakefield object.
    """
    warnings.warn(
        "pseudomode() is deprecated, use PseudomodeWakefield directly",
        DeprecationWarning,
        stacklevel=2,
    )
    return PseudomodeWakefield(A=A, d=d, k=k, phi=phi)


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
    geometry: str = "round"

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

        if self.geometry not in ("round", "flat"):
            raise ValueError(
                f"Unsupported geometry: {self.geometry}. Must be 'round' or 'flat'"
            )

    @classmethod
    def from_material(
        cls,
        material: str,
        radius: float,
        geometry: str = "round",
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
        if self.geometry == "round":
            return W0_round
        else:  # flat
            return W0_round * np.pi**2 / 16

    def _extract_z_weight(self, particle_group_or_z, weight=None):
        """Extract z and weight arrays from input."""
        if hasattr(particle_group_or_z, "in_t_coordinates"):
            particle_group = particle_group_or_z
            if particle_group.in_t_coordinates:
                z = np.asarray(particle_group.z)
            else:
                z = -c_light * np.asarray(particle_group.t)
            weight = np.asarray(particle_group.weight)
        else:
            z = np.asarray(particle_group_or_z)
            if weight is None:
                raise ValueError("weight must be provided when z is an array")
            weight = np.asarray(weight)
        return z, weight


@dataclass
class ResistiveWallPseudomode(ResistiveWallWakefieldBase):
    """
    Fast pseudomode-based resistive wall wakefield model.

    This class models the short-range resistive wall wakefield using a damped
    sinusoidal "pseudomode" approximation. When a relativistic charged particle
    travels through a conducting beam pipe, it induces image currents in the
    pipe walls. Due to the finite conductivity of the wall material, these
    currents penetrate into the conductor and dissipate energy, creating a
    wakefield that acts on trailing particles.

    Physics Background
    ------------------
    The resistive wall impedance arises from the skin effect in the conducting
    walls. At high frequencies (short distances), the AC conductivity of metals
    deviates from DC behavior due to the Drude relaxation time τ:

    $$\\sigma(\\omega) = \\frac{\\sigma_0}{1 - i\\omega\\tau}$$

    This frequency dependence causes the wakefield to oscillate and decay
    exponentially behind the source particle. The wakefield can be well
    approximated by a single damped sinusoid (pseudomode):

    $$W(z) = A \\cdot e^{d \\cdot z} \\cdot \\sin(k_r z + \\phi) \\quad (z \\le 0)$$

    where the parameters $k_r$ (oscillation wavenumber) and $Q_r$ (quality factor,
    related to decay rate $d = k_r / 2Q_r$) depend on the dimensionless relaxation
    parameter $\\Gamma = c\\tau / s_0$.

    Characteristic Scales
    ---------------------
    - **s₀**: Characteristic length scale where the wake transitions from
      the $1/\\sqrt{z}$ DC behavior to oscillatory AC behavior:

      $$s_0 = \\left( \\frac{2a^2}{Z_0 \\sigma_0} \\right)^{1/3}$$

    - **Γ**: Dimensionless relaxation parameter $\\Gamma = c\\tau / s_0$.
      For copper, Γ ≈ 0.8; for aluminum, Γ ≈ 0.2.

    - **W₀**: Characteristic wake amplitude at z = 0:
      - Round: $W_0 = c Z_0 / (\\pi a^2)$
      - Flat: $W_0 = c Z_0 \\pi / (16 a^2)$

    Algorithm
    ---------
    The pseudomode form enables an O(N) algorithm for computing particle kicks,
    compared to O(N²) or O(N log N) for general wakefields. This makes it
    suitable for multi-pass tracking simulations.

    Parameters
    ----------
    radius : float
        Radius of the beam pipe [m]. For flat geometry, this is half the gap.
    conductivity : float
        Electrical DC conductivity of the wall material [S/m].
    relaxation_time : float
        Drude-model relaxation time τ of the conductor [s].
        Typical values: Cu ≈ 27 fs, Al ≈ 8 fs.
    geometry : str, optional
        Geometry of the beam pipe: 'round' or 'flat'. Default is 'round'.

    Attributes
    ----------
    s0 : float
        Characteristic length scale [m]
    Gamma : float
        Dimensionless relaxation parameter Γ = cτ/s₀
    kr : float
        Resonant wavenumber [1/m]
    Qr : float
        Quality factor of the pseudomode

    Notes
    -----
    - The pseudomode approximation has ~10-20% error compared to the full
      impedance model, primarily in the first oscillation peak.
    - The polynomial fits for k_r and Q_r are valid for Γ ≲ 3. A warning is
      issued for larger values.
    - Materials with known conductivity and τ values are available via
      `from_material()`.

    See Also
    --------
    ResistiveWallWakefield : Accurate impedance-based model (slower)

    References
    ----------
    .. [1] K. Bane and G. Stupakov, "Resistive wall wakefield in the LCLS
       undulator beam pipe," SLAC-PUB-10707 (2004).
       https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf

    .. [2] D. Sagan, "Bmad Manual," Section 24.6: Short-Range Wakefields.
       https://www.classe.cornell.edu/bmad/manual.html

    .. [3] A. Chao, "Physics of Collective Beam Instabilities in High Energy
       Accelerators," Wiley, 1993, Chapter 2.

    Examples
    --------
    ::

        wake = ResistiveWallPseudomode.from_material(
            "copper-slac-pub-10707", radius=2.5e-3, geometry="round"
        )
        wake.wake(-10e-6)  # Wake at 10 µm behind source
    """

    def __post_init__(self):
        super().__post_init__()

        # Check if Gamma is in the valid range for the polynomial fits
        if self.Gamma > 3:
            warnings.warn(
                f"Γ = {self.Gamma:.3g} is above the validated range (Γ ≲ 3) for the "
                f"pseudomode polynomial fits. Results may be inaccurate. "
                f"Consider using ResistiveWallWakefield instead.",
                UserWarning,
                stacklevel=2,
            )

        # Create the internal pseudomode model
        self._internal_model = self._create_pseudomode()

    @property
    def Gamma(self):
        """Dimensionless relaxation time Γ = c * τ / s₀."""
        return Gammaf(self.relaxation_time, self.radius, self.conductivity)

    @property
    def Qr(self):
        """Dimensionless quality factor Q_r of the wakefield pseudomode."""
        if self.geometry == "round":
            return Qr_round(self.Gamma)
        if self.geometry == "flat":
            return Qr_flat(self.Gamma)
        else:
            raise NotImplementedError(f"{self.geometry=}")

    @property
    def kr(self):
        """Real-valued wave number k_r of the wakefield pseudomode [1/m]."""
        if self.geometry == "round":
            return krs0_round(self.Gamma) / self.s0
        if self.geometry == "flat":
            return krs0_flat(self.Gamma) / self.s0
        else:
            raise NotImplementedError(f"{self.geometry=}")

    def _create_pseudomode(self) -> PseudomodeWakefield:
        """Create the pseudomode representation."""
        # Amplitude A = c * Z0 / (π * a²), using Z0 = 1/(ε₀*c)
        A = 1 / (4 * np.pi * epsilon_0) * 4 / self.radius**2
        if self.geometry == "flat":
            A *= np.pi**2 / 16

        d = self.kr / (2 * self.Qr)
        mode = Pseudomode(A=A, d=d, k=self.kr, phi=np.pi / 2)
        return PseudomodeWakefield(modes=[mode])

    def __repr__(self):
        material = self.material_from_properties()
        material_str = f", material={material!r}" if material else ""
        return (
            f"{self.__class__.__name__}("
            f"radius={self.radius}, "
            f"conductivity={self.conductivity}, "
            f"relaxation_time={self.relaxation_time}, "
            f"geometry={self.geometry!r}"
            f"{material_str}) "
            f"→ s₀={self.s0:.3e} m, Γ={self.Gamma:.3f}, k_r={self.kr:.1f}/m, Q_r={self.Qr:.2f}"
        )

    @property
    def pseudomode(self) -> PseudomodeWakefield:
        """The internal PseudomodeWakefield model."""
        return self._internal_model

    def wake(self, z):
        """
        Evaluate the wakefield at position z.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position [m]. Negative z is behind the source.

        Returns
        -------
        W : float or np.ndarray
            Wakefield value [V/C/m]. Returns 0 for z > 0 (causality).
        """
        return self._internal_model.wake(z)

    def impedance(self, k):
        """
        Evaluate the impedance at wavenumber k.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber [1/m].

        Returns
        -------
        Z : complex or np.ndarray
            Impedance [Ohm/m].
        """
        return self._internal_model.impedance(k)

    def __call__(self, z):
        """Evaluate the wakefield at position z (convenience method)."""
        return self.wake(z)

    def convolve_density(
        self,
        density: np.ndarray,
        dz: float,
        offset: float = 0,
        include_self_kick: bool = True,
        plot: bool = False,
        ax=None,
    ) -> np.ndarray:
        """
        Compute integrated wakefield by convolving with charge density.

        Parameters
        ----------
        density : np.ndarray
            Charge density array [C/m].
        dz : float
            Grid spacing [m].
        offset : float, optional
            Offset for the z coordinate [m]. Default is 0.
        include_self_kick : bool, optional
            Whether to include the extra ½ self-kick. Default is True.
        plot : bool, optional
            If True, plot the density profile and wake potential. Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None and plot=True, creates a new figure.

        Returns
        -------
        integrated_wake : np.ndarray
            Integrated longitudinal wakefield [V/m].
        """
        result = self._internal_model.convolve_density(
            density, dz, offset=offset, include_self_kick=include_self_kick
        )
        if plot:
            self._plot_convolve_density(density, dz, result, ax=ax)
        return result

    def particle_kicks(
        self,
        particle_group_or_z,
        weight: np.ndarray = None,
        include_self_kick: bool = True,
        plot: bool = False,
        ax=None,
    ) -> np.ndarray:
        """
        Compute wakefield-induced longitudinal momentum kicks.

        Uses O(N) algorithm exploiting the pseudomode exponential form.

        Parameters
        ----------
        particle_group_or_z : ParticleGroup or np.ndarray
            Either a ParticleGroup object or an array of z positions [m].
        weight : np.ndarray, optional
            Particle charges [C]. Required if particle_group_or_z is an array.
        include_self_kick : bool, optional
            Whether to include the self-kick term. Default is True.
        plot : bool, optional
            If True, plot the per-particle kicks vs position. Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.

        Returns
        -------
        np.ndarray
            Array of longitudinal momentum kicks per unit length [eV/m].
        """
        z, weight = self._extract_z_weight(particle_group_or_z, weight)
        return self._internal_model.particle_kicks(
            z, weight, include_self_kick=include_self_kick, plot=plot, ax=ax
        )

    def apply_to_particles(
        self,
        particle_group,
        length: float,
        inplace: bool = False,
        include_self_kick: bool = True,
    ):
        """
        Apply the wakefield momentum kicks to a ParticleGroup.

        Parameters
        ----------
        particle_group : ParticleGroup
            The particle group to apply the wakefield to.
        length : float
            Length over which the wakefield acts [m].
        inplace : bool, optional
            If True, modifies in place. If False, returns a modified copy.
        include_self_kick : bool, optional
            Whether to include the self-kick term. Default is True.

        Returns
        -------
        ParticleGroup or None
            The modified ParticleGroup if `inplace=False`, otherwise None.
        """
        if not inplace:
            particle_group = particle_group.copy()

        kicks = self.particle_kicks(particle_group, include_self_kick=include_self_kick)
        particle_group.pz += kicks * length

        if not inplace:
            return particle_group

    def plot(self, zmax=None, zmin=0, n=200, normalized=False, ax=None):
        """
        Plot the resistive wall wakefield W(z).

        Parameters
        ----------
        zmax : float, optional
            Maximum trailing distance [m]. Defaults to 10 decay lengths.
        zmin : float, optional
            Minimum trailing distance [m]. Default is 0.
        n : int, optional
            Number of points. Default is 200.
        normalized : bool, optional
            If True, plot dimensionless Ŵ(z/s₀) = W(z)/W₀ vs z/s₀.
            Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        import matplotlib.pyplot as plt

        if zmax is None:
            zmax = 1 / (self.kr / (2 * self.Qr)) * 10

        zlist = np.linspace(zmin, zmax, n)
        Wz = self.wake(-zlist)

        if ax is None:
            _, ax = plt.subplots()

        if normalized:
            ax.plot(zlist / self.s0, Wz / self.W0)
            ax.set_xlabel(r"$|z|/s_0$")
            ax.set_ylabel(r"$W(z)/W_0$")
        else:
            ax.plot(zlist * 1e6, Wz * 1e-12)
            ax.set_xlabel(r"Distance behind source $|z|$ (µm)")
            ax.set_ylabel(r"$W(z)$ (V/pC/m)")

        ax.set_title("ResistiveWallPseudomode")

    def to_bmad(
        self, file=None, z_max=100, amp_scale=1, scale_with_length=True, z_scale=1
    ):
        """
        Export wakefield in Bmad format.

        Parameters
        ----------
        file : str, optional
            Output file path. If None, returns string only.
        z_max : float
            Trailing z distance for Bmad.
        amp_scale : float
            Amplitude scaling factor.
        scale_with_length : bool
            Whether to scale with length.
        z_scale : float
            Z scaling factor.

        Returns
        -------
        str
            Bmad-formatted wakefield string.
        """
        s = f"""! AC Resistive wall wakefield
! Adapted from SLAC-PUB-10707
!    Material        : {self.material_from_properties()}
!    Conductivity    : {self.conductivity} S/m
!    Relaxation time : {self.relaxation_time} s
!    Geometry        : {self.geometry}
"""

        if self.geometry == "round":
            s += f"!    Radius          : {self.radius} m\n"
        elif self.geometry == "flat":
            s += f"!    full gap        : {2*self.radius} m\n"

        s += f"!    s₀              : {self.s0}  m\n"
        s += f"!    Γ               : {self.Gamma} \n"
        s += "! sr_wake =  \n"

        s += f"{{{z_scale=}, {amp_scale=}, {scale_with_length=}, {z_max=},\n"
        s += self._internal_model.to_bmad() + "}\n"

        if file is not None:
            with open(file, "w") as f:
                f.write(s)

        return s


@dataclass
class ResistiveWallWakefield(ResistiveWallWakefieldBase):
    """
    Accurate impedance-based resistive wall wakefield model.

    This class computes the short-range resistive wall wakefield by numerically
    evaluating the full longitudinal impedance Z(k) and transforming to the
    wakefield W(z) via FFT. This approach captures all physical effects including
    the anomalous skin effect at high frequencies.

    Physics Background
    ------------------
    When a relativistic charged particle travels through a conducting beam pipe,
    electromagnetic fields penetrate into the conductor due to the finite
    conductivity. The skin depth δ decreases with frequency:

    $$\\delta = \\sqrt{\\frac{2}{\\omega \\mu_0 \\sigma}}$$

    At very high frequencies (relevant for short-range wakes), the electron
    mean free path becomes comparable to the skin depth, and the simple Ohmic
    model breaks down. The Drude model accounts for this through a frequency-
    dependent conductivity with relaxation time τ:

    $$\\sigma(k) = \\frac{\\sigma_0}{1 - ikc\\tau}$$

    The longitudinal impedance per unit length for a round pipe is:

    $$Z(k) = \\frac{Z_0}{2\\pi a} \\cdot \\frac{\\zeta(k)}{1 - i k a \\zeta(k) / 2}$$

    where ζ(k) is the surface impedance and a is the pipe radius.

    The wakefield is obtained via cosine transform (for z ≤ 0):

    $$W(z) = \\frac{2c}{\\pi} \\int_0^{\\infty} \\text{Re}[Z(k)] \\cos(kz) \\, dk$$

    Characteristic Scales
    ---------------------
    - **s₀**: Characteristic length scale:

      $$s_0 = \\left( \\frac{2a^2}{Z_0 \\sigma_0} \\right)^{1/3}$$

      For copper with a = 2.5 mm, s₀ ≈ 8 µm.

    - **W₀**: Characteristic wake amplitude at z = 0:
      - Round: $W_0 = c Z_0 / (\\pi a^2)$
      - Flat: $W_0 = c Z_0 \\pi / (16 a^2)$

    Geometry
    --------
    - **Round (circular pipe)**: Closed-form impedance expression.
    - **Flat (parallel plates)**: Requires numerical integration over
      transverse modes; more computationally expensive.

    Parameters
    ----------
    radius : float
        Radius of the beam pipe [m]. For flat geometry, this is the half-gap.
    conductivity : float
        Electrical DC conductivity of the wall material [S/m].
    relaxation_time : float
        Drude-model relaxation time τ of the conductor [s].
        Typical values: Cu ≈ 27 fs, Al ≈ 8 fs.
    geometry : str, optional
        Geometry of the beam pipe: 'round' or 'flat'. Default is 'round'.

    Attributes
    ----------
    s0 : float
        Characteristic length scale [m]
    W0 : float
        Characteristic wake amplitude [V/C/m]

    Notes
    -----
    - This model is ~10-20× slower than `ResistiveWallPseudomode` but more
      accurate, especially for the first oscillation peak.
    - The FFT-based convolution uses O(N log N) complexity for density
      convolution, compared to O(N) for the pseudomode model.
    - For flat geometry, impedance calculation requires numerical integration
      and is significantly slower than round geometry.

    See Also
    --------
    ResistiveWallPseudomode : Fast approximate model using damped sinusoid

    References
    ----------
    .. [1] K. Bane and G. Stupakov, "Resistive wall wakefield in the LCLS
       undulator beam pipe," SLAC-PUB-10707 (2004).
       https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf

    .. [2] A. Chao, "Physics of Collective Beam Instabilities in High Energy
       Accelerators," Wiley, 1993, Chapter 2.

    Examples
    --------
    ::

        wake = ResistiveWallWakefield.from_material(
            "copper-slac-pub-10707", radius=2.5e-3, geometry="round"
        )
        wake.wake(-10e-6)  # Wake at 10 µm behind source
        wake.impedance(1e5)  # Impedance at k = 100/mm
    """

    def __post_init__(self):
        super().__post_init__()
        # Precompute relaxation distance for impedance calculations
        self._ctau = c_light * self.relaxation_time

        # Create internal model for particle kicks and convolution
        self._internal_model = ImpedanceWakefield(
            impedance_func=self.impedance,
            wakefield_func=self._wakefield_internal,
        )

    def _wakefield_internal(
        self,
        z: float | np.ndarray,
        k_max: float = 1e7,
        n_fft: int = 4096,
    ) -> float | np.ndarray:
        """Internal wakefield computation using FFT."""
        z_arr = np.asarray(z)
        is_scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)

        result = wakefield_from_impedance_fft(
            z_arr, self.impedance, k_max=k_max, n_fft=n_fft
        )

        if is_scalar:
            return float(result[0])
        return result

    def __repr__(self):
        material = self.material_from_properties()
        material_str = f", material={material!r}" if material else ""
        return (
            f"{self.__class__.__name__}("
            f"radius={self.radius}, "
            f"conductivity={self.conductivity}, "
            f"relaxation_time={self.relaxation_time}, "
            f"geometry={self.geometry!r}"
            f"{material_str}) "
            f"→ s₀={self.s0:.3e} m"
        )

    def impedance(self, k):
        """
        Evaluate the impedance at wavenumber k.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber [1/m].

        Returns
        -------
        Z : complex or np.ndarray
            Impedance [Ohm/m].
        """
        if self.geometry == "round":
            return longitudinal_impedance_round(
                k, self.radius, self.conductivity, self._ctau
            )
        else:  # flat
            return longitudinal_impedance_flat(
                k, self.radius, self.conductivity, self._ctau
            )

    def wake(self, z, k_max: float = 1e7, method: str = "auto", n_fft: int = 4096):
        """
        Evaluate the wakefield at position z.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position [m]. Negative z is behind the source.
        k_max : float, optional
            Upper limit of k integration [1/m]. Default is 1e7.
        method : str, optional
            'auto', 'fft', or 'quad'. Default is 'auto'.
        n_fft : int, optional
            Number of FFT points for FFT method. Default is 4096.

        Returns
        -------
        W : float or np.ndarray
            Wakefield value [V/C/m]. Returns 0 for z > 0 (causality).
        """
        z_arr = np.asarray(z)
        is_scalar = z_arr.ndim == 0

        if method == "auto":
            method = "quad" if is_scalar else "fft"

        if method == "fft":
            z_arr = np.atleast_1d(z_arr)
            result = wakefield_from_impedance_fft(
                z_arr, self.impedance, k_max=k_max, n_fft=n_fft
            )
            if is_scalar:
                return float(result[0])
            return result
        else:  # quad
            return wakefield_from_impedance(z, self.impedance, k_max=k_max)

    def __call__(self, z):
        """Evaluate the wakefield at position z (convenience method)."""
        return self.wake(z)

    def convolve_density(
        self,
        density: np.ndarray,
        dz: float,
        offset: float = 0,
        include_self_kick: bool = True,
        plot: bool = False,
        ax=None,
    ) -> np.ndarray:
        """
        Compute integrated wakefield by convolving density with impedance.

        Uses FFT in frequency domain for efficiency.

        Parameters
        ----------
        density : np.ndarray
            Charge density array [C/m].
        dz : float
            Grid spacing [m].
        offset : float, optional
            Offset for the z coordinate [m]. Default is 0.
        include_self_kick : bool, optional
            Whether to include the extra ½ self-kick. Default is True.
        plot : bool, optional
            If True, plot the density profile and wake potential. Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None and plot=True, creates a new figure.

        Returns
        -------
        integrated_wake : np.ndarray
            Integrated longitudinal wakefield [V/m].
        """
        result = self._internal_model.convolve_density(
            density, dz, offset=offset, include_self_kick=include_self_kick
        )
        if plot:
            self._plot_convolve_density(density, dz, result, ax=ax)
        return result

    def particle_kicks(
        self,
        particle_group_or_z,
        weight: np.ndarray = None,
        include_self_kick: bool = True,
        n_bins: int = None,
        plot: bool = False,
        ax=None,
    ) -> np.ndarray:
        """
        Compute wakefield-induced longitudinal momentum kicks.

        Uses FFT-based density convolution with interpolation back to
        particle positions. This is O(N + M log M) where N is the number
        of particles and M is the number of grid points.

        Parameters
        ----------
        particle_group_or_z : ParticleGroup or np.ndarray
            Either a ParticleGroup object or an array of z positions [m].
        weight : np.ndarray, optional
            Particle charges [C]. Required if particle_group_or_z is an array.
        include_self_kick : bool, optional
            Whether to include the self-kick term. Default is True.
        n_bins : int, optional
            Number of bins for the density grid. Default is max(100, N//10).
        plot : bool, optional
            If True, plot the per-particle kicks vs position. Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.

        Returns
        -------
        np.ndarray
            Array of longitudinal momentum kicks per unit length [eV/m].
        """
        z, weight = self._extract_z_weight(particle_group_or_z, weight)
        return self._internal_model.particle_kicks(
            z,
            weight,
            include_self_kick=include_self_kick,
            n_bins=n_bins,
            plot=plot,
            ax=ax,
        )

    def apply_to_particles(
        self,
        particle_group,
        length: float,
        inplace: bool = False,
        include_self_kick: bool = True,
    ):
        """
        Apply the wakefield momentum kicks to a ParticleGroup.

        Parameters
        ----------
        particle_group : ParticleGroup
            The particle group to apply the wakefield to.
        length : float
            Length over which the wakefield acts [m].
        inplace : bool, optional
            If True, modifies in place. If False, returns a modified copy.
        include_self_kick : bool, optional
            Whether to include the self-kick term. Default is True.

        Returns
        -------
        ParticleGroup or None
            The modified ParticleGroup if `inplace=False`, otherwise None.
        """
        if not inplace:
            particle_group = particle_group.copy()

        kicks = self.particle_kicks(particle_group, include_self_kick=include_self_kick)
        particle_group.pz += kicks * length

        if not inplace:
            return particle_group

    def plot(self, zmax=None, zmin=0, n=200, normalized=False, ax=None):
        """
        Plot the resistive wall wakefield W(z).

        Parameters
        ----------
        zmax : float, optional
            Maximum trailing distance [m]. Defaults to 100 * s0.
        zmin : float, optional
            Minimum trailing distance [m]. Default is 0.
        n : int, optional
            Number of points. Default is 200.
        normalized : bool, optional
            If True, plot dimensionless Ŵ(z/s₀) = W(z)/W₀ vs z/s₀.
            Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        import matplotlib.pyplot as plt

        if zmax is None:
            zmax = 100 * self.s0

        zlist = np.linspace(zmin, zmax, n)
        Wz = self.wake(-zlist)

        if ax is None:
            _, ax = plt.subplots()

        if normalized:
            ax.plot(zlist / self.s0, Wz / self.W0)
            ax.set_xlabel(r"$|z|/s_0$")
            ax.set_ylabel(r"$W(z)/W_0$")
        else:
            ax.plot(zlist * 1e6, Wz * 1e-12)
            ax.set_xlabel(r"Distance behind source $|z|$ (µm)")
            ax.set_ylabel(r"$W(z)$ (V/pC/m)")
