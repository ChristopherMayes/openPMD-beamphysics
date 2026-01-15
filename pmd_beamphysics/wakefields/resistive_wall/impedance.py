"""
Impedance-based resistive wall wakefield model.

This module provides the ResistiveWallWakefield class, an accurate
impedance-based model for resistive wall wakefields.

Classes
-------
ResistiveWallWakefield
    Accurate impedance-based resistive wall wakefield model
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ...units import c_light
from ..impedance import ImpedanceWakefield
from .base import (
    Geometry,
    ResistiveWallWakefieldBase,
    longitudinal_impedance_round,
    longitudinal_impedance_flat,
)

__all__ = ["ResistiveWallWakefield"]


@dataclass
class ResistiveWallWakefield(ResistiveWallWakefieldBase, ImpedanceWakefield):
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

        # Initialize ImpedanceWakefield - wake() and impedance() are provided by this class
        ImpedanceWakefield.__init__(self, impedance_func=self.impedance)

    def __repr__(self):
        material = self.material_from_properties()
        material_str = f", material={material!r}" if material else ""
        return (
            f"{self.__class__.__name__}("
            f"radius={self.radius}, "
            f"conductivity={self.conductivity}, "
            f"relaxation_time={self.relaxation_time}, "
            f"geometry={self.geometry.value!r}"
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
        if self.geometry == Geometry.ROUND:
            return longitudinal_impedance_round(
                k, self.radius, self.conductivity, self._ctau
            )
        else:  # FLAT
            return longitudinal_impedance_flat(
                k, self.radius, self.conductivity, self._ctau
            )

    # wake inherited from ImpedanceWakefield (uses FFT for arrays, quad for scalars)
    # convolve_density inherited from WakefieldBase
    # particle_kicks inherited from ImpedanceWakefield

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
        if zmax is None:
            zmax = 20 * self.s0

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

    def plot_impedance(
        self, kmax: float = None, kmin: float = 0, n: int = 500, ax=None
    ):
        """
        Plot the resistive wall impedance Z(k).

        Parameters
        ----------
        kmax : float, optional
            Maximum wavenumber [1/m]. Defaults to 10/s0.
        kmin : float, optional
            Minimum wavenumber [1/m]. Default is 0.
        n : int, optional
            Number of points. Default is 500.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        if kmax is None:
            kmax = 10 / self.s0
        super().plot_impedance(kmax=kmax, kmin=kmin, n=n, ax=ax)
