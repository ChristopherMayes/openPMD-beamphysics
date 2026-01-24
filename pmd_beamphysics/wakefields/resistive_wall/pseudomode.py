"""
Pseudomode-based resistive wall wakefield model.

This module provides the ResistiveWallPseudomode class, a fast approximation
of the resistive wall wakefield using a damped sinusoidal representation.

Classes
-------
ResistiveWallPseudomode
    Fast pseudomode-based resistive wall wakefield model
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np

from ...units import epsilon_0
from ..pseudomode import Pseudomode, PseudomodeWakefield
from .base import (
    Geometry,
    ResistiveWallWakefieldBase,
    Gammaf,
    krs0_round,
    krs0_flat,
    Qr_round,
    Qr_flat,
)

__all__ = ["ResistiveWallPseudomode"]


@dataclass
class ResistiveWallPseudomode(ResistiveWallWakefieldBase, PseudomodeWakefield):
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

    @property
    def _modes(self) -> list:
        """Dynamically compute the pseudomode list from current parameters."""
        return [self._create_mode()]

    @property
    def Gamma(self):
        """Dimensionless relaxation time Γ = c * τ / s₀."""
        return Gammaf(self.relaxation_time, self.radius, self.conductivity)

    @property
    def Qr(self):
        """Dimensionless quality factor Q_r of the wakefield pseudomode."""
        if self.geometry == Geometry.ROUND:
            return Qr_round(self.Gamma)
        if self.geometry == Geometry.FLAT:
            return Qr_flat(self.Gamma)
        else:
            raise NotImplementedError(f"{self.geometry=}")

    @property
    def kr(self):
        """Real-valued wave number k_r of the wakefield pseudomode [1/m]."""
        if self.geometry == Geometry.ROUND:
            return krs0_round(self.Gamma) / self.s0
        if self.geometry == Geometry.FLAT:
            return krs0_flat(self.Gamma) / self.s0
        else:
            raise NotImplementedError(f"{self.geometry=}")

    def _create_mode(self) -> Pseudomode:
        """Create the pseudomode for this resistive wall wakefield."""
        # Amplitude A = c * Z0 / (π * a²), using Z0 = 1/(ε₀*c)
        A = 1 / (4 * np.pi * epsilon_0) * 4 / self.radius**2
        if self.geometry == Geometry.FLAT:
            A *= np.pi**2 / 16

        d = self.kr / (2 * self.Qr)
        return Pseudomode(A=A, d=d, k=self.kr, phi=np.pi / 2)

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
            f"→ s₀={self.s0:.3e} m, Γ={self.Gamma:.3f}, k_r={self.kr:.1f}/m, Q_r={self.Qr:.2f}"
        )

    # wake, impedance, particle_kicks, convolve_density inherited from PseudomodeWakefield

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
!    Geometry        : {self.geometry.value}
"""

        if self.geometry == Geometry.ROUND:
            s += f"!    Radius          : {self.radius} m\n"
        elif self.geometry == Geometry.FLAT:
            s += f"!    full gap        : {2*self.radius} m\n"

        s += f"!    s₀              : {self.s0}  m\n"
        s += f"!    Γ               : {self.Gamma} \n"
        s += "! sr_wake =  \n"

        s += f"{{{z_scale=}, {amp_scale=}, {scale_with_length=}, {z_max=},\n"
        s += PseudomodeWakefield.to_bmad(self) + "}\n"

        if file is not None:
            with open(file, "w") as f:
                f.write(s)

        return s
