"""
Resistive wall wakefield implementation.

This module provides an analytical model for short-range resistive wall wakefields
in accelerator beam pipes, based on the approach described in SLAC-PUB-10707.
It supports both round and flat geometries and includes effects of AC conductivity
through material relaxation times.

Classes
-------
pseudomode
    Single-mode analytic representation of a short-range wakefield
ResistiveWallWakefield
    Complete wakefield model with geometry and material properties

References
----------
Bane & Stupakov, SLAC-PUB-10707 (2004)
https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import functools

from ..units import c_light, epsilon_0, Z0
from scipy.signal import fftconvolve

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


def krs0_flat(G):
    """
    k_r*s_0 from SLAC-PUB-10707 Fig. 14 for flat geometry
    This is from a polynomial fit of the digitized data
    """
    return _krs0_flat_poly(G)


def Qr_round(G):
    """
    Qr from SLAC-PUB-10707 Fig. 14 for round geometry
    This is from a polynomial fit of the digitized data
    """
    return _Qr_round_poly(G)


def Qr_flat(G):
    """
    Qr from SLAC-PUB-10707 Fig. 14 for flat geometry
    This is from a polynomial fit of the digitized data
    """
    return _Qr_flat_poly(G)


def s0f(radius, conductivity):
    """
    Characteristic distance from SLAC-PUB-10707 Eq. 5
    """
    val = 2 * radius**2 / (Z0 * conductivity)
    return val ** (1 / 3.0)


def Gammaf(relaxation_time, radius, conductivity):
    """
    dimensionless relaxation time Γ = cτ/s0
    """

    return c_light * relaxation_time / s0f(radius, conductivity)


@dataclass(slots=True)
class pseudomode:
    """
    Single-mode analytic representation of a short-range wakefield.

    Models the longitudinal wakefield as a damped sinusoid:
        W(z) = A * exp(d * z) * sin(k * z + φ)

    This form is used to approximate short-range wakefields such as the resistive wall wake,
    and can be evaluated directly, plotted, or exported in Bmad format.

    Parameters
    ----------
    A : float
        Amplitude coefficient [V/C/m].
    d : float
        Exponential decay rate [1/m]. Typically negative.
    k : float
        Oscillation wavenumber [1/m].
    phi : float
        Phase offset [rad].

    Methods
    -------
    __call__(z)
        Evaluate W(z) for an array of z values.
    plot(zmax=..., zmin=..., n=...)
        Plot the pseudomode over a range of z values.
    to_bmad(type="longitudinal", transverse_dependence="none")
        Format pseudomode parameters as a Bmad-compatible string.
    particle_kicks(z, weight, include_self_kick=True)
        Compute energy kicks per unit length on a particle distribution.

    Notes
    -----
    - Wakefields are defined for z ≤ 0 (i.e., trailing the source particle).
    - This is a mathematical abstraction used to model physical wakefields.
    """

    A: float
    d: float
    k: float
    phi: float

    def to_bmad(self, type="longitudinal", transverse_dependence="none"):
        return f"{type} = {{{self.A}, {self.d}, {self.k}, {self.phi/(2*np.pi)}, {transverse_dependence}}}"

    def __call__(self, z):
        return self.A * np.exp(self.d * z) * np.sin(self.k * z + self.phi)

    def plot(self, zmax=0.001, zmin=0, n=200):
        zlist = np.linspace(zmin, zmax, n)
        Wz = self(-zlist)

        fig, ax = plt.subplots()
        ax.plot(zlist * 1e6, Wz * 1e-12)
        ax.set_xlabel(r"$-z$ (µm)")
        ax.set_ylabel(r"$W_z$ (V/pC/m)")

    def particle_kicks(
        self,
        z: np.ndarray,
        weight: np.ndarray,
        include_self_kick: bool = True,
    ) -> np.ndarray:
        """
        Compute short-range wakefield energy kicks per unit length

        Internally the particles will be sorted.
        This should take negligible time compared with the algorithm.

        Parameters
        ----------
        z : ndarray of shape (N,)
            Particle positions [m], sorted from tail to head (increasing).
        weight : ndarray of shape (N,)
            Particle charges [C].
        include_self_kick : bool, optional
            If True, applies the ½ A q sin(φ) self-kick. Default is True.

        Returns
        -------
        delta_E : ndarray of shape (N,)
            Wake-induced energy kick per unit length at each particle [eV/m].
        """

        z = np.asarray(z)
        weight = np.asarray(weight)

        if z.shape != weight.shape:
            raise ValueError(
                f"Mismatched shapes: z.shape={z.shape}, weight.shape={weight.shape}"
            )
        if z.ndim != 1:
            raise ValueError("z and weight must be 1D arrays")

        # Sort
        ix = z.argsort()
        z = z[ix]
        z -= z.max()  # Offset to avoid numerical problems
        weight = weight[ix]

        N = len(z)
        delta_E = np.zeros(N)

        s = self.d + 1j * self.k
        c = self.A * np.exp(1j * self.phi)

        b = 0.0 + 0.0j  # complex accumulator

        for i in range(N - 1, -1, -1):
            zi = z[i]
            qi = weight[i]

            # Wake from trailing particles
            delta_E[i] -= np.imag(c * np.exp(s * zi) * b)

            # Accumulate this particle's contribution
            b += qi * np.exp(-s * zi)

        if include_self_kick:
            delta_E -= 0.5 * self.A * weight * np.sin(self.phi)

        # Return kicks in the original particle order
        kicks = np.empty_like(delta_E)
        kicks[ix] = delta_E
        return kicks


@dataclass
class ResistiveWallWakefield:
    """
    Analytic short-range resistive wall wakefield model based on SLAC-PUB-10707 (Bane & Stupakov, 2004).

    Models the longitudinal wakefield trailing a charged particle moving through a
    conducting pipe, using a single damped sinusoidal pseudomode fit.

    Parameters
    ----------
    radius : float
        Radius of the beam pipe [m]. For flat geometry, this is half the gap.
    conductivity : float
        Electrical conductivity of the wall material [S/m].
    relaxation_time : float
        Drude-model relaxation time of the conductor [s].
    geometry : str, optional
        Geometry of the beam pipe: either 'round' or 'flat'. Default is 'round'.

    Notes
    -----
    - The model uses polynomial fits for k_r * s₀ and Q_r as functions of Γ = c * τ / s₀,
      based on digitized data from SLAC-PUB-10707 Fig. 14.
    - Wakefield output supports evaluation, convolution, and export in Bmad format.
    - Materials with known conductivity and τ values are available via `from_material()`.
    - Relaxation times for materials have a large uncertainty in this model. See Bane, Stupakov, Tu (2006).

    References
    ----------
    Bane & Stupakov, SLAC-PUB-10707 (2004)
    https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf

    Bane, Stupakov, Tu, Proceedings of EPAC 2006, Edinburgh, Scotland THPCH073 (2006)
    https://accelconf.web.cern.ch/e06/PAPERS/THPCH073.PDF
    """

    radius: float
    conductivity: float
    relaxation_time: float
    geometry: str = "round"

    # Internal material database (SI units)
    # Note conductivtity_SI = conductivtity_CGS / ( Z0 *c / (4*pi))
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

        if self.relaxation_time <= 0:
            raise ValueError(
                f"relaxation_time must be positive, got {self.relaxation_time}"
            )

        if self.geometry not in ("round", "flat"):
            raise ValueError(
                f"Unsupported geometry: {self.geometry}. Must be 'round' or 'flat'"
            )

    @classmethod
    def from_material(cls, material: str, radius: float, geometry: str = "round"):
        """
        Create a ResistiveWallWakefield from a known material preset.

        Parameters
        ----------
        material : str
            Material name. Must be in list(ResistiveWallWakefield.MATERIALS)
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
    def Gamma(self):
        """
        Dimensionless relaxation time Γ = c * τ / s₀.

        Describes the relative importance of material dispersion in the resistive wall model,
        where:
            τ   = relaxation time [s],
            c   = speed of light [m/s],
            s₀  = characteristic length scale [m].

        Used as the input variable for polynomial fits to Q_r and k_r * s₀ from SLAC-PUB-10707.
        """
        return Gammaf(self.relaxation_time, self.radius, self.conductivity)

    @property
    def Qr(self):
        """
        Dimensionless quality factor Q_r of the wakefield pseudomode.

        Obtained from a polynomial fit to SLAC-PUB-10707 Fig. 14 (bottom left) as a function of Γ.

        Depends on geometry:
            - 'round' or 'flat'

        Q_r characterizes the damping rate of the oscillatory pseudomode.
        """
        if self.geometry == "round":
            return Qr_round(self.Gamma)
        if self.geometry == "flat":
            return Qr_flat(self.Gamma)
        else:
            raise NotImplementedError(f"{self.geometry=}")

    @property
    def kr(self):
        """
        Real-valued wave number k_r of the wakefield pseudomode [1/m].

        Computed from:
            k_r = (k_r * s₀) / s₀

        where (k_r * s₀) is a dimensionless polynomial fit from SLAC-PUB-10707 Fig. 14 (top left).

        Depends on geometry:
            - 'round' or 'flat'

        k_r sets the frequency of oscillation of the short-range wakefield.
        """

        if self.geometry == "round":
            return krs0_round(self.Gamma) / self.s0
        if self.geometry == "flat":
            return krs0_flat(self.Gamma) / self.s0
        else:
            raise NotImplementedError(f"{self.geometry=}")

    @property
    def s0(self):
        """
        Characteristic length scale s₀ of the resistive wall wakefield [m].

        Defined by SLAC-PUB-10707 Eq. (5) as:
            s₀ = (2 * a² / (Z₀ * σ))^(1/3)

        where:
            a   = pipe radius [m],
            σ   = conductivity [S/m],
            Z₀  = vacuum impedance [Ω].

        s₀ sets the scale of the wakefield decay length and frequency.
        """
        return s0f(self.radius, self.conductivity)

    def to_bmad(
        self, file=None, z_max=100, amp_scale=1, scale_with_length=True, z_scale=1
    ):
        """

        Parameters
        ----------
        z_max: float
            trailing z distance
        """
        s = f"""! AC Resistive wall wakefield
! Adapted from SLAC-PUB-10707
!    Material        : {self.material_from_properties()}
!    Conductivity    : {self.conductivity} (Ωm)⁻¹
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
        s += self.pseudomode.to_bmad() + "}\n"

        if file is not None:
            with open(file, "w") as f:
                f.write(s)

        return s

    @property
    @functools.lru_cache
    def pseudomode(self):
        """
        Single pseudomode representing this wakefield (cached)
        """

        # Conversion from cgs units
        A = 1 / (4 * np.pi * epsilon_0) * 4 / self.radius**2
        if self.geometry == "flat":
            A *= np.pi**2 / 16

        d = self.kr / (2 * self.Qr)

        return pseudomode(A, d, self.kr, np.pi / 2)

    def plot(self, zmax=None):
        """
        Plot the resistive wall wakefield pseudomode W(z).

        The wake is plotted from z = 0 (source particle) to a specified negative trailing distance.

        Parameters
        ----------
        zmax : float, optional
            Maximum trailing distance [m] to plot.
            If not provided, defaults to 10 decay lengths:  zmax = 10 * (2 * Q_r / k_r)

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.
        """

        if zmax is None:
            zmax = 1 / (self.kr / (2 * self.Qr)) * 10

        return self.pseudomode.plot(zmax=zmax)

    def __call__(self, z):
        """
        Wakefield value at z relative to the source particle.

        z > 0 is the head of the bunch and returns 0.
        """
        z = np.asarray(z)
        out = np.empty_like(z, dtype=float)

        mask = z > 0
        out[mask] = 0
        out[~mask] = self.pseudomode(z[~mask])
        return out

    def particle_kicks(
        self,
        particle_group,
        include_self_kick: bool = True,
    ) -> np.ndarray:
        """
        Compute wakefield-induced longitudinal momentum kicks for a ParticleGroup.

        Parameters
        ----------
        particle_group : ParticleGroup
            The particle group to evaluate kicks for.
        include_self_kick : bool, optional
            Whether to include the ½ A q sin(φ) self-kick term. Default is True.

        Returns
        -------
        np.ndarray
            Array of longitudinal momentum kicks per unit length [eV/m], shape (N,).
        """
        if particle_group.in_t_coordinates:
            z = np.asarray(particle_group.z)
        else:
            z = -c_light * np.asarray(particle_group.t)

        weight = np.asarray(particle_group.weight)
        return self.pseudomode.particle_kicks(
            z, weight, include_self_kick=include_self_kick
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

        This modifies the pz component based on computed kicks and the specified length.

        Parameters
        ----------
        particle_group : ParticleGroup
            The particle group to apply the wakefield to.
        length : float
            Length over which the wakefield acts [m].
        inplace : bool, optional
            If True, modifies the ParticleGroup in place. If False, returns a modified copy. Default is False.
        include_self_kick : bool, optional
            Whether to include the ½ A q sin(φ) self-kick term. Default is True.

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

    def convolve_density(self, density, dz, offset=0):
        """

        Parameters
        ----------

        density: ndarray
            charge density array

        dz: float
            array spacing

        offset : float
            Offset coordinates for the center of the grid in [m]. Default: 0
            For example, an offset of -1.23 can be used to compute the trailing wake at z=-1.23 m relative to the density

        Returns
        -------
        integrated_wake: ndarray
            Integrated wakefield in eV / m

        """

        # make wakefield array
        n = len(density)

        # Make double sized density array
        density2 = np.zeros(2 * n)
        density2[0:n] = density

        # double-sized symmetric z vec
        z = np.arange(-n, n + 1, 1) * dz + dz / 2 + offset
        green2 = self(z)

        # Approximate (f * g)(t) = ∫ f(Δ) gz(z‑Δ) dΔ
        # Convolution of double-sized arrays
        conv = fftconvolve(density2, green2, mode="full")

        # The result is in a shifted location in the output array
        iwake = conv[n - 1 : 2 * n - 1] * dz

        return iwake
