from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from ..units import mu_0, c_light
from scipy.signal import fftconvolve

# Calculate these here for consistency
epsilon_0 = 1 / (mu_0 * c_light**2)
Z0 = mu_0 * c_light  # Ohm


# Conductivities (1/Ohm*1/meter)
CONDUCTIVITY = {"Cu": 6.5e7, "Al": 4.2e7, "SS": 1.5e6}
# Relaxation times (s)
RELAXATION_TIME = {"Cu": 27e-15, "Al": 7.5e-15, "SS": 8e-15}


# AC wake Fitting Formula Polynomial coefficients
krs0cRound = (
    1.81620118662482,
    0.29540336833708,
    -1.23728482888772,
    1.05410903517018,
    -0.38606826800684,
    0.05234403145974,
)

krs0cFlat = (
    1.29832152141677,
    0.18173822141741,
    -0.62770698511448,
    0.47383850057072,
    -0.15947258626548,
    0.02034464240245,
)

QrcRound = (1.09524274851589, 2.40729067134909, 0.06574992723432, -0.04766884506469)

QrcFlat = (1.02903443223445, 1.33341005467949, -0.16585375059715, 0.00075548123372)


def poly(data, x):
    v = 0
    for i in range(len(data)):
        v += data[i] * x**i
    return v


def krs0Round(G):
    return poly(krs0cRound, G)


def krs0Flat(G):
    return poly(krs0cFlat, G)


def QrRound(G):
    return poly(QrcRound, G)


def QrFlat(G):
    return poly(QrcFlat, G)


def z0f(radius, conductivity):
    val = 2 * radius**2 / (Z0 * conductivity)
    return val ** (1 / 3.0)


def Gammaf(relaxation_time, radius, conductivity):
    return c_light * relaxation_time / z0f(radius, conductivity)


def prefactor(radius):
    return 1 / (4 * np.pi * epsilon_0) * 4 / radius**2


# Bmad strings formatting
def str_bool(x):
    if x:
        return "T"
    else:
        return "F"


def bmad_sr_wake_header(z_scale=1.0, amp_scale=1.0, scale_with_length=False):
    x = "{"
    x += f"z_scale = {z_scale}, amp_scale = {amp_scale}, scale_with_length = {str_bool(scale_with_length)},\n"
    return x


def bmad_sr_wake_footer(z_max=1.0):
    return "  z_max = " + str(z_max) + "}\n"


@dataclass
class pseudomode:
    """
    Single Bmad short range wakefield pseudomode parameters
    """

    A: float
    d: float
    k: float
    phi: float

    def to_bmad(self, type="longitudinal", transverse_dependence="none"):
        x = type + " = {"
        x += f"{self.A}, {self.d}, {self.k}, {self.phi/(2*np.pi)}, "
        x += transverse_dependence
        return "  " + x + "},\n"

    def __call__(self, z):
        return self.A * np.exp(self.d * z) * np.sin(self.k * z + self.phi)

    def plot(self, zmax=0.001, zmin=0, n=200):
        zlist = np.linspace(zmin, zmax, n)
        Wz = self(-zlist)

        fig, ax = plt.subplots()
        ax.plot(zlist * 1e6, Wz * 1e-12)
        ax.set_xlabel(r"$-z$ (µm)")
        ax.set_ylabel(r"$W_z$ (V/pC)")


def apply_sr_wake(
    z: np.ndarray,
    weight: np.ndarray,
    mode,
    include_self_kick: bool = True,
) -> np.ndarray:
    """
    Compute short-range wakefield energy kicks using a single pseudomode.

    Wakefield is defined as:
        W(z) = A * exp(d * z) * sin(k * z + phi)

    Parameters
    ----------
    z : ndarray of shape (N,)
        Particle positions [m], sorted from tail to head (increasing).
    weight : ndarray of shape (N,)
        Particle charges [C].
    mode : pseudomode
        Object with attributes:
            A   : float, amplitude [V/C/m]
            d   : float, damping coefficient [1/m]
            k   : float, wave number [1/m]
            phi : float, phase offset [rad]
    include_self_kick : bool, optional
        If True, applies the ½ A q sin(φ) self-kick. Default is True.

    Returns
    -------
    delta_E : ndarray of shape (N,)
        Wake-induced energy change per particle [eV/m].
    """
    z = np.asarray(z)
    weight = np.asarray(weight)

    if z.shape != weight.shape:
        raise ValueError(
            f"Mismatched shapes: z.shape={z.shape}, weight.shape={weight.shape}"
        )
    if z.ndim != 1:
        raise ValueError("z and weight must be 1D arrays")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be sorted from tail to head (increasing)")

    N = len(z)
    delta_E = np.zeros(N)

    A = mode.A
    d = mode.d
    k = mode.k
    phi = mode.phi

    s = d + 1j * k
    c = A * np.exp(1j * phi)

    b = 0.0 + 0.0j  # complex accumulator

    for i in range(N - 1, -1, -1):
        zi = z[i]
        qi = abs(weight[i])

        # Wake from trailing particles
        delta_E[i] -= np.imag(c * np.exp(s * zi) * b)

        # Accumulate this particle's contribution
        b += qi * np.exp(-s * zi)

    if include_self_kick:
        delta_E -= 0.5 * A * weight * np.sin(phi)

    return delta_E


@dataclass
class ResistiveWallWakefield:
    """
    Trailing longitudinal wakefield from a charged particlein a conducting pipe

    Single oscillator fit according to Bane & Stupakov SLAC-PUB-10707 (2004)

    """

    radius: float
    material: str = "Cu"
    geometry: str = "round"

    def __post_init__(self):
        if not isinstance(self.radius, (int, float)) or self.radius <= 0:
            raise ValueError(f"radius must be a positive number, got {self.radius}")

        if self.material not in CONDUCTIVITY:
            raise ValueError(
                f"Unsupported material: {self.material}. Must be one of: {list(CONDUCTIVITY)}"
            )

        if self.geometry not in ("round", "flat"):
            raise ValueError(
                f"Unsupported geometry: {self.geometry}. Must be 'round' or 'flat'"
            )

    @property
    def conductivity(self):
        return CONDUCTIVITY[self.material]

    @property
    def relaxation_time(self):
        return RELAXATION_TIME[self.material]

    @property
    def Gamma(self):
        return Gammaf(self.relaxation_time, self.radius, self.conductivity)

    @property
    def Qr(self):
        if self.geometry == "round":
            return QrRound(self.Gamma)
        if self.geometry == "flat":
            return QrFlat(self.Gamma)
        else:
            raise NotImplementedError(f"{self.geometry=}")

    @property
    def kr(self):
        if self.geometry == "round":
            return krs0Round(self.Gamma) / self.z0
        if self.geometry == "flat":
            return krs0Flat(self.Gamma) / self.z0
        else:
            raise NotImplementedError(f"{self.geometry=}")

    @property
    def z0(self):
        return z0f(self.radius, self.conductivity)

    def to_bmad(self, z_max=100):
        """

        Parameters
        ----------
        z_max: float
            trailing z distance
        """
        s = f"""! AC Resistive wall wakefield
! Adapted from SLAC-PUB-10707
!    Material        : {self.material}
!    Conductivity    : {self.conductivity} Ohm^-1 m^-1
!    Relaxation time : {self.relaxation_time} s
!    Geometry        : {self.geometry}
"""

        if self.geometry == "round":
            s += f"!    Radius          : {self.radius} m\n"
        elif self.geometry == "flat":
            s += f"!    full gap        : {2*self.radius} m\n"

        s += f"! characteristic z0  : {self.z0}  m\n"
        s += f"!    Gamma           : {self.Gamma} \n"
        s += "! sr_wake =  \n"

        s += bmad_sr_wake_header()
        s += self.pseudomode.to_bmad()
        s += bmad_sr_wake_footer(z_max=z_max)

        return s

    @property
    def pseudomode(self):
        """
        Single pseudomode representing this wakefield
        """
        if self.geometry == "round":
            factor = 1
        elif self.geometry == "flat":
            factor = np.pi**2 / 16
        d = self.kr / (2 * self.Qr)

        return pseudomode(factor * prefactor(self.radius), d, self.kr, np.pi / 2)

    def plot(self, zmax=None):
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

    def convolve(self, density, dz, offset=0):
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
