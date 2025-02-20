from abc import ABC
from enum import Enum
from dataclasses import dataclass, replace
from copy import deepcopy

from math import pi
import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift, ifftn

from scipy.constants import epsilon_0, c

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pmd_beamphysics.statistics import mean_calc, mean_variance_calc
from pmd_beamphysics.plot import plot_2d_density_with_marginals


def fftfreq_max(n, d=1.0):
    """Return the maximum frequency in fftfreq for given n and d."""
    if n % 2 == 0:  # Even
        return (n / 2 - 1) / (d * n)
    else:  # Odd
        return ((n - 1) / 2) / (d * n)


def fftfreq_min(n, d=1.0):
    """Return the minimum frequency in fftfreq for given n and d."""
    if n % 2 == 0:  # Even
        return -(n / 2) / (d * n)
    else:  # Odd
        return -((n - 1) / 2) / (d * n)


class TemporalDomain(str, Enum):
    """
    Enumeration for temporal domain.
    """

    TIME = "time"
    FREQUENCY = "frequency"


class SpatialDomain(str, Enum):
    R = "r"
    K = "k"


# Axes to sum over for intensity/probability projections
# TODO: cleaner code.
_axis_for_sum = {
    "x": (1, 2),
    "y": (0, 2),
    "z": (0, 1),
    "kx": (1, 2),
    "ky": (0, 2),
    "kz": (0, 1),
}
_axis_for_sum["thetax"] = _axis_for_sum["kx"]
_axis_for_sum["thetay"] = _axis_for_sum["ky"]
_axis_for_sum["thetaz"] = _axis_for_sum["kz"]


class WavefrontBase(ABC):
    """
    Convenience functions for standard domain values
    """

    @property
    def shape(self):
        if self.Ex is None:
            return self.Ey.shape
        else:
            return self.Ex.shape

    def axis_index(self, key):
        """
        Returns axis index for a named axis label key.

        Examples:

        - `.axis_labels == ('x', 'y', 'z')`
        - `.axis_index('z')` returns `2`
        """
        return self.axis_labels.index(key)

    @property
    def k0(self):
        return 2 * pi / self.wavelength

    @property
    def nx(self):
        return self.shape[0]

    @property
    def ny(self):
        return self.shape[1]

    @property
    def nz(self):
        return self.shape[2]

    # x
    @property
    def xmin(self):
        """
        xmin = - (nx-1) * dx / 2
        """
        return -((self.nx - 1) * self.dx) / 2

    @property
    def xmax(self):
        """
        xmax =  (nx-1) * dx / 2
        """
        return ((self.nx - 1) * self.dx) / 2

    @property
    def xvec(self):
        return np.linspace(self.xmin, self.xmax, self.nx)

    # y
    @property
    def ymin(self):
        return -((self.ny - 1) * self.dy) / 2

    @property
    def ymax(self):
        return ((self.ny - 1) * self.dy) / 2

    @property
    def yvec(self):
        return np.linspace(self.ymin, self.ymax, self.ny)

    # z
    @property
    def zmin(self):
        return -((self.nz - 1) * self.dz) / 2

    @property
    def zmax(self):
        return ((self.nz - 1) * self.dz) / 2

    @property
    def zvec(self):
        return np.linspace(self.zmin, self.zmax, self.nz)

    # kx
    @property
    def dkx(self):
        """
        transverse wavevector components grid spacing in rad/m
        dkx = 2π / (nx * dx)
        """
        return 2 * pi / (self.nx * self.dx)

    @property
    def kxvec(self):
        return 2 * pi * fftshift(fftfreq(self.nx, d=self.dx))

    @property
    def kxmin(self):
        return 2 * pi * fftfreq_min(self.nx, self.dx)

    @property
    def kxmax(self):
        return 2 * pi * fftfreq_max(self.nx, self.dx)

    # ky
    @property
    def dky(self):
        return 2 * pi / (self.ny * self.dy)

    @property
    def kyvec(self):
        return 2 * pi * ifftshift(fftfreq(self.ny, d=self.dy))

    @property
    def kymin(self):
        return 2 * pi * fftfreq_min(self.ny, self.dy)

    @property
    def kymax(self):
        return 2 * pi * fftfreq_max(self.ny, self.dy)

    # kz
    @property
    def dkz(self):
        return 2 * pi / (self.nz * self.dz)

    @property
    def kzvec(self):
        return 2 * pi * ifftshift(fftfreq(self.nz, d=self.dz))

    @property
    def kzmin(self):
        return 2 * pi * fftfreq_min(self.nz, self.dz)

    @property
    def kzmax(self):
        return 2 * pi * fftfreq_max(self.nz, self.dz)

    # thetax = kx/k0
    @property
    def dthetax(self):
        """
        dthetax = wavelength / (nx * dx) = dkx/k0
        """
        return self.wavelength / (self.nx * self.dx)

    @property
    def thetaxvec(self):
        return self.kxvec / self.k0

    @property
    def thetaxmin(self):
        return self.kxmin / self.k0

    @property
    def thetaxmax(self):
        return self.kxmax / self.k0

    # thetay = ky/k0
    @property
    def dthetay(self):
        """
        dthetay = wavelength / (nx * dy) = dky/k0
        """
        return self.wavelength / (self.ny * self.dy)

    @property
    def thetayvec(self):
        return self.kyvec / self.k0

    @property
    def thetaymin(self):
        return self.kymin / self.k0

    @property
    def thetaymax(self):
        return self.kymax / self.k0

    # bools
    @property
    def in_rspace(self):
        return self.spatial_domain == SpatialDomain.R

    @property
    def in_kspace(self):
        return self.spatial_domain == SpatialDomain.K

    def _mean(self, key):
        """
        mean of a standard key

        Internal method
        """
        axis = _axis_for_sum[key]
        P = np.sum(self.intensity, axis=axis)
        x = getattr(self, key + "vec")
        return mean_calc(x, P)

    def _std(self, key):
        """
        Standard deviation of a standard key

        Internal method
        """
        axis = _axis_for_sum[key]
        P = np.sum(self.intensity, axis=axis)
        x = getattr(self, key + "vec")
        _, variance = mean_variance_calc(x, P)
        return np.sqrt(variance)

    def pad(self, nx=(0, 0), ny=(0, 0), nz=(0, 0)):
        """
        zero-pad the field arrays.

        If only a sincle number is given, the array will be symmetrically padded
        by this
        """

        nx = (nx, nx) if np.isscalar(nx) else nx
        ny = (ny, ny) if np.isscalar(ny) else ny
        nz = (nz, nz) if np.isscalar(nz) else nz

        Ex = np.pad(self.Ex, (nx, ny, nz)) if self.Ex is not None else None
        Ey = np.pad(self.Ey, (nx, ny, nz)) if self.Ey is not None else None

        return replace(self, Ex=Ex, Ey=Ey)

    def copy(self):
        """Returns a deep copy"""
        return deepcopy(self)


@dataclass
class WavefrontK(WavefrontBase):
    Ex: np.ndarray = None  # V m^2
    Ey: np.ndarray = None  # V m^2

    dkx: float = 1  # rad/m
    dky: float = 1  # rad/m
    dkz: float = 1  # rad/m

    wavelength: float = 1  # m

    spatial_domain = SpatialDomain.K
    axis_labels = ("kx", "ky", "kz")

    # Everything else is computed on the  fly

    # kx
    @property
    def dx(self):
        """
        grid spacing in m
        dx = 2π / (nx * dkx)
        """
        return 2 * pi / (self.nx * self.dkx)

    # ky
    @property
    def dy(self):
        """
        grid spacing in m
        dy = 2π / (nx * dky)
        """
        return 2 * pi / (self.ny * self.dky)

    # kz
    @property
    def dz(self):
        """
        grid spacing in m
        dz = 2π / (nx * dkz)
        """
        return 2 * pi / (self.nz * self.dkz)

    def to_rspace(self):
        """
        See Wavefront.to_kspace()
        """
        # Normalized for the Plancherel theorem (see def energy)
        norm = (
            self.dx
            * self.dy
            * self.dz
            * np.sqrt(self.nx * self.ny * self.nz / (2 * pi) ** 3)
        )
        Ex = (
            fftshift(ifftn(ifftshift(self.Ex), norm="ortho")) / norm
            if self.Ex is not None
            else None
        )
        Ey = (
            fftshift(ifftn(ifftshift(self.Ey), norm="ortho")) / norm
            if self.Ey is not None
            else None
        )

        return Wavefront(
            Ex=Ex,
            Ey=Ey,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            wavelength=self.wavelength,
        )

    @property
    def spectral_energy_density(self):
        """
        Spectral energy density ϵ0/2 |Ẽ|^2 in J * m^3

        3D real array

        """
        Ex = self.Ex if self.Ex is not None else 0
        Ey = self.Ey if self.Ey is not None else 0

        return epsilon_0 / 2 * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2)

    @property
    def energy(self):
        """
        Total (time-averaged) energy in J

        energy = ϵ0/2  ∫∫∫ |E|^2 dx dy dz

               = ϵ0/2  ∫∫∫ |Ẽ|^2 dkx dky dkz
        """

        return np.sum(self.spectral_energy_density) * self.dkx * self.dky * self.dkz

    @property
    def spectral_fluence(self):
        """
        K-space Fluence in the z direction:

        F(kx,ky) = ϵ0/2  ∫∫∫ |Ẽ(kx,ky,kz)|^2 dkz in J * m^2
        """
        return self.dkz * np.sum(self.intensity, axis=2) * epsilon_0 / 2

    @property
    def intensity_x(self):
        """
        x polarization field intensity in ??
        Intensity ~ |Ẽx|^2
        """
        return np.abs(self.Ex) ** 2 if self.Ex is not None else 0

    @property
    def intensity_y(self):
        """
        y polarization field intensity in ??
        Intensity ~ |Ẽy|^2
        """
        return np.abs(self.Ey) ** 2 if self.Ey is not None else 0

    @property
    def intensity(self):
        """
        total field intensity in ??
        ~ (|Ẽx|^2 + |Ẽy|^2)
        """
        return self.intensity_x + self.intensity_y

    def plot(self, cmap="inferno", logscale=False):
        """
        Simple projected intensity plot

        """

        xlabel = r"$\theta_x$ (µrad)"
        ylabel = r"$\theta_y$ (µrad)"
        xfactor = 1e6
        yfactor = 1e6
        zfactor = self.k0**2 / (1e6 * 1e6)
        label = r"Spectral $F$ (J/µrad$^2$)"

        extent = (
            self.thetaxmin * xfactor,
            self.thetaxmax * xfactor,
            self.thetaymin * yfactor,
            self.thetaymax * yfactor,
        )

        # Alternatively:
        # extent = (self.kxmin, self.kxmax, self.kymin, self.kymax)
        # xlabel = r'$k_x$ (rad/m)'
        # ylabel = r'$k_y$ (rad/m)'
        # zfactor = 1
        # label = r"Spectral $F$ (J$\cdot$m$^2$)"

        F = zfactor * self.spectral_fluence
        Fmax = np.max(F)

        fig, ax = plt.subplots()
        im = ax.imshow(
            F.T,
            cmap=cmap,
            extent=extent,
            origin="lower",
        )  # Note data.T and origin='lower' are required
        if logscale:
            norm = LogNorm(vmin=Fmax / 1e6, vmax=Fmax)
            im.set_norm(norm)

        fig.colorbar(im, ax=ax, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Statistics

    @property
    def mean_kx(self):
        """
        <kx> in rad/m
        """
        return self._mean("kx")

    @property
    def mean_ky(self):
        """
        <ky> in rad/m
        """
        return self._mean("ky")

    @property
    def mean_kz(self):
        """
        <kz> in rad/m
        """
        return self._mean("kz")

    @property
    def mean_thetax(self):
        """
        <thetax> in rad
        """
        return self._mean("thetax")

    @property
    def mean_thetay(self):
        """
        <thetay> in rad
        """
        return self._mean("thetay")

    @property
    def sigma_kx(self):
        """
        sqrt(<kx^2> - <kx>^2) in rad/m
        """
        return self._std("kx")

    @property
    def sigma_ky(self):
        """
        sqrt(<ky^2> - <ky>^2) in rad/m
        """
        return self._std("ky")

    @property
    def sigma_kz(self):
        """
        sqrt(<kz^2> - <kz>^2) in rad/m
        """
        return self._std("kz")

    @property
    def sigma_thetax(self):
        """
        sqrt(<thetax^2> - <thetax>^2) in rad
        """
        return self._std("thetax")

    @property
    def sigma_thetay(self):
        """
        sqrt(<thetay^2> - <thetay>^2) in rad
        """
        return self._std("thetay")


@dataclass
class Wavefront(WavefrontBase):
    """


    Principles
    ----------

    The code should be as simple as possible,
    with standard physics definitions and units for common symbols.

    Prefer explicit properties with equations directly in the functions.

    real space mesh spacing in meters
    k space mesh spacing in rad/meter,
    and are defined by real spacings and the shape of the 3d array

    Any other units can be simply converted from these from:
    c = omega/k
    E = hbar*omega = hbar*k*c
    etc.

    Only one representation for fields is stored: real space or k-space.
    The user can check with:
    .in_kspace
    .in_rspace
    or just look at .spatial_domain

    Fourier transforms have various conventions. Because of this, we cannot know
    the meaning of the field data in k-space without knowing the convention to convert
    to real-space.

    Use norm='ortho' in FFTs so that sum |E|^2 is the same value in both representations

    The fields can be padded directly as needed. No other data (spacing) needs to be changed.

    """

    Ex: np.ndarray = None  # V/m
    Ey: np.ndarray = None  # V/m

    dx: float = 1  # m
    dy: float = 1  # m
    dz: float = 1  # m

    wavelength: float = 1  # m

    spatial_domain = SpatialDomain.R
    axis_labels = ("x", "y", "z")

    # Everything else is computed on the  fly

    @property
    def intensity_x(self):
        """
        x polarization field intensity in W/m^2
        Intensity = c ϵ0/2 |Ex|^2
        """
        return c * epsilon_0 / 2 * np.abs(self.Ex) ** 2 if self.Ex is not None else 0

    @property
    def intensity_y(self):
        """
        y polarization field intensity in W/m^2
        Intensity = c ϵ0/2 |Ey|^2
        """
        return c * epsilon_0 / 2 * np.abs(self.Ey) ** 2 if self.Ey is not None else 0

    @property
    def intensity(self):
        """
        total field intensity in W/m^2
        I = c ϵ0/2 (|Ex|^2 + |Ey|^2)
        """
        return self.intensity_x + self.intensity_y

    @property
    def energy_density(self):
        """
        Energy density ϵ0/2 |E|^2 in J/m^3

        3D real array

        """
        Ex = self.Ex if self.Ex is not None else 0
        Ey = self.Ey if self.Ey is not None else 0

        return epsilon_0 / 2 * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2)

    @property
    def energy(self):
        """
        Total (time-averaged) energy in J

        energy = ϵ0/2  ∫∫∫ |E|^2 dx dy dz

               = ϵ0/2  ∫∫∫ |Ẽ|^2 dkx dky dkz
        """

        return np.sum(self.energy_density) * self.dx * self.dy * self.dz

    @property
    def fluence(self):
        """
        Fluence in the z direction:

        F(x,y) = ϵ0/2  ∫∫∫ |E(x,y,z)|^2 dz
        """
        return self.dz * np.sum(self.intensity, axis=2) / c  # J / m^2

    # @property
    # def marginal_intensity_x(self):
    #    """
    #    Marginal distribution of intensity along x (integrate over y, z) in W
    #    """
    #    return self.I.sum(axis=(1, 2)) * self.dy * self.dz

    # Representations
    def to_kspace(self, backend=np):
        """
        Transform to k-space according to the Fourier transform convention:

        Ẽ(kx,ky,kz) = 1/(2π)^(2/3) ∫∫∫ E(x,y,z) exp(-i kx x) exp(-i ky y) exp(-i kz z) dx dy dz

        E(x,y,z)   =  1/(2π)^(2/3) ∫∫∫ Ẽ(kx, ky, kz) exp(i kx x) exp(i ky y) exp(i kz z) dkx dky dkz

        This ensures that that Plancherel's theorm applies without any addition factors
            ∫∫∫ |E(x, y, z)|^2 dx dy dz =  ∫∫∫ |Ẽ(kx, ky, kz)|^2 dkx dky dkz

        And therefore ϵ0/2 |Ẽ|^2 can be directly interpreted as the spectral energy density.

        This is realized with factors so that:
        Ẽ = [dx sqrt(nx/(2π))] [...] [...]  fftn(E, norm = 'ortho') in units of V * m^2

        E = [dkx sqrt(nx/(2π))] [...] [...]  fftn(Ẽ, norm = 'ortho') in units of V/m

        Here  [...] is similar for y, z, etc. We use `norm='ortho'` to simplify the symmetry in the code.

        """

        fftn = backend.fft.fftn
        fftshift = backend.fft.fftshift
        ifftshift = backend.fft.ifftshift

        # Normalized for the Plancherel theorem (see energy def)
        norm = (
            self.dx
            * self.dy
            * self.dz
            * np.sqrt(self.nx * self.ny * self.nz / (2 * pi) ** 3)
        )
        Ex = (
            fftshift(fftn(ifftshift(self.Ex), norm="ortho")) * norm
            if self.Ex is not None
            else None
        )
        Ey = (
            fftshift(fftn(ifftshift(self.Ey), norm="ortho")) * norm
            if self.Ey is not None
            else None
        )

        return WavefrontK(
            Ex=Ex,
            Ey=Ey,
            dkx=self.dkx,
            dky=self.dky,
            dkz=self.dkz,
            wavelength=self.wavelength,
        )

    def plot(self, cmap="inferno", logscale=False):
        """
        Simple fluence plot

        """

        xlabel = r"$x$ (cm)"
        ylabel = r"$y$ (cm)"
        xfactor = 100
        yfactor = 100
        zfactor = 1 / (100 * 100)  # 1/m^2 -> 1/cm^2
        label = r"$F$ (J/cm$^2$)"
        extent = (
            self.xmin * xfactor,
            self.xmax * xfactor,
            self.ymin * yfactor,
            self.ymax * yfactor,
        )

        F = self.fluence * zfactor
        Fmax = np.max(F)

        fig, ax = plt.subplots()
        im = ax.imshow(
            F.T,
            cmap=cmap,
            extent=extent,
            origin="lower",
        )  # Note data.T and origin='lower' are required
        if logscale:
            norm = LogNorm(vmin=Fmax / 1e6, vmax=Fmax)
            im.set_norm(norm)

        fig.colorbar(im, ax=ax, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot2(self, cmap="inferno", logscale=False):
        """
        Simple fluence plot

        """

        # xlabel = r"$x$ (cm)"
        # ylabel = r"$y$ (cm)"
        xfactor = 100
        yfactor = 100
        zfactor = 1 / (100 * 100)  # 1/m^2 -> 1/cm^2
        # label = r"$F$ (J/cm$^2$)"
        F = self.fluence

        plot_2d_density_with_marginals(
            F * zfactor,
            dx=self.dx * xfactor,
            dy=self.dy * yfactor,
            xmin=self.xmin * xfactor,
            ymin=self.ymin * yfactor,
            x_name=r"$x$",
            x_units="cm",
            y_name=r"$y$",
            y_units="cm",
            z_name=r"$F$",
            z_units="J/cm$^2$",
        )

        # if logscale:
        # Fmax = np.max(F)
        #    norm = LogNorm(vmin=Fmax / 1e6, vmax=Fmax)
        #    im.set_norm(norm)

    # Statistics

    @property
    def mean_x(self):
        """
        <x> in meters
        """
        return self._mean("x")

    @property
    def mean_y(self):
        """
        <y> in meters
        """
        return self._mean("y")

    @property
    def mean_z(self):
        """
        <y> in meters
        """
        return self._mean("z")

    @property
    def sigma_x(self):
        """
        sqrt(<x^2> - <x>^2) in meters
        """
        return self._std("x")

    @property
    def sigma_y(self):
        """
        sqrt(<y^2> - <y>^2) in meters
        """
        return self._std("y")

    @property
    def sigma_z(self):
        """
        sqrt(<z^2> - <z>^2) in meters
        """
        return self._std("z")
