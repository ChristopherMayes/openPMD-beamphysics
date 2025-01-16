from abc import ABC
from enum import Enum
from dataclasses import dataclass, replace

from math import pi
import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift, fftn, ifftn

from scipy.constants import epsilon_0, c

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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
        # Normalized for the Plancherel theorem (see def energy)
        # (see to_kspace)
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
    def energy(self):
        """
        Total (time-averaged) energy in J

        energy = ϵ0/2  ∫∫∫ |E|^2 dx dy dz

               = ϵ0/2  ∫∫∫ |Ehat|^2 dkx dky dkz
        """
        Ex = self.Ex if self.Ex is not None else 0
        Ey = self.Ey if self.Ey is not None else 0
        sum_E2 = np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)

        return sum_E2 * self.dkx * self.dky * self.dkz * epsilon_0 / 2

    @property
    def Ix(self):
        """
        x polarization field intensity in ??
        Intensity ~ |Ex|^2
        """
        return np.abs(self.Ex) ** 2 if self.Ex is not None else 0

    @property
    def Iy(self):
        """
        y polarization field intensity in ??
        Intensity ~ |Ey|^2
        """
        return c * epsilon_0 / 2 * np.abs(self.Ex) ** 2 if self.Ex is not None else 0

    @property
    def intensity(self):
        """
        total field intensity in ??
        ~ (|Ex|^2 + |Ey|^2)
        """
        return self.Ix + self.Iy

    def plot(self, cmap="inferno", logscale=False):
        """
        Simple projected intensity plot

        """

        extent = (self.thetaxmin, self.thetaxmax, self.thetaymin, self.thetaymax)
        xlabel = r"$\theta_x$ (rad)"
        ylabel = r"$\theta_y$ (rad)"

        # Alternatively:
        # extent = (self.kxmin, self.kxmax, self.kymin, self.kymax)
        # xlabel = r'$k_x$ (rad/m)'
        # ylabel = r'$k_y$ (rad/m)'
        zfactor = self.dkz
        label = ""  # r'$I2$ (W/m?)' # TODO

        Ixy = zfactor * np.sum(self.intensity, axis=2)
        Imax = np.max(Ixy)

        fig, ax = plt.subplots()
        im = ax.imshow(
            Ixy.T,
            cmap=cmap,
            extent=extent,
            origin="lower",
        )  # Note data.T and origin='lower' are required
        if logscale:
            norm = LogNorm(vmin=Imax / 1e6, vmax=Imax)
            im.set_norm(norm)

        fig.colorbar(im, ax=ax, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


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
    def Ix(self):
        """
        x polarization field intensity in W/m^2
        Intensity = c ϵ0/2 |Ex|^2
        """
        return c * epsilon_0 / 2 * np.abs(self.Ex) ** 2 if self.Ex is not None else 0

    @property
    def Iy(self):
        """
        y polarization field intensity in W/m^2
        Intensity = c ϵ0/2 |Ey|^2
        """
        return c * epsilon_0 / 2 * np.abs(self.Ex) ** 2 if self.Ex is not None else 0

    @property
    def intensity(self):
        """
        total field intensity in W/m^2
        c ϵ0/2 (|Ex|^2 + |Ey|^2)
        """
        return self.Ix + self.Iy

    # @property
    # def marginal_intensity_x(self):
    #    """
    #    Marginal distribution of intensity along x (integrate over y, z) in W
    #    """
    #    return self.I.sum(axis=(1, 2)) * self.dy * self.dz

    # Representations
    def to_kspace(self):
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

    @property
    def energy(self):
        """
        Total (time-averaged) energy in J

        energy = ϵ0/2  ∫∫∫ |E|^2 dx dy dz

               = ϵ0/2  ∫∫∫ |Ehat|^2 dkx dky dkz
        """
        Ex = self.Ex if self.Ex is not None else 0
        Ey = self.Ey if self.Ey is not None else 0
        sum_E2 = np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)

        return sum_E2 * self.dx * self.dy * self.dz * epsilon_0 / 2

    def plot(self, cmap="inferno", logscale=False):
        """
        Simple projected intensity plot

        """
        extent = (self.xmin, self.xmax, self.ymin, self.ymax)
        xlabel = r"$x$ (m)"
        ylabel = r"$y$ (m)"
        zfactor = self.dz
        label = ""  # r'$I$ (W/m?)' # TODO

        # else:
        #    extent = (self.thetaxmin, self.thetaxmax, self.thetaymin, self.thetaymax)
        #    xlabel = r'$\theta_x$ (rad)'
        #    ylabel = r'$\theta_y$ (rad)'
        #    # Alternatively:
        #    #extent = (self.kxmin, self.kxmax, self.kymin, self.kymax)
        #    #xlabel = r'$k_x$ (rad/m)'
        #    #ylabel = r'$k_y$ (rad/m)'
        #    zfactor = self.dkz
        #    label = '' #r'$I2$ (W/m?)' # TODO

        Ixy = zfactor * np.sum(self.intensity, axis=2)
        Imax = np.max(Ixy)

        fig, ax = plt.subplots()
        im = ax.imshow(
            Ixy.T,
            cmap=cmap,
            extent=extent,
            origin="lower",
        )  # Note data.T and origin='lower' are required
        if logscale:
            norm = LogNorm(vmin=Imax / 1e6, vmax=Imax)
            im.set_norm(norm)

        fig.colorbar(im, ax=ax, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    @property
    def sigma_x(self):
        P = np.sum(self.intensity, axis=(1, 2))
        P = P / np.sum(P)
        x = self.xvec

        mean = np.sum(x * P)
        variance = np.sum((x - mean) ** 2 * P)
        sigma = np.sqrt(variance)

        return sigma
