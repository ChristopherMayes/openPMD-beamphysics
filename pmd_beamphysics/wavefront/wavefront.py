from abc import ABC
from enum import Enum
from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Union, ClassVar

from math import pi
import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift, ifftn

from scipy.constants import epsilon_0, c

import h5py
import pathlib

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pmd_beamphysics.statistics import mean_calc, mean_variance_calc
from pmd_beamphysics.plot import plot_2d_density_with_marginals
from pmd_beamphysics.units import Z0
from pmd_beamphysics.interfaces.genesis import (
    wavefront_write_genesis4,
    load_genesis4_fields,
)


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

    spatial_domain: ClassVar[SpatialDomain] = SpatialDomain.K
    axis_labels: ClassVar[tuple[str, str, str]] = ("kx", "ky", "kz")

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

    - The code should be as simple as possible, with standard physics definitions
      and units for common symbols.

    - Prefer explicit properties with equations directly in the functions.

    - Real-space mesh spacing is in meters; k-space mesh spacing is in rad/meter,
      both defined by the grid shape and spacings (dx, dy, dz).

    - Only one representation for the fields is stored at a time (either real or k-space).
      The user can check with `.in_kspace`, `.in_rspace`, or `.spatial_domain`.

    - Fourier transforms have multiple conventions. This implementation uses `norm='ortho'`
      to ensure that the total field energy (i.e., sum |E|^2) is preserved between
      real and k-space representations (Plancherel's theorem).

    - The head of the laser pulse is at `max(z)`. The pulse propagates in the +z direction.
      Avoid referring to time coordinates `t = ±z/c` to prevent confusion about
      the direction of time and the head/tail of the pulse.

    - The fields can be padded as needed. No other metadata (spacing, axes) must be adjusted.

    - Other physical quantities can be derived from the basic spatial and spectral variables
      using standard relations such as:

        c = omega / k
        E = hbar * omega = hbar * k * c

      These can be used to convert between spatial, temporal, and energy units as needed.


    """

    Ex: np.ndarray = None  # V/m
    Ey: np.ndarray = None  # V/m

    dx: float = 1  # m
    dy: float = 1  # m
    dz: float = 1  # m

    wavelength: float = 1  # m

    spatial_domain: ClassVar[SpatialDomain] = SpatialDomain.R
    axis_labels: ClassVar[tuple[str, str, str]] = ("x", "y", "z")

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

        F(x,y) = ϵ0/2  ∫ |E(x,y,z)|^2 dz
        """
        return self.dz * np.sum(self.intensity, axis=2) / c  # J / m^2

    @property
    def fluence_profile_x(self):
        """
        1D fluence profile along x, integrated over y and z (J/m).

        F_x(x) = ϵ0/2  ∫∫ |E(x,y,z)|^2 dy dz
        """

        return (
            np.sum(self.intensity, axis=_axis_for_sum["x"]) * self.dy * self.dz / c
        )  # J/m

    @property
    def fluence_profile_y(self):
        """
        1D fluence profile along y, integrated over x and z (J/m).

        F_y(y) = ϵ0/2  ∫∫ |E(x,y,z)|^2 dx dz
        """

        return (
            np.sum(self.intensity, axis=_axis_for_sum["y"]) * self.dx * self.dz / c
        )  # J/m

    @property
    def power(self):
        """
        Longitudinal power profile along z (W).

        P(z) = ∫∫ I(x, y, z) dx dy
             = ∫∫ (c ϵ0 / 2) |E(x, y, z)|² dx dy
        """
        return np.sum(self.intensity, axis=_axis_for_sum["z"]) * self.dx * self.dy  # W

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

    @classmethod
    def from_genesis4(
        cls,
        file: Union[pathlib.Path, str, h5py.Group],
    ):
        """
        Create a Wavefront instance from a Genesis4 field file.

        This method reads a Genesis4 field file (either from disk or an open HDF5 group)
        and initializes a `Wavefront` instance with the extracted field data.

        Parameters
        ----------
        file : Union[pathlib.Path, str, h5py.Group]
            The path to the Genesis4 HDF5 field file or an open `h5py.Group`.

        Returns
        -------
        Wavefront
            A new instance of `Wavefront` initialized with field data from the Genesis4 file.

        Raises
        ------
        ValueError
            If the provided file is not a valid path, string, or `h5py.Group`.

        Notes
        -----
        - The field data is extracted and converted into an electric field representation.
        - The grid spacing (`dx` and `dz`) and wavelength are obtained from the file metadata.
        """

        if isinstance(file, (str, pathlib.Path)):
            with h5py.File(file, "r") as h5:
                dfl, param = load_genesis4_fields(h5)

        elif isinstance(file, h5py.Group):
            dfl, param = load_genesis4_fields(h5)
        else:
            raise ValueError(f"{file=} must be a str, pathlib.Path, or h5py.Group")

        dx = param["gridsize"]
        dz = param["slicespacing"]
        wavelength = param["wavelength"]

        Ex = dfl * np.sqrt(2 * Z0) / dx

        return cls(Ex=Ex, dx=dx, dy=dx, dz=dz, wavelength=wavelength)

    def write_genesis4(
        self,
        file: Union[pathlib.Path, str, h5py.Group],
    ):
        """
        Write the Wavefront field data to a Genesis4-style HDF5 file.

        This method saves the wavefront field data into a Genesis4-compatible HDF5 file,
        either by creating a new file on disk or writing into an existing HDF5 group.

        Parameters
        ----------
        file : Union[pathlib.Path, str, h5py.Group]
            The path to the output HDF5 file or an open `h5py.Group` for writing.

        Raises
        ------
        ValueError
            If the provided file is not a valid path, string, or `h5py.Group`.

        Notes
        -----
        - If `file` is a path or string, a new HDF5 file is created and written to.
        - If `file` is an `h5py.Group`, the data is written directly into the provided group.
        """
        if isinstance(file, (str, pathlib.Path)):
            with h5py.File(file, "w") as h5:
                wavefront_write_genesis4(self, h5)
            return

        if isinstance(file, h5py.Group):
            wavefront_write_genesis4(self, h5)

        raise ValueError(type(file))  # type: ignore[unreachable]
