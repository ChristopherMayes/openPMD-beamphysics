from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Union, ClassVar

from math import pi
import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift, ifftn

from scipy.constants import epsilon_0, c, e, hbar

import h5py
import pathlib

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pmd_beamphysics.statistics import mean_calc, mean_variance_calc
from pmd_beamphysics.plot import plot_1d_density, plot_2d_density_with_marginals
from pmd_beamphysics.units import Z0, c_light
from pmd_beamphysics.interfaces.genesis import (
    wavefront_write_genesis4,
    load_genesis4_fields,
)

from pmd_beamphysics.wavefront.propagators import drift_wavefront


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


# Axes to sum over for intensity/probability projections.
# For projection onto axis i, sum over all other axes.
_axis_for_sum = (
    {key: (1, 2) for key in ("x", "kx", "thetax")}
    | {key: (0, 2) for key in ("y", "ky", "thetay")}
    | {key: (0, 1) for key in ("z", "kz", "thetaz")}
)


class WavefrontBase(ABC):
    """
    Convenience functions for standard domain values
    """

    Ex: np.ndarray | None = None
    Ey: np.ndarray | None = None

    wavelength: float = 1.0  # m
    axis_labels: ClassVar[tuple[str, str, str]] = ("", "", "")

    def __post_init__(self):
        """
        Validate inputs after dataclass initialization
        """
        # Validate field shapes match
        if self.Ex is not None and self.Ey is not None:
            if self.Ex.shape != self.Ey.shape:
                raise ValueError(
                    f"Ex shape {self.Ex.shape} != Ey shape {self.Ey.shape}"
                )

        # Validate at least one field exists
        if self.Ex is None and self.Ey is None:
            raise ValueError("At least one of Ex or Ey must be provided")

        # Validate positive spacing
        for attr in ["dx", "dy", "dz", "wavelength"]:
            val = getattr(self, attr)
            if val <= 0:
                raise ValueError(f"{attr} must be positive, got {val}")

        ## Validate field is complex
        # for field, name in [(self.Ex, 'Ex'), (self.Ey, 'Ey')]:
        #    if field is not None and not np.iscomplexobj(field):
        #        raise TypeError(f"{name} must be complex dtype, got {field.dtype}")

    @property
    @abstractmethod
    def spatial_domain(self) -> SpatialDomain: ...

    @property
    @abstractmethod
    def intensity(self) -> np.ndarray | float: ...

    @property
    @abstractmethod
    def dkx(self) -> float: ...

    @property
    @abstractmethod
    def dky(self) -> float: ...

    @property
    @abstractmethod
    def dkz(self) -> float: ...

    @property
    @abstractmethod
    def dx(self) -> float: ...

    @property
    @abstractmethod
    def dy(self) -> float: ...

    @property
    @abstractmethod
    def dz(self) -> float: ...

    @property
    def shape(self):
        """
        Shape of the field arrays as (nx, ny, nz).
        """
        if self.Ex is None:
            if self.Ey is None:
                raise ValueError("Neither Ex nor Ey is set")
            return self.Ey.shape
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
        """
        Central wavenumber in rad/m.

        k0 = 2π / wavelength
        """
        return 2 * pi / self.wavelength

    @property
    def photon_energy(self) -> float:
        """
        Central photon energy in eV
        """
        return self.k0 * hbar * c / e

    @property
    def nx(self):
        """Number of grid points in the x direction."""
        return self.shape[0]

    @property
    def ny(self):
        """Number of grid points in the y direction."""
        return self.shape[1]

    @property
    def nz(self):
        """Number of grid points in the z direction."""
        return self.shape[2]

    # x
    @property
    def xmin(self):
        """
        Minimum x coordinate in m.

        xmin = -(nx-1) * dx / 2
        """
        return -((self.nx - 1) * self.dx) / 2

    @property
    def xmax(self):
        """
        Maximum x coordinate in m.

        xmax = (nx-1) * dx / 2
        """
        return ((self.nx - 1) * self.dx) / 2

    @property
    def xvec(self):
        """
        Array of x coordinates in m.
        """
        return np.linspace(self.xmin, self.xmax, self.nx)

    # y
    @property
    def ymin(self):
        """
        Minimum y coordinate in m.

        ymin = -(ny-1) * dy / 2
        """
        return -((self.ny - 1) * self.dy) / 2

    @property
    def ymax(self):
        """
        Maximum y coordinate in m.

        ymax = (ny-1) * dy / 2
        """
        return ((self.ny - 1) * self.dy) / 2

    @property
    def yvec(self):
        """
        Array of y coordinates in m.
        """
        return np.linspace(self.ymin, self.ymax, self.ny)

    # z
    @property
    def zmin(self):
        """
        Minimum z coordinate in m.

        zmin = -(nz-1) * dz / 2
        """
        return -((self.nz - 1) * self.dz) / 2

    @property
    def zmax(self):
        """
        Maximum z coordinate in m.

        zmax = (nz-1) * dz / 2
        """
        return ((self.nz - 1) * self.dz) / 2

    @property
    def zvec(self):
        """
        Array of z coordinates in m.
        """
        return np.linspace(self.zmin, self.zmax, self.nz)

    # kx
    @property
    def kxvec(self):
        """
        Array of kx (transverse wavenumber) values in rad/m.
        """
        return 2 * pi * fftshift(fftfreq(self.nx, d=self.dx))

    @property
    def kxmin(self):
        """
        Minimum kx value in rad/m.
        """
        return 2 * pi * fftfreq_min(self.nx, self.dx)

    @property
    def kxmax(self):
        """
        Maximum kx value in rad/m.
        """
        return 2 * pi * fftfreq_max(self.nx, self.dx)

    # ky

    @property
    def kyvec(self):
        """
        Array of ky (transverse wavenumber) values in rad/m.
        """
        return 2 * pi * fftshift(fftfreq(self.ny, d=self.dy))

    @property
    def kymin(self):
        """
        Minimum ky value in rad/m.
        """
        return 2 * pi * fftfreq_min(self.ny, self.dy)

    @property
    def kymax(self):
        """
        Maximum ky value in rad/m.
        """
        return 2 * pi * fftfreq_max(self.ny, self.dy)

    # kz

    @property
    def kzvec(self):
        """
        Array of kz (longitudinal wavenumber) values in rad/m.
        """
        return 2 * pi * fftshift(fftfreq(self.nz, d=self.dz))

    @property
    def kzmin(self):
        """
        Minimum kz value in rad/m.
        """
        return 2 * pi * fftfreq_min(self.nz, self.dz)

    @property
    def kzmax(self):
        """
        Maximum kz value in rad/m.
        """
        return 2 * pi * fftfreq_max(self.nz, self.dz)

    # thetax = kx/k0
    @property
    def dthetax(self):
        """
        Angular spacing in thetax direction in rad.

        dthetax = wavelength / (nx * dx) = dkx / k0
        """
        return self.wavelength / (self.nx * self.dx)

    @property
    def thetaxvec(self):
        """
        Array of thetax (angular deviation) values in rad.

        thetax = kx / k0
        """
        return self.kxvec / self.k0

    @property
    def thetaxmin(self):
        """
        Minimum thetax value in rad.
        """
        return self.kxmin / self.k0

    @property
    def thetaxmax(self):
        """
        Maximum thetax value in rad.
        """
        return self.kxmax / self.k0

    # thetay = ky/k0
    @property
    def dthetay(self):
        """
        Angular spacing in thetay direction in rad.

        dthetay = wavelength / (ny * dy) = dky / k0
        """
        return self.wavelength / (self.ny * self.dy)

    @property
    def thetayvec(self):
        """
        Array of thetay (angular deviation) values in rad.

        thetay = ky / k0
        """
        return self.kyvec / self.k0

    @property
    def thetaymin(self):
        """
        Minimum thetay value in rad.
        """
        return self.kymin / self.k0

    @property
    def thetaymax(self):
        """
        Maximum thetay value in rad.
        """
        return self.kymax / self.k0

    # bools
    @property
    def in_rspace(self):
        """
        True if the wavefront is in real space (position domain).
        """
        return self.spatial_domain == SpatialDomain.R

    @property
    def in_kspace(self):
        """
        True if the wavefront is in k-space (Fourier domain).
        """
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

    def drift(self, z, curvature=0):
        return drift_wavefront(self, z, curvature=curvature)

    def pad(self, nx=(0, 0), ny=(0, 0), nz=(0, 0)):
        """
        zero-pad the field arrays.

        If only a single number is given, the array will be symmetrically padded
        by this
        """

        nx = (nx, nx) if np.isscalar(nx) else nx
        ny = (ny, ny) if np.isscalar(ny) else ny
        nz = (nz, nz) if np.isscalar(nz) else nz

        Ex = np.pad(self.Ex, (nx, ny, nz)) if self.Ex is not None else None
        Ey = np.pad(self.Ey, (nx, ny, nz)) if self.Ey is not None else None

        return replace(self, Ex=Ex, Ey=Ey)

    def crop(self, nx=(0, 0), ny=(0, 0), nz=(0, 0)):
        """
        Crop the field arrays by removing elements from edges.

        Parameters
        ----------
        nx : int or tuple of (int, int), default=(0, 0)
            Number of elements to remove from (start, end) of first axis.
            If a single int, removes symmetrically from both sides.
        ny : int or tuple of (int, int), default=(0, 0)
            Number of elements to remove from (start, end) of second axis.
            If a single int, removes symmetrically from both sides.
        nz : int or tuple of (int, int), default=(0, 0)
            Number of elements to remove from (start, end) of third axis.
            If a single int, removes symmetrically from both sides.

        Returns
        -------
        Wavefront or WavefrontK
            New wavefront with cropped field arrays.

        Raises
        ------
        ValueError
            If crop amounts exceed array dimensions.
        """
        nx = (nx, nx) if np.isscalar(nx) else nx
        ny = (ny, ny) if np.isscalar(ny) else ny
        nz = (nz, nz) if np.isscalar(nz) else nz

        # Calculate slice indices
        x_start = nx[0]
        x_end = self.shape[0] - nx[1] if nx[1] > 0 else None
        y_start = ny[0]
        y_end = self.shape[1] - ny[1] if ny[1] > 0 else None
        z_start = nz[0]
        z_end = self.shape[2] - nz[1] if nz[1] > 0 else None

        # Validate crop amounts
        if nx[0] + nx[1] >= self.shape[0]:
            raise ValueError(
                f"Crop in x ({nx}) exceeds array dimension ({self.shape[0]})"
            )
        if ny[0] + ny[1] >= self.shape[1]:
            raise ValueError(
                f"Crop in y ({ny}) exceeds array dimension ({self.shape[1]})"
            )
        if nz[0] + nz[1] >= self.shape[2]:
            raise ValueError(
                f"Crop in z ({nz}) exceeds array dimension ({self.shape[2]})"
            )

        Ex = (
            self.Ex[x_start:x_end, y_start:y_end, z_start:z_end]
            if self.Ex is not None
            else None
        )
        Ey = (
            self.Ey[x_start:x_end, y_start:y_end, z_start:z_end]
            if self.Ey is not None
            else None
        )

        return replace(self, Ex=Ex, Ey=Ey)

    def auto_crop(self, threshold: float = 1e-6, apply: bool = True):
        """
        Determine symmetric crop values based on intensity profile thresholds.

        Analyzes the 1D intensity projections along each axis to find
        regions where the signal falls below a relative threshold, then
        computes symmetric crop amounts for each axis.

        Parameters
        ----------
        threshold : float, default=1e-6
            Relative threshold (fraction of maximum) below which data
            is considered negligible and can be cropped.
        apply : bool, default=True
            If True, apply the crop and return the cropped wavefront.
            If False, return only the crop amounts as a dictionary.

        Returns
        -------
        Wavefront, WavefrontK, or dict
            If apply=True: New wavefront with cropped field arrays.
            If apply=False: Dictionary with keys 'nx', 'ny', 'nz' containing
            the symmetric crop amounts for each axis.

        Examples
        --------
        >>> # Get crop amounts without applying
        >>> crop_info = w.auto_crop(threshold=1e-4, apply=False)
        >>> print(crop_info)  # {'nx': 10, 'ny': 8, 'nz': 5}

        >>> # Apply auto-crop directly
        >>> w_cropped = w.auto_crop(threshold=1e-4)

        >>> # Manual two-step process
        >>> crop_info = w.auto_crop(threshold=1e-4, apply=False)
        >>> w_cropped = w.crop(**crop_info)
        """

        def find_symmetric_crop(profile: np.ndarray, threshold: float) -> int:
            """Find symmetric crop amount for a 1D profile."""
            if profile.size == 0:
                return 0

            max_val = np.max(profile)
            if max_val == 0:
                return 0

            # Normalize and find where signal is above threshold
            normalized = profile / max_val
            above_threshold = normalized > threshold

            if not np.any(above_threshold):
                return 0

            # Find first and last indices above threshold
            indices = np.where(above_threshold)[0]
            first_idx = indices[0]
            last_idx = indices[-1]

            # Calculate symmetric crop (use minimum of both sides)
            crop_start = first_idx
            crop_end = len(profile) - 1 - last_idx

            return min(crop_start, crop_end)

        # Get 1D intensity projections along each axis
        intensity = self.intensity
        profile_x = np.sum(intensity, axis=_axis_for_sum[self.axis_labels[0]])
        profile_y = np.sum(intensity, axis=_axis_for_sum[self.axis_labels[1]])
        profile_z = np.sum(intensity, axis=_axis_for_sum[self.axis_labels[2]])

        # Analyze each axis
        nx = find_symmetric_crop(profile_x, threshold)
        ny = find_symmetric_crop(profile_y, threshold)
        nz = find_symmetric_crop(profile_z, threshold)

        crop_amounts = {"nx": nx, "ny": ny, "nz": nz}

        if apply:
            return self.crop(**crop_amounts)
        else:
            return crop_amounts

    def copy(self):
        """Returns a deep copy"""
        return deepcopy(self)

    def _repr_pretty_(self, p, cycle):
        """IPython/Jupyter pretty-print representation"""
        if cycle:
            p.text(f"{self.__class__.__name__}(...)")
            return

        def summarize_field(field, name):
            if field is None:
                return f"{name}: None"
            return f"{name}: {field.shape}"

        lines = [
            f"{self.__class__.__name__}(",
            f"  wavelength: {self.wavelength:.6e} m",
            f"  shape: {self.shape}",
            f"  spacing: dx={self.dx:.3e}, dy={self.dy:.3e}, dz={self.dz:.3e} m",
            f"  {summarize_field(self.Ex, 'Ex')}",
            f"  {summarize_field(self.Ey, 'Ey')}",
            ")",
        ]
        p.text("\n".join(lines))

    def _repr_html_(self):
        """Rich HTML representation for Jupyter notebooks"""
        # Determine which fields exist
        fields = []
        if self.Ex is not None:
            fields.append("Ex")
        if self.Ey is not None:
            fields.append("Ey")
        field_str = ", ".join(fields) if fields else "None"

        # Add photon energy for easier reference
        photon_energy_str = ""
        if hasattr(self, "photon_energy"):
            photon_energy_str = f"""
                <tr>
                    <td><b>photon energy</b></td>
                    <td>{self.photon_energy:.6e} eV</td>
                </tr>
            """

        fmt = ""

        html = f"""
        <div style="font-family: monospace; border: 1px solid #ccc; padding: 10px; max-width: 600px;">
            <h4 style="margin-top: 0;">{self.__class__.__name__}</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f0f0f0;">
                    <td><b>wavelength</b></td>
                    <td>{self.wavelength:{fmt}} m</td>
                </tr>
                {photon_energy_str}
                <tr>
                    <td><b>grid shape</b></td>
                    <td>{self.shape}</td>
                </tr>
                <tr style="background-color: #f0f0f0;">
                    <td><b>dx</b></td>
                    <td>{self.dx:{fmt}} m</td>
                </tr>
                <tr>
                    <td><b>dy</b></td>
                    <td>{self.dy:{fmt}} m</td>
                </tr>
                <tr style="background-color: #f0f0f0;">
                    <td><b>dz</b></td>
                    <td>{self.dz:{fmt}} m</td>
                </tr>
                <tr>
                    <td><b>fields</b></td>
                    <td>{field_str}</td>
                </tr>
            </table>
        </div>
        """
        return html


@dataclass
class WavefrontK(WavefrontBase):
    """
    K-space (Fourier) representation of electromagnetic wavefront fields.

    This class represents electromagnetic fields in k-space (spatial frequency domain)
    for spectral analysis and propagation. It stores complex electric field Fourier
    components Ẽx and Ẽy on a 3D reciprocal-space grid with uniform spacing.

    The Fourier transform convention follows:
    Ẽ(kx,ky,kz) = (2π)^(-3/2) ∫∫∫ E(x,y,z) exp(-i·k·r) dx dy dz

    Parameters
    ----------
    Ex : np.ndarray, optional
        x-polarized electric field Fourier component in V·m², shape (nx, ny, nz)
    Ey : np.ndarray, optional
        y-polarized electric field Fourier component in V·m², shape (nx, ny, nz)
    dkx : float, default=1.0
        Grid spacing in kx direction (rad/m)
    dky : float, default=1.0
        Grid spacing in ky direction (rad/m)
    dkz : float, default=1.0
        Grid spacing in kz direction (rad/m)
    wavelength : float, default=1.0
        Central wavelength (m)

    Attributes
    ----------
    spatial_domain : SpatialDomain
        Always returns SpatialDomain.K (k-space)
    intensity : np.ndarray
        Spectral intensity |Ẽ|² (arbitrary units)
    spectral_energy_density : np.ndarray
        Spectral energy density ε₀/2 |Ẽ|² in J·m³
    energy : float
        Total energy ε₀/2 ∫∫∫ |Ẽ|² dkx dky dkz in J
    spectral_fluence : np.ndarray
        K-space fluence F(kx,ky) = ε₀/2 ∫ |Ẽ|² dkz in J·m²
    photon_energy_vec : np.ndarray
        Photon energy axis from kz in eV
    photon_energy_spectrum : np.ndarray
        Photon spectral energy density dU/dE in J/eV
    k0 : float
        Central wavenumber 2π/λ in rad/m
    photon_energy : float
        Central photon energy ℏck₀ in eV

    Methods
    -------
    to_rspace()
        Transform to real-space representation (Wavefront)
    plot_spectral_intensity(cmap, logscale)
        Plot angular spectrum intensity F(θx, θy)
    plot_photon_energy_spectrum(xlim, ax)
        Plot photon energy spectrum dU/dE
    pad(nx, ny, nz)
        Zero-pad the field arrays
    crop(nx, ny, nz)
        Crop the field arrays by removing elements from edges
    auto_crop(threshold, apply)
        Determine and optionally apply symmetric crop based on intensity thresholds

    Notes
    -----
    - At least one of Ex or Ey must be provided
    - All spacing parameters (dkx, dky, dkz, wavelength) must be positive
    - Field arrays must be complex dtype
    - The spectral energy density ε₀/2|Ẽ|² preserves total energy with real space
    - Derived real-space grid spacing: dx = 2π/(nx·dkx)
    - Angular coordinates: θx = kx/k₀, θy = ky/k₀
    - Statistical moments (mean, sigma) are intensity-weighted in k-space
    """

    Ex: np.ndarray | None = None  # V m^2
    Ey: np.ndarray | None = None  # V m^2

    dkx: float = 1.0  # rad/m  # type: ignore[override]
    dky: float = 1.0  # rad/m  # type: ignore[override]
    dkz: float = 1.0  # rad/m  # type: ignore[override]

    wavelength: float = 1.0  # m

    axis_labels: ClassVar[tuple[str, str, str]] = ("kx", "ky", "kz")

    @property
    def spatial_domain(self) -> SpatialDomain:
        return SpatialDomain.K

    # Everything else is computed on the  fly

    # kx
    @property
    def dx(self) -> float:
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

    def to_rspace(self, *, inplace: bool = False) -> Wavefront:
        """
        See Wavefront.to_kspace()
        """
        if inplace:
            raise NotImplementedError("inplace not yet implemented")

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
    def photon_energy_vec(self) -> np.ndarray:
        """
        Photon energy axis from kz in eV
        """
        return (self.kzvec + self.k0) * hbar * c / e

    @property
    def photon_energy_spectrum(self) -> np.ndarray:
        """
        Photon energy spectrum dU/dE in J/eV

        dU/dE = (ε0/2) ∫∫ |Ẽ|² dkx dky  / (ħc/e)


        See .photon_energy_vec for the corresponding photon energies in eV.
        """
        u_kz_J_m = (
            np.sum(self.spectral_energy_density, axis=(0, 1)) * self.dkx * self.dky
        )  # J·m
        return u_kz_J_m / (hbar * c / e)  # → J/eV

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
        x-polarization spectral intensity in V²·m⁴.

        Spectral intensity |Ẽx|²
        """
        return np.abs(self.Ex) ** 2 if self.Ex is not None else 0

    @property
    def intensity_y(self):
        """
        y-polarization spectral intensity in V²·m⁴.

        Spectral intensity |Ẽy|²
        """
        return np.abs(self.Ey) ** 2 if self.Ey is not None else 0

    @property
    def intensity(self) -> np.ndarray | float:
        """
        Total spectral intensity in V²·m⁴.

        |Ẽx|² + |Ẽy|²
        """
        return self.intensity_x + self.intensity_y

    def plot_spectral_intensity(self, cmap="inferno", logscale=False):
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

    def plot_photon_energy_spectrum(self, xlim=None, ax=None):
        x = self.photon_energy_vec  # eV
        y = self.photon_energy_spectrum  # J/eV

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, y * 1e6, color="purple")
        ax.set_xlabel("photon energy (eV)")
        ax.set_ylabel("photon spectral energy density (µJ/eV)")
        ax.set_ylim(0, None)
        ax.set_xlim(xlim)

    # Statistics

    @property
    def mean_kx(self):
        """
        Intensity-weighted mean kx in rad/m.

        <kx> = ∫ kx |Ẽ|² dkx dky dkz / ∫ |Ẽ|² dkx dky dkz
        """
        return self._mean("kx")

    @property
    def mean_ky(self):
        """
        Intensity-weighted mean ky in rad/m.

        <ky> = ∫ ky |Ẽ|² dkx dky dkz / ∫ |Ẽ|² dkx dky dkz
        """
        return self._mean("ky")

    @property
    def mean_kz(self):
        """
        Intensity-weighted mean kz in rad/m.

        <kz> = ∫ kz |Ẽ|² dkx dky dkz / ∫ |Ẽ|² dkx dky dkz
        """
        return self._mean("kz")

    @property
    def mean_thetax(self):
        """
        Intensity-weighted mean thetax in rad.

        <θx> = <kx> / k0
        """
        return self._mean("thetax")

    @property
    def mean_thetay(self):
        """
        Intensity-weighted mean thetay in rad.

        <θy> = <ky> / k0
        """
        return self._mean("thetay")

    @property
    def sigma_kx(self):
        """
        RMS width in kx in rad/m.

        σ_kx = sqrt(<kx²> - <kx>²)
        """
        return self._std("kx")

    @property
    def sigma_ky(self):
        """
        RMS width in ky in rad/m.

        σ_ky = sqrt(<ky²> - <ky>²)
        """
        return self._std("ky")

    @property
    def sigma_kz(self):
        """
        RMS width in kz in rad/m.

        σ_kz = sqrt(<kz²> - <kz>²)
        """
        return self._std("kz")

    @property
    def sigma_thetax(self):
        """
        RMS angular width in thetax in rad.

        σ_θx = sqrt(<θx²> - <θx>²)
        """
        return self._std("thetax")

    @property
    def sigma_thetay(self):
        """
        RMS angular width in thetay in rad.

        σ_θy = sqrt(<θy²> - <θy>²)
        """
        return self._std("thetay")


@dataclass
class Wavefront(WavefrontBase):
    """
    Real-space representation of electromagnetic wavefront fields.

    This class represents electromagnetic fields in real space (position domain) for
    optical wavefront propagation and analysis. It stores complex electric field
    components Ex and Ey on a 3D Cartesian grid with uniform spacing.

    The wavefront propagates in the +z direction, with the pulse head at max(z).
    Field intensities, energies, and other derived quantities follow standard
    electromagnetic definitions using SI units.

    Parameters
    ----------
    Ex : np.ndarray, optional
        x-polarized electric field component in V/m, shape (nx, ny, nz)
    Ey : np.ndarray, optional
        y-polarized electric field component in V/m, shape (nx, ny, nz)
    dx : float, default=1.0
        Grid spacing in x direction (m)
    dy : float, default=1.0
        Grid spacing in y direction (m)
    dz : float, default=1.0
        Grid spacing in z direction (m)
    wavelength : float, default=1.0
        Central wavelength (m)

    Attributes
    ----------
    spatial_domain : SpatialDomain
        Always returns SpatialDomain.R (real space)
    intensity : np.ndarray
        Total field intensity c ε₀/2 (|Ex|² + |Ey|²) in W/m²
    energy : float
        Total field energy ε₀/2 ∫∫∫ |E|² dx dy dz in J
    fluence : np.ndarray
        Fluence F(x,y) = ε₀/2 ∫ |E(x,y,z)|² dz in J/m²
    power : np.ndarray
        Power profile P(z) = ∫∫ I(x,y,z) dx dy in W
    k0 : float
        Central wavenumber 2π/λ in rad/m
    photon_energy : float
        Central photon energy ℏck₀ in eV

    Methods
    -------
    to_kspace()
        Transform to k-space representation (WavefrontK)
    drift(z, curvature=0)
        Propagate wavefront by distance z in free space
    pad(nx, ny, nz)
        Zero-pad the field arrays
    crop(nx, ny, nz)
        Crop the field arrays by removing elements from edges
    auto_crop(threshold, apply)
        Determine and optionally apply symmetric crop based on intensity thresholds
    estimate_curvature(axis, polarization, ...)
        Estimate wavefront radius of curvature
    plot_power()
        Plot longitudinal power profile
    plot_fluence()
        Plot transverse fluence distribution
    from_genesis4(file)
        Create Wavefront from Genesis4 HDF5 field file
    write_genesis4(file)
        Write Wavefront to Genesis4 HDF5 format

    Notes
    -----
    - At least one of Ex or Ey must be provided
    - All spacing parameters (dx, dy, dz, wavelength) must be positive
    - Field arrays must be complex dtype
    - Fourier transforms use norm='ortho' to preserve Plancherel's theorem:
      ∫∫∫ |E(x,y,z)|² dx dy dz = ∫∫∫ |Ẽ(kx,ky,kz)|² dkx dky dkz
    - Statistical moments (mean, sigma) are intensity-weighted

    """

    Ex: np.ndarray | None = None  # V/m
    Ey: np.ndarray | None = None  # V/m

    dx: float = 1.0  # m  # type: ignore[override]
    dy: float = 1.0  # m  # type: ignore[override]
    dz: float = 1.0  # m  # type: ignore[override]
    wavelength: float = 1.0  # m

    axis_labels: ClassVar[tuple[str, str, str]] = ("x", "y", "z")

    @property
    def spatial_domain(self) -> SpatialDomain:
        return SpatialDomain.R

    # Everything else is computed on the  fly

    @property
    def intensity_x(self) -> np.ndarray | float:
        """
        x polarization field intensity in W/m^2
        Intensity = c ϵ0/2 |Ex|^2
        """
        return c * epsilon_0 / 2 * np.abs(self.Ex) ** 2 if self.Ex is not None else 0

    @property
    def intensity_y(self) -> np.ndarray | float:
        """
        y polarization field intensity in W/m^2
        Intensity = c ϵ0/2 |Ey|^2
        """
        return c * epsilon_0 / 2 * np.abs(self.Ey) ** 2 if self.Ey is not None else 0

    @property
    def intensity(self) -> np.ndarray | float:
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
        Transverse fluence profile in J/m².

        F(x,y) = ϵ0/2 ∫ |E(x,y,z)|² dz

        2D real array with shape (nx, ny).
        """
        return self.dz * np.sum(self.intensity, axis=2) / c  # J / m^2

    @property
    def fluence_profile_x(self):
        """
        1D fluence profile along x in J/m.

        F_x(x) = ϵ0/2 ∫∫ |E(x,y,z)|² dy dz

        Integrated over y and z directions.
        """

        return (
            np.sum(self.intensity, axis=_axis_for_sum["x"]) * self.dy * self.dz / c
        )  # J/m

    @property
    def fluence_profile_y(self):
        """
        1D fluence profile along y in J/m.

        F_y(y) = ϵ0/2 ∫∫ |E(x,y,z)|² dx dz

        Integrated over x and z directions.
        """

        return (
            np.sum(self.intensity, axis=_axis_for_sum["y"]) * self.dx * self.dz / c
        )  # J/m

    @property
    def power(self):
        """
        Longitudinal power profile along z in W.

        P(z) = ∫∫ I(x, y, z) dx dy = ∫∫ (c ϵ0/2) |E(x, y, z)|² dx dy

        1D real array with shape (nz,).
        """
        return np.sum(self.intensity, axis=_axis_for_sum["z"]) * self.dx * self.dy  # W

    # Representations
    def to_kspace(self, backend=np, *, inplace: bool = False) -> WavefrontK:
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

        if inplace:
            raise NotImplementedError("inplace not yet implemented")

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

    def plot_power(
        self,
        ax=None,
        ylim=(0, None),
        xlim=None,
        nice=True,
        log_scale_y=False,
        show_cdf=False,
    ):
        x = self.zvec / c
        y = self.power

        data = {"z/c": x, "power": y}

        return plot_1d_density(
            "z/c",
            "power",
            data=data,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            auto_label=True,
            show_cdf=show_cdf,
            log_scale_y=log_scale_y,
            plot_style={"color": "purple"},
            kind="bar",
            nice=nice,
        )

    def plot_fluence(self, cmap="inferno", logscale=False):
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

        Notes
        -----
        This is experimental.
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
            cmap=cmap,
            log_scale_marginals=logscale,
            log_scale_z=logscale,
        )

        # if logscale:
        # Fmax = np.max(F)
        #    norm = LogNorm(vmin=Fmax / 1e6, vmax=Fmax)
        #    im.set_norm(norm)

    @property
    def dkx(self) -> float:
        """
        Transverse wavevector spacing in kx direction in rad/m.

        dkx = 2π / (nx * dx)
        """
        return 2 * pi / (self.nx * self.dx)

    @property
    def dky(self):
        """
        Transverse wavevector spacing in ky direction in rad/m.

        dky = 2π / (ny * dy)
        """
        return 2 * pi / (self.ny * self.dy)

    @property
    def dkz(self):
        """
        Longitudinal wavevector spacing in kz direction in rad/m.

        dkz = 2π / (nz * dz)
        """
        return 2 * pi / (self.nz * self.dz)

    # Statistics

    @property
    def mean_x(self):
        """
        Intensity-weighted mean x position in m.

        <x> = ∫ x I(x,y,z) dx dy dz / ∫ I(x,y,z) dx dy dz
        """
        return self._mean("x")

    @property
    def mean_y(self):
        """
        Intensity-weighted mean y position in m.

        <y> = ∫ y I(x,y,z) dx dy dz / ∫ I(x,y,z) dx dy dz
        """
        return self._mean("y")

    @property
    def mean_z(self):
        """
        Intensity-weighted mean z position in m.

        <z> = ∫ z I(x,y,z) dx dy dz / ∫ I(x,y,z) dx dy dz
        """
        return self._mean("z")

    @property
    def sigma_x(self):
        """
        RMS beam size in x in m.

        σ_x = sqrt(<x²> - <x>²)
        """
        return self._std("x")

    @property
    def sigma_y(self):
        """
        RMS beam size in y in m.

        σ_y = sqrt(<y²> - <y>²)
        """
        return self._std("y")

    @property
    def sigma_z(self):
        """
        RMS pulse length in z in m.

        σ_z = sqrt(<z²> - <z>²)
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

        return cls(
            Ex=Ex,
            dx=float(dx),
            dy=float(dx),
            dz=float(dz),
            wavelength=float(wavelength),
        )

    def write_genesis4(
        self,
        file: pathlib.Path | str | h5py.Group,
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
        elif isinstance(file, h5py.Group):
            wavefront_write_genesis4(self, file)
            return
        else:
            raise ValueError(
                f"file must be a str, pathlib.Path, or h5py.Group, got {type(file)}"
            )

    def estimate_curvature(
        self,
        axis: str = "x",
        polarization: str | None = None,
        ix: int | None = None,
        iy: int | None = None,
        iz: int | None = None,
        plot: bool = False,
        cutoff: float = 1e-4,
    ) -> float | tuple[float, plt.Figure]:
        """
        Estimates the wavefront curvature by looking at the phase across an axis.

        The method fits a parabola to the unwrapped phase: phase = k r^2 / 2R + phi0
        where R is the radius of curvature and k is the wavenumber.

        Parameters
        ----------
        axis : str, default='x'
            Axis along which to estimate curvature ('x' or 'y')
        polarization : str, optional
            Polarization to analyze ('x' or 'y'). If None, uses first available.
        ix, iy, iz : int, optional
            Indices for slicing. iz defaults to peak intensity slice.
        plot : bool, default=False
            Whether to create a diagnostic plot
        cutoff : float, default=1e-4
            Relative intensity cutoff for data selection

        Returns
        -------
        float
            Curvature in 1/m.

        Raises
        ------
        ValueError
            If insufficient data points or invalid parameters
        """
        w = self

        if polarization is None:
            polarization = "x" if w.Ex is not None else "y"
        field = getattr(w, "E" + polarization)
        if field is None:
            raise ValueError(
                f"E{polarization} field is None, cannot use polarization='{polarization}'"
            )

        if iz is None:
            iz = np.sum(np.abs(field) ** 2, axis=(0, 1)).argmax()

        field_xy = field[:, :, iz]
        field_xy2 = np.abs(field_xy) ** 2

        if axis == "x":
            if iy is None:
                iy = np.sum(field_xy2, axis=1).argmax()

            if ix is not None:
                raise ValueError(f"{ix=} must be None when requesting axis='x'")

            weights = field_xy2[:, iy]

            phase = np.unwrap(np.angle(field_xy[:, iy]))

        elif axis == "y":
            if ix is None:
                ix = np.sum(field_xy2, axis=0).argmax()

            if iy is not None:
                raise ValueError(f"{iy=} must be None when requesting axis='y'")

            weights = field_xy2[ix, :]

            phase = np.unwrap(np.angle(field_xy[ix, :]))
        else:
            raise ValueError(f"{axis=} must be 'x' or 'y'")

        x = getattr(w, axis + "vec")
        y = phase - phase.min()

        weights = weights / weights.max()
        mask = weights > cutoff

        x = x[mask]
        y = y[mask]
        weights = weights[mask]

        if len(x) < 3:
            raise ValueError(
                f"Insufficient points ({len(x)}) after cutoff filtering for polynomial fit"
            )

        a, b, c = np.polyfit(x, y, 2, w=weights)

        # curvature = 1/R = a λ / π

        curvature = a * w.wavelength / np.pi

        if plot:
            fig, ax = plt.subplots()
            norm = 360 / (2 * np.pi)
            ax.plot(x, y * norm)

            radius = 1 / curvature if abs(curvature) > 1e-10 else np.inf
            label = f"curvature_{axis} = {curvature:0.6f} 1/m \n radius_{axis} = {radius:0.1f} m"
            ax.plot(x, (a * x**2 + b * x + c) * norm, "--", label=label)
            ax.set_xlabel(f"{axis} (m)")

            ax.set_ylabel("phase (deg)")
            # ax.set_ylim(0, None)
            ax.legend()

            ax2 = ax.twinx()
            ax2.fill_between(x, 0, weights / weights.max(), alpha=0.5, color="purple")
            ax2.set_ylim(0, 2)
            ax2.set_ylabel("weight")

        return curvature

    @classmethod
    def from_gaussian(
        cls,
        shape: tuple[int, int, int],
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        wavelength: float = 1.0,
        sigma0: float | None = None,
        z0: float = 0.0,
        x0: float = 0.0,
        y0: float = 0.0,
        sigma_z: float | None = None,
        energy: float = 1.0,
        phase: float = 0.0,
        polarization: str = "x",
    ) -> "Wavefront":
        """
        Create a Wavefront with a Gaussian beam.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Grid shape (nx, ny, nz)
        dx : float, default=1.0
            Grid spacing in x direction (m)
        dy : float, default=1.0
            Grid spacing in y direction (m)
        dz : float, default=1.0
            Grid spacing in z direction (m)
        wavelength : float, default=1.0
            Central wavelength (m)
        sigma0 : float, optional
            RMS transverse beam size at the waist in meters (round beam).
            Related to waist size w0 via: w0 = 2·σ₀
            Related to Rayleigh length via: z_R = 4π·σ₀² / λ = π·w0² / λ
        z0 : float, default=0.0
            Distance from waist position (m)
        x0 : float, default=0.0
            Beam center position in x (m)
        y0 : float, default=0.0
            Beam center position in y (m)
        sigma_z : float or None, default=None
            Longitudinal Gaussian width (m). If None, use constant longitudinal profile.
        energy : float, default=1.0
            Total beam energy in Joules
        phase : float, default=0.0
            Global phase offset (radians)
        polarization : str, default='x'
            Polarization direction ('x' or 'y')

        Returns
        -------
        Wavefront
            New Wavefront instance with Gaussian beam

        Raises
        ------
        ValueError
            If sigma0 is not specified
            If polarization is not 'x' or 'y'
            If sigma_z is negative

        Notes
        -----
        The Gaussian beam uses the complex beam parameter formalism:

        q(z) = z + i·z_R  where z_R = 4π·σ₀² / λ = π·w0² / λ

        u_xy(x,y,z) = (1/q) · exp(-i·k·(x² + y²)/(2q))

        For sigma_z > 0:
            u_z(z) = exp(-z²/(4σ_z²)) / √(√(2π)·σ_z)
        For sigma_z = None:
            u_z(z) = 1  (constant profile, normalized over grid)

        The field is normalized such that:
        ∫∫∫ |u|² dx dy dz = energy / (2·Z₀·c)

        The relationship between sigma0 and common Gaussian beam parameters:
        - Waist size (1/e² intensity radius): w0 = 2·σ₀
        - Rayleigh length: z_R = π·w0²/λ = 4π·σ₀²/λ

        References
        ----------
        - Wikipedia: Gaussian beam, Complex beam parameter
        - Siegman "Lasers" 1986, Chapter 16.3
        """
        if sigma0 is None:
            raise ValueError("sigma0 must be specified")

        if polarization not in ("x", "y"):
            raise ValueError(f"polarization must be 'x' or 'y', got '{polarization}'")

        if sigma_z is not None and sigma_z < 0:
            raise ValueError(f"sigma_z must be non-negative or None, got {sigma_z}")

        # Convert sigma0 to Rayleigh length: z_R = π·w0²/λ = 4π·σ₀²/λ where w0 = 2·σ₀
        zR = 4 * pi * sigma0**2 / wavelength

        # Create coordinate vectors
        nx, ny, nz = shape
        xvec = np.linspace(-((nx - 1) * dx) / 2, ((nx - 1) * dx) / 2, nx)
        yvec = np.linspace(-((ny - 1) * dy) / 2, ((ny - 1) * dy) / 2, ny)
        zvec = np.linspace(-((nz - 1) * dz) / 2, ((nz - 1) * dz) / 2, nz)

        X, Y, Z = np.meshgrid(xvec, yvec, zvec, indexing="ij")

        k = 2 * pi / wavelength

        # Complex beam parameter at position z0 from waist
        q = z0 + 1j * zR

        # Transverse Gaussian beam profile
        uxy = (1 / q) * np.exp(-0.5j * k * ((X - x0) ** 2 + (Y - y0) ** 2) / q)

        # Analytic integral of |u_xy|² over transverse plane
        integral_uxy_squared = pi / (k * zR)

        # Longitudinal profile
        if sigma_z is None or sigma_z == 0:
            # Constant longitudinal profile
            uz = np.ones_like(Z)
        else:
            # Gaussian longitudinal profile (square root of Gaussian in intensity)
            uz = np.sqrt(1 / np.sqrt(2 * pi) / sigma_z) * np.exp(
                -(Z**2) / (4 * sigma_z**2)
            )

        # Combined field, normalized by transverse integral
        u = uxy * uz / np.sqrt(integral_uxy_squared)

        # Exact normalization over the discrete grid
        integral2 = np.sum(np.abs(u) ** 2) * dx * dy * dz
        u = u / np.sqrt(integral2)

        # Scale for desired energy using c_light for consistency
        u = u * np.sqrt(energy * 2 * Z0 * c_light) * np.exp(1j * phase)

        # Create wavefront with appropriate polarization
        if polarization == "x":
            return cls(Ex=u, dx=dx, dy=dy, dz=dz, wavelength=wavelength)
        else:
            return cls(Ey=u, dx=dx, dy=dy, dz=dz, wavelength=wavelength)
