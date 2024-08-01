#!/usr/bin/env python

from __future__ import annotations
from functools import cached_property
import logging
import pydantic

from typing import Annotated, Any, List, Optional, Tuple

import numpy as np
import scipy.constants

import scipy.fft

logger = logging.getLogger(__name__)


def _pad_array(wavefront: np.ndarray, shape):
    """
    Pad an array with complex zero elements.

    Parameters
    ----------
    wavefront : np.ndarray
        The array to pad
    shape : array_like
        Number of values padded to the edges of each axis

    Returns
    -------
    np.ndarray

    """

    return np.pad(
        wavefront,
        shape,
        mode="constant",
        constant_values=(0.0 + 1j * 0.0, 0.0 + 1j * 0.0),
    )


def fft_phased(
    array: np.ndarray,
    axes: Tuple[int, ...],
    phasors,
    workers=-1,
) -> np.ndarray:
    array_fft = scipy.fft.fftn(array, axes=axes, workers=workers, norm="ortho")
    for phasor in phasors:
        array_fft *= phasor
    return scipy.fft.fftshift(array_fft, axes=axes)


def ifft_phased(
    array: np.ndarray,
    axes: Tuple[int, ...],
    phasors,
    workers=-1,
) -> np.ndarray:
    array_fft = scipy.fft.ifftn(array, axes=axes, workers=workers, norm="ortho")
    for phasor in phasors:
        array_fft *= np.conj(phasor)
    return scipy.fft.ifftshift(array_fft, axes=axes)


def nd_kspace_domains(coeffs, sizes, pads, steps, shifted=True):
    """
    Generate reciprocal space domains for given grid sizes and steps.

    Parameters
    ----------
    coeffs : tuple
        Conversion coefficients for eV, rad
    sizes : tuple
        Grid sizes
    pads : tuple
        Number of padding points for each axis
    steps : tuple
        Grid step sizes
    shifted : bool
        Flag if np.fft.fftshift is applied

    Returns
    -------
    np.ndarray
    """

    if not shifted:
        return [
            coeff * np.fft.fftfreq(n + 2 * pad, step)
            for coeff, n, pad, step in zip(coeffs, sizes, pads, steps)
        ]

    return [
        np.fft.fftshift(coeff * np.fft.fftfreq(n + 2 * pad, step))
        for coeff, n, pad, step in zip(coeffs, sizes, pads, steps)
    ]


def nd_kspace_mesh(coeffs=(), sizes=(), pads=(), steps=(), shifted=True):
    """
    Generate reciprocal space for given grid sizes and steps.

    Parameters
    ----------
    coeffs : tuple
        Conversion coefficients for eV, rad
    sizes : tuple
        Grid sizes
    pads : tuple
        Number of padding points for each axis
    steps : tuple
        Grid step sizes
    shifted : bool
        Flag if np.fft.fftshift is applied

    Returns
    -------
    np.ndarray
    """

    domains = nd_kspace_domains(coeffs, sizes, pads, steps, shifted=shifted)
    meshes = np.meshgrid(*domains, indexing="ij")
    return meshes


def nd_space_mesh(mins=(), maxes=(), sizes=()):
    """
    Generate real space for given grid sizes and steps.

    Parameters
    ----------
    mins : tuple
        Minima of grid sizes
    maxes : tuple
        Maxima of grid sizes
    sizes : tuple
        Number of grid points in each axis

    Returns
    -------
    np.ndarray
    """
    domains = [np.linspace(min_, max_, n) for min_, max_, n in zip(mins, maxes, sizes)]
    return np.meshgrid(*domains, indexing="ij")


def _fix_fft_dimension(dim: int):
    """Get a dimension that's efficient for the FFT - and also odd for symmetry."""
    while True:
        dim = scipy.fft.next_fast_len(dim, real=False)
        if dim % 2 == 1:
            break
        dim += 1
    return dim


OddInteger = Annotated[int, pydantic.AfterValidator(_fix_fft_dimension)]


class WavefrontParams(pydantic.BaseModel, frozen=True):
    lambda0: float

    half_size: float
    sigma_t: float

    tmax: float

    tgrid: OddInteger
    xgrid: OddInteger
    ygrid: OddInteger

    tpad: int
    xpad: int
    ypad: int

    @property
    def xmax(self) -> float:
        return self.half_size

    @property
    def ymax(self) -> float:
        return self.half_size

    @property
    def t0(self) -> float:
        return self.tmax / 2.0

    @property
    def k0(self) -> float:
        return 2.0 * np.pi / self.lambda0

    @cached_property
    def dt(self) -> float:
        """Grid delta t."""
        t, _, _ = self.domains_txy
        return t[1] - t[0]

    @cached_property
    def dx(self) -> float:
        """Grid delta x."""
        _, x, _ = self.domains_txy
        return x[1] - x[0]

    @cached_property
    def dy(self) -> float:
        """Grid delta y."""
        _, _, y = self.domains_txy
        return y[1] - y[0]

    @cached_property
    def dkx(self) -> float:
        """Drift kernel delta x."""
        kx = self.domains_kxky[0]
        return kx[1, 0] - kx[0, 0]

    @cached_property
    def dky(self) -> float:
        """Drift kernel delta y."""
        ky = self.domains_kxky[1]
        return ky[0, 1] - ky[0, 0]

    @cached_property
    def shifts(self) -> Tuple[float, float, float]:
        return (
            self.t0 + self.tpad * self.dt,
            self.xmax + self.xpad * self.dx,
            self.ymax + self.ypad * self.dy,
        )

    @cached_property
    def pad_shape(self) -> Tuple[Tuple[int, int], ...]:
        return (
            (self.tpad, self.tpad),
            (self.xpad, self.xpad),
            (self.ypad, self.ypad),
        )

    def get_padded_shape(self, field_rspace: np.ndarray) -> Tuple[int, int, int]:
        if not len(field_rspace.shape) == 3:
            raise ValueError("`field_rspace` is not a 3D array")

        nt, nx, ny = field_rspace.shape
        return (
            nt + 2 * self.tpad,
            nx + 2 * self.xpad,
            ny + 2 * self.ypad,
        )

    @cached_property
    def coeffs(self) -> Tuple[float, float, float]:
        hbar = scipy.constants.hbar / scipy.constants.e * 1.0e15  # fs-eV
        return (
            2.0 * np.pi * hbar,
            2.0 * np.pi / self.k0,
            2.0 * np.pi / self.k0,
        )

    @property
    def domains_txy(self):
        return (
            np.linspace(0.0, self.tmax, self.tgrid),
            np.linspace(-self.xmax, self.xmax, self.xgrid),
            np.linspace(-self.ymax, self.ymax, self.ygrid),
        )

    @property
    def domains_wkxky(self):
        return nd_kspace_mesh(
            coeffs=self.coeffs,
            sizes=(self.tgrid, self.xgrid, self.ygrid),
            pads=(self.tpad, self.xpad, self.ypad),
            steps=(self.dt, self.dx, self.dy),
        )

    @property
    def domains_kxky(self):
        # Drift kernel meshes
        return nd_kspace_mesh(
            coeffs=(1.0, 1.0),
            sizes=(self.xgrid, self.ygrid),
            pads=(self.xpad, self.ypad),
            steps=(self.dx, self.dy),
        )

    def drift_kernel(self, z: float):
        kx, ky = self.domains_kxky
        return np.exp(-1j * z * np.pi * self.lambda0 * (kx**2 + ky**2))

    def drift_propagator(self, field_kspace: np.ndarray, z: float):
        return field_kspace * self.drift_kernel(z)

    def thin_lens_kernel(self, f_lens_x: float, f_lens_y: float):
        xx, yy = nd_space_mesh(
            (-self.xmax, -self.ymax), (self.xmax, self.ymax), (self.xgrid, self.ygrid)
        )
        return np.exp(-1j * self.k0 / 2.0 * (xx**2 / f_lens_x + yy**2 / f_lens_y))

    def create_gaussian_pulse_3d_with_q(self, nphotons: float, zR: float):
        """
        Generate a complex three-dimensional spatio-temporal Gaussian profile
        in terms of the q parameter.

        Returns
        -------
        np.ndarray
        """
        t_mesh, x_mesh, y_mesh = nd_space_mesh(
            mins=(0.0, -self.xmax, -self.ymax),
            maxes=(self.tmax, self.xmax, self.ymax),
            sizes=(self.tgrid, self.xgrid, self.ygrid),
        )
        qx = 1j * zR
        qy = 1j * zR

        ux = 1.0 / np.sqrt(qx) * np.exp(-1j * self.k0 * x_mesh**2 / 2.0 / qx)
        uy = 1.0 / np.sqrt(qy) * np.exp(-1j * self.k0 * y_mesh**2 / 2.0 / qy)
        ut = (
            1.0
            / (np.sqrt(2.0 * np.pi) * self.sigma_t)
            * np.exp(-((t_mesh - self.tmax / 2.0) ** 2) / 2.0 / self.sigma_t**2)
        )

        eta = 2.0 * self.k0 * zR * self.sigma_t / np.sqrt(np.pi)

        pulse = np.sqrt(eta) * np.sqrt(nphotons) * ux * uy * ut
        return pulse.astype(np.complex64)


class Wavefront:
    _field_rspace: Optional[np.ndarray]
    _field_kspace: Optional[np.ndarray]
    _phasors: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    _operations: List[Tuple[str, Any]]
    params: WavefrontParams

    def __init__(
        self,
        field_rspace: np.ndarray,
        *,
        params: Optional[WavefrontParams] = None,
        wavelength: float = 1.35e-8,
        half_size: float = 3e-4,
        tmax: int = 50,
        tgrid: int = 801,
        xgrid: int = 101,
        ygrid: int = 101,
        tpad: int = 400,
        xpad: int = 150,
        ypad: int = 150,
        sigma_t: int = 5,
    ) -> None:
        self._phasors = None
        self._field_rspace = field_rspace
        self._field_kspace = None
        self._operations = []
        if params is None:
            params = WavefrontParams(
                lambda0=wavelength,
                half_size=half_size,
                tmax=tmax,
                tgrid=tgrid,
                xgrid=xgrid,
                ygrid=ygrid,
                tpad=tpad,
                xpad=xpad,
                ypad=ypad,
                sigma_t=sigma_t,
            )

        self.params = params

    def _calc_phasors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = self.params
        meshes_wkxky = nd_kspace_mesh(
            params.coeffs,
            (params.tgrid, params.xgrid, params.ygrid),
            (params.tpad, params.xpad, params.ypad),
            (params.dt, params.dx, params.dy),
        )

        t, x, y = (
            np.exp(1j * 2.0 * np.pi * mesh * shift / coeff)
            for coeff, mesh, shift in zip(
                self.params.coeffs,
                meshes_wkxky,
                self.params.shifts,
            )
        )
        return (t, x, y)

    # TODO: tab completion with IPython can make slow properties a really bad
    # idea; would `get_phasors`, `get_dfl`, etc. be acceptable?

    @property
    def phasors(self):
        if self._phasors is None:
            self._phasors = self._calc_phasors()

        return self._phasors

    @property
    def field_rspace(self) -> np.ndarray:
        if self._field_rspace is None:
            self._field_rspace = self.ifft()
        return self._field_rspace

    @property
    def field_kspace(self) -> np.ndarray:
        if self._field_kspace is None:
            self._field_kspace = self.fft()
        return self._field_kspace

    def fft(self, workers=-1):
        assert self._field_rspace is not None
        self._record("fft", workers)
        dfl_pad = _pad_array(self.field_rspace, self.params.pad_shape)
        return fft_phased(
            dfl_pad,
            axes=(0, 1, 2),
            phasors=self.phasors,
            workers=workers,
        )

    def ifft(self, workers=-1):
        assert self._field_kspace is not None
        self._record("ifft", workers)
        return ifft_phased(
            self._field_kspace,
            axes=(0, 1, 2),
            phasors=self.phasors,
            workers=workers,
        )[
            self.params.tpad : -self.params.tpad,
            self.params.xpad : -self.params.xpad,
            self.params.ypad : -self.params.ypad,
        ]

    def _record(self, operation: str, params: Any):
        logger.debug(f"{operation}: {params}")
        self._operations.append((operation, params))

    def propagate_z(self, z_prop: float):
        z_prop = float(z_prop)
        self._record("propagate_z", z_prop)
        self._field_kspace = self.params.drift_propagator(self.field_kspace, z_prop)
        # Invalidate the real space data
        self._field_rspace = None
        return self._field_kspace

    def focusing_element(self, f_lens_x: float, f_lens_y: float):
        self._record("focusing_element", (f_lens_x, f_lens_y))
        self._field_rspace = self.field_rspace * self.params.thin_lens_kernel(
            f_lens_x, f_lens_y
        )
        # Invalidate the spectral data
        self._field_kspace = None
        return self._field_rspace
