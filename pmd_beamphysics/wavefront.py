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


_fft_workers = -1


def get_num_fft_workers() -> int:
    return _fft_workers


def set_num_fft_workers(workers: int):
    global _fft_workers

    _fft_workers = workers

    logger.info(f"Set number of FFT workers to: {workers}")


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


def _fix_grid_padding(grid: int, pad: int) -> Tuple[int, int]:
    # Grid must be odd for us:
    if grid % 2 == 0:
        grid += 1

    # Ensure that our FFT dimension is odd and optimal for scipy's FFT:
    dim = _fix_fft_dimension(grid + 2 * pad)
    assert dim % 2 == 1, "FFT dimension not odd?"

    # Fix padding based on our optimal dimension:
    pad = (dim - grid) // 2
    assert (dim - grid) % 2 == 0, "End dimension not even as expected?"
    return grid, pad


class WavefrontParams(pydantic.BaseModel, frozen=True):
    """
    Wavefront parameter settings.

    Grid and pad values will be automatically adjusted as follows:

    * grid will be made odd for the purposes of symmetry.
    * pad = (scipy_fft_fast_length - grid) // 2

    Such that the total array size will be: `grid + 2 * pad`.

    Parameters
    ----------
    lambda0 : float
        Wavelength [m].
    half_size : float
        Half-size of the domain [m].
    tmax : float
        Time window size [fs].
    tgrid : int
        Number of grid points in time.
    xgrid : int
        Number of grid points in the x-axis.
    ygrid : int
        Number of grid points in the y-axis.
    tpad : int
        Number of pad points in time on each side.
    xpad : int
        Number of pad points in the x-axis on each side.
    ypad : int
        Number of pad points in the y-axis on each side.
    """

    lambda0: float

    half_size: float
    tmax: float

    tgrid: int
    xgrid: int
    ygrid: int

    tpad: int
    xpad: int
    ypad: int

    @pydantic.model_validator(mode="before")
    @classmethod
    def _fix_dimensions(cls, values):
        tgrid, tpad = _fix_grid_padding(values["tgrid"], values["tpad"])
        xgrid, xpad = _fix_grid_padding(values["xgrid"], values["xpad"])
        ygrid, ypad = _fix_grid_padding(values["ygrid"], values["ypad"])

        logger.debug(
            "Fixing gridding: t %d -> %d padding %d -> %d",
            values["tgrid"],
            tgrid,
            values["tpad"],
            tpad,
        )
        logger.debug(
            "Fixing gridding: x %d -> %d padding %d -> %d",
            values["xgrid"],
            xgrid,
            values["xpad"],
            xpad,
        )
        logger.debug(
            "Fixing gridding: y %d -> %d padding %d -> %d",
            values["ygrid"],
            ygrid,
            values["ypad"],
            ypad,
        )
        values["tgrid"], values["tpad"] = tgrid, tpad
        values["xgrid"], values["xpad"] = xgrid, xpad
        values["ygrid"], values["ypad"] = ygrid, ypad
        return values

    @property
    def xmax(self) -> float:
        """Half-size of the domain in x [m]."""
        return self.half_size

    @property
    def ymax(self) -> float:
        """Half-size of the domain in y [m]."""
        return self.half_size

    @property
    def t_mid(self) -> float:
        """Center of the time window size [fs]."""
        return self.tmax / 2.0

    @property
    def k0(self) -> float:
        """K-value: 2 pi / lambda0"""
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
        """Fourier space step size in x."""
        kx = self.domains_kxky[0]
        return kx[1, 0] - kx[0, 0]

    @cached_property
    def dky(self) -> float:
        """Fourier space step size in y."""
        ky = self.domains_kxky[1]
        return ky[0, 1] - ky[0, 0]

    @cached_property
    def shifts(self) -> Tuple[float, float, float]:
        """Effective half sizes with padding in t, x, and y."""
        return (
            self.t_mid + self.tpad * self.dt,
            self.xmax + self.xpad * self.dx,
            self.ymax + self.ypad * self.dy,
        )

    @cached_property
    def pad_shape(self) -> Tuple[Tuple[int, int], ...]:
        """Padding in each dimension."""
        return (
            (self.tpad, self.tpad),
            (self.xpad, self.xpad),
            (self.ypad, self.ypad),
        )

    def get_padded_shape(self, field_rspace: np.ndarray) -> Tuple[int, int, int]:
        """Get the padded shape given a 3D field rspace array."""
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
        """
        Conversion coefficients to (eV, radians, radians).

        Omega, theta-x, theta-y.
        """
        hbar = scipy.constants.hbar / scipy.constants.e * 1.0e15  # fs-eV
        return (
            2.0 * np.pi * hbar,
            2.0 * np.pi / self.k0,
            2.0 * np.pi / self.k0,
        )

    @property
    def domains_txy(self):
        """Real-space domain of the non-padded field."""
        return (
            np.linspace(0.0, self.tmax, self.tgrid),
            np.linspace(-self.xmax, self.xmax, self.xgrid),
            np.linspace(-self.ymax, self.ymax, self.ygrid),
        )

    @property
    def domains_omega_thx_thy(self):
        """
        Fourier space domain of the padded field.

        Units of (eV, radians, radians).
        """
        return nd_kspace_mesh(
            coeffs=self.coeffs,
            sizes=(self.tgrid, self.xgrid, self.ygrid),
            pads=(self.tpad, self.xpad, self.ypad),
            steps=(self.dt, self.dx, self.dy),
        )

    @property
    def domains_kxky(self):
        """
        Transverse Fourier space domain x and y.

        In units of the scipy FFT.
        """
        # Drift kernel meshes
        return nd_kspace_mesh(
            coeffs=(1.0, 1.0),
            sizes=(self.xgrid, self.ygrid),
            pads=(self.xpad, self.ypad),
            steps=(self.dx, self.dy),
        )

    def drift_kernel(self, z: float):
        """Drift transfer function in Z [m] paraxial approximation."""
        kx, ky = self.domains_kxky
        return np.exp(-1j * z * np.pi * self.lambda0 * (kx**2 + ky**2))

    def drift_propagator(self, field_kspace: np.ndarray, z: float):
        """Fresnel propagator in paraxial approximation to distance z [m]."""
        return field_kspace * self.drift_kernel(z)

    def thin_lens_kernel(self, f_lens_x: float, f_lens_y: float):
        """
        Transfer function for thin lens focusing.

        Parameters
        ----------
        f_lens_x : float
            Focal length of the lens in x [m].
        f_lens_y : float
            Focal length of the lens in y [m].

        Returns
        -------
        np.ndarray
        """
        xx, yy = nd_space_mesh(
            (-self.xmax, -self.ymax), (self.xmax, self.ymax), (self.xgrid, self.ygrid)
        )
        return np.exp(-1j * self.k0 / 2.0 * (xx**2 / f_lens_x + yy**2 / f_lens_y))

    def create_gaussian_pulse_3d_with_q(
        self,
        nphotons: float,
        zR: float,
        sigma_t: float,
    ):
        """
        Generate a complex three-dimensional spatio-temporal Gaussian profile
        in terms of the q parameter.

        Parameters
        ----------
        nphotons : float
            Number of photons.
        zR : float
            Rayleigh range [m].
        sigma_t : float
            Time RMS [fs]

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
            / (np.sqrt(2.0 * np.pi) * sigma_t)
            * np.exp(-((t_mesh - self.t_mid) ** 2) / 2.0 / sigma_t**2)
        )

        eta = 2.0 * self.k0 * zR * sigma_t / np.sqrt(np.pi)

        pulse = np.sqrt(eta) * np.sqrt(nphotons) * ux * uy * ut
        return pulse.astype(np.complex64)


class Wavefront:
    """
    Particle field wavefront.

    Parameters
    ----------
    field_rspace : np.ndarray
        Real-space field data.  3D array with dimensions of (time, x, y).
    params : WavefrontParams, optional
        Gridding, padding, and FFT-related parameters.  May be shared
        among multiple Wavefront objects.
    **kwargs :
        If `params` is unspecified, `**kwargs` will be passed to a new
        `WavefrontParams` instance.

    Examples
    --------

    >>> params = WavefrontParams(lambda0=1.35e-8, ...)
    >>> wave1 = Wavefront(dfl1, params=params)
    >>> wave2 = Wavefront(dfl2, params=params)

    >>> wave3 = Wavefront(dfl, lambda0=1.35e-8, ...)
    """

    _field_rspace: Optional[np.ndarray]
    _field_kspace: Optional[np.ndarray]
    _phasors: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    _operations: List[Tuple[str, Any]]
    params: WavefrontParams

    def __init__(
        self,
        field_rspace: np.ndarray,
        params: Optional[WavefrontParams] = None,
        **kwargs,
    ) -> None:
        self._phasors = None
        self._field_rspace = field_rspace
        self._field_kspace = None
        self._operations = []
        if params is None:
            params = WavefrontParams(**kwargs)

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

    def fft(self):
        assert self._field_rspace is not None
        workers = get_num_fft_workers()
        self._record("fft", workers)
        dfl_pad = _pad_array(self.field_rspace, self.params.pad_shape)
        return fft_phased(
            dfl_pad,
            axes=(0, 1, 2),
            phasors=self.phasors,
            workers=workers,
        )

    def ifft(self):
        assert self._field_kspace is not None
        workers = get_num_fft_workers()
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
