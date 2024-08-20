#!/usr/bin/env python

from __future__ import annotations

import copy
import dataclasses
import logging
import pathlib
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import scipy.fft

logger = logging.getLogger(__name__)


_fft_workers = -1
Ranges = Sequence[Tuple[float, float]]
AnyPath = Union[str, pathlib.Path]
Plane = Union[str, Tuple[int, int]]


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
    axes: Sequence[int],
    phasors,
    workers=-1,
) -> np.ndarray:
    """
    Compute the N-D discrete Fourier Transform with phasors applied.

    Parameters
    ----------
    array : np.ndarray
        Input array which can be complex.
    axes : sequence of int
        Axis indices to apply the FFT to.
    phasors : np.ndarray
        Apply these per-dimension phasors after performing the FFT.
    workers : int, default=-1
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
    """
    array_fft = scipy.fft.fftn(array, axes=axes, workers=workers, norm="ortho")
    for phasor in phasors:
        array_fft *= phasor
    return scipy.fft.fftshift(array_fft, axes=axes)


def ifft_phased(
    array: np.ndarray,
    axes: Sequence[int],
    phasors,
    workers=-1,
) -> np.ndarray:
    """
    Compute the N-D inverse discrete Fourier Transform with phasors applied.

    Parameters
    ----------
    array : np.ndarray
        Input array which can be complex.
    axes : Sequence[int]
        Axis indices to apply the FFT to.
    phasors : np.ndarray
        Apply the complex conjugate of these per-dimension phasors after the
        inverse FFT.
    workers : int, default=-1
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
    """
    array_fft = scipy.fft.ifftn(array, axes=axes, workers=workers, norm="ortho")
    for phasor in phasors:
        array_fft *= np.conj(phasor)
    return scipy.fft.ifftshift(array_fft, axes=axes)


def nd_kspace_domains(
    coeffs: Sequence[float],
    sizes: Sequence[int],
    pads: Sequence[int],
    steps: Sequence[float],
    shifted: bool = True,
):
    """
    Generate reciprocal space domains for given grid sizes and steps.

    Parameters
    ----------
    coeffs : tuple
        Conversion coefficients for eV, rad
    sizes : tuple of ints
        Grid sizes
    pads : tuple of ints
        Number of padding points for each axis
    steps : tuple of ints
        Grid step sizes in cartesian space.
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


def nd_kspace_mesh(
    coeffs: Sequence[float],
    sizes: Sequence[int],
    pads: Sequence[int],
    steps: Sequence[float],
    shifted: bool = True,
):
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
        Grid step sizes in cartesian space.
    shifted : bool
        Flag if np.fft.fftshift is applied

    Returns
    -------
    np.ndarray
    """

    domains = nd_kspace_domains(
        coeffs=coeffs,
        sizes=sizes,
        pads=pads,
        steps=steps,
        shifted=shifted,
    )
    meshes = np.meshgrid(*domains, indexing="ij")
    return meshes


def nd_space_mesh(
    ranges: Sequence[Tuple[float, float]],
    sizes: Sequence[int],
):
    """
    Generate a cartesian space mesh for given grid sizes and steps.

    Parameters
    ----------
    ranges : tuple of (float, float) pairs
        Low and high domain range for each dimension of the wavefront.
    sizes : list of float
        Number of grid points in each axis.

    Returns
    -------
    np.ndarray
    """
    domains = [np.linspace(min_, max_, n) for (min_, max_), n in zip(ranges, sizes)]
    return np.meshgrid(*domains, indexing="ij")


def is_odd(value: int) -> bool:
    """
    Is `value` an odd integer?

    Parameters
    ----------
    value : int

    Returns
    -------
    bool
    """
    return value % 2 == 1


def _fix_fft_dimension(dim: int):
    """Get a dimension that's efficient for the FFT - and also odd for symmetry."""
    while True:
        next_dim = scipy.fft.next_fast_len(dim, real=False)
        if next_dim is None:
            raise ValueError(f"Unable to get the next valid dimension for: {dim}")
        dim = next_dim
        if is_odd(dim):
            break
        dim += 1
    return dim


def _fix_grid_padding(grid: int, pad: int) -> Tuple[int, int]:
    """
    Fix gridding and padding values for symmetry and FFT efficiency.

    This works on a single dimension.

    Parameters
    ----------
    grid : int
        Data gridding.
    pad : int
        Data padding.

    Returns
    -------
    int
        Adjusted data gridding.
    int
        Adjusted data padding.
    """
    # Grid must be odd for us:
    if not is_odd(grid):
        grid += 1

    # Ensure that our FFT dimension is odd and optimal for scipy's FFT:
    dim = _fix_fft_dimension(grid + 2 * pad)
    assert is_odd(dim), "FFT dimension not odd?"

    # Fix padding based on our optimal dimension:
    pad = (dim - grid) // 2
    assert not is_odd(dim - grid), "End dimension not even as expected?"
    return grid, pad


def fix_padding(
    grid: Sequence[int],
    pad: Sequence[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Fix gridding and padding values for symmetry and FFT efficiency.

    This works on all dimensions of gridding and padding.

    Parameters
    ----------
    grid : tuple of ints
        Data gridding.
    pad : tuple of ints
        Data padding.

    Returns
    -------
    tuple of ints
        Adjusted data gridding.
    tuple of ints
        Adjusted data padding.
    """
    if len(grid) != len(pad):
        raise ValueError(
            f"Gridding and padding must be of the same dimension. "
            f"Got: {len(grid)} and {len(pad)}"
        )

    result = [[], []]
    for dim, (dim_grid, dim_pad) in enumerate(zip(grid, pad)):
        new_grid, new_pad = _fix_grid_padding(dim_grid, dim_pad)
        logger.debug(
            "Grid[%d] %d -> %d pad %d -> %d",
            dim,
            dim_grid,
            new_grid,
            dim_pad,
            new_pad,
        )
        result[0].append(new_grid)
        result[1].append(new_pad)

    return tuple(result[0]), tuple(result[1])


def get_shifts(
    ranges: Ranges,
    pads: Sequence[int],
    deltas: Sequence[float],
) -> Tuple[float, ...]:
    """
    Effective half sizes with padding in all dimensions.

    Parameters
    ----------
    ranges : tuple of (float, float) pairs
        Low and high domain range for each dimension of the wavefront.
    pads : tuple of ints
        Number of padding points for each axis
    deltas : tuple of floats
        Grid delta steps in cartesian space.

    Returns
    -------
    tuple of floats
        Effective half sizes with padding in all dimensions.
    """

    assert len(ranges) == len(pads) == len(deltas) > 1
    return tuple(
        (domain[1] - domain[0]) / 2.0 + pad * delta
        for domain, pad, delta in zip(ranges, pads, deltas)
    )


def calculate_k0(wavelength: float) -> float:
    """K-value angular wavenumber: 2 pi / wavelength."""
    return 2.0 * np.pi / wavelength


def conversion_coeffs(wavelength: float, dim: int) -> Tuple[float, ...]:
    """
    Conversion coefficients to (eV, radians, radians, ...).

    Omega, theta-x, theta-y.
    """
    k0 = calculate_k0(wavelength)
    hbar = scipy.constants.hbar / scipy.constants.e * 1.0e15  # fs-eV
    return tuple([2.0 * np.pi * hbar] + [2.0 * np.pi / k0] * (dim - 1))


def cartesian_domain(
    ranges: Ranges,
    grids: Sequence[int],
):
    """Real-space domain of the non-padded field."""
    return tuple(
        np.linspace(range_[0], range_[1], grid) for range_, grid in zip(ranges, grids)
    )


def domains_omega_thx_thy(
    wavelength: float,
    grids: Sequence[int],
    pads: Sequence[int],
    deltas: Sequence[float],
):
    """
    Fourier space domain of the padded field.

    Units of (eV, radians, ...)
    """
    coeffs = conversion_coeffs(wavelength=wavelength, dim=len(grids))

    assert len(grids) == len(pads) == len(deltas) == len(coeffs)
    return nd_kspace_mesh(
        coeffs=coeffs,
        sizes=grids,
        pads=pads,
        steps=deltas,
    )


def domains_kxky(
    grids: Sequence[int],
    pads: Sequence[int],
    deltas: Sequence[float],
):
    """
    Transverse Fourier space domain x and y.

    In units of the scipy FFT.

    Parameters
    ----------
    grids : tuple of ints
        Data gridding.
    pads : tuple of ints
        Number of padding points for each axis
    deltas : tuple of floats
        Grid delta steps in cartesian space.
    """
    assert len(grids) == len(pads) == len(deltas)
    return nd_kspace_mesh(
        coeffs=(1.0,) * (len(grids) - 1),
        sizes=grids[1:],
        pads=pads[1:],
        steps=deltas[1:],
    )


def drift_kernel_z(
    domains_kxky: List[np.ndarray],
    z: float,
    wavelength: float,
):
    """Drift transfer function in Z [m] paraxial approximation."""
    kx, ky = domains_kxky
    return np.exp(-1j * z * np.pi * wavelength * (kx**2 + ky**2))


def drift_propagator_z(
    field_kspace: np.ndarray,
    domains_kxky: List[np.ndarray],
    z: float,
    wavelength: float,
):
    """Fresnel propagator in paraxial approximation to distance z [m]."""
    return field_kspace * drift_kernel_z(
        domains_kxky=domains_kxky, z=z, wavelength=wavelength
    )


def thin_lens_kernel_xy(
    wavelength: float,
    ranges: Ranges,
    grid: Sequence[int],
    f_lens_x: float,
    f_lens_y: float,
):
    """
    Transfer function for thin lens focusing.

    Parameters
    ----------
    wavelength : float
        Wavelength (lambda0) [m].
    ranges : tuple of (float, float) pairs
        Low and high domain range for each dimension of the wavefront.
    f_lens_x : float
        Focal length of the lens in x [m].
    f_lens_y : float
        Focal length of the lens in y [m].

    Returns
    -------
    np.ndarray
    """
    k0 = calculate_k0(wavelength)
    xx, yy = nd_space_mesh(ranges[1:], grid[1:])
    return np.exp(-1j * k0 / 2.0 * (xx**2 / f_lens_x + yy**2 / f_lens_y))


def create_gaussian_pulse_3d_with_q(
    wavelength: float,
    nphotons: float,
    zR: float,
    sigma_t: float,
    ranges: Ranges,
    grid: Tuple[int, int, int],
    dtype=np.complex64,
):
    """
    Generate a complex three-dimensional spatio-temporal Gaussian profile in terms of the q parameter.

    Parameters
    ----------
    wavelength : float
        Wavelength (lambda0) [m].
    nphotons : float
        Number of photons.
    zR : float
        Rayleigh range [m].
    sigma_t : float
        Time RMS [s]
    ranges : tuple of (float, float) pairs
        Low and high domain range for each dimension of the wavefront.
        First axis must be time [s].
        Remaining axes are expected to be spatial (x, y) [m].
    grid : tuple of ints
        Data gridding.
    dtype : np.dtype, default=np.complex64

    Returns
    -------
    np.ndarray
    """
    k0 = calculate_k0(wavelength)
    (min_t, max_t), *_ = ranges
    t_mid = (max_t + min_t) / 2.0
    t_mesh, x_mesh, y_mesh = nd_space_mesh(ranges=ranges, sizes=grid)
    qx = 1j * zR
    qy = 1j * zR

    ux = 1.0 / np.sqrt(qx) * np.exp(-1j * k0 * x_mesh**2 / 2.0 / qx)
    uy = 1.0 / np.sqrt(qy) * np.exp(-1j * k0 * y_mesh**2 / 2.0 / qy)
    ut = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_t)) * np.exp(
        -((t_mesh - t_mid) ** 2) / 2.0 / sigma_t**2
    )

    eta = 2.0 * k0 * zR * sigma_t / np.sqrt(np.pi)

    pulse = np.sqrt(eta) * np.sqrt(nphotons) * ux * uy * ut
    return pulse.astype(dtype)


@dataclasses.dataclass(frozen=True)
class WavefrontPadding:
    """
    Wavefront padding settings.

    Parameters
    ----------
    grid : tuple of int
        Number of grid points.
    pad : tuple of int
        Number of pad points in time on each side.
    """

    grid: Tuple[int, ...]
    pad: Tuple[int, ...]

    @property
    def ifft_slices(self):
        """Slices to extract the cartesian space data from its padded form (i.e., post-ifft)."""
        return tuple(slice(pad, -pad) for pad in self.pad)

    @property
    def pad_shape(self) -> Tuple[Tuple[int, int], ...]:
        """Padding in each dimension."""
        return tuple((pad, pad) for pad in self.pad)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        pad: Sequence[int],
        fix: bool = True,
    ) -> WavefrontPadding:
        if data.ndim != len(pad):
            raise ValueError("Dimensions of the grid and the padding must be identical")

        padding = WavefrontPadding(data.shape, tuple(pad))
        return padding.fix() if fix else padding

    def fix(self) -> WavefrontPadding:
        """
        Grid and pad values will be automatically adjusted as follows:

        * grid will be made odd for the purposes of symmetry.
        * pad = (scipy_fft_fast_length - grid) // 2

        Such that the total array size will be: `grid + 2 * pad`.
        """
        grid, pad = fix_padding(self.grid, self.pad)
        return WavefrontPadding(grid, pad)

    def get_padded_shape(self, field_rspace: np.ndarray) -> Tuple[int, ...]:
        """Get the padded shape given a 3D field rspace array."""
        nd = len(self.grid)
        if field_rspace.ndim != nd:
            raise ValueError(f"`field_rspace` is not an {nd}D array")

        if not all(is_odd(dim) for dim in field_rspace.shape):
            raise ValueError(
                f"`field_rspace` dimensions are not all odd numbers: {field_rspace.shape}"
            )

        return tuple(dim + 2 * pad for dim, pad in zip(field_rspace.shape, self.pad))


class Wavefront:
    """
    Particle field wavefront.

    Parameters
    ----------
    field_rspace : np.ndarray
        Cartesian space field data.  3D array with dimensions of (time, x, y).
    wavelength : float
        Wavelength (lambda0) [m].
    ranges : tuple of (float, float) pairs
        Low and high domain range for each dimension of the wavefront.
        First axis must be time [s].
        Remaining axes are expected to be spatial (x, y) [m].
    pad : tuple of int, optional
        Padding for each of the dimensions.  Defaults to 40 for the time
        dimension and 100 for the remaining dimensions.
    """

    _field_rspace: Optional[np.ndarray]
    _field_kspace: Optional[np.ndarray]
    _phasors: Optional[Tuple[np.ndarray, ...]]
    _ranges: Ranges
    _wavelength: float
    _pad: WavefrontPadding

    def __init__(
        self,
        field_rspace: np.ndarray,
        wavelength: float,
        ranges: Ranges,
        pad: Optional[Sequence[int]] = None,
    ) -> None:
        if not pad:
            pad = (40,) + (100,) * (field_rspace.ndim - 1)
        self._phasors = None
        self._field_rspace = field_rspace
        self._field_rspace_shape = field_rspace.shape
        self._field_kspace = None
        self._wavelength = wavelength
        self._ranges = tuple(ranges)
        self._pad = WavefrontPadding.from_array(field_rspace, pad=pad, fix=True)

    def __copy__(self) -> Wavefront:
        res = Wavefront.__new__(Wavefront)
        res._phasors = self._phasors
        res._field_rspace_shape = self._field_rspace_shape
        res._field_rspace = self._field_rspace
        res._field_kspace = self._field_kspace
        res._wavelength = self._wavelength
        res._ranges = self._ranges
        res._pad = self._pad
        return res

    def __deepcopy__(self, memo) -> Wavefront:
        res = Wavefront.__new__(Wavefront)
        res._phasors = self._phasors
        res._field_rspace_shape = self._field_rspace_shape
        res._field_rspace = (
            np.copy(self._field_rspace) if self._field_rspace is not None else None
        )
        res._field_kspace = (
            np.copy(self._field_kspace) if self._field_kspace is not None else None
        )
        res._wavelength = self._wavelength
        res._ranges = self._ranges
        res._pad = self._pad
        return res

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return all(
            (
                self._field_rspace_shape == other._field_rspace_shape,
                np.all(self._field_rspace == other._field_rspace),
                np.all(self._field_kspace == other._field_kspace),
                self._wavelength == other._wavelength,
                self._ranges == other._ranges,
                self._pad == other._pad,
            )
        )

    @classmethod
    def gaussian_pulse(
        cls,
        dims: Tuple[int, int, int],
        wavelength: float,
        nphotons: float,
        zR: float,
        sigma_t: float,
        ranges: Ranges,
        pad: Optional[Sequence[int]] = None,
        dtype=np.complex64,
    ):
        """
        Generate a complex three-dimensional spatio-temporal Gaussian profile
        in terms of the q parameter.

        Parameters
        ----------
        wavelength : float
            Wavelength (lambda0) [m].
        nphotons : float
            Number of photons.
        zR : float
            Rayleigh range [m].
        sigma_t : float
            Time RMS [s]

        Returns
        -------
        Wavefront
        """
        pulse = create_gaussian_pulse_3d_with_q(
            wavelength=wavelength,
            nphotons=nphotons,
            zR=zR,
            sigma_t=sigma_t,
            ranges=ranges,
            grid=dims,
            dtype=dtype,
        )
        return cls(
            field_rspace=pulse,
            wavelength=wavelength,
            ranges=ranges,
            pad=pad,
        )

    @property
    def rspace_domain(self):
        """
        Cartesian space domain values in all dimensions.

        For each dimension of the wavefront, this is the evenly-spaced set of values over
        its specified range.
        """
        return cartesian_domain(ranges=self._ranges, grids=self._pad.grid)

    @property
    def _rspace_deltas(self) -> Tuple[float, ...]:
        """Spacing for each dimension of the cartesian domain."""
        return tuple(dim[1] - dim[0] for dim in self.rspace_domain)

    def _calc_phasors(self) -> Tuple[np.ndarray, ...]:
        """Calculate phasors for each dimension of the cartesian domain."""
        coeffs = conversion_coeffs(
            wavelength=self._wavelength, dim=len(self._field_rspace_shape)
        )

        rspace_deltas = self._rspace_deltas

        shifts = get_shifts(
            ranges=self._ranges,
            pads=self._pad.pad,
            deltas=rspace_deltas,
        )
        meshes_wkxky = nd_kspace_mesh(
            coeffs=coeffs,
            sizes=self._pad.grid,
            pads=self._pad.pad,
            steps=rspace_deltas,
        )

        return tuple(
            np.exp(1j * 2.0 * np.pi * mesh * shift / coeff)
            for coeff, mesh, shift in zip(coeffs, meshes_wkxky, shifts)
        )

    @property
    def phasors(self) -> Tuple[np.ndarray, ...]:
        """Phasors for each dimension of the real-space domain."""
        if self._phasors is None:
            self._phasors = self._calc_phasors()

        return self._phasors

    @property
    def field_rspace(self) -> np.ndarray:
        """Real-space wavefront field data."""
        if self._field_rspace is None:
            self._field_rspace = self._ifft()
        return self._field_rspace

    @property
    def field_kspace(self) -> np.ndarray:
        """K-space wavefront field data."""
        if self._field_kspace is None:
            self._field_kspace = self._fft()
        return self._field_kspace

    def _fft(self):
        """
        Calculate the FFT (rspace -> kspace) on the user data.

        Requires that the `_field_rspace` data is available.

        This is intended to be handled by the `Wavefront` class itself, such
        that the user does not need to pay attention to whether the real-space
        or k-space wavefront data is up-to-date.
        """
        assert self._field_rspace is not None
        workers = get_num_fft_workers()
        dfl_pad = _pad_array(self.field_rspace, self._pad.pad_shape)
        return fft_phased(
            dfl_pad,
            axes=(0, 1, 2),
            phasors=self.phasors,
            workers=workers,
        )

    def _ifft(self):
        """
        Calculate the inverse FFT (kspace -> rspace) on the user data.

        Requires that the `_field_kspace` data is available.

        This is intended to be handled by the `Wavefront` class itself, such
        that the user does not need to pay attention to whether the real-space
        or k-space wavefront data is up-to-date.
        """
        assert self._field_kspace is not None
        workers = get_num_fft_workers()
        return ifft_phased(
            self._field_kspace,
            axes=(0, 1, 2),
            phasors=self.phasors,
            workers=workers,
        )[self._pad.ifft_slices]

    @property
    def wavelength(self) -> float:
        """Wavelength of the wavefront [m]."""
        return self._wavelength

    @property
    def pad(self):
        """Padding settings."""
        return self._pad

    @property
    def ranges(self):
        return self._ranges

    def focus(
        self,
        plane: Plane,
        focus: Tuple[float, float],
        *,
        inplace: bool = False,
    ) -> Wavefront:
        """
        Apply thin lens focusing.

        Parameters
        ----------
        plane : str or (int, int)
            Plane identifier (e.g., "xy") or dimension indices (e.g., ``(1, 2)``)
        focus : (float, float)
            Focal length of the lens in each dimension [m].
        inplace : bool, default=False
            Perform the operation in-place on this wavefront object.

        Returns
        -------
        Wavefront
            This object if `inplace=True` or a new copy if `inplace=False`.
        """
        if plane not in ("xy", (1, 2)):
            raise NotImplementedError(f"Unsupported plane: {plane}")

        if not inplace:
            wavefront = copy.copy(self)
            return wavefront.focus(plane, focus, inplace=True)

        self._field_rspace = self.field_rspace * thin_lens_kernel_xy(
            wavelength=self.wavelength,
            ranges=self._ranges,
            grid=self._pad.grid,
            f_lens_x=focus[0],
            f_lens_y=focus[1],
        )
        # Invalidate the spectral data
        self._field_kspace = None
        return self

    def drift(
        self,
        direction: Union[str, int],
        distance: float,
        *,
        inplace: bool = False,
    ) -> Wavefront:
        """
        Drift this Wavefront along `direction` in meters.

        Parameters
        ----------
        direction : str or (int, int)
            Propagation direction dimension name (e.g., "z") or dimension index (e.g., `2`)
        z_prop : float
            Distance in meters.
        inplace : bool, default=False
            Perform the operation in-place on this wavefront object.

        Returns
        -------
        Wavefront
            This object if `inplace=True` or a new copy if `inplace=False`.
        """

        if direction not in {"z", 2}:
            raise NotImplementedError(f"Unsupported propagation direction: {direction}")

        if not inplace:
            wavefront = copy.copy(self)
            return wavefront.drift(direction, distance, inplace=True)

        self._field_kspace = drift_propagator_z(
            field_kspace=self.field_kspace,
            domains_kxky=domains_kxky(
                grids=self._pad.grid,
                pads=self._pad.pad,
                deltas=self._rspace_deltas,
            ),
            wavelength=self._wavelength,
            z=float(distance),
        )
        # Invalidate the real space data
        self._field_rspace = None
        return self

    def plot(
        self,
        plane: Plane,
        *,
        rspace: bool = True,
        show_real: bool = True,
        show_imaginary: bool = True,
        show_abs: bool = True,
        show_phase: bool = True,
        axs: Optional[List[matplotlib.axes.Axes]] = None,
        cmap: str = "viridis",
        figsize: Optional[Tuple[int, int]] = None,
        nrows: int = 2,
        ncols: int = 2,
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None,
        tight_layout: bool = True,
        save: Optional[AnyPath] = None,
    ):
        """
        Plot the projection onto the given plane.

        Parameters
        ----------
        plane : str or (int, int)
            Plane to plot, e.g., "xy" or (1, 2).
        rspace : bool, default=True
            Plot the real/cartesian space data.
        show_real : bool
            Show the projection of the real portion of the data.
        show_imaginary : bool
            Show the projection of the imaginary portion of the data.
        show_abs : bool
            Show the projection of the absolute value of the data.
        show_phase : bool
            Show the projection of the phase of the data.
        figsize : (int, int), optional
            Figure size for the axes.
            Defaults to Matplotlib's `rcParams["figure.figsize"]``.
        axs : List[matplotlib.axes.Axes], optional
            Plot the data in the provided matplotlib Axes.
            Creates a new figure and Axes if not specified.
        cmap : str, default="viridis"
            Color map to use.
        nrows : int, default=2
            Number of rows for the plot.
        ncols : int, default=2
            Number of columns for the plot.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        tight_layout : bool, default=True
            Set a tight layout.

        Returns
        -------
        Figure
        list of Axes
        """
        if rspace:
            data = self.field_rspace
        else:
            data = self.field_kspace

        sum_axis = {
            # TODO: when standardized, this will be xyz instead of txy
            "xy": 0,
            (1, 2): 0,
        }[plane]

        if axs is None:
            fig, gs = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                squeeze=False,
                figsize=figsize,
            )
            axs = list(gs.flatten())
            fig.suptitle(f"{plane}")
        else:
            fig = axs[0].get_figure()
            assert fig is not None

        remaining_axes = list(axs)

        def plot(dat, title: str):
            ax = remaining_axes.pop(0)
            ax.imshow(np.sum(dat, axis=sum_axis), cmap=cmap)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if not ax.get_title():
                ax.set_title(title)

        if show_real:
            plot(np.real(data), title="Real")

        if show_imaginary:
            plot(np.imag(data), title="Imaginary")

        if show_abs:
            plot(np.abs(data), f"|{plane}|")

        if show_phase:
            plot(np.angle(data), title="Phase")

        if fig is not None:
            if tight_layout:
                fig.tight_layout()

            if save:
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save)

        return fig, axs
