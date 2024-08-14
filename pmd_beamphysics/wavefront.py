#!/usr/bin/env python

from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any, List, Optional, Tuple

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
    """
    Compute the N-D discrete Fourier Transform with phasors applied.

    Parameters
    ----------
    array : np.ndarray
        Input array which can be complex.
    axes : Tuple[int, ...]
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
    axes: Tuple[int, ...],
    phasors,
    workers=-1,
) -> np.ndarray:
    """
    Compute the N-D inverse discrete Fourier Transform with phasors applied.

    Parameters
    ----------
    array : np.ndarray
        Input array which can be complex.
    axes : Tuple[int, ...]
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


def nd_kspace_domains(coeffs, sizes, pads, steps, shifted=True):
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
    grid: Tuple[int, ...],
    pad: Tuple[int, ...],
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


RealSpaceRanges = Tuple[Tuple[float, float], ...]


def get_shifts(
    ranges: RealSpaceRanges,
    pads: Tuple[int, ...],
    deltas: Tuple[float, ...],
) -> Tuple[float, ...]:
    """
    Effective half sizes with padding in all dimensions.

    Parameters
    ----------
    ranges : tuple of (float, float) pairs
        Low and high domain range for each dimension of the wavefront.
        First axis must be time [fs].
        Remaining axes are expected to be spatial (x, y) [m].
    pads : tuple of ints
        Number of padding points for each axis
    deltas : tuple of floats
        Grid delta steps.

    Returns
    -------
    tuple of floats
        Effective half sizes with padding in all dimensions.
    """

    assert len(ranges) == len(pads) == len(deltas) > 1

    # Time domain: (0, tmax) -> centered at tmax / 2
    mid_t = (ranges[0][1] + ranges[0][0]) / 2
    tpad = pads[0]
    dt = deltas[0]

    return (
        mid_t + tpad * dt,
        *(
            # Spatial domains are symmetric around 0: (-value, 0, value)
            domain[1] + pad * delta
            for domain, pad, delta in zip(ranges[1:], pads[1:], deltas[1:])
        ),
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


def real_space_domain(
    ranges: RealSpaceRanges,
    grids: Tuple[int, ...],
):
    """Real-space domain of the non-padded field."""
    return tuple(
        np.linspace(range_[0], range_[1], grid) for range_, grid in zip(ranges, grids)
    )


def domains_omega_thx_thy(
    wavelength: float,
    grids: Tuple[int, ...],
    pads: Tuple[int, ...],
    deltas: Tuple[float, ...],
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
    grids: Tuple[int, ...],
    pads: Tuple[int, ...],
    deltas: Tuple[float, ...],
):
    """
    Transverse Fourier space domain x and y.

    In units of the scipy FFT.
    """
    assert len(grids) == len(pads) == len(deltas)
    return nd_kspace_mesh(
        coeffs=(1.0,) * (len(grids) - 1),
        sizes=grids[1:],
        pads=pads[1:],
        steps=deltas[1:],
    )


def drift_kernel(
    domains_kxky: List[np.ndarray],
    z: float,
    wavelength: float,
):
    """Drift transfer function in Z [m] paraxial approximation."""
    kx, ky = domains_kxky
    return np.exp(-1j * z * np.pi * wavelength * (kx**2 + ky**2))


def drift_propagator(
    field_kspace: np.ndarray,
    domains_kxky: List[np.ndarray],
    z: float,
    wavelength: float,
):
    """Fresnel propagator in paraxial approximation to distance z [m]."""
    return field_kspace * drift_kernel(
        domains_kxky=domains_kxky, z=z, wavelength=wavelength
    )


def thin_lens_kernel(
    wavelength: float,
    ranges: RealSpaceRanges,
    grid: Tuple[int, ...],
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
        First axis must be time [fs].
        Remaining axes are expected to be spatial (x, y) [m].
    f_lens_x : float
        Focal length of the lens in x [m].
    f_lens_y : float
        Focal length of the lens in y [m].

    Returns
    -------
    np.ndarray
    """
    k0 = calculate_k0(wavelength)
    mins = [range_[0] for range_ in ranges]
    maxes = [range_[1] for range_ in ranges]
    xx, yy = nd_space_mesh(mins, maxes, grid[1:])
    return np.exp(-1j * k0 / 2.0 * (xx**2 / f_lens_x + yy**2 / f_lens_y))


def create_gaussian_pulse_3d_with_q(
    wavelength: float,
    nphotons: float,
    zR: float,
    sigma_t: float,
    ranges: RealSpaceRanges,
    grid: Tuple[int, int, int],
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
        Time RMS [fs]

    Returns
    -------
    np.ndarray
    """
    k0 = calculate_k0(wavelength)
    (min_t, max_t), (min_x, max_x), (min_y, max_y) = ranges
    tgrid, xgrid, ygrid = grid
    t_mid = (max_t + min_t) / 2.0
    t_mesh, x_mesh, y_mesh = nd_space_mesh(
        mins=(min_t, min_x, min_y),
        maxes=(max_t, max_x, max_y),
        sizes=(tgrid, xgrid, ygrid),
    )
    qx = 1j * zR
    qy = 1j * zR

    ux = 1.0 / np.sqrt(qx) * np.exp(-1j * k0 * x_mesh**2 / 2.0 / qx)
    uy = 1.0 / np.sqrt(qy) * np.exp(-1j * k0 * y_mesh**2 / 2.0 / qy)
    ut = (
        1.0
        / (np.sqrt(2.0 * np.pi) * sigma_t)
        * np.exp(-((t_mesh - t_mid) ** 2) / 2.0 / sigma_t**2)
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
        """Slices to extract the real space data from its padded form (i.e., post-ifft)."""
        return [slice(pad, -pad) for pad in self.pad]

    @property
    def pad_shape(self) -> Tuple[Tuple[int, int], ...]:
        """Padding in each dimension."""
        return tuple((pad, pad) for pad in self.pad)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        pad: Tuple[int, ...],
        fix: bool = True,
    ) -> WavefrontPadding:
        if data.ndim != len(pad):
            raise ValueError("Dimensions of the grid and the padding must be identical")

        padding = WavefrontPadding(data.shape, pad)
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
        Real-space field data.  3D array with dimensions of (time, x, y).
    wavelength : float
        Wavelength (lambda0) [m].
    ranges : tuple of (float, float) pairs
        Low and high domain range for each dimension of the wavefront.
        First axis must be time [fs].
        Remaining axes are expected to be spatial (x, y) [m].
    pad : tuple of int, optional
        Padding for each of the dimensions.  Defaults to 40 for the time
        dimension and 100 for the remaining dimensions.
    """

    _field_rspace: Optional[np.ndarray]
    _field_kspace: Optional[np.ndarray]
    _phasors: Optional[Tuple[np.ndarray, ...]]
    _ranges: RealSpaceRanges
    _wavelength: float
    _pad: WavefrontPadding

    def __init__(
        self,
        field_rspace: np.ndarray,
        wavelength: float,
        ranges: RealSpaceRanges,
        pad: Optional[Tuple[int, ...]] = None,
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

    @property
    def rspace_domain(self):
        """
        Real-space domain values in all dimensions.

        For each dimension of the wavefront, this is the evenly-spaced set of values over
        its specified range.
        """
        return real_space_domain(ranges=self._ranges, grids=self._pad.grid)

    @property
    def _real_domain_deltas(self) -> Tuple[float, ...]:
        """Spacing for each dimension of the real space domain."""
        return tuple(dim[1] - dim[0] for dim in self.rspace_domain)

    def _calc_phasors(self) -> Tuple[np.ndarray, ...]:
        """Calculate phasors for each dimension of the real-space domain."""
        coeffs = conversion_coeffs(
            wavelength=self._wavelength, dim=len(self._field_rspace_shape)
        )

        deltas = self._real_domain_deltas

        shifts = get_shifts(ranges=self._ranges, pads=self._pad.pad, deltas=deltas)
        meshes_wkxky = nd_kspace_mesh(coeffs, self._pad.grid, self._pad.pad, deltas)

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
        )[*self._pad.ifft_slices]

    def propagate_z(self, z_prop: float):
        """
        Propagate this Wavefront in-place along Z in meters.

        Parameters
        ----------
        z_prop : float
            Distance in meters.

        Returns
        -------
        np.ndarray
            Propagated k-space data.

        See Also
        --------
        `propagate_z`
            For a version which returns a propagated copy of the wavefront,
            instead of performing it in-place.
        """
        z_prop = float(z_prop)
        self._field_kspace = drift_propagator(
            field_kspace=self.field_kspace,
            domains_kxky=domains_kxky(
                grids=self._pad.grid,
                pads=self._pad.pad,
                deltas=self._real_domain_deltas,
            ),
            wavelength=self._wavelength,
            z=z_prop,
        )
        # Invalidate the real space data
        self._field_rspace = None
        return self._field_kspace

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

    def focusing_element(self, f_lens_x: float, f_lens_y: float):
        """
        Apply thin lens focusing.

        Parameters
        ----------
        f_lens_x : float
            Focal length of the lens in x [m].
        f_lens_y : float
            Focal length of the lens in y [m].

        Returns
        -------
        np.ndarray
            Focused r-space data.

        See Also
        --------
        `focusing_element`
            For a version which returns a focused copy of the wavefront,
            instead of performing it in-place.
        """
        self._field_rspace = self.field_rspace * thin_lens_kernel(
            wavelength=self.wavelength,
            ranges=self._ranges,
            grid=self._pad.grid,
            f_lens_x=f_lens_x,
            f_lens_y=f_lens_y,
        )
        # Invalidate the spectral data
        self._field_kspace = None
        return self._field_rspace

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
        ranges: RealSpaceRanges,
        pad: Optional[Tuple[int, ...]] = None,
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
            Time RMS [fs]
        t_mid : float
            Center of the time window size [fs].

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


def propagate_z(wavefront: Wavefront, z_prop: float) -> Wavefront:
    """
    Propagate a Wavefront along Z in meters and get a new `Wavefront` object.

    Parameters
    ----------
    wavefront : Wavefront
        The Wavefront object to propagate.
    z_prop : float
        Distance in meters.

    Returns
    -------
    Wavefront
        Propagated Wavefront object.

    See Also
    --------
    `Wavefront.propagate_z`
        For an in-place version.
    """
    wavefront = copy.copy(wavefront)
    wavefront.propagate_z(z_prop)
    return wavefront


def focusing_element(
    wavefront: Wavefront, f_lens_x: float, f_lens_y: float
) -> Wavefront:
    """
    Apply thin lens focusing to `wavefront` and get a new `Wavefront` object.

    Parameters
    ----------
    wavefront : Wavefront
        The Wavefront object to focus.
    f_lens_x : float
        Focal length of the lens in x [m].
    f_lens_y : float
        Focal length of the lens in y [m].

    Returns
    -------
    Wavefront
        Focused Wavefront.

    See Also
    --------
    `Wavefront.focusing_element`
        For an in-place version.
    """
    wavefront = copy.copy(wavefront)
    wavefront.focusing_element(f_lens_x, f_lens_y)
    return wavefront
