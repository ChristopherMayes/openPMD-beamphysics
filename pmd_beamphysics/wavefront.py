from __future__ import annotations

import copy
import dataclasses
import logging
import pathlib
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import h5py
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import scipy.fft

from mpl_toolkits.axes_grid1 import make_axes_locatable

from .metadata import PolarizationDirection, WavefrontMetadata
from . import writers
from .units import known_unit, nice_array

logger = logging.getLogger(__name__)


_fft_workers = -1
Ranges = Sequence[Tuple[float, float]]
AnyPath = Union[str, pathlib.Path]
Plane = Union[str, Tuple[int, int]]

kspace_labels = {
    "x": r"\theta_x",
    "y": r"\theta_y",
    "z": r"\omega",
}


def get_num_fft_workers() -> int:
    return _fft_workers


def set_num_fft_workers(workers: int):
    global _fft_workers

    _fft_workers = workers

    logger.info(f"Set number of FFT workers to: {workers}")


def get_axis_index(axis_labels: Sequence[str], axis: Union[str, int]):
    if isinstance(axis, int):
        if axis >= len(axis_labels):
            raise ValueError(f"Axis out of bounds: {axis} ({len(axis_labels)}D array)")
        return axis
    return axis_labels.index(axis)


def get_axis_indices(
    axis_labels: Tuple[str, ...],
    axes: Union[Sequence[str], Sequence[int]],
):
    return tuple(get_axis_index(axis_labels, axis) for axis in axes)


def get_rspace_label(
    axis_labels: Tuple[str, ...],
    axis: Union[int, str],
):
    return axis_labels[get_axis_index(axis_labels, axis)]


def get_rspace_labels(
    axis_labels: Tuple[str, ...],
    axes: Union[Sequence[str], Sequence[int]],
):
    return tuple(get_rspace_label(axis_labels, axis) for axis in axes)


def get_kspace_label(
    axis_labels: Tuple[str, ...],
    axis: Union[int, str],
):
    rspace_label = get_rspace_label(axis_labels, axis)
    return kspace_labels.get(rspace_label, rspace_label)


def get_kspace_labels(
    axis_labels: Tuple[str, ...],
    axes: Union[Sequence[str], Sequence[int]],
):
    return tuple(get_kspace_label(axis_labels, axis) for axis in axes)


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


class _NiceXYZ(NamedTuple):
    """
    `nice_array`-modified values.

    Axes X and Y are always of the same units (rspace and kspace), but Z may be
    of different units (kspace) so they are separated here.
    """

    x: np.ndarray
    y: np.ndarray
    xy_scale: float
    xy_unit_prefix: str

    z: np.ndarray
    z_scale: float
    z_unit_prefix: str


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
) -> List[np.ndarray]:
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


def _fix_grid_padding(grid: int, pad: int) -> int:
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
        Adjusted data padding.
    """

    def is_good(dim: int) -> bool:
        return is_odd(dim) and scipy.fft.next_fast_len(dim, real=False) == dim

    while not is_good(grid + pad):
        dim = scipy.fft.next_fast_len(grid + pad + 1, real=False)
        if dim is None:
            raise ValueError(
                f"Unable to get the next valid FFT length for: {grid=} {pad=}"
            )

        pad = dim - grid

    return pad


def fix_padding(
    grid: Sequence[int],
    pad: Sequence[int],
) -> Tuple[int, ...]:
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
        Adjusted data padding.
    """
    if len(grid) != len(pad):
        raise ValueError(
            f"Gridding and padding must be of the same dimension. "
            f"Got: {len(grid)} and {len(pad)}"
        )

    final_padding = []
    for dim, (dim_grid, dim_pad) in enumerate(zip(grid, pad)):
        new_pad = _fix_grid_padding(dim_grid, dim_pad)
        if new_pad != dim_pad:
            logger.debug(
                "Grid[dim=%d] grid=%d pad=%d -> adjusted padding=%d",
                dim,
                dim_grid,
                dim_pad,
                new_pad,
            )
        final_padding.append(new_pad)

    return tuple(final_padding)


def get_shifts(
    dims: Sequence[int],
    ranges: Ranges,
    pads: Sequence[int],
    deltas: Sequence[float],
) -> Tuple[float, ...]:
    """
    Effective half sizes with padding in all dimensions.

    Parameters
    ----------
    dims : sequence of int
        Grid dimensions.
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

    assert len(pads) == len(deltas) > 1
    spans = tuple(domain[1] - domain[0] for domain in ranges)

    def fix_even(dim: int, delta: float) -> float:
        # For odd number of grid points, no fix is required
        if dim % 2 == 1:
            return 0.0
        # For an even number of grid points, we add half a grid step
        return delta / 2.0

    return tuple(
        span / 2.0 + pad * delta + fix_even(dim, delta)
        for dim, span, pad, delta in zip(dims, spans, pads, deltas)
    )


def calculate_k0(wavelength: float) -> float:
    """K-value angular wavenumber: 2 pi / wavelength."""
    return 2.0 * np.pi / wavelength


def conversion_coeffs(wavelength: float, dim: int) -> Tuple[float, ...]:
    """
    Conversion coefficients to (radians, radians, eV).

    Theta-x, theta-y, omega.
    """
    k0 = calculate_k0(wavelength)
    hbar = scipy.constants.hbar / scipy.constants.e * scipy.constants.c  # eV-m
    return tuple([2.0 * np.pi / k0] * (dim - 1) + [2.0 * np.pi * hbar])


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


def drift_kernel_paraxial(
    transverse_kspace_grid: List[np.ndarray],
    z: float,
    wavelength: float,
):
    """Drift transfer function in Z [m] paraxial approximation."""
    kx, ky = transverse_kspace_grid
    return np.exp(-1j * z * np.pi * wavelength * (kx**2 + ky**2))


def drift_propagator_paraxial(
    kmesh: np.ndarray,
    transverse_kspace_grid: List[np.ndarray],
    z: float,
    wavelength: float,
):
    """Fresnel propagator in paraxial approximation to distance z [m]."""
    kernel = drift_kernel_paraxial(
        transverse_kspace_grid=transverse_kspace_grid,
        z=z,
        wavelength=wavelength,
    )
    return kmesh * kernel[:, :, np.newaxis]


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
    xx, yy = nd_space_mesh(ranges[:2], grid[:2])
    return np.exp(-1j * k0 / 2.0 * (xx**2 / f_lens_x + yy**2 / f_lens_y))


def create_gaussian_pulse_3d_with_q(
    wavelength: float,
    nphotons: float,
    zR: float,
    sigma_z: float,
    grid_spacing: Sequence[float],
    grid: Sequence[int],
    dtype=np.complex64,
):
    """
    Generate a complex three-dimensional spatio-temporal Gaussian profile in terms of the q parameter.

    Parameters
    ----------
    wavelength : float
        Wavelength (lambda0). [m]
    nphotons : float
        Number of photons.
    zR : float
        Rayleigh range. [m]
    sigma_z : float
        Pulse length RMS. [m]
    grid_spacing : tuple of floats
        Per-axis grid spacing.
    grid : tuple of ints
        Data gridding.
    dtype : np.dtype, default=np.complex64

    Returns
    -------
    np.ndarray
    """
    if len(grid_spacing) != 3:
        raise ValueError("`grid_spacing` must be of length 3 for a 3D gaussian")
    if len(grid) != 3:
        raise ValueError("`grid` must be of length 3 for a 3D gaussian")

    ranges = get_ranges_for_grid_spacing(grid_spacing=grid_spacing, dims=grid)
    min_z, max_z = ranges[-1]

    k0 = calculate_k0(wavelength)
    z_mid = (max_z + min_z) / 2.0
    x_mesh, y_mesh, z_mesh = nd_space_mesh(ranges=ranges, sizes=grid)
    qx = 1j * zR
    qy = 1j * zR

    ux = 1.0 / np.sqrt(qx) * np.exp(-1j * k0 * x_mesh**2 / 2.0 / qx)
    uy = 1.0 / np.sqrt(qy) * np.exp(-1j * k0 * y_mesh**2 / 2.0 / qy)
    uz = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_z)) * np.exp(
        -((z_mesh - z_mid) ** 2) / 2.0 / sigma_z**2
    )

    eta = 2.0 * k0 * zR * sigma_z / np.sqrt(np.pi)

    pulse = np.sqrt(eta) * np.sqrt(nphotons) * ux * uy * uz
    return pulse.astype(dtype)


def transverse_divergence_padding_factor(
    theta_max: float,
    drift_distance: float,
    beam_size: float,
) -> float:
    """
    Calculate the padding factor for the maximum divergence scenario.

    Parameters
    ----------
    theta_max : float
        Maximum transverse divergence [rad]
    drift_distance : float
        Drift propagation distance [m]
    beam_size : float
        Size of the beam at z=0 [m]

    Returns
    -------
    float
        Factor to increase the initial number of grid points, per dimension.
    """
    return 2.0 * (theta_max * drift_distance) / beam_size


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
        return WavefrontPadding(self.grid, fix_padding(self.grid, self.pad))

    def get_padded_shape(self, rmesh: np.ndarray) -> Tuple[int, ...]:
        """Get the padded shape given a 3D field rspace array."""
        nd = len(self.grid)
        if rmesh.ndim != nd:
            raise ValueError(f"`rmesh` is not an {nd}D array")

        # if not all(is_odd(dim) for dim in rmesh.shape):
        #     raise ValueError(
        #         f"`rmesh` dimensions are not all odd numbers: {rmesh.shape}"
        #     )

        return tuple(dim + 2 * pad for dim, pad in zip(rmesh.shape, self.pad))


def get_range_for_grid_spacing(grid_spacing: float, dim: int) -> Tuple[float, float]:
    """
    Given a grid spacing and array dimension, get the range the entire array represents.

    Parameters
    ----------
    grid_spacing : float
    dim : int

    Returns
    -------
    low : float
    high : float
    """
    # Generally, we expect an odd-dimension array which is centered around 0.
    # e.g.: 11 -> (-5, 5)
    half_dim = dim // 2
    if dim % 2 == 1:
        return (-grid_spacing * half_dim, grid_spacing * half_dim)

    # However, we may have an even dimension, in which case we choose to clip
    # the upper bound.
    # e.g., 10 -> (-5, 4)
    # TODO: confirm this is desirable
    clipped_half_dim = (dim - 1) // 2
    return (-grid_spacing * half_dim, grid_spacing * clipped_half_dim)


def get_ranges_for_grid_spacing(
    grid_spacing: Sequence[float],
    dims: Sequence[int],
) -> Sequence[Tuple[float, float]]:
    """
    Given a grid spacing and array dimension, get the range the entire array represents.

    Parameters
    ----------
    grid_spacing : list of float
    dim : list of int

    Returns
    -------
    tuples of (low, high) pairs
    """
    return tuple(
        get_range_for_grid_spacing(grid_spacing=delta, dim=dim)
        for delta, dim in zip(grid_spacing, dims)
    )


def _get_projection(img, axis: int) -> np.ndarray:
    sum_ = np.sum(img, axis=axis)
    return sum_ / np.max(sum_)


class Wavefront:
    """
    Particle field wavefront.

    Parameters
    ----------
    rmesh : np.ndarray
        Cartesian space field data. [V/m]
    wavelength : float
        Wavelength. [m]
    grid_spacing : sequence of float
        Grid spacing for the corresponding dimensions. [m]
    polarization : {"x", "y", "z"}, default="x"
        Direction of polarization.  The default assumes a planar undulator
        with electric field polarization in the X direction.
        Circular or generalized polarization is not currently supported.
    metadata : Metadata, optional
        OpenPMD-specific metadata.
    pad : int or tuple of int, optional
    pad_theta_max : float, default=5e-5
    pad_drift_distance : float, default=1.0
    pad_beam_size : float, default=1e-4

    See Also
    --------
    [OpenPMD standard](https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_Wavefront.md)
    """

    _rmesh: Optional[np.ndarray]
    _kmesh: Optional[np.ndarray]
    _phasors: Optional[Tuple[np.ndarray, ...]]
    # TODO:
    # time snapshot in Z
    # attributes on mesh show where it is in 3D space
    _wavelength: float
    _pad: WavefrontPadding
    _metadata: WavefrontMetadata

    def __init__(
        self,
        rmesh: np.ndarray,
        *,
        wavelength: float,
        grid_spacing: Optional[Sequence[float]] = None,
        pad: Optional[Union[int, Sequence[int]]] = None,
        polarization: Optional[PolarizationDirection] = None,
        axis_labels: Optional[Sequence[str]] = None,
        metadata: Optional[Union[WavefrontMetadata, dict]] = None,
        pad_theta_max: float = 5e-5,
        pad_drift_distance: float = 1.0,
        pad_beam_size: float = 1e-4,
        longitudinal_axis: Optional[str] = None,
    ) -> None:
        self._phasors = None
        self._rmesh = rmesh
        self._rmesh_shape = rmesh.shape
        self._kmesh = None
        self._wavelength = wavelength

        if pad is None:
            pad_factor = transverse_divergence_padding_factor(
                theta_max=pad_theta_max,
                drift_distance=pad_drift_distance,
                beam_size=pad_beam_size,
            )
            # TODO: current factor is just for transverse, we need to calculate
            # for longitudinal as well
            if pad_factor < 1:
                pad = (0, 0, 0)
            else:
                pad = tuple((dim * pad_factor) - dim for dim in rmesh.shape)

        self._set_metadata(
            metadata,
            pad=pad,
            polarization=polarization,
            axis_labels=axis_labels,
            grid_spacing=grid_spacing,
        )
        self._check_metadata()
        self._longitudinal_axis = longitudinal_axis or self.axis_labels[-1]

    def _set_metadata(
        self,
        metadata: Any,
        grid_spacing: Optional[Sequence[float]] = None,
        pad: Optional[Union[int, Sequence[int]]] = None,
        polarization: Optional[PolarizationDirection] = None,
        axis_labels: Optional[Sequence[str]] = None,
        # units: Optional[pmd_unit] = None,
    ) -> None:
        if metadata is None:
            metadata = WavefrontMetadata()

        if isinstance(metadata, dict):
            md = WavefrontMetadata.from_dict(metadata)
        elif isinstance(metadata, WavefrontMetadata):
            md = metadata
        else:
            raise ValueError(
                f"Unsupported type for metadata: {type(metadata).__name__}. Expected "
                f"'WavefrontMetadata', 'dict', or None to reset the metadata"
            )

        ndim = len(self._rmesh_shape)
        if isinstance(pad, int):
            pad = (pad,) * ndim
        if pad is None:
            if md.pads:
                pad = md.pads
            else:
                pad = (100,) * ndim

        self._pad = WavefrontPadding.from_array(self.rmesh, pad=pad, fix=True)
        md.pads = self._pad.pad
        if polarization is not None:
            md.polarization = polarization
        if axis_labels is not None:
            md.mesh.axis_labels = tuple(axis_labels)
        if grid_spacing is not None:
            md.mesh.grid_spacing = tuple(grid_spacing)
        # if units is not None:
        md.units = known_unit["V/m"]

        self._metadata = md

    def _check_metadata(self) -> None:
        if len(self.metadata.mesh.grid_spacing) != len(self._rmesh_shape):
            raise ValueError(
                "'grid_spacing' must have the same number of dimensions as `rmesh`; "
                "each should describe the cartesian range of the corresponding axis."
            )

        if len(self.metadata.mesh.axis_labels) != len(self._rmesh_shape):
            raise ValueError(
                "'axis_labels' must have the same number of dimensions as `rmesh`"
            )

    def __copy__(self) -> Wavefront:
        res = Wavefront.__new__(Wavefront)
        res._phasors = self._phasors
        res._rmesh_shape = self._rmesh_shape
        res._rmesh = self._rmesh
        res._kmesh = self._kmesh
        res._wavelength = self._wavelength
        res._pad = self._pad
        res._metadata = copy.deepcopy(self._metadata)
        # TODO there are more fields now
        res._longitudinal_axis = self._longitudinal_axis
        return res

    def __deepcopy__(self, memo) -> Wavefront:
        res = self.__copy__()
        res._rmesh = np.copy(self._rmesh) if self._rmesh is not None else None
        res._kmesh = np.copy(self._kmesh) if self._kmesh is not None else None
        return res

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False

        return all(
            (
                self._rmesh_shape == other._rmesh_shape,
                np.all(self._rmesh == other._rmesh),
                np.all(self._kmesh == other._kmesh),
                self._wavelength == other._wavelength,
                self._pad == other._pad,
                self.metadata == other.metadata,
            )
        )

    @classmethod
    def gaussian_pulse(
        cls,
        dims: Sequence[int],
        wavelength: float,
        nphotons: float,
        zR: float,
        sigma_z: float,
        grid_spacing: Sequence[float],
        pad: Optional[Sequence[int]] = None,
        dtype=np.complex64,
    ):
        """
        Generate a complex three-dimensional spatio-temporal Gaussian profile
        in terms of the q parameter.

        Parameters
        ----------
        dims :
            TODO update params
        wavelength : float
            Wavelength (lambda0) [m].
        nphotons : float
            Number of photons.
        zR : float
            Rayleigh range [m].
        sigma_z : float
            Pulse length RMS. [m]

        Returns
        -------
        Wavefront
        """
        pulse = create_gaussian_pulse_3d_with_q(
            wavelength=wavelength,
            nphotons=nphotons,
            zR=zR,
            sigma_z=sigma_z,
            grid_spacing=grid_spacing,
            grid=dims,
            dtype=dtype,
        )
        return cls(
            rmesh=pulse,
            wavelength=wavelength,
            grid_spacing=grid_spacing,
            pad=pad,
            axis_labels="xyz",
            longitudinal_axis="z",
        )

    @property
    def rspace_domain(self):
        """
        Cartesian space domain values in all dimensions.

        For each dimension of the wavefront, this is the evenly-spaced set of values over
        its specified range.
        """
        return cartesian_domain(ranges=self.ranges, grids=self._pad.grid)

    @property
    def kspace_domain(self):
        """
        Reciprocal space domain values in all dimensions.

        For each dimension of the wavefront, this is the evenly-spaced set of values over
        its specified range.
        """

        coeffs = conversion_coeffs(
            wavelength=self.wavelength,
            dim=len(self._rmesh_shape),
        )
        return nd_kspace_domains(
            coeffs=coeffs,
            sizes=self._rmesh_shape,
            pads=self.pad.pad,
            steps=self.grid_spacing,
            shifted=True,
        )

    @property
    def _nice_kspace_domain(self) -> _NiceXYZ:
        """
        `nice_array`-modified kspace domain.

        X and Y share units (m) and Z (eV) remains separate.
        """
        (domain_x, domain_y), xy_scale, xy_unit_prefix = nice_array(
            np.vstack(self.kspace_domain[:2])
        )
        domain_z, z_scale, z_unit_prefix = nice_array(self.kspace_domain[2])
        assert isinstance(domain_x, np.ndarray)
        assert isinstance(domain_y, np.ndarray)
        assert isinstance(domain_z, np.ndarray)
        return _NiceXYZ(
            x=domain_x,
            y=domain_y,
            xy_scale=xy_scale,
            xy_unit_prefix=xy_unit_prefix,
            z=domain_z,
            z_scale=z_scale,
            z_unit_prefix=z_unit_prefix,
        )

    @property
    def _k_center_indices(self) -> Tuple[int, ...]:
        return tuple(
            grid // 2 + pad for grid, pad in zip(self.rmesh.shape, self.pad.pad)
        )

    def _calc_phasors(self) -> Tuple[np.ndarray, ...]:
        """Calculate phasors for each dimension of the cartesian domain."""
        coeffs = conversion_coeffs(
            wavelength=self._wavelength,
            dim=len(self._rmesh_shape),
        )
        shifts = get_shifts(
            dims=self._pad.grid,
            ranges=self.ranges,
            pads=self._pad.pad,
            deltas=self.grid_spacing,
        )
        meshes_wkxky = nd_kspace_mesh(
            coeffs=coeffs,
            sizes=self._pad.grid,
            pads=self._pad.pad,
            steps=self.grid_spacing,
        )

        return tuple(
            np.exp(1j * 2.0 * np.pi * mesh * shift / coeff)
            for coeff, mesh, shift in zip(coeffs, meshes_wkxky, shifts)
        )

    @property
    def phasors(self) -> Tuple[np.ndarray, ...]:
        """Phasors for each dimension of the cartesian domain."""
        if self._phasors is None:
            self._phasors = self._calc_phasors()

        return self._phasors

    @property
    def rmesh(self) -> np.ndarray:
        """Real-space (cartesian space) wavefront field data."""
        if self._rmesh is None:
            self._rmesh = self._ifft()
        return self._rmesh

    @property
    def kmesh(self) -> np.ndarray:
        """K-space (reciprocal space) wavefront field data."""
        if self._kmesh is None:
            self._kmesh = self._fft()
        return self._kmesh

    @property
    def polarization(self) -> PolarizationDirection:
        """Polarization direction."""
        return self.metadata.polarization

    @polarization.setter
    def polarization(self, polarization: PolarizationDirection) -> None:
        if polarization not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported polarization direction: {polarization}")
        self.metadata.polarization = polarization

    @property
    def metadata(self) -> WavefrontMetadata:
        return self._metadata

    def _fft(self):
        """
        Calculate the FFT (rspace -> kspace) on the user data.

        Requires that the `_rmesh` data is available.

        This is intended to be handled by the `Wavefront` class itself, such
        that the user does not need to pay attention to whether the real-space
        or k-space wavefront data is up-to-date.
        """
        assert self._rmesh is not None
        workers = get_num_fft_workers()
        dfl_pad = _pad_array(self.rmesh, self._pad.pad_shape)
        return fft_phased(
            dfl_pad,
            axes=(0, 1, 2),
            phasors=self.phasors,
            workers=workers,
        )

    def _ifft(self):
        """
        Calculate the inverse FFT (kspace -> rspace) on the user data.

        Requires that the `_kmesh` data is available.

        This is intended to be handled by the `Wavefront` class itself, such
        that the user does not need to pay attention to whether the real-space
        or k-space wavefront data is up-to-date.
        """
        assert self._kmesh is not None
        workers = get_num_fft_workers()
        return ifft_phased(
            self._kmesh,
            axes=(0, 1, 2),
            phasors=self.phasors,
            workers=workers,
        )[self._pad.ifft_slices]

    @property
    def wavelength(self) -> float:
        """Wavelength of the wavefront [m]."""
        return self._wavelength

    @property
    def photon_energy(self) -> float:
        """Photon energy [eV]."""
        h = scipy.constants.value("Planck constant in eV/Hz") / (2 * np.pi)
        freq = scipy.constants.speed_of_light / self.wavelength
        return h * freq

    @property
    def pad(self):
        """Padding settings."""
        return self._pad

    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        return self.metadata.mesh.grid_spacing

    @property
    def ranges(self):
        return get_ranges_for_grid_spacing(
            grid_spacing=self.metadata.mesh.grid_spacing,
            dims=self.rmesh.shape,
        )

    @property
    def axis_labels(self):
        return self.metadata.mesh.axis_labels

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

        self._rmesh = (
            self.rmesh
            * thin_lens_kernel_xy(
                wavelength=self.wavelength,
                ranges=self.ranges,
                grid=self._pad.grid,
                f_lens_x=focus[0],
                f_lens_y=focus[1],
            )[:, :, np.newaxis]
        )
        # Invalidate the spectral data
        self._kmesh = None
        return self

    def drift(
        self,
        distance: float,
        *,
        inplace: bool = False,
    ) -> Wavefront:
        """
        Drift this Wavefront along the longitudinal direction in meters.

        Parameters
        ----------
        distance : float
            Distance in meters.
        inplace : bool, default=False
            Perform the operation in-place on this wavefront object.

        Returns
        -------
        Wavefront
            This object if `inplace=True` or a new copy if `inplace=False`.
        """

        if not inplace:
            wavefront = copy.copy(self)
            return wavefront.drift(distance, inplace=True)

        indices = [
            idx
            for idx, label in enumerate(self.axis_labels)
            if label != self._longitudinal_axis
        ]
        transverse_kspace_grid = nd_kspace_mesh(
            coeffs=(1.0,) * len(self._rmesh_shape),
            sizes=[self._pad.grid[idx] for idx in indices],
            pads=[self._pad.pad[idx] for idx in indices],
            steps=[self.grid_spacing[idx] for idx in indices],
        )

        self._kmesh = drift_propagator_paraxial(
            kmesh=self.kmesh,
            transverse_kspace_grid=transverse_kspace_grid,
            wavelength=self._wavelength,
            z=float(distance),
        )
        # Invalidate the real space data
        self._rmesh = None
        return self

    def _get_wigner_zphasor(self):
        coeff = conversion_coeffs(self.wavelength, dim=1)
        zlow, zhigh = self.ranges[2]
        shift = (zlow + zhigh) / 2.0
        (mesh,) = nd_kspace_mesh(
            coeffs=coeff,
            sizes=[self.pad.grid[2]],
            pads=[self.pad.pad[2]],
            steps=[self.grid_spacing[2]],
        )
        return np.exp(1j * 2.0 * np.pi * mesh * shift / coeff)

    def plot(
        self,
        plane: Plane,
        *,
        rspace: bool = True,
        show_real: bool = False,
        show_imaginary: bool = False,
        show_abs: bool = True,
        show_phase: bool = True,
        axs: Optional[List[matplotlib.axes.Axes]] = None,
        cmap: str = "viridis",
        figsize: Optional[Tuple[float, float]] = None,
        nrows: int = 1,
        ncols: int = 2,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        tight_layout: bool = True,
        save: Optional[AnyPath] = None,
        transpose: bool = False,
        colorbar: bool = True,
        # contour: bool = True,
    ):
        """
        Plot the projection onto the given plane.

        Parameters
        ----------
        plane : str, (int, int), or sequence of str
            Plane to plot. With axis_labels of "xyz", the following would be equivalent:
            * ``"xy"``
            * (1, 2)
            * ("x", "y")
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
        figsize : (float, float), optional
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
        transpose : bool, default=False
            Transpose the data for plotting.

        Returns
        -------
        Figure
        list of Axes
        """
        axis_indices = get_axis_indices(self.metadata.mesh.axis_labels, plane)

        if rspace:
            data = self.rmesh
            labels = get_rspace_labels(self.axis_labels, axis_indices)
            domain = [self.rspace_domain[idx] for idx in axis_indices]
            units = ["m", "m"]
        else:
            data = self.kmesh
            labels = get_kspace_labels(self.axis_labels, axis_indices)
            domain = [self.kspace_domain[idx] for idx in axis_indices]
            units = ["rad", "rad"]

        (domain_x, domain_y), _scale, unit_prefix = nice_array(np.vstack(domain))
        extent = (domain_x[0], domain_x[-1], domain_y[-1], domain_y[0])

        if transpose:
            data = data.T
            labels = tuple(reversed(labels))
            extent = tuple(reversed(extent))
            units = tuple(reversed(units))

        sum_axis = tuple(axis for axis in range(data.ndim) if axis not in axis_indices)
        plane_label = " ".join(labels)

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
        else:
            fig = axs[0].get_figure()
            assert fig is not None

        remaining_axes = list(axs)

        def plot(dat, title: str):
            ax = remaining_axes.pop(0)
            img = ax.imshow(np.mean(dat, axis=sum_axis), cmap=cmap, extent=extent)

            ax.set_xlabel(f"${labels[0]}$ ({unit_prefix}{units[0]})")
            ax.set_ylabel(f"${labels[1]}$ ({unit_prefix}{units[1]})")
            if colorbar:
                divider = make_axes_locatable(ax)
                fig = ax.get_figure()
                assert fig is not None
                cax = divider.append_axes("right", size="10%")
                fig.colorbar(img, cax=cax, orientation="vertical")
                cax.set_ylabel(title)
            else:
                ax.set_title(title)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            images.append(img)
            return img

        images = []
        if show_real:
            plot(np.real(data), title="Real")

        if show_imaginary:
            plot(np.imag(data), title="Imaginary")

        if show_abs:
            plot(np.abs(data) ** 2, f"$|{plane_label}|^2$")

        if show_phase:
            plot(np.angle(data), title="Phase")

        if fig is not None:
            if tight_layout:
                fig.tight_layout()

            if save:
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save)

        return fig, axs, images

    def plot_1d_kmesh_projections(
        self,
        *,
        axs: Optional[List[matplotlib.axes.Axes]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        tight_layout: bool = True,
        save: Optional[AnyPath] = None,
    ):
        if axs:
            (ax1, ax2) = axs
            fig = ax1.get_figure()
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        img_xy = np.sum(np.abs(self.kmesh) ** 2, axis=2)
        # TODO fix naming in other function
        proj_y = _get_projection(img_xy, axis=0)
        proj_x = _get_projection(img_xy, axis=1)

        kdomain = self._nice_kspace_domain
        ax1.plot(kdomain.y, proj_y, label="Vertical")
        ax1.plot(kdomain.x, proj_x, label="Horizontal")
        ax1.set_xlabel(rf"$\theta$ (${kdomain.xy_unit_prefix} rad$)")
        ax1.set_ylabel("Angular Divergence [arb units]")
        ax1.legend(loc="best")

        ax2.set_xlabel(rf"$\omega - \omega_0$ (${kdomain.z_unit_prefix} eV$)")
        ax2.set_ylabel("Spectrum [arb units]")
        img_xz = np.sum(np.abs(self.kmesh) ** 2, axis=1)
        proj_z = _get_projection(img_xz, axis=0)
        ax2.plot(kdomain.z, proj_z)

        if xlim is not None:
            ax1.set_xlim(xlim)
            ax2.set_ylim(xlim)
        if ylim is not None:
            ax2.set_xlim(ylim)
            ax1.set_ylim(ylim)
        if fig is not None:
            if tight_layout:
                fig.tight_layout()
            # TODO Bounding box options? Higher DPI than default?
            if save:
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save)
        return fig, (ax1, ax2)

    def _plot_reciprocal_thy_vs_thx(
        self,
        ax: matplotlib.axes.Axes,
        cmap: str = "viridis",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        kdomain = self._nice_kspace_domain

        extent = (
            np.min(kdomain.y),
            np.max(kdomain.y),
            np.min(kdomain.x),
            np.max(kdomain.x),
        )

        if not xlim:
            xlim = extent[0:2]
        if not ylim:
            ylim = extent[2:4]
        (xmin, xmax), (ymin, ymax) = xlim, ylim

        img = np.sum(np.abs(self.kmesh) ** 2, axis=2)
        proj_x = _get_projection(img, axis=0)
        proj_y = _get_projection(img, axis=1)

        # TODO: change these to subplots one th side
        im = ax.imshow(img, cmap=cmap, extent=extent, aspect="auto")
        ax.plot(
            kdomain.y,
            0.3 * xmax * proj_x + 0.9 * xmin,
            color="#17baca",
            linewidth=2.0,
        )
        ax.plot(
            0.3 * ymax * proj_y + 0.9 * ymin,
            kdomain.x,
            color="#17baca",
            linewidth=2.0,
        )

        xlabel, ylabel = get_kspace_labels(axis_labels=self.axis_labels, axes=(0, 1))
        ax.set_xlabel(rf"${xlabel}$ (${kdomain.xy_unit_prefix} rad$)")
        ax.set_ylabel(rf"${ylabel}$ (${kdomain.xy_unit_prefix} rad$)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return im

    def _plot_reciprocal_energy_vs_thetax(
        self,
        ax: matplotlib.axes.Axes,
        cmap: str = "viridis",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        kdomain = self._nice_kspace_domain
        extent = (
            np.min(kdomain.y),
            np.max(kdomain.y),
            np.min(kdomain.z),
            np.max(kdomain.z),
        )

        if not xlim:
            xlim = extent[0:2]
        if not ylim:
            ylim = extent[2:4]
        (xmin, xmax), (ymin, ymax) = xlim, ylim

        img = np.sum(np.abs(self.kmesh) ** 2, axis=1)
        proj_x = _get_projection(img, axis=0)
        proj_z = _get_projection(img, axis=1)

        im = ax.imshow(img.T, cmap=cmap, extent=extent, aspect="auto")
        step_kz = kdomain.z[1] - kdomain.z[0]
        ax.plot(
            kdomain.x,
            2 * xmax * proj_z + 8 * xmin + step_kz * 0.8,
            color="#17baca",
            linewidth=2.0,
        )
        ax.plot(
            np.flip(0.02 * ymax * proj_x + 0.9 * xmin),
            kdomain.z,
            color="#17baca",
            linewidth=2.0,
        )

        xlabel, ylabel = get_kspace_labels(axis_labels=self.axis_labels, axes=(0, 2))
        ax.set_ylabel(
            rf"Photon Energy, $(\Delta{ylabel} = {ylabel}-{ylabel}_0)$ (${kdomain.z_unit_prefix} eV$)"
        )
        ax.set_xlabel(rf"${xlabel}$ (${kdomain.xy_unit_prefix} rad$)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return im

    def plot_reciprocal(
        self,
        *,
        axs: Optional[List[matplotlib.axes.Axes]] = None,
        cmap: str = "viridis",
        figsize: Optional[Tuple[float, float]] = None,
        xlim_theta: Optional[Tuple[float, float]] = None,
        ylim_theta: Optional[Tuple[float, float]] = None,
        xlim_theta_w: Optional[Tuple[float, float]] = None,
        ylim_theta_w: Optional[Tuple[float, float]] = None,
        tight_layout: bool = True,
        save: Optional[AnyPath] = None,
        colorbar: bool = True,
    ):
        """
        Plot reciprocal space projections.

        Parameters
        ----------
        axs : List[matplotlib.axes.Axes], optional
            Plot the data in the provided matplotlib Axes.
            Creates a new figure and Axes if not specified.
        cmap : str, default="viridis"
            Color map to use.
        figsize : (float, float), optional
            Figure size for the axes.
            Defaults to Matplotlib's `rcParams["figure.figsize"]``.
        xlim_theta : (float, float), optional
            X axis limits for the thetay vs thetax plot.
        ylim_theta : (float, float), optional
            Y axis limits for the thetay vs thetax plot.
        xlim_theta_w : (float, float), optional
            X axis limits for the omega vs thetax plot.
        ylim_theta_w : (float, float), optional
            Y axis limits for the omega vs thetax plot.
        tight_layout : bool
            Use a tight layout.
        save : pathlib.Path or str, optional
            Save the resulting image to a file.
        colorbar : bool
            Add a colorbar to each image.

        Returns
        -------
        ax1 : matplotlib.axes.Axes
        ax2 : matplotlib.axes.Axes
        """
        if axs:
            ax1, ax2 = axs
        else:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        im1 = self._plot_reciprocal_thy_vs_thx(
            ax=ax1,
            cmap=cmap,
            xlim=xlim_theta,
            ylim=ylim_theta,
        )
        im2 = self._plot_reciprocal_energy_vs_thetax(
            ax=ax2,
            cmap=cmap,
            xlim=xlim_theta_w,
            ylim=ylim_theta_w,
        )

        for ax, im in [(ax1, im1), (ax2, im2)]:
            fig = ax.get_figure()
            if fig is not None:
                if colorbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="10%")
                    fig.colorbar(im, cax=cax, orientation="vertical")
                if tight_layout:
                    fig.tight_layout()

        if save:
            logger.info(f"Saving plot to {save!r}")
            fig = ax1.get_figure()
            assert fig is not None
            fig.savefig(save)

        return ax1, ax2

    @classmethod
    def from_genesis4(
        cls,
        h5: Union[h5py.File, pathlib.Path, str],
        pad: Union[int, Tuple[int, int, int]] = 100,
    ) -> Wavefront:
        """
        Load a Genesis4-format field file as a `Wavefront`.

        Parameters
        ----------
        h5 : h5py.File, pathlib.Path, or str
            The opened h5py File or a path to it on disk.
        pad : int or (int, int, int), default=100
            Specify either (padx, pady, padz) or all-around padding.

        Returns
        -------
        Wavefront
        """
        from genesis.version4 import FieldFile

        field = FieldFile.from_file(h5)

        # NOTE: refer to here for more information:
        # https://github.com/slaclab/lume-genesis/blob/master/docs/notes/genesis_fields.pdf
        Z0 = np.pi * 119.9169832  # V^2/W exactly
        genesis_to_v_over_m = np.sqrt(2.0 * Z0) / field.param.gridsize

        # TODO: to test:
        #   1. write gaussian field
        #   2. have genesis propagate it
        #   3. read the output from genesis, roughly compare `imported_wf.drift("z", ...)`
        #
        # Split:
        #   1. Run half genesis sim - get output
        #   2. Run full genesis sim - get output
        #   3. Drift (1) and compare with (2)

        # field.param.gridsize = 2 * field.dgrid / (ngrid - 1)
        wf = cls(
            rmesh=field.dfl * genesis_to_v_over_m,
            wavelength=field.param.wavelength,
            grid_spacing=(
                field.param.gridsize,
                field.param.gridsize,
                field.param.slicespacing,
            ),
            pad=pad,
            polarization="x",
            axis_labels="xyz",
            longitudinal_axis="z",
        )
        wf.metadata.mesh.grid_global_offset = (0.0, 0.0, field.param.refposition)
        return wf

    @classmethod
    def _from_h5_file(cls, h5: h5py.File) -> Wavefront:
        return cls()

    @classmethod
    def from_file(
        cls,
        h5: Union[h5py.File, pathlib.Path, str],
        identifier: int = 0,
    ) -> Wavefront:
        """Load a Wavefront from a file in the OpenPMD format."""
        if isinstance(h5, h5py.File):
            return cls._from_h5_file(h5)
        with h5py.File(h5) as h5p:
            return cls._from_h5_file(h5p)

    # names = get_wavefront_names_from_file("something.h5")
    # for name in names:
    #    wavefront = Wavefront.from_file("something.h5", identifier=name)
    #    wavefront.rmesh  # <-- single wavefront
    #
    # wavefront = Wavefront.from_file("something.h5", identifier=5)

    def _write_file(self, h5: h5py.File):
        md = self.metadata
        base_path_template = "/data/%T/"
        if md.index is not None:
            wavefront_base_path_template = "/wavefront/%T/"
        else:
            # For us, at least, second %T doesn't make much sense:
            wavefront_base_path_template = "/wavefront"
        wavefront_base_path = wavefront_base_path_template.replace("%T", str(md.index))
        base_path = base_path_template.replace("%T", str(md.iteration.iteration))

        # TODO: yes, this belongs in 'writers' - but I'm not sure what could/should be
        # reused/pulled apart/rewritten/etc just yet.
        writers.write_attrs(
            h5,
            {
                "basePath": base_path_template,
                "dataType": "openPMD",
                "openPMD": md.base.spec_version,
                # No particles, meshes in the file - per PMD spec:
                # - note: if this attribute is missing, the file is interpreted as if it
                #   contains *no particle records*! If the attribute is set, the group behind
                #   it *must* exist!
                # "meshesPath": meshes_path,
                # "particlesPath": particles_path,
                # TODO
                # "openPMDextension": "BeamPhysics;SpeciesType",
                "openPMDextension": "Wavefront",
                "wavefrontFieldPath": wavefront_base_path_template,  # TODO: add to standard
                **md.base.attrs,
            },
        )

        wavefront_path = base_path + wavefront_base_path
        base_group = h5.create_group(base_path)
        writers.write_attrs(base_group, md.iteration.attrs)

        electric_field_path = wavefront_path + "electricField/"
        efield_group = h5.create_group(electric_field_path)
        self.write_group(efield_group)

    def write(self, h5: Union[h5py.File, pathlib.Path, str]) -> None:
        """Write the Wavefront in OpenPMD format."""
        if isinstance(h5, h5py.File):
            return self._write_file(h5)

        with h5py.File(h5, "w") as h5p:
            return self._write_file(h5p)

    def write_group(self, group: h5py.Group) -> None:
        """Write the Wavefront in OpenPMD format to a specific HDF group."""

        writers.write_attrs(
            group,
            {
                "photonEnergy": self.photon_energy,
                "photonEnergyUnitSI": known_unit["eV"].unitSI,
                "photonEnergyUnitDimension": known_unit["eV"].unitDimension,
                "temporalDomain": "time",
                "spatialDomain": "r",
                **self.metadata.attrs,
            },
        )

        writers.write_component_data(
            group,
            name=self.polarization,
            data=self.rmesh,
            unit=self.metadata.units,
            attrs=self.metadata.mesh.attrs,
        )
