from __future__ import annotations

import copy
import logging
import pathlib
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Union

import h5py
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import scipy.fft
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import readers, tools, writers
from .metadata import PolarizationDirection, WavefrontMetadata
from .units import known_unit, nice_array

if TYPE_CHECKING:
    from genesis.version4 import FieldFile as Genesis4FieldFile

logger = logging.getLogger(__name__)


Ranges = Sequence[tuple[float, float]]
AnyPath = Union[str, pathlib.Path]
Z0 = np.pi * 119.9169832  # V^2/W exactly
HBAR_EV_M = scipy.constants.hbar / scipy.constants.e * scipy.constants.c  # eV-m
_rspace_labels = (
    "x",
    "y",
    "z",
)
_kspace_labels = (
    r"\theta_x",
    r"\theta_y",
    r"\omega",
)
PlotKey = Literal[
    "re",
    "im",
    "power_density",
    "phase",
]
Plane = Literal[
    "xy",
    "yz",
    "xz",
    "yx",
    "zy",
    "zx",
    "kxky",
    "kykz",
    "kxkz",
    "kykx",
    "kzky",
    "kzkx",
]
projection_key_to_indices = {
    "xy": ("rspace", (0, 1)),
    "yz": ("rspace", (1, 2)),
    "xz": ("rspace", (0, 2)),
    "yx": ("rspace", (1, 0)),
    "zy": ("rspace", (2, 1)),
    "zx": ("rspace", (2, 0)),
    "kxky": ("kspace", (0, 1)),
    "kykz": ("kspace", (1, 2)),
    "kxkz": ("kspace", (0, 2)),
    "kykx": ("kspace", (1, 0)),
    "kzky": ("kspace", (2, 1)),
    "kzkx": ("kspace", (2, 0)),
}


def pad_array(wavefront: np.ndarray, pads: Sequence[int] | int):
    """
    Pad an array with complex zero elements.

    Parameters
    ----------
    wavefront : np.ndarray
        The array to pad
    pads : array_like
        Number of values padded to the edges of each axis.

    Returns
    -------
    np.ndarray
    """

    if isinstance(pads, int):
        pad_shape = ((pads, pads),)
    else:
        pad_shape = tuple((pad, pad) for pad in pads)

    return np.pad(
        wavefront,
        pad_shape,
        mode="constant",
        constant_values=(0.0 + 1j * 0.0, 0.0 + 1j * 0.0),
    )


class _NiceXYZ(NamedTuple):
    """
    `nice_array`-modified values for x, y, and z domains.
    """

    x: np.ndarray
    x_scale: float
    x_unit_prefix: str

    y: np.ndarray
    y_scale: float
    y_unit_prefix: str

    z: np.ndarray
    z_scale: float
    z_unit_prefix: str

    @classmethod
    def from_domain(cls, domain: tuple[np.ndarray, ...]) -> _NiceXYZ:
        """`nice_array`-modified kspace domain."""
        assert len(domain) == 3

        domain_x, x_scale, x_unit_prefix = nice_array(domain[0])
        domain_y, y_scale, y_unit_prefix = nice_array(domain[1])
        domain_z, z_scale, z_unit_prefix = nice_array(domain[2])
        assert isinstance(domain_x, np.ndarray)
        assert isinstance(domain_y, np.ndarray)
        assert isinstance(domain_z, np.ndarray)
        return _NiceXYZ(
            x=domain_x,
            y=domain_y,
            z=domain_z,
            x_scale=x_scale,
            y_scale=y_scale,
            z_scale=z_scale,
            x_unit_prefix=x_unit_prefix,
            y_unit_prefix=y_unit_prefix,
            z_unit_prefix=z_unit_prefix,
        )


def nd_kspace_domains(
    coeffs: Sequence[float],
    sizes: Sequence[int],
    pads: Sequence[int],
    steps: Sequence[float],
    shifted: bool = True,
) -> tuple[np.ndarray, ...]:
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
    tuple of np.ndarray
    """

    if not shifted:
        return tuple(
            coeff * np.fft.fftfreq(n + 2 * pad, step)
            for coeff, n, pad, step in zip(coeffs, sizes, pads, steps)
        )

    return tuple(
        np.fft.fftshift(coeff * np.fft.fftfreq(n + 2 * pad, step))
        for coeff, n, pad, step in zip(coeffs, sizes, pads, steps)
    )


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
    ranges: Sequence[tuple[float, float]],
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
) -> tuple[int, ...]:
    """
    Fix gridding and padding values for symmetry and FFT efficiency.

    This works on all dimensions of gridding and padding.

    pad = (scipy_fft_fast_length - grid) // 2

    Such that the total array size will be: `grid + 2 * pad`.

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
) -> tuple[float, ...]:
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


def conversion_coeffs(wavelength: float, dim: int) -> tuple[float, ...]:
    """
    Conversion coefficients to (radians, radians, eV).

    Theta-x, theta-y, omega.
    """
    k0 = 2.0 * np.pi / wavelength
    return tuple([2.0 * np.pi / k0] * (dim - 1) + [2.0 * np.pi * HBAR_EV_M])


def cartesian_domain(
    ranges: Ranges,
    grids: Sequence[int],
):
    """Real-space domain of the non-padded field."""
    return tuple(
        np.linspace(range_[0], range_[1], grid) for range_, grid in zip(ranges, grids)
    )


def drift_kernel_paraxial(
    transverse_kspace_grid: list[np.ndarray],
    z: float,
    wavelength: float,
):
    """Drift transfer function in Z [m] paraxial approximation."""
    kx, ky = transverse_kspace_grid
    return np.exp(-1j * z * np.pi * wavelength * (kx**2 + ky**2))


def drift_propagator_paraxial(
    kmesh: np.ndarray,
    transverse_kspace_grid: list[np.ndarray],
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

    TODO: Alex - provide a reference and an actual LaTeX equation.

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
    k0 = 2.0 * np.pi / wavelength
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

    k0 = 2.0 * np.pi / wavelength
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


PadShape = tuple[tuple[int, int], ...]


def get_padded_shape(
    rmesh_shape: Sequence[int], padding: Sequence[int]
) -> tuple[int, ...]:
    """Get the padded shape given a 3D field rspace array."""
    rmesh_shape = tuple(rmesh_shape)
    padding = tuple(padding)
    if len(rmesh_shape) != len(padding):
        raise ValueError(
            f"Padding shape must be equal to the array dimensions. "
            f"Got {len(padding)} but expected {len(rmesh_shape)}"
        )

    return tuple(dim + 2 * pad for dim, pad in zip(rmesh_shape, padding))


def get_range_for_grid_spacing(grid_spacing: float, dim: int) -> tuple[float, float]:
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
) -> Sequence[tuple[float, float]]:
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


def calculate_phasors(
    wavelength: float,
    rmesh_grid_spacing: tuple[float, ...],
    rmesh_grid: tuple[int, ...],
    pad: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Calculate phasors for each dimension of the cartesian domain."""
    ranges = get_ranges_for_grid_spacing(
        grid_spacing=rmesh_grid_spacing,
        dims=rmesh_grid,
    )
    coeffs = conversion_coeffs(
        wavelength=wavelength,
        dim=len(rmesh_grid),
    )
    shifts = get_shifts(
        dims=rmesh_grid,
        ranges=ranges,
        pads=pad,
        deltas=rmesh_grid_spacing,
    )
    meshes_wkxky = nd_kspace_mesh(
        coeffs=coeffs,
        sizes=rmesh_grid,
        pads=pad,
        steps=rmesh_grid_spacing,
    )

    return tuple(
        np.exp(1j * 2.0 * np.pi * mesh * shift / coeff)
        for coeff, mesh, shift in zip(coeffs, meshes_wkxky, shifts)
    )


def wavelength_to_photon_energy(wavelength: float) -> float:
    h = scipy.constants.value("Planck constant in eV/Hz") / (2 * np.pi)
    freq = scipy.constants.speed_of_light / wavelength
    return h * freq


def wigner_zphasor(wavelength: float, pad: float, grid: int, grid_spacing: float):
    """
    Calculate the Wigner z-Phasor.

    Parameters
    ----------
    wavelength : float
        The wavelength of the wavefront.
    pad : float
        Padding size for the grid in Z.
    grid : int
        Number of grid points in Z.
    grid_spacing : float
        Spacing between grid points in Z.

    Returns
    -------
    ndarray
        The computed Wigner z-Phasor.
    """
    coeff = conversion_coeffs(wavelength, dim=1)
    shift = (grid + pad) * grid_spacing
    (mesh,) = nd_kspace_mesh(
        coeffs=coeff,
        sizes=[grid],
        pads=[0],  # self.pad[2]],
        steps=[grid_spacing],
    )
    return np.exp(1j * 2.0 * np.pi * mesh * shift / coeff)


def _wavefront_ids_from_h5_file(h5: h5py.File) -> list[str]:
    extensions = tools.require_h5_string_attr(h5, "openPMDextension").split(";")
    if "wavefront" in extensions:
        data = tools.require_h5_group(h5, "/data")
        ids = []
        for name, group in data.items():
            if isinstance(group, h5py.Group) and "meshes" in group:
                ids.append(name)
        return ids

    if "Wavefront" not in extensions:
        raise ValueError(
            f"Wavefront extension not enabled in file. "
            f"Extensions configured: {extensions}"
        )

    # TODO this will work in the normal case but perhaps not in other weird configurations
    base_path = tools.require_h5_string_attr(h5, "basePath").replace("%T", "")
    base_group = tools.require_h5_group(h5, base_path)
    ids = []
    for name, group in base_group.items():
        if isinstance(group, h5py.Group) and "wavefront" in group:
            ids.append(name)
    return ids


def wavefront_ids_from_file(
    h5: h5py.File | pathlib.Path | str,
) -> list[str]:
    """
    Get a list of Wavefront identifiers available in the given file.

    Parameters
    ----------
    h5 : h5py.File or pathlib.Path or str
        The HDF5 file instance or a path to the file.

    Returns
    -------
    list of str
    """
    if isinstance(h5, h5py.File):
        return _wavefront_ids_from_h5_file(h5)

    with h5py.File(h5, "r") as h5p:
        return _wavefront_ids_from_h5_file(h5p)


class Wavefront:
    """
    Particle field wavefront.

    Only 3D Wavefront meshes are currently supported.

    Axis labels are implicitly "x", "y", and "z", with Z as the longitudinal
    axis.

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

    See Also
    --------
    [OpenPMD standard](https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_Wavefront.md)
    """

    # Saved into OpenPMD-compatible file:
    _rmesh: np.ndarray | None
    wavelength: float
    _metadata: WavefrontMetadata

    # Internal state for Wavefront:
    _kmesh: np.ndarray | None
    _grid: tuple[int, ...]  # rmesh shape
    _padding: tuple[int, ...]
    # TODO:
    # time snapshot in Z
    # attributes on mesh show where it is in 3D space

    def __init__(
        self,
        rmesh: np.ndarray,
        *,
        wavelength: float,
        grid_spacing: Sequence[float] | None = None,
        polarization: PolarizationDirection | None = None,
        metadata: WavefrontMetadata | dict | None = None,
        pad: Sequence[int] | int | None = None,
        fix_pad: bool = True,
    ) -> None:
        if rmesh.ndim != 3:
            raise NotImplementedError(
                "Only 3D Wavefront meshes are currently supported "
                "with implicit axis labels of {x,y,z} with Z as the longitudinal axis."
            )

        self._rmesh = rmesh
        self._grid = rmesh.shape
        self._kmesh = None
        self.wavelength = wavelength
        if isinstance(pad, int):
            self._padding = (pad,) * rmesh.ndim
        elif pad is None:
            self._padding = (0,) * rmesh.ndim
        else:
            self._padding = tuple(pad)

        if fix_pad:
            self._padding = fix_padding(self._grid, pad=self._padding)

        self._set_metadata(
            metadata,
            polarization=polarization,
            axis_labels=("x", "y", "z"),
            grid_spacing=grid_spacing,
        )
        self._check_metadata()

    def _set_metadata(
        self,
        metadata: Any,
        grid_spacing: Sequence[float] | None = None,
        polarization: PolarizationDirection | None = None,
        axis_labels: Sequence[str] | None = None,
        # units: Optional[pmd_unit] = None,
    ) -> None:
        if metadata is None:
            md = WavefrontMetadata()
        elif isinstance(metadata, dict):
            md = WavefrontMetadata.from_dict(metadata)
        elif isinstance(metadata, WavefrontMetadata):
            md = copy.deepcopy(metadata)
        else:
            raise ValueError(
                f"Unsupported type for metadata: {type(metadata).__name__}. Expected "
                f"'WavefrontMetadata', 'dict', or None to reset the metadata"
            )

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
        if len(self.metadata.mesh.grid_spacing) != len(self._grid):
            raise ValueError(
                "'grid_spacing' must have the same number of dimensions as `rmesh`; "
                "each should describe the cartesian range of the corresponding axis."
            )

        if len(self.metadata.mesh.axis_labels) != len(self._grid):
            raise ValueError(
                "'axis_labels' must have the same number of dimensions as `rmesh`"
            )

    def __copy__(self) -> Wavefront:
        res = Wavefront.__new__(Wavefront)
        res._grid = self._grid
        res._rmesh = self._rmesh
        res._kmesh = self._kmesh
        res.wavelength = self.wavelength
        res._padding = self._padding
        res._metadata = copy.deepcopy(self._metadata)
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
                self._grid == other._grid,
                np.all(self._rmesh == other._rmesh),
                np.all(self._kmesh == other._kmesh),
                self.wavelength == other.wavelength,
                self.pad == other.pad,
                self.metadata == other.metadata,
            )
        )

    def __repr__(self) -> str:
        def describe_arr(arr: np.ndarray | None):
            if arr is None:
                return "<to be calculated>"
            return f"<np.ndarray of shape {arr.shape} (nbytes={arr.nbytes})>"

        rmesh = describe_arr(self._rmesh)
        kmesh = describe_arr(self._kmesh)
        wavelength = self.wavelength
        return f"<{type(self).__name__} {wavelength=} rmesh={rmesh} grid={self.grid} kmesh={kmesh}>"

    @classmethod
    def gaussian_pulse(
        cls,
        dims: tuple[int, int, int],
        wavelength: float,
        nphotons: float,
        zR: float,
        sigma_z: float,
        grid_spacing: Sequence[float],
        pad: Sequence[int] | None = None,
        dtype=np.complex64,
    ):
        """
        Generate a complex three-dimensional spatio-temporal Gaussian profile
        in terms of the q parameter.

        Parameters
        ----------
        dims : tuple of (nx, ny, nz)
            Shape of the generated pulse.
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
        if len(dims) != 3:
            raise ValueError("Only 3D wavefronts are supported currently")
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
        )

    def with_rmesh(
        self,
        rmesh: np.ndarray,
        pad: Sequence[int] | int | None = None,
        fix_pad: bool = True,
    ) -> Wavefront:
        """
        Create a new Wavefront instance, replacing the `rmesh`.

        Parameters
        ----------
        rmesh : np.ndarray
        pad : int or sequence of int, optional
            New padding settings.
        fix : bool, optional
            Fix padding for efficiency.

        Returns
        -------
        Wavefront
        """
        if pad is None:
            pad = self.pad
        return Wavefront(
            rmesh=rmesh,
            wavelength=self.wavelength,
            metadata=self.metadata,
            pad=self.pad,
            fix_pad=fix_pad,
        )

    def with_padding(
        self,
        pad: int | Sequence[int],
        fix: bool = True,
    ) -> Wavefront:
        """
        Get a new Wavefront instance with adjusted padding settings.

        Parameters
        ----------
        pad : int or sequence of int
            The new padding settings.
        fix : bool, optional
            Fix padding for efficiency.

        Returns
        -------
        Wavefront
        """
        ndim = len(self._grid)
        if isinstance(pad, int):
            pad = (pad,) * ndim
        else:
            if len(pad) != ndim:
                raise ValueError(
                    f"Padding shape must be equal to the array dimensions. "
                    f"Got {len(pad)} but expected {ndim}"
                )

        return Wavefront(
            rmesh=self.rmesh,
            wavelength=self.wavelength,
            metadata=self.metadata,
            pad=pad,
            fix_pad=fix,
        )

    def with_padding_divergence(
        self,
        theta_max: float = 5e-5,
        drift_distance: float = 1.0,
        beam_size: float = 1e-4,
        fix: bool = True,
    ) -> Wavefront:
        """
        Using divergence settings, create a new Wavefront with padding adjusted by the factor:

        2.0 * (theta_max * drift_distance) / beam_size

        Parameters
        ----------
        theta_max : float, optional
            Maximum divergence angle in radians. Default is 5e-5.
        drift_distance : float, optional
            Maximum distance over which the beam will be drifted in meters.
            Default is 1.0.
        beam_size : float, optional
            Size of the beam in meters. Default is 1e-4.
        fix : bool, optional
            Fix padding for efficiency.

        Returns
        -------
        Wavefront
            The new wavefront instance with the padding settings.
        """
        pad_factor = transverse_divergence_padding_factor(
            theta_max=theta_max,
            drift_distance=drift_distance,
            beam_size=beam_size,
        )
        # TODO: current factor is just for transverse, we need to calculate
        # for longitudinal as well
        if pad_factor < 1:
            pad = (0, 0, 0)
        else:
            pad = tuple(int((dim * pad_factor) - dim) for dim in self._grid)

        return self.with_padding(pad, fix=fix)

    @property
    def grid(self) -> tuple[int, ...]:
        """The rmesh shape, without padding."""
        return self._grid

    @property
    def pad(self) -> tuple[int, ...]:
        """
        Per-dimension padding used in the FFT.

        To change this, use `.with_padding` or `.with_padding_divergence` and
        instantiate a new `Wavefront` instance.
        """
        return tuple(self._padding)

    @property
    def rspace_domain(self):
        """
        Cartesian space domain values in all dimensions.

        For each dimension of the wavefront, this is the evenly-spaced set of values over
        its specified range.
        """
        return cartesian_domain(ranges=self.ranges, grids=self._grid)

    @property
    def kspace_domain(self) -> tuple[np.ndarray, ...]:
        """
        Reciprocal space domain values in all dimensions.

        For each dimension of the wavefront, this is the evenly-spaced set of values over
        its specified range.
        """

        coeffs = conversion_coeffs(
            wavelength=self.wavelength,
            dim=len(self._grid),
        )
        return nd_kspace_domains(
            coeffs=coeffs,
            sizes=self._grid,
            pads=self.pad,
            steps=self.grid_spacing,
            shifted=True,
        )

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

    @property
    def fft_unit_coeff(self) -> float:
        """
        FFT unit conversion coefficient.

        See tech note section 3.4 (Plotting field intensity angular distribution for n-th slice)
        """
        padded_shape = get_padded_shape(self._grid, padding=self._padding)
        return float(
            np.prod(np.asarray(self.grid_spacing) ** 2)
            * np.prod(padded_shape)
            * self.k0**2
            / (8 * np.pi**3 * HBAR_EV_M * scipy.constants.c**2)
        )

    def _fft(self):
        """
        Calculate the FFT (rspace -> kspace) on the user data.

        Requires that the `_rmesh` data is available.

        This is intended to be handled by the `Wavefront` class itself, such
        that the user does not need to pay attention to whether the real-space
        or k-space wavefront data is up-to-date.
        """
        assert self._rmesh is not None
        workers = tools.get_num_fft_workers()
        dfl_pad = pad_array(self._rmesh, self._padding)

        phasors = calculate_phasors(
            wavelength=self.wavelength,
            rmesh_grid_spacing=self.grid_spacing,
            rmesh_grid=self._grid,
            pad=self._padding,
        )
        axes = (0, 1, 2)
        array_fft = scipy.fft.fftn(dfl_pad, axes=axes, workers=workers, norm="ortho")
        for phasor in phasors:
            array_fft *= phasor
        return scipy.fft.fftshift(array_fft, axes=axes)

    def _ifft(self):
        """
        Calculate the inverse FFT (kspace -> rspace) on the user data.

        Requires that the `_kmesh` data is available.

        This is intended to be handled by the `Wavefront` class itself, such
        that the user does not need to pay attention to whether the real-space
        or k-space wavefront data is up-to-date.
        """
        assert self._kmesh is not None
        workers = tools.get_num_fft_workers()
        phasors = calculate_phasors(
            wavelength=self.wavelength,
            rmesh_grid_spacing=self.grid_spacing,
            rmesh_grid=self._grid,
            pad=self._padding,
        )

        array_fft = scipy.fft.ifftn(
            self._kmesh,
            axes=(0, 1, 2),
            workers=workers,
            norm="ortho",
        )
        for phasor in phasors:
            array_fft *= np.conj(phasor)
        full_ifft = scipy.fft.ifftshift(array_fft, axes=(0, 1, 2))

        # Remove padding from the inverse fft result:
        ifft_slices = tuple(slice(pad, -pad) for pad in self._padding)
        return full_ifft[ifft_slices]

    @property
    def k0(self) -> float:
        """Wave number. [m^-1]"""
        return 2.0 * np.pi / self.wavelength

    @property
    def photon_energy(self) -> float:
        """Photon energy [eV]."""
        return wavelength_to_photon_energy(self.wavelength)

    @property
    def grid_spacing(self) -> tuple[float, ...]:
        """Per-dimension grid spacing in rspace. [m]"""
        return self.metadata.mesh.grid_spacing

    @property
    def ranges(self):
        """
        Get the physical range the entire rmesh represents in (low, high) pairs
        for each dimension.

        Returns
        -------
        (low_x, high_x)
            Range of X values [m].
        (low_y, high_y)
            Range of Y values [m].
        (low_z, high_z)
            Range of Z values [eV].
        """
        return get_ranges_for_grid_spacing(
            grid_spacing=self.metadata.mesh.grid_spacing,
            dims=self.rmesh.shape,
        )

    def wigner_distribution(self) -> np.ndarray:
        """
        Compute the Wigner distribution.

        Returns
        -------
        np.ndarray
        """
        xgrid, ygrid, zgrid = self.grid
        _xpad, _ypad, zpad = self.pad

        sig_z_corr = np.zeros((zgrid + 2 * zpad, zgrid), dtype=complex)

        sig_z = self.rmesh[xgrid // 2 + 1, ygrid // 2 + 1, :]

        sig_z_pad = pad_array(sig_z, zpad)
        sig_zconj_pad = np.conj(sig_z_pad)

        plt.plot(np.abs(sig_z_pad) ** 2)
        plt.show()

        for i in range(0, zgrid):
            j = zpad - i
            sig_z_corr[:, i] = np.roll(sig_z_pad, j) * np.roll(sig_zconj_pad, -j)

        zphasor = wigner_zphasor(
            wavelength=self.wavelength,
            pad=self.pad[2],
            grid=self.grid[2],
            grid_spacing=self.grid_spacing[2],
        )
        WT = np.fft.fftshift(zphasor * np.fft.fft(sig_z_corr, axis=1), axes=1)
        return np.real(WT[zpad : zgrid + zpad, :])

    def plot_wigner_distribution(
        self,
        ax: matplotlib.axes.Axes | None = None,
        cmap: str = "bwr",
        figsize: tuple[float, float] | None = None,
        save: AnyPath | None = None,
        colorbar: bool = True,
    ):
        """
        Plot the Wigner distribution for the wavefront.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            assert ax is not None
        else:
            fig = ax.get_figure()
            assert fig is not None

        _, _, z = self.rspace_domain
        kdomain = _NiceXYZ.from_domain(self.kspace_domain)
        extent = (
            float(np.min(z)),  # left
            float(np.max(z)),  # right
            float(np.min(kdomain.z) / 2.0),  # bottom
            float(np.max(kdomain.z) / 2.0),  # top
        )
        WT = self.wigner_distribution()
        im = ax.imshow(WT, cmap=cmap, extent=extent, aspect="auto")
        ax.set_xlabel(f"Time (${kdomain.z_unit_prefix} s$)")
        ax.set_ylabel("Photon energy ($eV$)")
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%")
            fig.colorbar(im, cax=cax, orientation="vertical")

        if fig is not None:
            if save:
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save, dpi=writers.savefig_dpi, bbox_inches="tight")

        return fig, ax

    def plot(
        self,
        key: PlotKey,
        *,
        projection: Plane = "xy",
        isophase_contour: bool = False,
        ax: matplotlib.axes.Axes | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        save: AnyPath | None = None,
        colorbar: bool = True,
        aspect: float | Literal["equal", "auto"] = "auto",
    ):
        """
        Plot the projection onto the given plane.

        Parameters
        ----------
        key : {"re", "im", "power_density", "phase"}
            The type of data to plot.
        projection : {"xy", "yz", "xz", "kxky", "kykz", "kxkz"}
            The plane to project onto.  Flipped versions are also supported.
        isophase_contour : bool, default=False
            Add isophase contour to the phase plot.
        figsize : (float, float), optional
            Figure size for the axes.
            Defaults to Matplotlib's `rcParams["figure.figsize"]``.
        ax : matplotlib.axes.Axes, optional
            Plot the data in the provided matplotlib Axes.
            Creates a new figure and Axes if not specified.
        cmap : str, default="viridis"
            Color map to use.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        aspect : float, "equal", or "auto", optional
            Aspect ratio for the plot. Default is "auto".

        Returns
        -------
        Figure
        list of Axes
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            assert ax is not None
        else:
            fig = ax.get_figure()

        try:
            r_or_k, axis_indices = projection_key_to_indices[projection]
        except KeyError:
            raise ValueError(
                f"Unsupported projection: {projection} choose from {list(projection_key_to_indices)}"
            ) from None

        if r_or_k == "rspace":
            mesh_data = self.rmesh
            labels = [_rspace_labels[idx] for idx in axis_indices]
            domain = [self.rspace_domain[idx] for idx in axis_indices]
            units = ["m", "m"]
        else:
            mesh_data = self.kmesh
            labels = [_kspace_labels[idx] for idx in axis_indices]
            domain = [self.kspace_domain[idx] for idx in axis_indices]
            units = ["rad", "rad"]

        domain_x, _scale, x_unit_prefix = nice_array(domain[0])
        domain_y, _scale, y_unit_prefix = nice_array(domain[1])
        assert isinstance(domain_x, np.ndarray)
        assert isinstance(domain_y, np.ndarray)
        extent = (domain_x[0], domain_x[-1], domain_y[-1], domain_y[0])

        sum_axis = tuple(
            axis for axis in range(mesh_data.ndim) if axis not in axis_indices
        )

        assert len(sum_axis) == 1
        sum_axis = sum_axis[0]

        def plot(dat, title: str):
            # TODO: balticfish will double-check
            # _z_min, z_max = self.ranges[sum_axis]
            # dz = self.grid_spacing[sum_axis]
            # dat = np.sum(dat, axis=sum_axis) * dz / (2.0 * z_max)
            img = ax.imshow(
                np.mean(dat, axis=sum_axis),
                cmap=cmap,
                extent=extent,
                aspect=aspect,
            )

            ax.set_xlabel(f"${labels[0]}$ ({x_unit_prefix}{units[0]})")
            ax.set_ylabel(f"${labels[1]}$ ({y_unit_prefix}{units[1]})")
            if colorbar:
                divider = make_axes_locatable(ax)
                fig = ax.get_figure()
                assert fig is not None
                cax = divider.append_axes("right", size="10%", pad=0.1)
                fig.colorbar(img, cax=cax, orientation="vertical")
                cax.set_ylabel(title)
            else:
                ax.set_title(title)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            return img

        if key == "re":
            img = plot(np.real(mesh_data), title=r"$\Re(\text{mesh})$")

        elif key == "im":
            img = plot(np.imag(mesh_data), title=r"$\Im(\text{mesh})$")

        elif key == "power_density":
            if r_or_k == "rspace":
                # See tech note 3.2
                power_density = 1 / 1e4 * np.abs(mesh_data) ** 2 / (2.0 * Z0)
                img = plot(power_density, "Power density (W/cm$^2$)")
            else:
                # See tech note 3.6 (coefficient is in 3.4)
                power_density = (
                    np.abs(mesh_data) ** 2 / (2.0 * Z0) * self.fft_unit_coeff
                )
                img = plot(power_density, "Power density (J/eV/rad$^2$)")

        elif key == "phase":
            phase = np.angle(mesh_data)
            if isophase_contour:
                img = plot(phase, title="Phase")
                ax.contour(
                    np.mean(phase, axis=sum_axis),
                    cmap="Greys",
                    extent=extent,
                )
            else:
                img = plot(phase, title=r"$Phase(\text{mesh})$")

        else:
            valid_keys = typing.get_args(PlotKey)
            raise ValueError(
                f"Unsupported plot key: {key}. Supported keys: {valid_keys}"
            )

        if fig is not None:
            if save:
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save, dpi=writers.savefig_dpi, bbox_inches="tight")

        return fig, ax, img

    def plot_1d_far_field_spectral_density(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        tight_layout: bool = True,
        save: AnyPath | None = None,
    ):
        if ax is not None:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

            assert ax is not None

        density = (
            self.fft_unit_coeff * np.abs(self.kmesh * self.kmesh.conj()) / (2.0 * Z0)
        )
        nx, ny, _nz = density.shape
        data = density[nx // 2, ny // 2, :]

        kdomain = _NiceXYZ.from_domain(self.kspace_domain)
        xlabel = _kspace_labels[2]
        ax.plot(kdomain.z, data)
        ax.set_xlabel(f"${xlabel} ({kdomain.z_unit_prefix} eV)$")
        ax.set_ylabel("Far-field spectral intensity ($J/eV$)")

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_xlim(ylim)
        if fig is not None:
            if tight_layout:
                fig.tight_layout()
            if save:
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save, dpi=writers.savefig_dpi, bbox_inches="tight")
        return fig, ax

    def plot_1d_kmesh_angular_divergence(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        save: AnyPath | None = None,
    ):
        if ax is not None:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(figsize=figsize)
            assert ax is not None

        img_xy = np.sum(np.abs(self.kmesh) ** 2, axis=2)
        # TODO fix naming in other function
        proj_y = _get_projection(img_xy, axis=0)
        proj_x = _get_projection(img_xy, axis=1)

        kdomain = _NiceXYZ.from_domain(self.kspace_domain)
        ax.plot(kdomain.y, proj_y, label="Vertical")
        ax.plot(kdomain.x, proj_x, label="Horizontal")
        ax.set_xlabel(rf"$\theta$ (${kdomain.x_unit_prefix} rad$)")
        ax.set_ylabel("Angular Divergence [arb units]")
        ax.legend(loc="best")

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if fig is not None:
            if save:
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save, dpi=writers.savefig_dpi, bbox_inches="tight")
        return fig, ax

    def plot_1d_kmesh_spectrum(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        save: AnyPath | None = None,
    ):
        if ax is not None:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(figsize=figsize)
            assert ax is not None

        kdomain = _NiceXYZ.from_domain(self.kspace_domain)

        ax.set_xlabel(rf"$\omega - \omega_0$ (${kdomain.z_unit_prefix} eV$)")
        ax.set_ylabel("Spectrum [arb units]")
        img_xz = np.sum(np.abs(self.kmesh) ** 2, axis=1)
        proj_z = _get_projection(img_xz, axis=0)
        ax.plot(kdomain.z, proj_z)

        if xlim is not None:
            ax.set_ylim(xlim)
        if ylim is not None:
            ax.set_xlim(ylim)
        if fig is not None:
            if save:
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save, dpi=writers.savefig_dpi, bbox_inches="tight")
        return fig, ax

    def _plot_reciprocal_thy_vs_thx(
        self,
        ax: matplotlib.axes.Axes,
        cmap: str = "viridis",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ):
        kdomain = _NiceXYZ.from_domain(self.kspace_domain)

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

        xlabel, ylabel = _kspace_labels[:2]
        ax.set_xlabel(rf"${xlabel}$ (${kdomain.x_unit_prefix} rad$)")
        ax.set_ylabel(rf"${ylabel}$ (${kdomain.y_unit_prefix} rad$)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return im

    def _plot_reciprocal_energy_vs_thetax(
        self,
        ax: matplotlib.axes.Axes,
        cmap: str = "viridis",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ):
        kdomain = _NiceXYZ.from_domain(self.kspace_domain)
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

        xlabel, ylabel = _kspace_labels[0], _kspace_labels[2]
        ax.set_ylabel(
            rf"Photon Energy, $(\Delta{ylabel} = {ylabel}-{ylabel}_0)$ (${kdomain.z_unit_prefix} eV$)"
        )
        ax.set_xlabel(rf"${xlabel}$ (${kdomain.x_unit_prefix} rad$)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return im

    def plot_reciprocal(
        self,
        *,
        axs: list[matplotlib.axes.Axes] | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] | None = None,
        xlim_theta: tuple[float, float] | None = None,
        ylim_theta: tuple[float, float] | None = None,
        xlim_theta_w: tuple[float, float] | None = None,
        ylim_theta_w: tuple[float, float] | None = None,
        tight_layout: bool = True,
        save: AnyPath | None = None,
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
            fig.savefig(save, dpi=writers.savefig_dpi, bbox_inches="tight")

        return ax1, ax2

    @classmethod
    def from_kmesh(
        cls,
        kmesh: np.ndarray,
        padding: Sequence[int] | None,
        *,
        wavelength: float,
        grid_spacing: Sequence[float] | None = None,
        polarization: PolarizationDirection | None = None,
        metadata: WavefrontMetadata | dict | None = None,
    ) -> Wavefront:
        """
        Create a Wavefront from a k-space mesh.

        Parameters
        ----------
        kmesh : np.ndarray
            The k-space mesh.
        padding : Sequence[int] or None
            Padding to be applied to the mesh.
        wavelength : float
            Wavelength of the wavefront.
        grid_spacing : Sequence[float] or None, optional
            Grid spacing for the wavefront.
        polarization : PolarizationDirection or None, optional
            Polarization direction of the wavefront.
        metadata : WavefrontMetadata or dict or None, optional
            Metadata associated with the wavefront.

        Returns
        -------
        Wavefront
        """
        if padding is None:
            padding = (0,) * kmesh.ndim
        else:
            padding = tuple(padding)

        if kmesh.ndim != len(padding):
            raise ValueError(
                f"Padding shape must be equal to the array dimensions. "
                f"Got {len(padding)} but expected {kmesh.ndim}"
            )

        self = Wavefront.__new__(Wavefront)
        self._rmesh = None
        self._kmesh = kmesh
        self._grid = tuple(dim - 2 * pad for dim, pad in zip(kmesh.shape, padding))
        self._padding = padding
        self.wavelength = wavelength

        self._set_metadata(
            metadata,
            polarization=polarization,
            axis_labels=("x", "y", "z"),
            grid_spacing=grid_spacing,
        )
        self._check_metadata()
        return self

    @classmethod
    def from_genesis4_fieldfile(
        cls,
        field: Genesis4FieldFile,
        pad: int | tuple[int, int, int] = 0,
    ) -> Wavefront:
        """
        Load a Genesis4 `FieldFile` as a `Wavefront`.

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
        from .interfaces.genesis import genesis4_fieldfile_to_wavefront

        return genesis4_fieldfile_to_wavefront(field, pad=pad)

    @classmethod
    def from_genesis4(
        cls,
        h5: h5py.File | pathlib.Path | str,
        pad: int | tuple[int, int, int] = 0,
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

        field_file = FieldFile.from_file(h5)
        return cls.from_genesis4_fieldfile(field_file, pad=pad)

    def to_genesis4_fieldfile(self) -> Genesis4FieldFile:
        """Convert to an in-memory Genesis4 `FieldFile` instance."""
        from .interfaces.genesis import wavefront_to_genesis4_fieldfile

        return wavefront_to_genesis4_fieldfile(self)

    def write_genesis4(self, h5: h5py.File | pathlib.Path | str) -> None:
        """
        Save a Genesis4-format field file.

        Parameters
        ----------
        h5 : h5py.File, pathlib.Path, or str
            The opened h5py File or a path to it on disk.
        """

        field_file = self.to_genesis4_fieldfile()
        field_file.write_genesis4(h5)

    @classmethod
    def _from_h5_file(
        cls,
        h5: h5py.File,
        identifier: str | int,
        iteration: str | int = 0,
    ) -> Wavefront:
        base_path = tools.require_h5_string_attr(h5, "basePath")
        # data_type = h5.attrs["dataType"]
        extensions = tools.require_h5_string_attr(h5, "openPMDextension").split(";")
        if "wavefront" in extensions:
            from .interfaces.genesis import read_legacy_genesis_pmd_wavefront

            return read_legacy_genesis_pmd_wavefront(h5, identifier=identifier)

        if "Wavefront" not in extensions:
            raise ValueError(
                f"Wavefront extension not enabled in file. "
                f"Extensions configured: {extensions}"
            )

        iteration_path = base_path.replace("%T", str(iteration))
        wavefront_field_path = tools.require_h5_string_attr(h5, "wavefrontFieldPath")

        # {iteration group}/{wavefront group}/{efield_group}
        iteration_group = tools.require_h5_group(h5, iteration_path)
        wavefront_group = tools.require_h5_group(iteration_group, wavefront_field_path)
        efield_group = tools.require_h5_group(wavefront_group, "electricField")

        photon_energy = efield_group.attrs["photonEnergy"]
        assert isinstance(photon_energy, float)
        # efield_group["photonEnergyUnitSI"]
        # efield_group["photonEnergyUnitDimension"]
        # efield_group["temporalDomain"]
        # efield_group["spatialDomain"]
        for polarization in "xyz":
            try:
                rmesh_dataset = tools.require_h5_dataset(efield_group, polarization)
            except KeyError:
                pass
            else:
                break
        else:
            raise ValueError("No supported polarization direction group found")

        metadata = WavefrontMetadata.from_hdf5(h5, efield_group, rmesh_dataset)

        rmesh = readers.component_data(rmesh_dataset)
        return cls(
            rmesh=rmesh,
            wavelength=wavelength_to_photon_energy(photon_energy),
            metadata=metadata,
        )

    @classmethod
    def from_file(
        cls,
        h5: h5py.File | pathlib.Path | str,
        iteration: int = 0,
        identifier: int = 0,
    ) -> Wavefront:
        """Load a Wavefront from a file in the OpenPMD format.

        Parameters
        ----------
        h5 : h5py.File or pathlib.Path or str
            The HDF5 file object (or path to the file) to load the Wavefront from.
        iteration : int, optional
            The iteration number to load, by default 0.
        identifier : int, optional
            The identifier for the wavefront for the given iteration, by default 0.

        Returns
        -------
        Wavefront
        """
        if isinstance(h5, h5py.File):
            return cls._from_h5_file(h5, iteration=iteration, identifier=identifier)
        with h5py.File(h5) as h5p:
            return cls._from_h5_file(h5p, iteration=iteration, identifier=identifier)

    def _write_file(self, h5: h5py.File):
        """Write the Wavefront to the given HDF5 File."""
        md = self.metadata
        base_path_template = "/data/%T/"
        # if md.index is not None:
        #     wavefront_base_path_template = "wavefront/%T/"
        # else:
        # For us, at least, second %T doesn't make much sense. Each iteration
        # stores one wavefront for now.
        wavefront_base_path_template = "wavefront"
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

        electric_field_path = wavefront_path + "/electricField/"
        efield_group = h5.create_group(electric_field_path)
        self._write_group(efield_group)

    def write(self, h5: h5py.File | pathlib.Path | str) -> None:
        """
        Write the Wavefront in OpenPMD format.

        Parameters
        ----------
        h5 : h5py.File or pathlib.Path or str
            The h5py File handle or filename to write.
        """
        if isinstance(h5, h5py.File):
            return self._write_file(h5)

        with h5py.File(h5, "w") as h5p:
            return self._write_file(h5p)

    def _write_group(self, group: h5py.Group) -> None:
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


def focus(
    wf: Wavefront,
    plane: Plane,
    focus: tuple[float, float],
) -> Wavefront:
    """
    Apply thin lens focusing to a Wavefront.

    Parameters
    ----------
    wf : Wavefront
    plane : str
        Plane identifier (e.g., "xy").
    focus : (float, float)
        Focal length of the lens in each dimension [m].

    Returns
    -------
    Wavefront
    """
    if plane != "xy":
        raise NotImplementedError(f"Unsupported plane: {plane}")

    new_rmesh = (
        wf.rmesh
        * thin_lens_kernel_xy(
            wavelength=wf.wavelength,
            ranges=wf.ranges,
            grid=wf.grid,
            f_lens_x=focus[0],
            f_lens_y=focus[1],
        )[:, :, np.newaxis]
    )

    return wf.with_rmesh(new_rmesh)


def drift(wf: Wavefront, distance: float) -> Wavefront:
    """
    Drift a Wavefront along the longitudinal direction in meters.

    Parameters
    ----------
    wf : Wavefront
    distance : float
        Distance in meters.

    Returns
    -------
    Wavefront
    """
    # indices = [
    #     idx
    #     for idx, label in enumerate(wf.axis_labels)
    #     if label != wf.longitudinal_axis
    # ]
    indices = (0, 1)
    transverse_kspace_grid = nd_kspace_mesh(
        coeffs=(1.0,) * len(wf.grid),
        sizes=[wf.rmesh.shape[idx] for idx in indices],
        pads=[wf.pad[idx] for idx in indices],
        steps=[wf.grid_spacing[idx] for idx in indices],
    )

    kmesh = drift_propagator_paraxial(
        kmesh=wf.kmesh,
        transverse_kspace_grid=transverse_kspace_grid,
        wavelength=wf.wavelength,
        z=float(distance),
    )
    return Wavefront.from_kmesh(
        kmesh,
        wavelength=wf.wavelength,
        metadata=wf.metadata,
        padding=wf.pad,
    )
