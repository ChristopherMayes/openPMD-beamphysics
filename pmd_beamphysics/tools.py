import datetime
import logging
from typing import Sequence

import h5py
import numpy as np
import scipy.fft

logger = logging.getLogger(__name__)
_global_fft_workers = -1


def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.bytes_(s)


def data_are_equal(d1, d2):
    """
    Simple utility to compare data in dicts

    Returns True only if all keys are the same, and all np.all data are the same
    """

    if set(d1) != set(d2):
        return False

    for k in d1:
        if not np.all(d1[k] == d2[k]):
            return False

    return True


# -----------------------------------------
# HDF5 utilities


def decode_attr(a):
    """
    Decodes:
        ASCII strings and arrays of them to str and arrays of str
        single-length arrays to scalar (Bmad writes this)

    """
    if isinstance(a, bytes):
        return a.decode("utf-8")

    if isinstance(a, np.ndarray):
        if a.dtype.type is np.bytes_:
            a = a.astype(str)
        if len(a) == 1:
            return a[0]

    return a


def decode_attrs(attrs):
    return {k: decode_attr(v) for k, v in attrs.items()}


def encode_attr(a):
    """
    Encodes attribute

    See the inverse function:
        decode_attr

    """

    if isinstance(a, str):
        a = fstr(a)

    if isinstance(a, list) or isinstance(a, tuple):
        a = np.array(a)

    if isinstance(a, np.ndarray):
        if a.dtype.type is np.str_:
            a = a.astype(np.bytes_)

    return a


def encode_attrs(attrs):
    return {k: encode_attr(v) for k, v in attrs.items()}


def get_version() -> str:
    """Get the installed pmd-beamphysics version."""
    from . import __version__

    return __version__


def current_date_with_tzinfo() -> datetime.datetime:
    from dateutil.tz import tzlocal

    return datetime.datetime.now(tzlocal())


def pmd_format_date(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %z")


def require_h5_dataset(h5: h5py.Group, name: str) -> h5py.Dataset:
    """
    Require a dataset from an HDF5 group.

    Parameters
    ----------
    h5 : h5py.Group
        The HDF5 group from which the dataset is required.
    name : str
        The key name of the dataset to retrieve.

    Returns
    -------
    h5py.Dataset
        The requested dataset.

    Raises
    ------
    KeyError
        If the dataset named `name` is not found in the HDF5 group `h5`.
    ValueError
        If the found object named `name` is not a dataset.
    """
    try:
        dataset = h5[name]
    except KeyError:
        raise KeyError(f"Expected HDF dataset {name} not found in {h5.name}")

    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(
            f"Key {name} expected to be a dataset, but is a {type(dataset)}"
        )
    return dataset


def require_h5_group(h5: h5py.Group, name: str) -> h5py.Group:
    """
    Require a subgroup from an HDF5 group.

    Parameters
    ----------
    h5 : h5py.Group
        The HDF5 group from which the subgroup is required.
    name : str
        The key name of the subgroup to retrieve.

    Returns
    -------
    h5py.Group
        The requested group.

    Raises
    ------
    KeyError
        If the subgroup named `name` is not found in the HDF5 group `h5`.
    ValueError
        If the found object named `name` is not a dataset.
    """
    try:
        group = h5[name]
    except KeyError:
        raise KeyError(f"Expected HDF group {name} not found in {h5.name}")

    if not isinstance(group, h5py.Group):
        raise ValueError(f"Key {name} expected to be a group, but is a {type(group)}")
    return group


def require_h5_string_attr(h5: h5py.Group, attr: str) -> str:
    """
    Retrieve a required string attribute from an HDF5 group.

    Parameters
    ----------
    h5 : h5py.Group
    attr : str
        The name of the attribute to retrieve.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the attribute is not a string or bytes.
    """
    value = h5.attrs[attr]
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode()
    raise ValueError(
        f"Expected bytes or strings for {h5.name} key {attr}; got {type(value).__name__}"
    )


def get_num_fft_workers() -> int:
    """Get the global number of FFT workers."""
    return _global_fft_workers


def set_num_fft_workers(workers: int):
    """Set the global number of FFT workers."""
    global _global_fft_workers

    _global_fft_workers = workers

    logger.info(f"Set number of FFT workers to: {workers}")


def fft_phased(
    array: np.ndarray,
    axes: Sequence[int],
    phasors,
    workers=-1,
) -> np.ndarray:
    """
    Compute the N-D discrete Fourier Transform with phasors applied.

    Ortho normalization is used.

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

    Ortho normalization is used.

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
