import datetime
import logging

import h5py
import numpy as np

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


def require_h5_group(h5: h5py.Group, name: str) -> h5py.Group:
    try:
        group = h5[name]
    except KeyError:
        raise KeyError(f"Expected HDF group {name} not found in {h5.name}")

    if not isinstance(group, h5py.Group):
        raise ValueError(f"Key {group} expected to be a group, but is a {type(group)}")
    return group


def require_h5_string_attr(h5: h5py.Group, attr: str) -> str:
    value = h5.attrs[attr]
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode()
    raise ValueError(
        f"Expected bytes or strings for {h5.name} key {attr}; got {type(value).__name__}"
    )


def get_num_fft_workers() -> int:
    return _global_fft_workers


def set_num_fft_workers(workers: int):
    global _global_fft_workers

    _global_fft_workers = workers

    logger.info(f"Set number of FFT workers to: {workers}")
