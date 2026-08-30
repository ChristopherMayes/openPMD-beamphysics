import logging
import os
import pathlib
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from h5py import File, Group

from .exceptions import (
    MultipleIterationsError,
    MultipleSpeciesError,
    NoIterationsError,
    NoSpeciesError,
    NotOpenPMDError,
)
from .tools import decode_attr, decode_attrs
from .units import SI_symbol, c_light, dimension, dimension_name, e_charge

logger = logging.getLogger(__name__)

# -----------------------------------------
# General Utilities

# Root attributes recommended from the openPMD base standard
root_attrs = (
    "author",
    "software",
    "softwareVersion",
    "date",
    "openPMDextension",
    "softwareDependencies",
    "machine",
    "comment",
    "dataType",
)


def check_openpmd_root(h5: File | Group, warn: bool = False) -> dict:
    """
    Check that `h5` is the root of an openPMD series and log its metadata.

    Parameters
    ----------
    h5 : h5py.File or h5py.Group
        Handle carrying the openPMD root attributes: a file root, or a group
        holding a series inside a larger file.
    warn : bool, optional
        Warn instead of raising when the "openPMD" attribute is missing, so
        that the caller can read the handle anyway. Default is False.

    Returns
    -------
    dict
        Decoded root attributes.

    Raises
    ------
    NotOpenPMDError
        If `h5` has no "openPMD" attribute and `warn` is False.
    """
    attrs = decode_attrs(h5.attrs)
    if "openPMD" not in attrs:
        message = f"No 'openPMD' attribute in {h5.file.filename}:{h5.name}"
        if not warn:
            raise NotOpenPMDError(message)

        warnings.warn(
            f"{message}. This is not a standards compliant openPMD file, but "
            "reading will be attempted anyway. This warning will become an "
            "exception in a later version of beamphysics.",
            category=FutureWarning,
            stacklevel=2,
        )
        return attrs

    metadata = {key: attrs[key] for key in root_attrs if key in attrs}
    logger.debug(
        "Reading openPMD %s from %s:%s with metadata %s",
        attrs["openPMD"],
        h5.file.filename,
        h5.name,
        metadata,
    )
    return attrs


# -----------------------------------------
# Records, components, units

particle_record_components = {
    "branchIndex": None,
    "chargeState": None,
    "electricField": ["x", "y", "z"],
    "elementIndex": None,
    "magneticField": ["x", "y", "z"],
    "locationInElement": None,
    "momentum": ["x", "y", "z"],
    "momentumOffset": ["x", "y", "z"],
    "photonPolarizationAmplitude": ["x", "y"],
    "photonPolarizationPhase": ["x", "y"],
    "sPosition": None,
    "totalMomentum": None,
    "totalMomentumOffset": None,
    #'particleCoordinatesToGlobalTransformation': ??
    "particleStatus": None,
    "pathLength": None,
    "position": ["x", "y", "z"],
    "positionOffset": ["x", "y", "z"],
    "spin": ["x", "y", "z", "theta", "phi", "psi"],
    "time": None,
    "timeOffset": None,
    "velocity": ["x", "y", "z"],
    "weight": None,
}

field_record_components = {
    "electricField": ["x", "y", "z", "r", "theta"],
    "magneticField": ["x", "y", "z", "r", "theta"],
}


# Expected unit dimensions for particle and field records
expected_record_unit_dimension = {
    "branchIndex": dimension("1"),
    "chargeState": dimension("1"),
    "electricField": dimension("electric_field"),
    "magneticField": dimension("magnetic_field"),
    "elementIndex": dimension("1"),
    "locationInElement": dimension("1"),
    "momentum": dimension("momentum"),
    "momentumOffset": dimension("momentum"),
    "photonPolarizationAmplitude": dimension("electric_field"),
    "photonPolarizationPhase": dimension("1"),
    "sPosition": dimension("length"),
    "totalMomentum": dimension("momentum"),
    "totalMomentumOffset": dimension("momentum"),
    #'particleCoordinatesToGlobalTransformation': ??
    "particleStatus": dimension("1"),
    "pathLength": dimension("length"),
    "position": dimension("length"),
    "positionOffset": dimension("length"),
    "spin": dimension("1"),
    "time": dimension("time"),
    "timeOffset": dimension("time"),
    "velocity": dimension("velocity"),
    "weight": dimension("charge"),
}

# Convenient aliases for components
component_from_alias = {
    # 'x':'position/x',
    # 'y':'position/y',
    # 'z':'position/z',
    # 'px':'momentum/x',
    # 'py':'momentum/y',
    # 'pz':'momentum/z',
    "t": "time",
    "weight": "weight",
    "status": "particleStatus",
}
# Aliases for particles and fields
for g, prefix in zip(
    ["position", "momentum", "electricField", "magneticField"], ["", "p", "E", "B"]
):
    for c in ["x", "y", "z", "r", "theta"]:
        alias = prefix + c
        component_from_alias[alias] = g + "/" + c
# Inverse
component_alias = {v: k for k, v in component_from_alias.items()}


def particle_paths(h5, key="particlesPath"):
    """
    Uses the basePath and particlesPath to find where openPMD particles should be

    """
    basePath = h5.attrs["basePath"].decode("utf-8")
    particlesPath = h5.attrs[key].decode("utf-8")

    if "%T" not in basePath:
        return [basePath + particlesPath]
    path1, path2 = basePath.split("%T")
    tlist = list(h5[path1])
    paths = [path1 + t + path2 + particlesPath for t in tlist]
    return paths


def field_paths(h5, key="externalFieldPath"):
    """
    Looks for the External Fields

    """
    if key not in h5.attrs:
        return []

    fpath = h5.attrs[key].decode("utf-8")

    if "%T" not in fpath:
        return [fpath]

    path1 = fpath.split("%T")[0]
    tlist = list(h5[path1])
    paths = [path1 + t for t in tlist]
    return paths


def is_constant_component(h5):
    """
    Constant record component should have 'value' and 'shape'
    """
    return "value" in h5.attrs and "shape" in h5.attrs


def constant_component_value(h5):
    """
    Constant record component should have 'value' and 'shape'
    """
    unitSI = h5.attrs["unitSI"]
    val = h5.attrs["value"]
    if unitSI == 1.0:
        return val
    else:
        return val * unitSI


def component_unit_dimension(h5):
    """
    Return the unit dimension tuple
    """
    return tuple(h5.attrs["unitDimension"])


def is_legacy_fortran_data_ordering(component_data_attrs):
    if "gridDataOrder" in component_data_attrs:
        warnings.warn(
            "Legacy gridDataOrder in component. Please remove and use "
            "axisLabels at the group level.",
            category=UserWarning,
            stacklevel=2,
        )
        if decode_attr(component_data_attrs["gridDataOrder"]) == "F":
            return True
    return False


def component_data(h5, slice=slice(None), unit_factor=1, axis_labels=None):
    """
    Returns a numpy array from an h5 component.

    Parameters
    ----------
    h5 : h5py.Dataset or h5py.Group
        The HDF5 component to extract data from.
    slice : slice or tuple, optional
        Slice or tuple of slices to retrieve parts of the array, by default slice(None).
    unit_factor : float, optional
        Additional factor to convert from SI units to output units, by default 1.
    axis_labels : tuple of str
        Required for multidimensional arrays.
        Supported options are:
        * ("z", "y", "x")
        * ("z", "theta", "r")
        * ("x", "y", "z")
        * ("r", "theta", "z")

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Determines whether a component has constant data or array data and handles both cases.
    Checks for legacy gridDataOrder attribute: F or C. If F, the numpy array is transposed.
    Applies unitSI factor from h5 attributes if available.
    """

    # look for unitSI factor.
    if "unitSI" in h5.attrs:
        factor = h5.attrs["unitSI"]
    else:
        factor = 1

    # Additional conversion factor
    if unit_factor:
        factor *= unit_factor

    if is_constant_component(h5):
        dat = np.full(h5.attrs["shape"], h5.attrs["value"])[slice]

    # Check multidimensional for data ordering, convert to 'x', 'y', 'z' order
    elif len(h5.shape) > 1:
        if axis_labels is None:
            raise ValueError("axis_labels required for multidimensional arrays")

        # Reorder to x, y, z
        if axis_labels in [("z", "y", "x"), ("z", "theta", "r")]:
            if isinstance(slice, tuple):
                # Need to transpose the slice ordering
                slice = slice[::-1]

            # Retrieve dataset and transpose for C order
            dat = h5[slice]
            dat = np.transpose(dat)
        elif axis_labels in [("x", "y", "z"), ("r", "theta", "z")]:
            dat = h5[slice]
        else:
            # C-order
            dat = h5[slice]
    else:
        # 1-D array
        dat = h5[slice]

    if factor != 1:
        dat *= factor

    return dat


def component_scalar_or_array_data(
    h5: Group, name: str, default: float | None = None
) -> float | np.ndarray:
    """
    Read a record component as a float if it is constant, otherwise as an array.

    Parameters
    ----------
    h5 : h5py.Group
        Group holding the record components.
    name : str
        Name of the component.
    default : float, optional
        Value to return when `h5` has no component `name`.

    Returns
    -------
    float or numpy.ndarray
        Component data in SI units. A constant component is returned as a
        scalar rather than broadcast to the shape it stands in for.

    Raises
    ------
    KeyError
        If `h5` has no component `name` and no `default` is given.
    """
    if name not in h5:
        if default is None:
            raise KeyError(f"No component {name} in {h5.name}")
        return default

    component = h5[name]
    if is_constant_component(component):
        return float(constant_component_value(component))

    return component_data(component)


def offset_component_name(component_name):
    """
    Many components can also have an offset, as in:

        position/x
        positionOffset/c

    Return the appropriate name.
    """
    x = component_name.split("/")
    if len(x) == 1:
        return x[0] + "Offset"
    else:
        return x[0] + "Offset/" + x[1]


def particle_array(h5, component, slice=slice(None), include_offset=True):
    """
    Main routine to return particle arrays in fixed units.
    All units are SI except momentum, which will be in eV/c.

    Example:
        particle_array(h5['data/00001/particles/'], 'px')
        Will return the momentum/x + momentumOffset/x in eV/c.


    """

    # Handle aliases
    if component in component_from_alias:
        component = component_from_alias[component]

    if component in ["momentum/x", "momentum/y", "momentum/z"]:
        unit_factor = c_light / e_charge  # convert J/(m/s) to eV/c
    else:
        unit_factor = 1.0

    # Get data
    dat = component_data(h5[component], slice=slice, unit_factor=unit_factor)

    # Look for offset component
    ocomponent = offset_component_name(component)
    if include_offset and ocomponent in h5:
        offset = component_data(h5[ocomponent], slice=slice, unit_factor=unit_factor)
        dat += offset

    return dat


def _scalar_maybe_from_array(value):
    if np.isscalar(value):
        return value

    if len(value) != 1:
        raise ValueError(
            f"Expected a scalar or length-1 array, got length {len(value)}"
        )
    return value[0]


def _only_iteration_group(h5: File | Group) -> Group:
    """
    Get the HDF5 group of the only openPMD iteration. Raise if none or multiple in series.

    Parameters
    ----------
    h5 : h5py.File or h5py.Group
        Handle carrying the openPMD "basePath" and "particlesPath" attributes.

    Returns
    -------
    h5py.Group
        Particle group of the single iteration, resolved relative to `h5`.
    """
    missing = {"basePath", "particlesPath"} - set(h5.attrs)
    if missing:
        raise NoIterationsError(
            f"Missing openPMD attributes in {h5.name}: {sorted(missing)}"
        )

    paths = particle_paths(h5)
    if not paths:
        raise NoIterationsError(f"No openPMD iterations in {h5.name}")
    if len(paths) > 1:
        raise MultipleIterationsError(
            f"Multiple openPMD iterations in {h5.name}: {paths}"
        )

    logger.debug("Loading iteration %s from %s", paths[0], h5.name)

    # particle_paths returns absolute paths. Strip the leading separator so that
    # they resolve relative to h5, which may itself be a group within a file.
    path = paths[0].strip("/")
    return h5[path] if path else h5["."]


def _only_species_group(h5: Group) -> Group:
    """
    Return the HDF5 group of the only species in the OpenPMD iteration. Raise if none or more than one.

    Parameters
    ----------
    h5 : h5py.Group
        Particle group. Legacy-style groups hold the records directly instead of
        nesting them in a species subgroup.

    Returns
    -------
    h5py.Group
        Group holding the particle records.
    """
    # Legacy-style particles with no species
    if "position" in h5:
        logger.debug("Loading species from records in %s", h5.name)
        return h5

    species = list(h5)
    if not species:
        raise NoSpeciesError(f"No species in particle group: {h5.name}")
    if len(species) > 1:
        raise MultipleSpeciesError(
            f"Multiple species in particle group {h5.name}: {species}"
        )

    logger.debug("Loading species %s from %s", species[0], h5.name)
    return h5[species[0]]


@contextmanager
def _only_iteration_only_species_group(
    h5: str | pathlib.Path | File | Group,
    warn: bool = False,
) -> Iterator[Group]:
    """
    Yield the HDF5 group of the only species in the only iteration of the OpenPMD series / file. Fall back to attempting to
    treat `h5` as an iteration (legacy behavior).

    Parameters
    ----------
    h5 : str, pathlib.Path, h5py.File, or h5py.Group
        Filename of an openPMD file, or an open handle. A handle carrying the
        openPMD attributes is resolved to its single iteration; one that does
        not is taken to be the particle group itself.
    warn : bool, optional
        Warn instead of raising when the handle is not an openPMD root, and
        read it anyway. Default is False.

    Yields
    ------
    h5py.Group
        Group holding the particle records. A file opened from a filename is
        closed on exit.
    """
    if isinstance(h5, (str, pathlib.Path)):
        with (
            File(os.path.expandvars(h5), "r") as h5file,
            _only_iteration_only_species_group(h5file, warn=warn) as group,
        ):
            yield group

    else:
        # h5py.File is itself a Group
        if not isinstance(h5, Group):
            raise TypeError(f"Unsupported type for h5: {type(h5).__name__}")

        check_openpmd_root(h5, warn=warn)

        try:
            group = _only_iteration_group(h5)
        except NoIterationsError:
            # The root has no iterations, so h5 holds the particle records
            group = h5

        yield _only_species_group(group)


def load_species_data(h5: Group, include_time_offset: bool = True) -> dict:
    """
    Load a single species into a dict of numpy arrays.

    Parameters
    ----------
    h5 : h5py.Group
        Group holding the particle records.
    include_time_offset : bool, optional
        Add the "timeOffset" record to `t`. The position and momentum offsets
        are always included. Default is True.

    Returns
    -------
    dict
        Keys 'x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight' (arrays),
        'species' (str), 'total_charge' (float), and optionally 'id' (array).
    """
    attrs = dict(h5.attrs)
    data = {}

    species_type = attrs["speciesType"]
    data["species"] = (
        species_type.decode("utf-8")
        if isinstance(species_type, bytes)
        else species_type
    )

    n_particle = int(_scalar_maybe_from_array(attrs["numParticles"]))

    data["total_charge"] = attrs["totalCharge"] * attrs["chargeUnitSI"]

    for key in ["x", "px", "y", "py", "z", "pz"]:
        data[key] = particle_array(h5, key)
    data["t"] = particle_array(h5, "t", include_offset=include_time_offset)

    if "particleStatus" in h5:
        data["status"] = particle_array(h5, "particleStatus")
    else:
        data["status"] = np.full(n_particle, 1)

    # Make sure weight is populated
    if "weight" in h5:
        weight = particle_array(h5, "weight")
        if len(weight) == 1:
            weight = np.full(n_particle, weight[0])
    else:
        weight = np.full(n_particle, data["total_charge"] / n_particle)
    data["weight"] = weight

    # id should be a unique integer, no units
    # optional
    if "id" in h5:
        data["id"] = h5["id"][:]

    return data


def load_time_offset(h5: Group) -> float | np.ndarray:
    """
    Load the time offset of a single species.

    Parameters
    ----------
    h5 : h5py.Group
        Group holding the particle records.

    Returns
    -------
    float or numpy.ndarray
        Offset in seconds: a float for a constant component, an array of length
        n_particle for a per-particle component, and 0.0 when the group has no
        "timeOffset" record.
    """
    return component_scalar_or_array_data(h5, "timeOffset", default=0.0)


def load_bunch_data(h5: Group, include_time_offset: bool = True) -> dict:
    """
    Load particles from the only species in this iteration of an OpenPMD BeamPhysics file into a dict of numpy arrays.
    Raises if more than one or no species.

    Parameters
    ----------
    h5 : h5py.Group
        Particle group, one iteration holding either a single species subgroup or the records
        themselves (legacy).
    include_time_offset : bool, optional
        Add the "timeOffset" record to `t`. The position and momentum offsets
        are always included. Default is True.

    Returns
    -------
    dict
        See `beamphysics.readers.load_species_data`.
    """
    return load_species_data(
        _only_species_group(h5), include_time_offset=include_time_offset
    )


def load_only_time_offset(h5: str | pathlib.Path | File | Group) -> float | np.ndarray:
    """
    Load the time offset of the only species of the only iteration.

    Parameters
    ----------
    h5 : str, pathlib.Path, h5py.File, or h5py.Group
        Filename of an openPMD file, or an open handle. A handle carrying the
        openPMD attributes is resolved to its single iteration; one that does
        not is taken to be the particle group itself.

    Returns
    -------
    float or numpy.ndarray
        See `load_time_offset`.
    """
    with _only_iteration_only_species_group(h5) as group:
        return load_time_offset(group)


def all_components(h5):
    """
    Look for possible components in a particle group
    """
    components = []
    for record_name in h5:
        if record_name not in particle_record_components:
            continue

        # Look for components
        possible_components = particle_record_components[record_name]

        if not possible_components:
            # Record is a component
            components.append(record_name)
        else:
            g = h5[record_name]
            for cname in possible_components:
                if cname in g:
                    components.append(record_name + "/" + cname)

    return components


def component_str(particle_group, name):
    """
    Informational string from a component in a particle group (h5)
    """

    g = particle_group[name]
    record_name = name.split("/")[0]
    expected_dimension = expected_record_unit_dimension[record_name]
    this_dimension = component_unit_dimension(g)
    # A file may carry a dimension this package has no name for (e.g. from
    # another openPMD extension); describe it rather than raising KeyError.
    try:
        dname = dimension_name(this_dimension)
        symbol = SI_symbol[dname]
    except KeyError:
        dname = str(tuple(this_dimension))
        symbol = "?"

    s = name + " "

    if is_constant_component(g):
        val = constant_component_value(g)
        shape = g.attrs["shape"]
        s += f"[constant {val} with shape {shape}]"
    else:
        s += "[" + str(len(g)) + " items]"

    if symbol != "1":
        s += f" is a {dname} with units: {symbol}"

    if expected_dimension != this_dimension:
        s += ", but expected units: " + SI_symbol[dimension_name(expected_dimension)]

    return s


# ----------------------------------
# Fields

required_field_attrs = [
    # strings
    "eleAnchorPt",
    "gridGeometry",
    "axisLabels",
    # reals and ints
    "gridLowerBound",
    "gridOriginOffset",
    "gridSpacing",
    "gridSize",
    "harmonic",
]

# Dict with options
optional_field_attrs = {
    "name": None,
    "gridCurvatureRadius": None,
    "fundamentalFrequency": 0,
    "RFphase": 0,
    "fieldScale": 1.0,
    "masterParameter": None,
}


def load_field_attrs(attr, verbose=False):
    """
    Loads FieldMesh required and optional attributes from a dict_like object.

    Non-standard attributes will be collected in an 'other' dict.

    Returns dicts:
        attrs, other

    """
    # Get all attrs. Will pop.
    a = dict(attr)

    attrs = {}
    other = {}

    # Required
    for k in required_field_attrs:
        attrs[k] = a.pop(k)

    # Optional, filling in some defaults
    for k in optional_field_attrs:
        if k in a:
            attrs[k] = a.pop(k)
        else:
            v = optional_field_attrs[k]
            if v is not None:
                attrs[k] = v

    # Collect other.
    for k, v in a.items():
        other[k] = v
        if verbose:
            print("Nonstandard attr:", k, v)

    # Decode
    attrs = decode_attrs(attrs)

    # Error checking
    # if attrs['harmonic'] > 0:
    #    assert 'fundamentalFrequency' in attrs, 'fundamentalFrequency required if harmonic > 0'

    return attrs, other
