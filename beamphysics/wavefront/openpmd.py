"""
openPMD EXT_Wavefront reading and writing for the `Wavefront` class.

The layout below follows the ``Wavefront`` extension of the openPMD standard
(branch ``upcoming-2.0.0``), which is authoritative::

    /                             openPMD "2.0.0", openPMDextension "Wavefront",
                                  basePath "/data/%T/", meshesPath "meshes/",
                                  iterationEncoding "groupBased"
    /data/<iteration>/            one iteration per file. Slices of a single pulse are
                                  simultaneous, so the slice axis is a *mesh* axis and
                                  never the openPMD iteration.
    .../meshes/electricField      the mesh record. All required attributes live on the
                                  record: the base standard's geometry, axisLabels,
                                  gridSpacing, gridGlobalOffset, gridUnitSI,
                                  gridUnitDimension, unitDimension and timeOffset, plus
                                  the extension's photonEnergy [J], temporalDomain,
                                  spatialDomain and zCoordinate.
    .../electricField/x, y        complex compound ``{r, i}`` datasets in V/m, which
                                  h5py maps to complex dtypes natively. The ``z``
                                  component is never written: a paraxial field has none.

Datasets are stored in ``(z, y, x)`` order -- declared by ``axisLabels`` -- so that each
transverse slice is one contiguous block in the file. The class's ``(nx, ny, nz)``
convention is recovered by a transpose that is applied one slice at a time, keeping peak
memory at one field plus one transverse slice rather than two full copies.

Only the real-space, time-domain case is implemented; ``spatialDomain='k'`` and
``temporalDomain='frequency'`` are refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
from scipy.constants import h as h_planck

from ..readers import constant_component_value, is_constant_component
from ..tools import decode_attr, encode_attr
from ..units import c_light, dimension

__all__ = [
    "WavefrontAttrs",
    "load_wavefront_openpmd",
    "write_wavefront_openpmd",
]


# openPMD defaults, used when a file omits these root attributes.
_DEFAULT_BASE_PATH = "/data/%T/"
_DEFAULT_MESHES_PATH = "meshes/"

_RECORD_NAME = "electricField"

# Stored (file) axis order. See the module docstring for why.
_STORED_AXIS_LABELS = ("z", "y", "x")

# Axis order of the `Wavefront` class arrays.
_CLASS_AXIS_LABELS = ("x", "y", "z")

# 7-tuples of base-SI exponents, from the package's own table.
_UNIT_DIMENSION_E_FIELD = dimension("electric_field")
_UNIT_DIMENSION_LENGTH = dimension("length")

# Attributes on the mesh record that this module writes from the wavefront itself.
# They may not be supplied by the caller, and are not carried in `WavefrontAttrs.other`
# on read, because the value in the file is redundant with the class's own state.
_COMPUTED_RECORD_ATTRS = (
    # Extension.
    "photonEnergy",  # derived from `wavelength`
    "temporalDomain",  # always 'time' for this class
    "spatialDomain",  # always 'r' for this class
    "zCoordinate",  # `Wavefront.s_position`, which `drift` advances
    # Base standard, written from the grid and the field dtype.
    "geometry",
    "geometryParameters",
    "axisLabels",
    "gridSpacing",
    "gridGlobalOffset",
    "gridUnitSI",
    "gridUnitDimension",
    "unitDimension",
    "timeOffset",
    "dataOrder",  # openPMD 1.x only; 2.0 uses axisLabels alone
)


def _pmd(key):
    """
    Dataclass field metadata tagging a field with its openPMD attribute name.

    Parameters
    ----------
    key : str
        The attribute name as it appears in the file.

    Returns
    -------
    dict
    """
    return {"pmd_key": key}


@dataclass
class WavefrontAttrs:
    """
    openPMD EXT_Wavefront attributes carried alongside a `Wavefront`.

    Field names are Python-style; the corresponding openPMD attribute names, used in
    the file, are camelCase and recorded in each field's metadata. Attributes the
    class derives from its own state -- `photonEnergy`, `temporalDomain`,
    `spatialDomain`, `zCoordinate` and the grid attributes -- are deliberately
    absent. `zCoordinate` in particular is `Wavefront.s_position`, which is a
    coordinate the propagators advance rather than provenance carried through I/O.

    Parameters
    ----------
    beamline : str, optional
        Name of the beamline this wavefront belongs to.
    radius_of_curvature_x : float, optional
        Radius of curvature in x, in m.
    radius_of_curvature_y : float, optional
        Radius of curvature in y, in m.
    delta_radius_of_curvature_x : float, optional
        Uncertainty in `radius_of_curvature_x`, in m.
    delta_radius_of_curvature_y : float, optional
        Uncertainty in `radius_of_curvature_y`, in m.
    other : dict, optional
        Record attributes that are not part of the extension as this module knows
        it, keyed by their openPMD name. Populated on read and written back
        verbatim, so that a file using a newer revision of the extension survives a
        round trip. Names this module computes are refused. Named after the `other`
        dict that `readers.load_field_attrs` uses for the same purpose.

    Raises
    ------
    ValueError
        If `other` holds a name that has a field of its own, or one that the writer
        computes.

    Examples
    --------
    >>> attrs = WavefrontAttrs(beamline="SXR", radius_of_curvature_x=12.5)
    >>> attrs.to_pmd()["radiusOfCurvatureX"]
    12.5

    A misspelled attribute is a `TypeError` at construction rather than a silent
    omission at write time:

    >>> WavefrontAttrs(radius_of_curvature_z=1.0)
    Traceback (most recent call last):
        ...
    TypeError: ...
    """

    beamline: str | None = field(default=None, metadata=_pmd("beamline"))
    radius_of_curvature_x: float | None = field(
        default=None, metadata=_pmd("radiusOfCurvatureX")
    )
    radius_of_curvature_y: float | None = field(
        default=None, metadata=_pmd("radiusOfCurvatureY")
    )
    delta_radius_of_curvature_x: float | None = field(
        default=None, metadata=_pmd("deltaRadiusOfCurvatureX")
    )
    delta_radius_of_curvature_y: float | None = field(
        default=None, metadata=_pmd("deltaRadiusOfCurvatureY")
    )
    other: dict = field(default_factory=dict)

    def __post_init__(self):
        known = self.pmd_keys()
        for name in self.other:
            if name in known.values():
                raise ValueError(
                    f"{name!r} has a field of its own; set it directly rather than "
                    "through `other`"
                )
            if name in _COMPUTED_RECORD_ATTRS:
                raise ValueError(
                    f"{name!r} is written from the wavefront itself and cannot be "
                    "set through `other`"
                )

    @classmethod
    def pmd_keys(cls):
        """
        Map field names to their openPMD attribute names.

        Returns
        -------
        dict
            ``{field_name: pmd_key}``, excluding `other`.
        """
        return {
            fld.name: fld.metadata["pmd_key"]
            for fld in fields(cls)
            if "pmd_key" in fld.metadata
        }

    @classmethod
    def from_pmd(cls, attrs):
        """
        Build from a mapping keyed by openPMD attribute names.

        Parameters
        ----------
        attrs : mapping
            Attribute names to values. Field names are also accepted, so that a
            dict written in Python style round-trips. A nested `other` mapping is
            merged into `other` rather than nested inside it. Anything unrecognized
            is kept in `other`.

        Returns
        -------
        WavefrontAttrs

        Raises
        ------
        ValueError
            If an attribute is given under both its openPMD name and its field
            name. Silently preferring one would write the other to the file as a
            nonstandard attribute holding a conflicting value.
        """
        if isinstance(attrs, cls):
            return attrs.copy()

        remaining = dict(attrs)
        kwargs = {}
        for name, key in cls.pmd_keys().items():
            has_key = key in remaining
            # Some fields spell the two the same way, which is not a collision.
            has_name = name != key and name in remaining
            if has_key and has_name:
                raise ValueError(
                    f"{key!r} and {name!r} are two spellings of the same attribute "
                    f"and both were given, with values {remaining[key]!r} and "
                    f"{remaining[name]!r}"
                )
            if has_key:
                kwargs[name] = remaining.pop(key)
            elif has_name:
                kwargs[name] = remaining.pop(name)

        # `other` is a field of this class, not a record attribute, so a mapping
        # carrying one means Python-style input rather than something read from a
        # file. Merge it instead of nesting it, which would produce an `other`
        # entry whose value is a dict and fail at write time.
        other = dict(remaining.pop("other", {}))
        other.update(remaining)

        return cls(**kwargs, other=other)

    def to_pmd(self):
        """
        Render as a mapping keyed by openPMD attribute names.

        Returns
        -------
        dict
            Set fields plus `other`. Fields left as None are omitted, since the
            extension treats them as absent rather than zero.
        """
        out = {}
        for name, key in self.pmd_keys().items():
            value = getattr(self, name)
            if value is not None:
                out[key] = value
        out.update(self.other)
        return out

    def copy(self):
        """
        Return an independent copy, including `other`.

        Returns
        -------
        WavefrontAttrs
        """
        return type(self)(
            **{name: getattr(self, name) for name in self.pmd_keys()},
            other=dict(self.other),
        )


def photon_energy_joules(wavelength):
    """
    Central photon energy in joules for a given wavelength.

    Parameters
    ----------
    wavelength : float
        Central wavelength in m.

    Returns
    -------
    float
        Photon energy ``h c / wavelength`` in J.

    Notes
    -----
    The extension gives `photonEnergy` a `unitDimension` of energy but, as an
    attribute, it carries no `unitSI`. SI (joules) is written here.
    """
    return h_planck * c_light / wavelength


def wavelength_from_photon_energy(photon_energy):
    """
    Central wavelength for a photon energy in joules.

    Parameters
    ----------
    photon_energy : float
        Photon energy in J.

    Returns
    -------
    float
        Wavelength ``h c / photon_energy`` in m.
    """
    return h_planck * c_light / photon_energy


def _axis_permutation(source, target):
    """
    Permutation taking an array indexed by `source` axes to `target` axis order.

    Parameters
    ----------
    source : sequence of str
        Axis labels of the array being permuted, in its own axis order.
    target : sequence of str
        Desired axis label order.

    Returns
    -------
    tuple of int
        Argument for `numpy.ndarray.transpose`.
    """
    return tuple(source.index(label) for label in target)


def _write_attr(group, name, value):
    """
    Write one attribute, encoded the way the rest of the package encodes them.

    Parameters
    ----------
    group : h5py.Group
        Target group.
    name : str
        Attribute name.
    value : object
        Attribute value.
    """
    group.attrs[name] = encode_attr(value)


def write_wavefront_openpmd(
    wavefront,
    h5,
    iteration=1,
    **extension_attrs,
):
    """
    Write a `Wavefront` into an open HDF5 group as an openPMD EXT_Wavefront series.

    Parameters
    ----------
    wavefront : Wavefront
        Real-space, time-domain wavefront. `Ex` and `Ey` are in V/m with shape
        ``(nx, ny, nz)``; an absent polarization is not written. Its `s_position`
        is written as the extension's required `zCoordinate`.
    h5 : h5py.Group
        Group to use as the openPMD series root.
    iteration : int, default=1
        openPMD iteration index to write under `basePath`.
    **extension_attrs
        `WavefrontAttrs` field names (`beamline`, `radius_of_curvature_x`, ...),
        taking precedence over the same fields in ``wavefront.attrs``.

    Raises
    ------
    TypeError
        If an attribute name is not a `WavefrontAttrs` field.

    Notes
    -----
    Datasets are stored in ``(z, y, x)`` order and written one transverse slice at a
    time, so no full transposed copy of the field is ever materialized.
    """
    attrs = WavefrontAttrs.from_pmd(wavefront.attrs)
    for name, value in extension_attrs.items():
        # Assigning an unknown name would silently stick a new instance attribute on
        # the dataclass, so check first and let the constructor raise TypeError.
        if name not in attrs.pmd_keys():
            WavefrontAttrs(**{name: value})
        setattr(attrs, name, value)

    carried = attrs.to_pmd()

    # Series (root) attributes.
    _write_attr(h5, "openPMD", "2.0.0")
    _write_attr(h5, "openPMDextension", "Wavefront")
    _write_attr(h5, "basePath", _DEFAULT_BASE_PATH)
    _write_attr(h5, "meshesPath", _DEFAULT_MESHES_PATH)
    _write_attr(h5, "iterationEncoding", "groupBased")
    _write_attr(h5, "iterationFormat", _DEFAULT_BASE_PATH)

    base = _DEFAULT_BASE_PATH.replace("%T", str(int(iteration))).strip("/")
    iteration_group = h5.require_group(base)
    iteration_group.attrs["time"] = 0.0
    iteration_group.attrs["dt"] = 0.0
    iteration_group.attrs["timeUnitSI"] = 1.0

    mesh = h5.require_group(f"{base}/{_DEFAULT_MESHES_PATH}{_RECORD_NAME}")

    # Grid quantities, ordered like the stored axes. `gridGlobalOffset` is the
    # position of the beginning of the first cell; the components declare
    # `position = 0`, so that is exactly the first sample of each axis.
    spacing = {"x": wavefront.dx, "y": wavefront.dy, "z": wavefront.dz}
    offset = {"x": wavefront.xmin, "y": wavefront.ymin, "z": wavefront.zmin}

    # Base standard, required on the mesh record.
    _write_attr(mesh, "geometry", "cartesian")
    mesh.attrs["axisLabels"] = encode_attr(_STORED_AXIS_LABELS)
    mesh.attrs["gridSpacing"] = np.array(
        [spacing[label] for label in _STORED_AXIS_LABELS], dtype=float
    )
    mesh.attrs["gridGlobalOffset"] = np.array(
        [offset[label] for label in _STORED_AXIS_LABELS], dtype=float
    )
    # openPMD 2.0 makes gridUnitSI one value per axis.
    mesh.attrs["gridUnitSI"] = np.ones(len(_STORED_AXIS_LABELS), dtype=float)
    mesh.attrs["gridUnitDimension"] = np.array(
        _UNIT_DIMENSION_LENGTH * len(_STORED_AXIS_LABELS), dtype=float
    )
    mesh.attrs["unitDimension"] = np.array(_UNIT_DIMENSION_E_FIELD, dtype=float)
    mesh.attrs["timeOffset"] = 0.0

    # Extension, required on the mesh record.
    mesh.attrs["photonEnergy"] = photon_energy_joules(wavefront.wavelength)
    _write_attr(mesh, "temporalDomain", "time")
    _write_attr(mesh, "spatialDomain", "r")
    mesh.attrs["zCoordinate"] = float(wavefront.s_position)

    # Extension, optional.
    for name, value in carried.items():
        _write_attr(mesh, name, value)

    # Components. `z` is never written.
    to_stored = _axis_permutation(_CLASS_AXIS_LABELS, _STORED_AXIS_LABELS)
    for name, field_array in (("x", wavefront.Ex), ("y", wavefront.Ey)):
        if field_array is None:
            continue

        # A view, not a copy: the transpose is realized one slice at a time below.
        stored_view = np.asarray(field_array).transpose(to_stored)

        dataset = mesh.create_dataset(
            name, shape=stored_view.shape, dtype=field_array.dtype
        )
        for islice in range(stored_view.shape[0]):
            dataset[islice] = stored_view[islice]

        dataset.attrs["unitSI"] = 1.0
        dataset.attrs["position"] = np.zeros(len(_STORED_AXIS_LABELS), dtype=float)


def _iteration_group(h5, iteration=None):
    """
    Return the openPMD iteration group, honoring `basePath`.

    Parameters
    ----------
    h5 : h5py.Group
        Series root.
    iteration : int, optional
        Iteration to select. If None, the sole iteration is used.

    Returns
    -------
    h5py.Group

    Raises
    ------
    ValueError
        If the base path is missing, if there are no iterations, or if `iteration`
        is None and the file holds more than one.
    """
    base_path = decode_attr(h5.attrs.get("basePath", _DEFAULT_BASE_PATH))

    if "%T" not in base_path:
        raise ValueError(f"basePath {base_path!r} has no %T iteration placeholder")

    parent_path = base_path.split("%T")[0].strip("/")
    if parent_path not in h5:
        raise ValueError(
            f"basePath {base_path!r} points at {parent_path!r}, which is not in the file"
        )
    parent = h5[parent_path]

    if iteration is not None:
        key = str(int(iteration))
        if key not in parent:
            raise ValueError(
                f"iteration {key} not in the file. Available: {sorted(parent)}"
            )
        return parent[key]

    available = sorted(parent)
    if not available:
        raise ValueError(f"no iterations under {parent_path!r}")
    if len(available) > 1:
        raise ValueError(
            f"file holds {len(available)} iterations {available}; "
            "pass iteration= to select one"
        )
    return parent[available[0]]


def _mesh_record(h5, iteration=None):
    """
    Return the `electricField` mesh record group.

    Parameters
    ----------
    h5 : h5py.Group
        Series root.
    iteration : int, optional
        Iteration to select.

    Returns
    -------
    h5py.Group

    Raises
    ------
    ValueError
        If the record is absent.
    """
    iteration_group = _iteration_group(h5, iteration)
    meshes_path = decode_attr(h5.attrs.get("meshesPath", _DEFAULT_MESHES_PATH))

    record_path = f"{meshes_path}{_RECORD_NAME}"
    if record_path not in iteration_group:
        raise ValueError(
            f"no {_RECORD_NAME!r} mesh record at "
            f"{iteration_group.name}/{record_path}: not an EXT_Wavefront file"
        )
    return iteration_group[record_path]


def _required_attr(mesh, name, fallback=None):
    """
    Read a required attribute from the mesh record, with an optional fallback group.

    Parameters
    ----------
    mesh : h5py.Group
        The mesh record.
    name : str
        Attribute name.
    fallback : h5py.Group, optional
        Group to consult if `mesh` lacks the attribute. Used for `photonEnergy`,
        which the extension places ambiguously.

    Returns
    -------
    object
        The decoded attribute value.

    Raises
    ------
    ValueError
        If the attribute is present in neither location.
    """
    if name in mesh.attrs:
        return decode_attr(mesh.attrs[name])
    if fallback is not None and name in fallback.attrs:
        return decode_attr(fallback.attrs[name])
    raise ValueError(f"required EXT_Wavefront attribute {name!r} is missing")


def load_wavefront_openpmd(h5, iteration=None):
    """
    Read an openPMD EXT_Wavefront series into `Wavefront` constructor arguments.

    Parameters
    ----------
    h5 : h5py.Group
        Series root.
    iteration : int, optional
        Iteration to read. If None, the file must hold exactly one.

    Returns
    -------
    dict
        Keyword arguments for `Wavefront`: `Ex`, `Ey`, `dx`, `dy`, `dz`,
        `wavelength`, `attrs` (a `WavefrontAttrs`) and, when the file declares a
        `gridGlobalOffset`, `xmid`, `ymid` and `zmid`.

    Raises
    ------
    ValueError
        If a required attribute is missing, or if the file holds something this
        class cannot represent: a frequency-domain field, a k-space field, or
        `axisLabels` that are not a permutation of x, y and z.

    Notes
    -----
    Any `axisLabels` permutation of ``(x, y, z)`` is honored, with `gridSpacing`
    permuted alongside it. Scalar attributes stored as length-1 arrays and strings
    stored as bytes are both tolerated. `gridUnitSI`, `gridGlobalOffset` and the
    per-component `unitSI` are applied, so a file written in non-SI units reads
    correctly; `gridUnitSI` is accepted either as the openPMD 2.0 per-axis array or
    as the 1.x scalar.
    """
    mesh = _mesh_record(h5, iteration)

    temporal_domain = _required_attr(mesh, "temporalDomain")
    if temporal_domain != "time":
        raise ValueError(
            f"temporalDomain {temporal_domain!r}: only 'time' (a field in V/m) "
            "is implemented"
        )

    spatial_domain = _required_attr(mesh, "spatialDomain")
    if spatial_domain != "r":
        raise ValueError(
            f"spatialDomain {spatial_domain!r}: only 'r' (cartesian space) "
            "is implemented"
        )

    photon_energy = float(_required_attr(mesh, "photonEnergy", fallback=h5))

    raw_labels = _required_attr(mesh, "axisLabels")
    labels = tuple(decode_attr(label) for label in np.atleast_1d(raw_labels))
    if sorted(labels) != sorted(_CLASS_AXIS_LABELS):
        raise ValueError(
            f"axisLabels {labels}: only a permutation of "
            f"{_CLASS_AXIS_LABELS} is implemented"
        )

    raw_spacing = np.atleast_1d(_required_attr(mesh, "gridSpacing"))
    if len(raw_spacing) != len(labels):
        raise ValueError(
            f"gridSpacing has {len(raw_spacing)} values but axisLabels has "
            f"{len(labels)}"
        )

    # gridSpacing is in the file's own units; gridUnitSI converts it to meters.
    # openPMD 2.0 makes gridUnitSI one value per axis, but implementations written
    # against 1.x emit a single scalar, so accept either.
    raw_grid_unit = np.atleast_1d(decode_attr(mesh.attrs.get("gridUnitSI", 1.0)))
    if raw_grid_unit.size == 1:
        grid_unit = np.full(len(labels), float(raw_grid_unit[0]))
    elif raw_grid_unit.size == len(labels):
        grid_unit = raw_grid_unit.astype(float)
    else:
        raise ValueError(
            f"gridUnitSI has {raw_grid_unit.size} values but axisLabels has "
            f"{len(labels)}"
        )

    spacing = {
        label: float(value) * float(unit)
        for label, value, unit in zip(labels, raw_spacing, grid_unit)
    }

    # gridGlobalOffset is the position of the first cell, in the same units as
    # gridSpacing. A file that omits it is read as a centered grid, which is this
    # class's own default rather than an assertion about the file.
    global_offset = None
    if "gridGlobalOffset" in mesh.attrs:
        raw_offset = np.atleast_1d(decode_attr(mesh.attrs["gridGlobalOffset"]))
        if len(raw_offset) != len(labels):
            raise ValueError(
                f"gridGlobalOffset has {len(raw_offset)} values but axisLabels has "
                f"{len(labels)}"
            )
        global_offset = {
            label: float(value) * float(unit)
            for label, value, unit in zip(labels, raw_offset, grid_unit)
        }

    # Everything on the record that this module does not write from the wavefront
    # itself, including names it does not know: those land in `other` so that a file
    # written against a newer revision of the extension survives a round trip.
    pmd_attrs = {
        name: decode_attr(value)
        for name, value in mesh.attrs.items()
        if name not in _COMPUTED_RECORD_ATTRS
    }
    if "zCoordinate" not in mesh.attrs:
        raise ValueError("required EXT_Wavefront attribute 'zCoordinate' is missing")
    attrs = WavefrontAttrs.from_pmd(pmd_attrs)

    kwargs = {
        "dx": spacing["x"],
        "dy": spacing["y"],
        "dz": spacing["z"],
        "wavelength": wavelength_from_photon_energy(photon_energy),
        "s_position": float(decode_attr(mesh.attrs["zCoordinate"])),
        "Ex": None,
        "Ey": None,
        "attrs": attrs,
    }

    # Permutation of the class's (x, y, z) axes into the file's stored order, so the
    # transposed view of the output array lines up with the dataset slice by slice.
    class_to_stored = _axis_permutation(_CLASS_AXIS_LABELS, labels)
    for name, key in (("x", "Ex"), ("y", "Ey")):
        if name not in mesh:
            continue
        dataset = mesh[name]

        # openPMD allows a uniform component to be stored as a group carrying only
        # `value` and `shape` instead of a dataset. It has no axes to permute.
        if is_constant_component(dataset):
            stored_shape = tuple(decode_attr(dataset.attrs["shape"]))
            kwargs[key] = np.full(
                tuple(
                    stored_shape[labels.index(label)] for label in _CLASS_AXIS_LABELS
                ),
                constant_component_value(dataset),
            )
            continue

        shape = tuple(
            dataset.shape[labels.index(label)] for label in _CLASS_AXIS_LABELS
        )
        out = np.empty(shape, dtype=dataset.dtype)

        # A view of `out` in the file's axis order. Filling it slice by slice avoids
        # allocating a second full-size array for the transpose.
        stored_view = out.transpose(class_to_stored)
        for islice in range(dataset.shape[0]):
            stored_view[islice] = dataset[islice]

        # Component values are in the file's own units; unitSI converts to V/m.
        # Multiply in place so that a complex64 field is not silently widened.
        unit_si = float(np.atleast_1d(decode_attr(dataset.attrs.get("unitSI", 1.0)))[0])
        if unit_si != 1.0:
            out *= unit_si

        kwargs[key] = out

    # The class's grid fields are the axis *midpoint*, while gridGlobalOffset is
    # the first sample, so converting between them needs the sample count.
    sample = kwargs["Ex"] if kwargs["Ex"] is not None else kwargs["Ey"]
    if global_offset is not None and sample is not None:
        counts = dict(zip(_CLASS_AXIS_LABELS, sample.shape))
        for label in _CLASS_AXIS_LABELS:
            kwargs[f"{label}mid"] = (
                global_offset[label] + (counts[label] - 1) * spacing[label] / 2
            )

    return kwargs
