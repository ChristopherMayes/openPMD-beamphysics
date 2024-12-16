from typing import Any, Optional, Union

import h5py
import numpy as np

from .readers import component_from_alias, load_field_attrs
from .tools import encode_attrs, fstr
from .units import pg_units, pmd_unit

savefig_dpi = 450


def write_attrs(h5: Union[h5py.Group, h5py.Dataset], dct: dict[str, Any]) -> None:
    """
    Write attributes to the given `h5py.Group` or `h5py.Dataset`.

    Parameters
    ----------
    h5 : h5py.Group or h5py.Dataset
    dct : Dict[str, Any]
    """
    for k, v in dct.items():
        h5.attrs[k] = fstr(v) if isinstance(v, str) else v


def pmd_init(h5, basePath="/data/%T/", particlesPath="./"):
    """
    Root attribute initialization.

    h5 should be the root of the file.
    """
    d = {
        "basePath": basePath,
        "dataType": "openPMD",
        "openPMD": "2.0.0",
        "openPMDextension": "BeamPhysics;SpeciesType",
        # TODO: only write particlesPath if particles exist in the output file
        "particlesPath": particlesPath,
    }
    write_attrs(h5, d)


def pmd_field_init(h5, externalFieldPath="/ExternalFieldPath/%T/"):
    """
    Root attribute initialization for an openPMD-beamphysics External Field Mesh

    h5 should be the root of the file.
    """
    d = {
        "dataType": "openPMD",
        "openPMD": "2.0.0",
        "openPMDextension": "BeamPhysics",
        # TODO: only write externalFieldPath if external fields exist in the output file
        "externalFieldPath": externalFieldPath,
    }
    write_attrs(h5, d)


def pmd_wavefront_init(h5):
    """
    Root attribute initialization for an openPMD-beamphysics Wavefront init.

    h5 should be the root of the file.
    """
    d = {
        "dataType": "openPMD",
        "openPMD": "2.0.0",
        "openPMDextension": "Wavefront",
    }
    write_attrs(h5, d)


def write_pmd_bunch(h5, data, name=None):
    """
    Data is a dict with:
        np.array: 'x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight'
        str: 'species'
        int: n_particle

    Optional data:
        np.array: 'id'

    See inverse routine:
        .particles.load_bunch_data

    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    # Attributes
    g.attrs["speciesType"] = fstr(data["species"])
    g.attrs["numParticles"] = data["n_particle"]
    g.attrs["totalCharge"] = data["charge"]
    g.attrs["chargeUnitSI"] = 1.0

    # Required Datasets
    for key in ["x", "px", "y", "py", "z", "pz", "t", "status", "weight"]:
        # Get full name, write data
        g2_name = component_from_alias[key]

        # Units
        u = pg_units(key)

        # Write
        write_component_data(g, g2_name, data[key], unit=u)

    # Optional id. This does not have any units.
    if "id" in data:
        g["id"] = data["id"]


def write_pmd_field(h5, data, name=None):
    """
    Data is a dict with:
        attrs: flat dict of attributes.
        components: flat dict of components

    See inverse routine:
        .readers.load_field_data

    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    # Validate attrs
    attrs, other = load_field_attrs(data["attrs"])

    # Encode and write required and optional
    attrs = encode_attrs(attrs)
    for k, v in attrs.items():
        g.attrs[k] = v

    # All other attributes (don't change them)
    for k, v in other.items():
        g.attrs[k] = v

    # write components (datasets)
    for key, val in data["components"].items():
        # Units
        u = pg_units(key)

        # Ensure complex
        val = val.astype(complex)

        # Write
        write_component_data(g, key, val, unit=u)


def write_component_data(
    h5: h5py.Group,
    name: str,
    data,
    unit: Optional[pmd_unit] = None,
    attrs: Optional[dict[str, Any]] = None,
):
    """
    Writes data to a dataset h5[name].

    May create a `h5py.Group` or `h5py.Dataset` depending on if `data` is
    constant (or all the same value).

    Parameters
    ----------
    h5 : h5py.Group
    name : str
    data :
        If data is a constant array, a group is created with the constant value
        and shape.
    unit : pmd_unit, optional
        Units for `data`.
    attrs : dict, optional
        Additional attributes for the group.
    """
    if len(data) and np.all(data == data[0]):
        g = h5.create_group(name)
        g.attrs["value"] = data[0]
        g.attrs["shape"] = data.shape
    else:
        g = h5.create_dataset(name, data=data)

    if unit is not None:
        g.attrs["unitSI"] = unit.unitSI
        g.attrs["unitDimension"] = unit.unitDimension
        g.attrs["unitSymbol"] = fstr(unit.unitSymbol)

    if attrs:
        write_attrs(h5=g, dct=attrs)
    return g
