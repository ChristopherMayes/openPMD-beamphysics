from __future__ import annotations

import dataclasses
import datetime
import getpass
import platform
from collections.abc import Sequence
from typing import Any, TypeVar

from typing_extensions import Literal

import h5py
import numpy as np

from . import tools
from .types import Dataclass
from .units import pmd_unit, known_unit

PolarizationDirection = Literal["x", "y", "z"]


def python_attrs_to_pmd_keys(obj: Dataclass | type[Dataclass]) -> dict[str, str]:
    assert dataclasses.is_dataclass(obj)
    return {
        fld.name: fld.metadata.get("pmd_key", fld.name)
        for fld in dataclasses.fields(obj)
    }


def hdf_to_python_attrs(obj: Dataclass | type[Dataclass]) -> dict[str, str]:
    assert dataclasses.is_dataclass(obj)
    return {
        fld.metadata.get("pmd_key", fld.name): fld.name
        for fld in dataclasses.fields(obj)
    }


def get_pmd_metadata_dict(
    obj: Dataclass,
    attrs: Sequence[str],
) -> dict[str, str | float | None]:
    attr_to_field = python_attrs_to_pmd_keys(obj)
    return {
        attr_to_field[attr]: getattr(obj, attr)
        for attr in attrs
        if getattr(obj, attr) is not None
    }


def _key(pmd_key: str):
    return {"pmd_key": pmd_key}


_T = TypeVar("_T", bound=Dataclass)


def _dataclass_from_hdf5(cls: type[_T], h5: h5py.Group | h5py.Dataset) -> _T:
    hdf_key_to_attr = hdf_to_python_attrs(cls)

    def maybe_decode(value):
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        return value

    values = {
        attr: maybe_decode(h5.attrs[hdf_key])
        for hdf_key, attr in hdf_key_to_attr.items()
        if hdf_key in h5.attrs
    }
    return cls(**values)


def _to_dataclass_dict(cls: type[_T], dct: dict[str, Any]) -> dict[str, Any]:
    hdf_key_to_attr = hdf_to_python_attrs(cls)

    result: dict[str, Any] = {}
    for hdf_key, attr in hdf_key_to_attr.items():
        if hdf_key in dct:
            result[attr] = dct.pop(hdf_key)
        if attr in dct:
            result[attr] = dct.pop(attr)

    return result


@dataclasses.dataclass
class BaseMetadata:
    """Base metadata for OpenPMD spec files."""

    spec_version: str = "2.0.0"

    author: str = dataclasses.field(default_factory=getpass.getuser)
    machine: str = dataclasses.field(default_factory=platform.node)
    comment: str = dataclasses.field(default="")

    software: str = dataclasses.field(default="pmd_beamphysics")
    software_version: str = dataclasses.field(
        default_factory=tools.get_version, metadata=_key("softwareVersion")
    )
    software_dependencies: str = dataclasses.field(
        default="", metadata=_key("softwareDependencies")
    )
    iteration_encoding: Literal["fileBased", "groupBased"] = dataclasses.field(
        default="groupBased", metadata=_key("iterationEncoding")
    )
    iteration_format: str = dataclasses.field(
        default="/data/%T/", metadata=_key("iterationFormat")
    )
    date: datetime.datetime = dataclasses.field(
        default_factory=tools.current_date_with_tzinfo,
        metadata=_key("date"),
    )

    def __post_init__(self):
        if isinstance(self.date, str):
            try:
                self.date = datetime.datetime.fromisoformat(self.date)
            except Exception:
                self.date = tools.current_date_with_tzinfo()

    @property
    def attrs(self):
        res = get_pmd_metadata_dict(
            self,
            [
                "author",
                "machine",
                "comment",
                "software",
                "software_version",
                "software_dependencies",
                "iteration_format",
                "iteration_encoding",
            ],
        )
        return {
            **res,
            "date": tools.pmd_format_date(self.date),
        }


@dataclasses.dataclass
class IterationMetadata:
    """Per-iteration metadata for OpenPMD spec files."""

    iteration: int = 0

    time: float = dataclasses.field(default=0.0, metadata=_key("time"))
    dt: float = dataclasses.field(default=0.0, metadata=_key("dt"))
    time_unit_si: float = dataclasses.field(default=1.0, metadata=_key("timeUnitSI"))

    @property
    def attrs(self):
        return get_pmd_metadata_dict(
            self,
            [
                "time",
                "dt",
                "time_unit_si",
            ],
        )


Geometry = Literal["cartesian", "thetaMode", "cylindrical", "spherical", "other"]


@dataclasses.dataclass
class MeshMetadata:
    """Per-Mesh metadata for OpenPMD spec files."""

    geometry: Geometry = "cartesian"
    geometry_parameters: str | None = dataclasses.field(
        default=None, metadata=_key("geometryParameters")
    )
    axis_labels: tuple[str, ...] = dataclasses.field(
        default=("x", "y", "z"), metadata=_key("axisLabels")
    )
    grid_spacing: tuple[float, ...] = dataclasses.field(
        default=(), metadata=_key("gridSpacing")
    )
    grid_global_offset: tuple[float, ...] = dataclasses.field(
        default=(), metadata=_key("gridGlobalOffset")
    )
    grid_unit_dimension: tuple[float, ...] = dataclasses.field(
        default=(), metadata=_key("gridUnitDimension")
    )
    position: tuple[float, ...] = dataclasses.field(
        default_factory=tuple,
    )
    particle_list: tuple[str, ...] = dataclasses.field(
        default_factory=tuple, metadata=_key("particleList")
    )

    @property
    def attrs(self):
        dct = get_pmd_metadata_dict(
            self,
            [
                "geometry",
                "geometry_parameters",
                "axis_labels",
                "grid_spacing",
                "grid_global_offset",
                "grid_unit_dimension",
                "position",
            ],
        )

        if self.geometry == "thetaMode":
            if not self.geometry_parameters:
                raise ValueError("geometry_parameters is required in thetaMode")

        if self.particle_list:
            dct["particleList"] = ";".join(self.particle_list)
        return dct


@dataclasses.dataclass
class WavefrontMetadata:
    """Per-Wavefront metadata for OpenPMD spec files."""

    index: int | None = None
    units: pmd_unit = dataclasses.field(default_factory=lambda: known_unit["V/m"])

    base: BaseMetadata = dataclasses.field(default_factory=BaseMetadata)
    iteration: IterationMetadata = dataclasses.field(default_factory=IterationMetadata)
    mesh: MeshMetadata = dataclasses.field(default_factory=MeshMetadata)
    polarization: PolarizationDirection = dataclasses.field(default="x")
    beamline: str = dataclasses.field(default="")
    radius_of_curvature_x: float | None = dataclasses.field(
        default=None,
        metadata=_key("radiusOfCurvatureX"),
    )
    radius_of_curvature_y: float | None = dataclasses.field(
        default=None,
        metadata=_key("radiusOfCurvatureY"),
    )
    delta_radius_of_curvature_x: float | None = dataclasses.field(
        default=None,
        metadata=_key("deltaRadiusOfCurvatureX"),
    )
    delta_radius_of_curvature_y: float | None = dataclasses.field(
        default=None,
        metadata=_key("deltaRadiusOfCurvatureY"),
    )
    z_coordinate: float = dataclasses.field(default=0.0, metadata=_key("zCoordinate"))
    # pads: tuple[int, ...] = dataclasses.field(
    #     default_factory=tuple, metadata=_key("pads")
    # )

    @property
    def attrs(self) -> dict[str, str | float | None]:
        """electricField attributes."""
        return get_pmd_metadata_dict(
            self,
            [
                "beamline",
                "radius_of_curvature_x",
                "radius_of_curvature_y",
                "delta_radius_of_curvature_x",
                "delta_radius_of_curvature_y",
                # "pads",
            ],
        )

    def to_dict(self):
        return {
            "base": self.base.attrs,
            "iteration": self.iteration.attrs,
            "mesh": self.mesh.attrs,
        }

    @classmethod
    def from_dict(cls, md: dict) -> WavefrontMetadata:
        md = dict(md)
        base_md = _to_dataclass_dict(
            BaseMetadata,
            md.pop("base") if "base" in md else md,
        )
        iteration_md = _to_dataclass_dict(
            IterationMetadata,
            md.pop("iteration") if "iteration" in md else md,
        )
        mesh_md = _to_dataclass_dict(
            MeshMetadata,
            md.pop("mesh") if "mesh" in md else md,
        )

        return WavefrontMetadata(
            base=BaseMetadata(**base_md),
            iteration=IterationMetadata(**iteration_md),
            mesh=MeshMetadata(**mesh_md),
            **md,
        )

    @classmethod
    def from_hdf5(
        cls, base_h5: h5py.Group, field_h5: h5py.Group, rmesh_h5: h5py.Dataset
    ):
        md = _dataclass_from_hdf5(cls, field_h5)
        md.base = _dataclass_from_hdf5(BaseMetadata, base_h5.require_group("/"))
        md.iteration = _dataclass_from_hdf5(IterationMetadata, base_h5)
        md.mesh = _dataclass_from_hdf5(MeshMetadata, rmesh_h5)
        if isinstance(md.base.date, str):
            try:
                md.base.date = datetime.datetime.fromisoformat(md.base.date)
            except Exception:
                md.base.date = tools.current_date_with_tzinfo()
        return md
