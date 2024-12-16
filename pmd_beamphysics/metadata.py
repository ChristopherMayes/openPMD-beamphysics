from __future__ import annotations

import dataclasses
import datetime
import getpass
import platform
from collections.abc import Sequence

from typing_extensions import Literal

from . import tools
from .types import Dataclass
from .units import pmd_unit, known_unit

PolarizationDirection = Literal["x", "y", "z"]


def get_pmd_metadata_dict(
    obj: Dataclass,
    attrs: Sequence[str],
) -> dict[str, str | float | None]:
    assert dataclasses.is_dataclass(obj)
    attr_to_field = {
        fld.name: fld.metadata.get("pmd_key", fld.name)
        for fld in dataclasses.fields(obj)
    }
    return {
        attr_to_field[attr]: getattr(obj, attr)
        for attr in attrs
        if getattr(obj, attr) is not None
    }


def _key(pmd_key: str):
    return {"pmd_key": pmd_key}


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
    pads: tuple[int, ...] = dataclasses.field(
        default_factory=tuple, metadata=_key("pads")
    )

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
                "pads",
            ],
        )

    @classmethod
    def from_dict(cls, md: dict) -> WavefrontMetadata:
        md = dict(md)
        if "base" in md:
            base_md = md.pop("base")
        else:
            base_md = {
                key: value
                for key, value in md.items()
                if key in BaseMetadata.__dataclass_fields__
            }
        if "iteration" in md:
            iteration_md = md.pop("base")
        else:
            iteration_md = {
                key: value
                for key, value in md.items()
                if key in IterationMetadata.__dataclass_fields__
            }
        for key in list(base_md) + list(iteration_md):
            md.pop(key)
        return WavefrontMetadata(
            base=BaseMetadata(**base_md),
            iteration=IterationMetadata(**iteration_md),
            **md,
        )
