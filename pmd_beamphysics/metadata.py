from __future__ import annotations

import dataclasses
import datetime
import getpass
import platform
from typing import Dict, Optional, Sequence, Tuple, Union

from typing_extensions import Literal

from . import tools
from .types import Dataclass


def get_pmd_metadata_dict(
    obj: Dataclass,
    attrs: Sequence[str],
) -> Dict[str, Union[str, float, None]]:
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


@dataclasses.dataclass
class BaseMetadata:
    # Base pmd wavefront file attrs:
    spec_version: str = "2.0.0"

    author: str = dataclasses.field(default_factory=getpass.getuser)
    machine: str = dataclasses.field(default_factory=platform.node)
    comment: str = dataclasses.field(default="")

    software: str = dataclasses.field(default="pmd_beamphysics")
    software_version: str = dataclasses.field(
        default_factory=tools.get_version, metadata={"pmd_key": "softwareVersion"}
    )
    software_dependencies: str = dataclasses.field(
        default="", metadata={"pmd_key": "softwareDependencies"}
    )
    iteration_encoding: Literal["fileBased", "groupBased"] = dataclasses.field(
        default="groupBased", metadata={"pmd_key": "iterationEncoding"}
    )
    iteration_format: str = dataclasses.field(
        default="/data/%T/", metadata={"pmd_key": "iterationFormat"}
    )
    date: datetime.datetime = dataclasses.field(
        default_factory=tools.current_date_with_tzinfo,
        metadata={"pmd_key": "date"},
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
    iteration: int = 0

    time: float = dataclasses.field(default=0.0, metadata={"pmd_key": "time"})
    dt: float = dataclasses.field(default=0.0, metadata={"pmd_key": "dt"})
    time_unit_si: float = dataclasses.field(
        default=1.0, metadata={"pmd_key": "timeUnitSI"}
    )

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


@dataclasses.dataclass
class WavefrontMetadata:
    base: BaseMetadata = dataclasses.field(default_factory=BaseMetadata)
    iteration: IterationMetadata = dataclasses.field(default_factory=IterationMetadata)
    wavefront_index: Optional[int] = None

    beamline: str = dataclasses.field(default="")
    radius_of_curvature_x: Optional[float] = dataclasses.field(
        default=None,
        metadata={"pmd_key": "radiusOfCurvatureX"},
    )
    radius_of_curvature_y: Optional[float] = dataclasses.field(
        default=None,
        metadata={"pmd_key": "radiusOfCurvatureY"},
    )
    delta_radius_of_curvature_x: Optional[float] = dataclasses.field(
        default=None,
        metadata={"pmd_key": "deltaRadiusOfCurvatureX"},
    )
    delta_radius_of_curvature_y: Optional[float] = dataclasses.field(
        default=None,
        metadata={"pmd_key": "deltaRadiusOfCurvatureY"},
    )
    z_coordinate: float = dataclasses.field(
        default=0.0, metadata={"pmd_key": "zCoordinate"}
    )
    pads: Tuple[float, ...] = dataclasses.field(
        default_factory=tuple, metadata={"pmd_key": "pads"}
    )

    @property
    def attrs(self) -> Dict[str, Union[str, float, None]]:
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
