from __future__ import annotations

import datetime
import platform
import getpass
import dataclasses

from typing import Dict, Sequence, Union
from typing_extensions import Literal

from . import tools


@dataclasses.dataclass
class BaseMetadata:
    data_index: int = 0
    object_index: int = 0

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

    # Per iteration
    iteration_time: float = dataclasses.field(default=0.0, metadata={"pmd_key": "time"})
    iteration_dt: float = dataclasses.field(default=0.0, metadata={"pmd_key": "dt"})
    iteration_time_unit_si: float = dataclasses.field(
        default=1.0, metadata={"pmd_key": "timeUnitSI"}
    )

    def _get_pmd_dict(self, attrs: Sequence[str]) -> Dict[str, Union[str, float, None]]:
        attr_to_field = {
            fld.name: fld.metadata.get("pmd_key", fld.name)
            for fld in dataclasses.fields(self)
        }
        return {
            attr_to_field[attr]: getattr(self, attr)
            for attr in attrs
            if getattr(self, attr) is not None
        }

    @property
    def iteration_attrs(self):
        return self._get_pmd_dict(
            [
                "iteration_time",
                "iteration_dt",
                "iteration_time_unit_si",
            ]
        )

    @property
    def base_attrs(self):
        res = self._get_pmd_dict(
            [
                "author",
                "machine",
                "comment",
                "software",
                "software_version",
                "software_dependencies",
                "iteration_format",
                "iteration_encoding",
            ]
        )
        return {
            **res,
            "date": tools.pmd_format_date(self.date),
        }
