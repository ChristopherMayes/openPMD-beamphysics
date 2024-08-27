from typing import Any, ClassVar, Dict

from typing_extensions import Protocol


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]
