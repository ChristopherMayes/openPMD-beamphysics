from __future__ import annotations

import typing as _typing

if _typing.TYPE_CHECKING:
    from .fields import FieldMesh
    from .particles import ParticleGroup, single_particle
    from .readers import particle_paths
    from .status import ParticleStatus
    from .wavefront import Wavefront, WavefrontK
    from .writers import pmd_init

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"


_LAZY_IMPORTS = {
    "FieldMesh": ".fields",
    "ParticleGroup": ".particles",
    "single_particle": ".particles",
    "particle_paths": ".readers",
    "ParticleStatus": ".status",
    "Wavefront": ".wavefront",
    "WavefrontK": ".wavefront",
    "pmd_init": ".writers",
    "set_default_backend": ".plot_dispatch",
    "get_default_backend": ".plot_dispatch",
}


__all__ = [
    "FieldMesh",
    "ParticleGroup",
    "ParticleStatus",
    "particle_paths",
    "pmd_init",
    "single_particle",
    "set_default_backend",
    "get_default_backend",
    "Wavefront",
    "WavefrontK",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], package=__name__)

        obj = getattr(module, name)

        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
