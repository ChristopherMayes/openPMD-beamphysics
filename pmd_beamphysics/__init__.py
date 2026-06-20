import sys
import warnings
from beamphysics import *  # noqa: F403

sys.modules["pmd_beamphysics"] = sys.modules["beamphysics"]

warnings.warn(
    "The 'pmd_beamphysics' package name is deprecated and will be removed in a future version. "
    "Please use 'beamphysics' instead.",
    DeprecationWarning,
    stacklevel=2,
)
