from enum import IntEnum


class ParticleStatus(IntEnum):
    """
    Particle Status Enum
    This is defined by the openPMD-beamphysics standard as integers.
    """

    CATHODE = 0
    ALIVE = 1


class LocationInElement(IntEnum):
    """
    Where a particle sits relative to a lattice element's hard edges.

    This is the openPMD-beamphysics `locationInElement` record. It matters only for
    hard-edge or zero-length elements, where the s-position alone cannot tell
    whether a particle is inside the fringe field.
    """

    UPSTREAM_END = -1
    INSIDE = 0
    DOWNSTREAM_END = 1
