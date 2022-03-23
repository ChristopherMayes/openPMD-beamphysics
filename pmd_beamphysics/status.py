from enum import IntEnum

class ParticleStatus(IntEnum):
    """
    Particle Status Enum
    This is defined by the openPMD-beamphysics standard as integers.
    """
    CATHODE =  0 
    ALIVE   =  1