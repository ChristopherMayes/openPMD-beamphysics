class BeamPhysicsError(Exception):
    """Base class for errors reading openPMD data."""


class NoIterationsError(BeamPhysicsError):
    """No openPMD iteration was found."""


class MultipleIterationsError(BeamPhysicsError):
    """More than one openPMD iteration was found."""


class NoSpeciesError(BeamPhysicsError):
    """No particle species was found."""


class MultipleSpeciesError(BeamPhysicsError):
    """More than one particle species was found."""


class NotOpenPMDError(BeamPhysicsError):
    """The group is not the root of an openPMD series."""
