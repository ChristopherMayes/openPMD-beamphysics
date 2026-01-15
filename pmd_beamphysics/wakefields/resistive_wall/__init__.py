"""
Resistive wall wakefield models.

This subpackage provides analytical models for short-range resistive wall
wakefields in accelerator beam pipes, based on the approach described in
SLAC-PUB-10707.

Two models are available:

- **ResistiveWallWakefield**: Accurate impedance-based model using FFT
  convolution. Recommended for most applications.

- **ResistiveWallPseudomode**: Fast pseudomode-based model using polynomial fits.
  Good for quick calculations, ~10-20% difference from full impedance.

Low-level functions are also exported for direct impedance/wakefield evaluation.

Classes
-------
ResistiveWallWakefieldBase
    Abstract base class with shared properties
ResistiveWallWakefield
    Accurate impedance-based wakefield model
ResistiveWallPseudomode
    Fast pseudomode-based wakefield model

Functions
---------
sinhc
    Numerically stable sinh(x)/x
ac_conductivity
    Drude-model AC conductivity
surface_impedance
    Surface impedance for conducting wall
longitudinal_impedance_round
    Longitudinal impedance Z(k) for round pipe
longitudinal_impedance_flat
    Longitudinal impedance Z(k) for flat geometry
wakefield_from_impedance
    Wakefield W(z) via cosine transform (quadrature)
wakefield_from_impedance_fft
    Wakefield W(z) via FFT (fast)
characteristic_length
    Characteristic length s₀

References
----------
.. [1] K. Bane and G. Stupakov, "Resistive wall wakefield in the LCLS
   undulator beam pipe," SLAC-PUB-10707 (2004).
   https://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-10707.pdf
"""

from .base import (
    Geometry,
    ResistiveWallWakefieldBase,
    # Low-level functions
    sinhc,
    ac_conductivity,
    surface_impedance,
    longitudinal_impedance_round,
    longitudinal_impedance_flat,
    wakefield_from_impedance,
    wakefield_from_impedance_fft,
    characteristic_length,
    s0f,
    Gammaf,
    krs0_round,
    krs0_flat,
    Qr_round,
    Qr_flat,
)
from .impedance import ResistiveWallWakefield
from .pseudomode import ResistiveWallPseudomode

__all__ = [
    # Enum
    "Geometry",
    # Classes
    "ResistiveWallWakefieldBase",
    "ResistiveWallWakefield",
    "ResistiveWallPseudomode",
    # Low-level functions
    "sinhc",
    "ac_conductivity",
    "surface_impedance",
    "longitudinal_impedance_round",
    "longitudinal_impedance_flat",
    "wakefield_from_impedance",
    "wakefield_from_impedance_fft",
    "characteristic_length",
    "s0f",
    "Gammaf",
    "krs0_round",
    "krs0_flat",
    "Qr_round",
    "Qr_flat",
]
