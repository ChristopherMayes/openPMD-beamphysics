"""
Wakefield models for particle beam physics.

This package provides various wakefield implementations for modeling
collective effects in particle accelerators.

Classes
-------
WakefieldBase
    Abstract base class for all wakefield models
Pseudomode
    Single pseudomode parameters (amplitude, decay, wavenumber, phase)
PseudomodeWakefield
    Wakefield represented as a sum of damped sinusoidal modes
TabularWakefield
    Interpolation-based wakefield from user-supplied data
ImpedanceWakefield
    Wakefield defined through its impedance Z(k)
TaylorWakefield
    Second-order Taylor-expanded 3D wakefield (longitudinal + transverse
    kicks from wake tables, ocelot/Zagorodnov style)
TaylorWakeComponent
    A single one-dimensional component h_ab(s) of a TaylorWakefield
ResistiveWallWakefieldBase
    Base class for resistive wall wakefield models
ResistiveWallWakefield
    Accurate impedance-based resistive wall model
ResistiveWallPseudomode
    Fast pseudomode-based resistive wall model

Subpackages
-----------
resistive_wall
    Resistive wall wakefield models and low-level functions
"""

from .base import WakefieldBase
from .impedance import ImpedanceWakefield
from .pseudomode import Pseudomode, PseudomodeWakefield
from .resistive_wall import (
    Geometry,
    ResistiveWallPseudomode,
    ResistiveWallWakefield,
    ResistiveWallWakefieldBase,
    ac_conductivity,
    characteristic_length,
    longitudinal_impedance_flat,
    longitudinal_impedance_round,
    # Low-level impedance functions
    sinhc,
    surface_impedance,
    wakefield_from_impedance,
    wakefield_from_impedance_fft,
)
from .tabular import TabularWakefield
from .taylor import TaylorWakeComponent, TaylorWakefield

__all__ = [
    # Base classes
    "WakefieldBase",
    "Pseudomode",
    "PseudomodeWakefield",
    "TabularWakefield",
    "ImpedanceWakefield",
    "TaylorWakefield",
    "TaylorWakeComponent",
    # Resistive wall
    "Geometry",
    "ResistiveWallWakefieldBase",
    "ResistiveWallWakefield",
    "ResistiveWallPseudomode",
    # Low-level impedance functions
    "sinhc",
    "ac_conductivity",
    "surface_impedance",
    "longitudinal_impedance_flat",
    "longitudinal_impedance_round",
    "wakefield_from_impedance",
    "wakefield_from_impedance_fft",
    "characteristic_length",
]
