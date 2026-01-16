"""
XSuite Interface for openPMD-beamphysics

This module provides utilities for working with XSuite simulation data in
OpenPMD format, including:

- Input/Output operations (xsuite_io)
- Parameter scan management (xsuite_scans)
- TMCI analysis (xsuite_tmci)

Usage:
    >>> from pmd_beamphysics.interfaces import xsuite_io, xsuite_scans, xsuite_tmci
    >>> # or import specific functions:
    >>> from pmd_beamphysics.interfaces.xsuite_io import load_machine_parameters
    >>> params = load_machine_parameters('fcc_ee_params.h5', mode='z')

Author: Adnan Ghribi, XSuite Collaboration
License: BSD-3-Clause
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Adnan Ghribi, XSuite Collaboration'

# Import main modules
from . import xsuite_io
from . import xsuite_scans
from . import xsuite_tmci

# Import commonly used functions for convenience
from .xsuite_io import (
    load_machine_parameters,
    load_wake_potential,
    load_impedance,
    load_particles,
    save_particles,
    validate_openpmd_file,
    compute_file_checksum,
)

from .xsuite_scans import (
    create_scan_manifest,
    update_scan_point,
    get_scan_status,
    print_scan_status,
    load_scan_results,
    find_threshold,
)

from .xsuite_tmci import (
    TMCIAnalyzer,
    TMCIResults,
    ModeData,
    compute_fft_spectrum,
    find_coherent_modes,
    fit_growth_rate,
)

# Define public API
__all__ = [
    # Modules
    'xsuite_io',
    'xsuite_scans',
    'xsuite_tmci',
    
    # I/O functions
    'load_machine_parameters',
    'load_wake_potential',
    'load_impedance',
    'load_particles',
    'save_particles',
    'validate_openpmd_file',
    'compute_file_checksum',
    
    # Scan functions
    'create_scan_manifest',
    'update_scan_point',
    'get_scan_status',
    'print_scan_status',
    'load_scan_results',
    'find_threshold',
    
    # TMCI classes and functions
    'TMCIAnalyzer',
    'TMCIResults',
    'ModeData',
    'compute_fft_spectrum',
    'find_coherent_modes',
    'fit_growth_rate',
]


def get_version():
    """Return XSuite interface version."""
    return __version__


def list_available_functions():
    """Print available XSuite interface functions."""
    print("\n" + "="*70)
    print("XSuite Interface for openPMD-beamphysics")
    print(f"Version: {__version__}")
    print("="*70)
    
    print("\n📥 INPUT/OUTPUT (xsuite_io):")
    print("  • load_machine_parameters(file, mode=None)")
    print("  • load_wake_potential(file)")
    print("  • load_impedance(file)")
    print("  • load_particles(file, iteration=0)")
    print("  • save_particles(file, particles, metadata)")
    print("  • validate_openpmd_file(file)")
    
    print("\n📊 PARAMETER SCANS (xsuite_scans):")
    print("  • create_scan_manifest(file, parameters, ...)")
    print("  • update_scan_point(manifest, point_index, results)")
    print("  • get_scan_status(manifest)")
    print("  • print_scan_status(manifest)")
    print("  • load_scan_results(manifest, result_names)")
    print("  • find_threshold(manifest, result_name, criterion)")
    
    print("\n🔬 TMCI ANALYSIS (xsuite_tmci):")
    print("  • TMCIAnalyzer() - Main analysis class")
    print("  • TMCIResults - Results container")
    print("  • ModeData - Mode information container")
    print("  • compute_fft_spectrum(signal, ...)")
    print("  • find_coherent_modes(frequencies, power, ...)")
    print("  • fit_growth_rate(amplitude_evolution, ...)")
    
    print("\n📚 Documentation:")
    print("  • See docs/examples/xsuite/ for Jupyter notebooks")
    print("  • See pmd_beamphysics/interfaces/EXT_XSuite*.md for specifications")
    print()


# Optional: Add version check on import
def _check_dependencies():
    """Check if required dependencies are available."""
    try:
        import h5py
        import numpy
    except ImportError as e:
        import warnings
        warnings.warn(
            f"XSuite interface requires h5py and numpy: {e}",
            ImportWarning
        )


_check_dependencies()


# Print version info when module is imported with verbose mode
import os
if os.environ.get('XSUITE_VERBOSE', '').lower() in ('1', 'true', 'yes'):
    print(f"XSuite interface v{__version__} loaded")