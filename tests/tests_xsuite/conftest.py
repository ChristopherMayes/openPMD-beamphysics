"""
Pytest configuration and shared fixtures for XSuite tests.

This module provides shared pytest fixtures and configuration for the
XSuite ↔ openPMD conversion test suite.
"""

import pytest
import json
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture(scope="session")
def xsuite_origin_path():
    """
    Path to xsuite_origin test data directory.
    
    Returns the path to the xsuite_origin folder containing simulation inputs.
    """
    test_dir = Path(__file__).parent
    origin_path = test_dir / "xsuite_origin"
    
    if origin_path.exists():
        return origin_path
    else:
        pytest.skip(f"xsuite_origin directory not found at {origin_path}")


@pytest.fixture(scope="session")
def xsuite_output_path(tmp_path_factory):
    """
    Path to xsuite_pmd output directory.
    
    Returns a temporary directory for converted openPMD data.
    """
    output_dir = tmp_path_factory.mktemp("xsuite_pmd")
    return output_dir


@pytest.fixture
def temp_hdf5_dir(tmp_path):
    """Create temporary directory for HDF5 test files."""
    return tmp_path / "h5_temp"


@pytest.fixture
def sample_machine_params():
    """Sample FCC-ee machine parameters for testing."""
    return {
        'energy': 182.5e9,  # eV
        'circumference': 97750.0,  # m
        'harmonic_number': 132500,
        'particles_per_bunch': 1.7e11,
        'emittance_x': 1.46e-9,  # m (normalized)
        'emittance_y': 2.9e-12,  # m (normalized)
        'bunch_length': 3.5e-3,  # m
        'energy_spread': 0.001,  # ΔE/E
        'tune_x': 414.225,
        'tune_y': 410.29,
        'chromaticity_x': 2.057,
        'chromaticity_y': 1.779,
        'momentum_compaction': 7.12e-6,
        'RF_frequency': 800e6,  # Hz
        'RF_voltage': 50e6,  # V
        'beam_pipe_diameter': 0.06,  # m
        'beam_pipe_material': 'Copper',
    }


@pytest.fixture
def sample_wake_z_array():
    """Sample z coordinate array for wake functions (100 points)."""
    return np.linspace(0, 0.01, 100)


@pytest.fixture
def sample_wake_long(sample_wake_z_array):
    """Sample longitudinal wake potential."""
    z = sample_wake_z_array
    return np.exp(-z/0.001) * 1e6


@pytest.fixture
def sample_wake_dipole_x(sample_wake_z_array):
    """Sample transverse (x) wake potential."""
    z = sample_wake_z_array
    return np.exp(-z/0.002) * 5e5


@pytest.fixture
def sample_wake_dipole_y(sample_wake_z_array):
    """Sample transverse (y) wake potential."""
    z = sample_wake_z_array
    return np.exp(-z/0.002) * 5e5


@pytest.fixture
def sample_impedance_frequency():
    """Sample frequency array for impedance (100 points log-spaced)."""
    return np.logspace(6, 12, 100)


@pytest.fixture
def sample_impedance_real(sample_impedance_frequency):
    """Sample real impedance component."""
    f = sample_impedance_frequency
    return 1e3 / (1 + (f/1e9)**2)


@pytest.fixture
def sample_impedance_imag(sample_impedance_frequency):
    """Sample imaginary impedance component."""
    f = sample_impedance_frequency
    return f * 1e-6


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "conversion: marks tests as conversion tests"
    )
    config.addinivalue_line(
        "markers",
        "io: marks tests as I/O tests"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )


# ============================================================================
# Parameterized Fixtures
# ============================================================================

@pytest.fixture(params=['injection', 'z', 'w', 'zh', 'ttbar'])
def energy_point(request):
    """Parameterized fixture for all FCC-ee energy points."""
    return request.param


@pytest.fixture(params=['copper', 'stainless_steel'])
def material_type(request):
    """Parameterized fixture for material types."""
    return request.param


@pytest.fixture(params=['longitudinal', 'x', 'y'])
def impedance_plane(request):
    """Parameterized fixture for impedance planes."""
    return request.param


# ============================================================================
# Utility Functions
# ============================================================================

def create_mock_machine_params_json(tmp_path, energy_point='ttbar'):
    """
    Create a mock Booster_parameter_table.json file.
    
    Parameters
    ----------
    tmp_path : Path
        Temporary directory
    energy_point : str
        Which energy to use in the mock parameters
    
    Returns
    -------
    Path
        Path to created JSON file
    """
    json_file = tmp_path / "Booster_parameter_table.json"
    
    params = {
        "version": "PA31-3.0",
        "C": {
            "value": 90.658745,
            "unit": "km",
            "comments": "circumference"
        },
        "Np": {
            "injection": 10e9,
            "z": 25e9,
            "w": 25e9,
            "zh": 10e9,
            "ttbar": 10e9,
            "unit": ""
        },
        "Nb": {
            "z": 1120,
            "w": 890,
            "zh": 380,
            "ttbar": 56,
            "unit": ""
        },
        "E": {
            "injection": 20e9,
            "z": 45.6e9,
            "w": 80e9,
            "zh": 120e9,
            "ttbar": 182.5e9,
            "unit": "eV"
        },
        "bunch": {
            "epsnx": {"value": 1e-5, "unit": "m"},
            "epsny": {"value": 1e-5, "unit": "m"},
            "sigmaz": {"value": 0.004, "unit": "m"},
            "sigmae": {"value": 0.001, "unit": ""}
        },
        "optics": {
            "Qx": {"injection": 414.225, "z": 414.225, "w": 414.225, "zh": 414.225, "ttbar": 414.225, "unit": ""},
            "Qy": {"injection": 410.29, "z": 410.29, "w": 410.29, "zh": 410.29, "ttbar": 410.29, "unit": ""},
            "chix": {"injection": 2.057, "z": 2.057, "w": 2.057, "zh": 2.057, "ttbar": 2.057, "unit": ""},
            "chiy": {"injection": 1.779, "z": 1.779, "w": 1.779, "zh": 1.779, "ttbar": 1.779, "unit": ""},
            "alpha": {"injection": 7.12e-6, "z": 7.12e-6, "w": 7.12e-6, "zh": 7.12e-6, "ttbar": 7.12e-6, "unit": ""},
            "I2": {"injection": 5.94e-4, "z": 5.94e-4, "w": 5.94e-4, "zh": 5.94e-4, "ttbar": 5.94e-4, "unit": ""},
            "I3": {"injection": 5.68e-8, "z": 5.68e-8, "w": 5.68e-8, "zh": 5.68e-8, "ttbar": 5.68e-8, "unit": ""},
            "I5": {"injection": 1.70e-11, "z": 1.70e-11, "w": 1.70e-11, "zh": 1.70e-11, "ttbar": 1.70e-11, "unit": ""},
            "damp_xy": {"injection": 9.04, "z": 9.04, "w": 9.04, "zh": 9.04, "ttbar": 9.04, "unit": "s"},
            "damp_s": {"injection": 4.52, "z": 4.52, "w": 4.52, "zh": 4.52, "ttbar": 4.52, "unit": "s"},
            "coupling": {"injection": 0.002, "z": 0.002, "w": 0.002, "zh": 0.002, "ttbar": 0.002, "unit": ""}
        },
        "RF": {
            "RF_freq": {"injection": 800e6, "z": 800e6, "w": 800e6, "zh": 800e6, "ttbar": 800e6, "unit": "Hz"},
            "Vtot": {"injection": 50e6, "z": 50e6, "w": 50e6, "zh": 50e6, "ttbar": 50e6, "unit": "Volt"}
        },
        "beam_pipe": {
            "shape": "Circular",
            "material": "Copper",
            "D": {"value": 0.06, "unit": "m"}
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    return json_file


@pytest.fixture
def mock_machine_params_json(tmp_path):
    """Create mock machine parameters JSON file."""
    return create_mock_machine_params_json(tmp_path, energy_point='ttbar')
