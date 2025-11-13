"""
Tests for xsuite_io module.

Test coverage:
- Machine parameter I/O
- Wake table I/O
- Impedance table I/O
- File validation
- Component listing
- Error handling
- Edge cases
"""

import pytest
import h5py
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path to import the module
# This allows running tests from the tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import module to test
try:
    # Try package import first (when installed)
    from pmd_beamphysics.interfaces import xsuite_io
except ImportError:
    # Fall back to direct import (when running from repository)
    try:
        from pmd_beamphysics.interfaces import xsuite_io
    except ImportError:
        # Last resort: import from same directory
        import xsuite_io


class TestMachineParameters:
    """Tests for machine parameter I/O."""
    
    def test_write_and_read_machine_parameters(self, tmp_path):
        """Test writing and reading machine parameters."""
        h5_file = tmp_path / "test_params.h5"
        
        # Define test parameters
        params = {
            'energy': 182.5e9,  # eV
            'circumference': 97750.0,  # m
            'harmonic_number': 132500,
            'particles_per_bunch': 1.7e11,
            'beta_x': 0.125,
            'beta_y': 0.068,
            'emittance_x': 1.46e-9,
            'emittance_y': 2.9e-12,
            'bunch_length': 3.5e-3  # m
        }
        
        # Write parameters
        xsuite_io.write_machine_parameters(str(h5_file), params)
        
        # Read parameters back
        read_params = xsuite_io.read_machine_parameters(str(h5_file))
        
        # Verify all parameters match
        assert len(read_params) == len(params)
        for key in params:
            assert key in read_params
            assert read_params[key] == pytest.approx(params[key])
    
    def test_write_with_file_object(self, tmp_path):
        """Test writing parameters with h5py.File object."""
        h5_file = tmp_path / "test_params.h5"
        
        params = {'energy': 45.6e9, 'circumference': 26658.88}
        
        with h5py.File(h5_file, 'w') as f:
            xsuite_io.write_machine_parameters(f, params)
        
        # Verify
        read_params = xsuite_io.read_machine_parameters(str(h5_file))
        assert read_params['energy'] == pytest.approx(45.6e9)
    
    def test_read_nonexistent_path(self, tmp_path):
        """Test reading from nonexistent path raises error."""
        h5_file = tmp_path / "test.h5"
        
        # Create empty file
        with h5py.File(h5_file, 'w') as f:
            pass
        
        # Try to read from nonexistent path
        with pytest.raises(xsuite_io.XSuiteIOError):
            xsuite_io.read_machine_parameters(str(h5_file))
    
    def test_custom_base_path(self, tmp_path):
        """Test using custom base path."""
        h5_file = tmp_path / "test.h5"
        custom_path = "/custom/machine/params/"
        
        params = {'energy': 7e12}
        
        xsuite_io.write_machine_parameters(str(h5_file), params, base_path=custom_path)
        read_params = xsuite_io.read_machine_parameters(str(h5_file), base_path=custom_path)
        
        assert read_params['energy'] == pytest.approx(7e12)
    
    def test_overwrite_parameters(self, tmp_path):
        """Test overwriting existing parameters."""
        h5_file = tmp_path / "test.h5"
        
        # Write first set
        params1 = {'energy': 1e9, 'circumference': 1000.0}
        xsuite_io.write_machine_parameters(str(h5_file), params1)
        
        # Write second set (should overwrite)
        params2 = {'energy': 2e9, 'particles_per_bunch': 1e10}
        xsuite_io.write_machine_parameters(str(h5_file), params2)
        
        # Read back
        read_params = xsuite_io.read_machine_parameters(str(h5_file))
        
        # Should have both old and new parameters
        assert read_params['energy'] == pytest.approx(2e9)
        assert read_params['circumference'] == pytest.approx(1000.0)
        assert read_params['particles_per_bunch'] == pytest.approx(1e10)


class TestWakeTables:
    """Tests for wake table I/O."""
    
    def test_write_and_read_wake_table(self, tmp_path):
        """Test writing and reading wake table."""
        h5_file = tmp_path / "test_wakes.h5"
        
        # Create test wake data
        z = np.linspace(0, 1, 1000)  # m
        wake = np.exp(-z/0.1) * 1e6  # V/C
        
        # Write wake table
        xsuite_io.write_wake_table(str(h5_file), z, wake, component='longitudinal')
        
        # Read back
        z_read, wake_read, metadata = xsuite_io.read_wake_table(
            str(h5_file), component='longitudinal'
        )
        
        # Verify
        np.testing.assert_array_almost_equal(z, z_read)
        np.testing.assert_array_almost_equal(wake, wake_read)
        assert metadata['component'] == 'longitudinal'
        assert metadata['z_unit'] == 'm'
        assert metadata['wake_unit'] == 'V/C'
    
    def test_multiple_wake_components(self, tmp_path):
        """Test writing multiple wake components."""
        h5_file = tmp_path / "test_wakes.h5"
        
        z = np.linspace(0, 0.5, 500)
        
        # Write longitudinal
        wake_long = np.sin(z * 10) * 1e6
        xsuite_io.write_wake_table(str(h5_file), z, wake_long, component='longitudinal')
        
        # Write dipole_x
        wake_x = np.cos(z * 5) * 5e5
        xsuite_io.write_wake_table(str(h5_file), z, wake_x, component='dipole_x')
        
        # Write dipole_y
        wake_y = np.sin(z * 8) * 3e5
        xsuite_io.write_wake_table(str(h5_file), z, wake_y, component='dipole_y')
        
        # Read back each component
        z_long, wake_long_read, _ = xsuite_io.read_wake_table(str(h5_file), 'longitudinal')
        z_x, wake_x_read, _ = xsuite_io.read_wake_table(str(h5_file), 'dipole_x')
        z_y, wake_y_read, _ = xsuite_io.read_wake_table(str(h5_file), 'dipole_y')
        
        # Verify
        np.testing.assert_array_almost_equal(wake_long, wake_long_read)
        np.testing.assert_array_almost_equal(wake_x, wake_x_read)
        np.testing.assert_array_almost_equal(wake_y, wake_y_read)
    
    def test_wake_with_metadata(self, tmp_path):
        """Test writing wake table with metadata."""
        h5_file = tmp_path / "test_wakes.h5"
        
        z = np.linspace(0, 1, 100)
        wake = np.zeros_like(z)
        
        metadata = {
            'source': 'CST simulation',
            'date': '2025-11-12',
            'model': 'FCC-ee_chamber',
            'resolution': 1e-3
        }
        
        xsuite_io.write_wake_table(
            str(h5_file), z, wake, 
            component='longitudinal',
            metadata=metadata
        )
        
        # Read back
        _, _, read_metadata = xsuite_io.read_wake_table(str(h5_file), 'longitudinal')
        
        # Verify metadata
        for key in metadata:
            assert key in read_metadata
            assert read_metadata[key] == metadata[key]
    
    def test_transverse_wake_units(self, tmp_path):
        """Test that transverse wakes have correct units."""
        h5_file = tmp_path / "test_wakes.h5"
        
        z = np.linspace(0, 0.5, 100)
        wake = np.ones_like(z) * 1e5
        
        xsuite_io.write_wake_table(str(h5_file), z, wake, component='dipole_x')
        
        _, _, metadata = xsuite_io.read_wake_table(str(h5_file), 'dipole_x')
        
        assert metadata['wake_unit'] == 'V/C/m'
    
    def test_read_nonexistent_component(self, tmp_path):
        """Test reading nonexistent wake component raises error."""
        h5_file = tmp_path / "test_wakes.h5"
        
        # Create file with one component
        z = np.linspace(0, 1, 100)
        wake = np.zeros_like(z)
        xsuite_io.write_wake_table(str(h5_file), z, wake, component='longitudinal')
        
        # Try to read nonexistent component
        with pytest.raises(xsuite_io.XSuiteIOError):
            xsuite_io.read_wake_table(str(h5_file), component='quadrupole')
    
    def test_overwrite_wake_component(self, tmp_path):
        """Test overwriting existing wake component."""
        h5_file = tmp_path / "test_wakes.h5"
        
        z = np.linspace(0, 1, 100)
        
        # Write first version
        wake1 = np.ones_like(z)
        xsuite_io.write_wake_table(str(h5_file), z, wake1, component='longitudinal')
        
        # Overwrite
        wake2 = np.ones_like(z) * 2
        xsuite_io.write_wake_table(str(h5_file), z, wake2, component='longitudinal')
        
        # Read back
        _, wake_read, _ = xsuite_io.read_wake_table(str(h5_file), 'longitudinal')
        
        # Should be the second version
        np.testing.assert_array_almost_equal(wake2, wake_read)


class TestImpedanceTables:
    """Tests for impedance table I/O."""
    
    def test_write_and_read_impedance_table(self, tmp_path):
        """Test writing and reading impedance table."""
        h5_file = tmp_path / "test_impedance.h5"
        
        # Create test impedance data
        frequency = np.logspace(6, 12, 1000)  # Hz
        real_Z = 1e3 / (1 + (frequency/1e9)**2)  # Ohm
        imag_Z = frequency * 1e-6  # Ohm
        
        # Write
        xsuite_io.write_impedance_table(
            str(h5_file), frequency, real_Z, imag_Z, 
            plane='longitudinal'
        )
        
        # Read back
        freq_read, real_read, imag_read, metadata = xsuite_io.read_impedance_table(
            str(h5_file), plane='longitudinal'
        )
        
        # Verify
        np.testing.assert_array_almost_equal(frequency, freq_read)
        np.testing.assert_array_almost_equal(real_Z, real_read)
        np.testing.assert_array_almost_equal(imag_Z, imag_read)
        assert metadata['plane'] == 'longitudinal'
        assert metadata['frequency_unit'] == 'Hz'
        assert metadata['impedance_unit'] == 'Ohm'
    
    def test_multiple_impedance_planes(self, tmp_path):
        """Test writing impedances for multiple planes."""
        h5_file = tmp_path / "test_impedance.h5"
        
        frequency = np.logspace(6, 10, 500)
        
        # Longitudinal
        real_long = np.ones_like(frequency) * 100
        imag_long = np.zeros_like(frequency)
        xsuite_io.write_impedance_table(
            str(h5_file), frequency, real_long, imag_long, plane='longitudinal'
        )
        
        # X plane
        real_x = np.ones_like(frequency) * 1e6
        imag_x = frequency * 1e-3
        xsuite_io.write_impedance_table(
            str(h5_file), frequency, real_x, imag_x, plane='x'
        )
        
        # Y plane
        real_y = np.ones_like(frequency) * 5e5
        imag_y = -frequency * 5e-4
        xsuite_io.write_impedance_table(
            str(h5_file), frequency, real_y, imag_y, plane='y'
        )
        
        # Read back and verify
        _, real_long_read, _, _ = xsuite_io.read_impedance_table(str(h5_file), 'longitudinal')
        _, real_x_read, _, _ = xsuite_io.read_impedance_table(str(h5_file), 'x')
        _, real_y_read, _, _ = xsuite_io.read_impedance_table(str(h5_file), 'y')
        
        np.testing.assert_array_almost_equal(real_long, real_long_read)
        np.testing.assert_array_almost_equal(real_x, real_x_read)
        np.testing.assert_array_almost_equal(real_y, real_y_read)
    
    def test_impedance_with_metadata(self, tmp_path):
        """Test writing impedance with metadata."""
        h5_file = tmp_path / "test_impedance.h5"
        
        frequency = np.logspace(6, 10, 100)
        real_Z = np.ones_like(frequency) * 50
        imag_Z = np.zeros_like(frequency)
        
        metadata = {
            'source': 'IW2D simulation',
            'element': 'pumping_port',
            'location': 's=1250.5m'
        }
        
        xsuite_io.write_impedance_table(
            str(h5_file), frequency, real_Z, imag_Z,
            plane='x', metadata=metadata
        )
        
        # Read and verify
        _, _, _, read_metadata = xsuite_io.read_impedance_table(str(h5_file), 'x')
        
        for key in metadata:
            assert key in read_metadata
    
    def test_complex_impedance_values(self, tmp_path):
        """Test impedance with complex resonator model."""
        h5_file = tmp_path / "test_impedance.h5"
        
        # Resonator model: Z = R / (1 + jQ(f/f0 - f0/f))
        # Create frequency array that includes resonance
        f0 = 1e9  # resonance frequency
        freq_low = np.logspace(8, np.log10(f0*0.9), 400)
        freq_high = np.logspace(np.log10(f0*1.1), 10, 400)
        frequency = np.concatenate([freq_low, [f0], freq_high])
        
        R = 1e3  # shunt impedance
        Q = 100  # quality factor
        
        delta = frequency/f0 - f0/frequency
        Z_complex = R / (1 + 1j * Q * delta)
        
        real_Z = np.real(Z_complex)
        imag_Z = np.imag(Z_complex)
        
        # Write
        xsuite_io.write_impedance_table(
            str(h5_file), frequency, real_Z, imag_Z, plane='longitudinal'
        )
        
        # Read back
        freq_read, real_read, imag_read, _ = xsuite_io.read_impedance_table(
            str(h5_file), plane='longitudinal'
        )
        
        # Verify arrays match exactly
        np.testing.assert_array_almost_equal(frequency, freq_read)
        np.testing.assert_array_almost_equal(real_Z, real_read)
        np.testing.assert_array_almost_equal(imag_Z, imag_read)
        
        # Check that impedance at resonance (where f = f0) equals R
        idx_resonance = np.where(frequency == f0)[0][0]
        assert real_read[idx_resonance] == pytest.approx(R, rel=1e-6)
        assert imag_read[idx_resonance] == pytest.approx(0.0, abs=1e-6)


class TestFileValidation:
    """Tests for file validation."""
    
    def test_validate_empty_file(self, tmp_path):
        """Test validation of empty file."""
        h5_file = tmp_path / "empty.h5"
        
        with h5py.File(h5_file, 'w') as f:
            pass
        
        validation = xsuite_io.validate_xsuite_file(str(h5_file))
        
        assert not validation['has_machine_params']
        assert not validation['has_wake_data']
        assert not validation['has_impedance_data']
        assert not validation['has_particle_data']
        assert not validation['is_valid_openpmd']
    
    def test_validate_file_with_machine_params(self, tmp_path):
        """Test validation with machine parameters."""
        h5_file = tmp_path / "test.h5"
        
        params = {'energy': 1e9}
        xsuite_io.write_machine_parameters(str(h5_file), params)
        
        validation = xsuite_io.validate_xsuite_file(str(h5_file))
        
        assert validation['has_machine_params']
    
    def test_validate_file_with_wakes(self, tmp_path):
        """Test validation with wake data."""
        h5_file = tmp_path / "test.h5"
        
        z = np.linspace(0, 1, 100)
        wake = np.zeros_like(z)
        xsuite_io.write_wake_table(str(h5_file), z, wake)
        
        validation = xsuite_io.validate_xsuite_file(str(h5_file))
        
        assert validation['has_wake_data']
    
    def test_validate_file_with_impedance(self, tmp_path):
        """Test validation with impedance data."""
        h5_file = tmp_path / "test.h5"
        
        freq = np.logspace(6, 10, 100)
        real_Z = np.ones_like(freq)
        imag_Z = np.zeros_like(freq)
        xsuite_io.write_impedance_table(str(h5_file), freq, real_Z, imag_Z)
        
        validation = xsuite_io.validate_xsuite_file(str(h5_file))
        
        assert validation['has_impedance_data']
    
    def test_validate_openpmd_file(self, tmp_path):
        """Test validation of openPMD compliant file."""
        h5_file = tmp_path / "test.h5"
        
        with h5py.File(h5_file, 'w') as f:
            f.attrs['openPMD'] = '1.1.0'
        
        validation = xsuite_io.validate_xsuite_file(str(h5_file))
        
        assert validation['is_valid_openpmd']
    
    def test_validate_complete_file(self, tmp_path):
        """Test validation of complete XSuite file."""
        h5_file = tmp_path / "complete.h5"
        
        # Add machine parameters
        params = {'energy': 1e9, 'circumference': 1000}
        xsuite_io.write_machine_parameters(str(h5_file), params)
        
        # Add wake data
        z = np.linspace(0, 1, 100)
        wake = np.zeros_like(z)
        xsuite_io.write_wake_table(str(h5_file), z, wake)
        
        # Add impedance data
        freq = np.logspace(6, 10, 100)
        real_Z = np.ones_like(freq)
        imag_Z = np.zeros_like(freq)
        xsuite_io.write_impedance_table(str(h5_file), freq, real_Z, imag_Z)
        
        # Add openPMD attribute
        with h5py.File(h5_file, 'a') as f:
            f.attrs['openPMD'] = '1.1.0'
            # Add particle data group
            f.create_group('/data/particles/')
        
        validation = xsuite_io.validate_xsuite_file(str(h5_file))
        
        assert validation['has_machine_params']
        assert validation['has_wake_data']
        assert validation['has_impedance_data']
        assert validation['has_particle_data']
        assert validation['is_valid_openpmd']


class TestComponentListing:
    """Tests for listing components."""
    
    def test_list_wake_components(self, tmp_path):
        """Test listing wake components."""
        h5_file = tmp_path / "test.h5"
        
        z = np.linspace(0, 1, 100)
        wake = np.zeros_like(z)
        
        # Add multiple components
        xsuite_io.write_wake_table(str(h5_file), z, wake, component='longitudinal')
        xsuite_io.write_wake_table(str(h5_file), z, wake, component='dipole_x')
        xsuite_io.write_wake_table(str(h5_file), z, wake, component='dipole_y')
        
        # List components
        components = xsuite_io.list_components(str(h5_file), data_type='wake')
        
        assert len(components) == 3
        assert 'longitudinal' in components
        assert 'dipole_x' in components
        assert 'dipole_y' in components
    
    def test_list_impedance_planes(self, tmp_path):
        """Test listing impedance planes."""
        h5_file = tmp_path / "test.h5"
        
        freq = np.logspace(6, 10, 100)
        real_Z = np.ones_like(freq)
        imag_Z = np.zeros_like(freq)
        
        xsuite_io.write_impedance_table(str(h5_file), freq, real_Z, imag_Z, plane='longitudinal')
        xsuite_io.write_impedance_table(str(h5_file), freq, real_Z, imag_Z, plane='x')
        
        planes = xsuite_io.list_components(str(h5_file), data_type='impedance')
        
        assert len(planes) == 2
        assert 'longitudinal' in planes
        assert 'x' in planes
    
    def test_list_components_empty_file(self, tmp_path):
        """Test listing components in empty file."""
        h5_file = tmp_path / "empty.h5"
        
        with h5py.File(h5_file, 'w') as f:
            pass
        
        components = xsuite_io.list_components(str(h5_file), data_type='wake')
        
        assert len(components) == 0
    
    def test_list_components_invalid_type(self, tmp_path):
        """Test listing with invalid data type."""
        h5_file = tmp_path / "test.h5"
        
        with h5py.File(h5_file, 'w') as f:
            pass
        
        with pytest.raises(ValueError):
            xsuite_io.list_components(str(h5_file), data_type='invalid')


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_arrays(self, tmp_path):
        """Test handling of empty arrays."""
        h5_file = tmp_path / "test.h5"
        
        z = np.array([])
        wake = np.array([])
        
        xsuite_io.write_wake_table(str(h5_file), z, wake)
        
        z_read, wake_read, _ = xsuite_io.read_wake_table(str(h5_file))
        
        assert len(z_read) == 0
        assert len(wake_read) == 0
    
    def test_single_point_data(self, tmp_path):
        """Test handling of single data point."""
        h5_file = tmp_path / "test.h5"
        
        z = np.array([0.5])
        wake = np.array([1000.0])
        
        xsuite_io.write_wake_table(str(h5_file), z, wake)
        
        z_read, wake_read, _ = xsuite_io.read_wake_table(str(h5_file))
        
        assert len(z_read) == 1
        assert z_read[0] == pytest.approx(0.5)
        assert wake_read[0] == pytest.approx(1000.0)
    
    def test_very_large_arrays(self, tmp_path):
        """Test handling of large arrays."""
        h5_file = tmp_path / "test.h5"
        
        # Create large arrays (10M points)
        n_points = 10_000_000
        z = np.linspace(0, 100, n_points)
        wake = np.random.randn(n_points)
        
        xsuite_io.write_wake_table(str(h5_file), z, wake)
        
        z_read, wake_read, _ = xsuite_io.read_wake_table(str(h5_file))
        
        assert len(z_read) == n_points
        np.testing.assert_array_almost_equal(z[:100], z_read[:100])
    
    def test_special_float_values(self, tmp_path):
        """Test handling of special float values (inf, nan)."""
        h5_file = tmp_path / "test.h5"
        
        z = np.array([0, 1, 2, 3, 4])
        wake = np.array([1.0, np.inf, -np.inf, np.nan, 0.0])
        
        xsuite_io.write_wake_table(str(h5_file), z, wake)
        
        _, wake_read, _ = xsuite_io.read_wake_table(str(h5_file))
        
        assert np.isinf(wake_read[1])
        assert np.isinf(wake_read[2])
        assert np.isnan(wake_read[3])
    
    def test_unicode_metadata(self, tmp_path):
        """Test handling of unicode in metadata."""
        h5_file = tmp_path / "test.h5"
        
        z = np.linspace(0, 1, 100)
        wake = np.zeros_like(z)
        
        metadata = {
            'source': 'Simulation à Paris',
            'author': 'José García',
            'notes': '测试数据'
        }
        
        xsuite_io.write_wake_table(str(h5_file), z, wake, metadata=metadata)
        
        _, _, read_metadata = xsuite_io.read_wake_table(str(h5_file))
        
        # HDF5 handles unicode automatically
        assert 'source' in read_metadata
    
    def test_concurrent_read_access(self, tmp_path):
        """Test concurrent read access to file."""
        h5_file = tmp_path / "test.h5"
        
        # Create file with data
        z = np.linspace(0, 1, 100)
        wake = np.zeros_like(z)
        xsuite_io.write_wake_table(str(h5_file), z, wake)
        
        # Open file multiple times for reading
        with h5py.File(h5_file, 'r') as f1:
            with h5py.File(h5_file, 'r') as f2:
                z1, wake1, _ = xsuite_io.read_wake_table(f1)
                z2, wake2, _ = xsuite_io.read_wake_table(f2)
                
                np.testing.assert_array_equal(z1, z2)
                np.testing.assert_array_equal(wake1, wake2)


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_complete_xsuite_workflow(self, tmp_path):
        """Test complete workflow: write all data types, validate, read."""
        h5_file = tmp_path / "complete_workflow.h5"
        
        # 1. Write machine parameters
        machine_params = {
            'energy': 182.5e9,
            'circumference': 97750.0,
            'harmonic_number': 132500,
            'particles_per_bunch': 1.7e11
        }
        xsuite_io.write_machine_parameters(str(h5_file), machine_params)
        
        # 2. Write wake data
        z = np.linspace(0, 1, 1000)
        wake_long = np.exp(-z/0.1) * 1e6
        wake_x = np.exp(-z/0.2) * 5e5
        
        xsuite_io.write_wake_table(str(h5_file), z, wake_long, component='longitudinal')
        xsuite_io.write_wake_table(str(h5_file), z, wake_x, component='dipole_x')
        
        # 3. Write impedance data
        freq = np.logspace(6, 12, 1000)
        real_Z = 1e3 / (1 + (freq/1e9)**2)
        imag_Z = freq * 1e-6
        
        xsuite_io.write_impedance_table(str(h5_file), freq, real_Z, imag_Z, plane='longitudinal')
        
        # 4. Add openPMD compliance
        with h5py.File(h5_file, 'a') as f:
            f.attrs['openPMD'] = '1.1.0'
            f.attrs['openPMDextension'] = 'XSuite'
        
        # 5. Validate
        validation = xsuite_io.validate_xsuite_file(str(h5_file))
        assert validation['has_machine_params']
        assert validation['has_wake_data']
        assert validation['has_impedance_data']
        assert validation['is_valid_openpmd']
        
        # 6. List components
        wake_components = xsuite_io.list_components(str(h5_file), data_type='wake')
        assert len(wake_components) == 2
        
        # 7. Read everything back
        params_read = xsuite_io.read_machine_parameters(str(h5_file))
        assert params_read['energy'] == pytest.approx(182.5e9)
        
        z_read, wake_read, _ = xsuite_io.read_wake_table(str(h5_file), 'longitudinal')
        np.testing.assert_array_almost_equal(z, z_read)
        
        freq_read, real_read, imag_read, _ = xsuite_io.read_impedance_table(str(h5_file), 'longitudinal')
        np.testing.assert_array_almost_equal(freq, freq_read)
    
    def test_file_compatibility(self, tmp_path):
        """Test that files can be read with both string paths and file objects."""
        h5_file = tmp_path / "compatibility.h5"
        
        # Write with string path
        params = {'energy': 1e9}
        xsuite_io.write_machine_parameters(str(h5_file), params)
        
        # Read with string path
        params1 = xsuite_io.read_machine_parameters(str(h5_file))
        
        # Read with file object
        with h5py.File(h5_file, 'r') as f:
            params2 = xsuite_io.read_machine_parameters(f)
        
        assert params1 == params2


# Fixtures
@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create temporary directory for tests."""
    return tmp_path_factory.mktemp("xsuite_test")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])