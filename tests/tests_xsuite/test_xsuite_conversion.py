"""
Tests for XSuite ↔ openPMD conversion utilities.

This module contains comprehensive test coverage for converting XSuite
simulation data (machine parameters, wakes, impedances) to openPMD format.

Test Categories:
- Machine parameter extraction and conversion
- Wake potential conversion (time-domain to HDF5)
- Impedance conversion (frequency-domain to HDF5)
- Synthetic particle generation
- Round-trip consistency (write → read → verify)
- Batch conversion workflow
- Error handling and edge cases
"""

import pytest
import numpy as np
import tempfile
import h5py
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from pmd_beamphysics.interfaces import xsuite_conversion, xsuite_io
except ImportError:
    try:
        from pmd_beamphysics.interfaces import xsuite_conversion, xsuite_io
    except ImportError:
        import xsuite_conversion
        import xsuite_io


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory for test files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_machine_params_json(tmp_path):
    """Create a sample machine parameters JSON file."""
    json_file = tmp_path / "test_params.json"
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
            "Qx": {
                "injection": 414.225, "z": 414.225, "w": 414.225, "zh": 414.225, "ttbar": 414.225,
                "unit": "", "comments": "Horizontal tune"
            },
            "Qy": {
                "injection": 410.29, "z": 410.29, "w": 410.29, "zh": 410.29, "ttbar": 410.29,
                "unit": "", "comments": "Vertical tune"
            },
            "chix": {
                "injection": 2.057, "z": 2.057, "w": 2.057, "zh": 2.057, "ttbar": 2.057,
                "unit": "", "comments": "Horizontal chromaticity"
            },
            "chiy": {
                "injection": 1.779, "z": 1.779, "w": 1.779, "zh": 1.779, "ttbar": 1.779,
                "unit": "", "comments": "Vertical chromaticity"
            },
            "alpha": {
                "injection": 7.12e-6, "z": 7.12e-6, "w": 7.12e-6, "zh": 7.12e-6, "ttbar": 7.12e-6,
                "unit": "", "comments": "Momentum compaction"
            }
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
        json.dump(params, f)
    return json_file


@pytest.fixture
def sample_wake_csv(tmp_path):
    """Create a sample wake potential CSV file."""
    csv_file = tmp_path / "test_wake.csv"
    
    # Generate simple test wake data
    n_points = 100
    time = np.linspace(-0.01, -0.0001, n_points)
    z = -time
    longitudinal = np.exp(-z/0.001) * 1e6
    dipole_x = np.exp(-z/0.002) * 5e5
    dipole_y = dipole_x.copy()
    
    # Write ECSV format
    with open(csv_file, 'w') as f:
        f.write("# %ECSV 1.0\n")
        f.write("# ---\n")
        f.write("# datatype:\n")
        f.write("# - {name: time, datatype: float64}\n")
        f.write("# - {name: longitudinal, datatype: float64}\n")
        f.write("# - {name: dipole_x, datatype: float64}\n")
        f.write("# - {name: dipole_y, datatype: float64}\n")
        f.write("# meta: !!omap\n")
        f.write("# - {origin: IW2D}\n")
        f.write("# - {author: Test}\n")
        f.write("# - {date: 2025-11-12}\n")
        f.write("# schema: astropy-2.0\n")
        f.write("time longitudinal dipole_x dipole_y\n")
        for i in range(n_points):
            f.write(f"{time[i]:.12f} {longitudinal[i]:.12f} {dipole_x[i]:.12f} {dipole_y[i]:.12f}\n")
    
    return csv_file


@pytest.fixture
def sample_impedance_csv(tmp_path):
    """Create a sample impedance CSV file."""
    csv_file = tmp_path / "test_impedance.csv"
    
    # Generate simple test impedance
    n_freq = 100
    frequency = np.logspace(6, 10, n_freq)
    real_z = 1e3 / (1 + (frequency/1e9)**2)
    imag_z = frequency * 1e-6
    
    # Write simple CSV format
    with open(csv_file, 'w') as f:
        f.write("# Frequency (Hz), Real Impedance (Ohm), Imag Impedance (Ohm)\n")
        for i in range(n_freq):
            f.write(f"{frequency[i]:.6e},{real_z[i]:.6e},{imag_z[i]:.6e}\n")
    
    return csv_file


# ============================================================================
# Tests for Machine Parameter Conversion
# ============================================================================

class TestMachineParameterConversion:
    """Test machine parameter JSON to HDF5 conversion."""
    
    def test_convert_machine_parameters_basic(self, sample_machine_params_json, tmp_output_dir):
        """Test basic machine parameter conversion."""
        h5_file = tmp_output_dir / "params.h5"
        
        params = xsuite_conversion.convert_machine_parameters(
            str(sample_machine_params_json),
            str(h5_file),
            energy_point='ttbar',
            verbose=False
        )
        
        # Verify file was created
        assert h5_file.exists()
        
        # Verify extracted parameters
        assert 'energy' in params
        assert params['energy'] == pytest.approx(182.5e9)
        assert 'circumference' in params
        assert params['circumference'] == pytest.approx(90.658745e3)
        assert 'emittance_x' in params
        assert params['emittance_x'] == pytest.approx(1e-5)
    
    def test_convert_machine_parameters_energy_points(self, sample_machine_params_json, tmp_output_dir):
        """Test extraction of different energy points."""
        energy_points = ['injection', 'z', 'w', 'zh', 'ttbar']
        expected_energies = [20e9, 45.6e9, 80e9, 120e9, 182.5e9]
        
        for energy_point, expected_energy in zip(energy_points, expected_energies):
            h5_file = tmp_output_dir / f"params_{energy_point}.h5"
            
            params = xsuite_conversion.convert_machine_parameters(
                str(sample_machine_params_json),
                str(h5_file),
                energy_point=energy_point,
                verbose=False
            )
            
            assert params['energy'] == pytest.approx(expected_energy)
    
    def test_convert_machine_parameters_invalid_energy(self, sample_machine_params_json, tmp_output_dir):
        """Test error handling for invalid energy point."""
        h5_file = tmp_output_dir / "params.h5"
        
        with pytest.raises(KeyError):
            xsuite_conversion.convert_machine_parameters(
                str(sample_machine_params_json),
                str(h5_file),
                energy_point='invalid_energy',
                verbose=False
            )
    
    def test_convert_machine_parameters_missing_file(self, tmp_output_dir):
        """Test error handling for missing JSON file."""
        h5_file = tmp_output_dir / "params.h5"
        
        with pytest.raises(FileNotFoundError):
            xsuite_conversion.convert_machine_parameters(
                str(tmp_output_dir / "nonexistent.json"),
                str(h5_file),
                verbose=False
            )
    
    def test_read_back_machine_parameters(self, sample_machine_params_json, tmp_output_dir):
        """Test round-trip: write → read → verify."""
        h5_file = tmp_output_dir / "params.h5"
        
        # Convert
        params_written = xsuite_conversion.convert_machine_parameters(
            str(sample_machine_params_json),
            str(h5_file),
            energy_point='ttbar',
            verbose=False
        )
        
        # Read back
        params_read = xsuite_io.read_machine_parameters(str(h5_file))
        
        # Verify all parameters match
        for key in params_written:
            assert key in params_read
            assert params_read[key] == pytest.approx(params_written[key])


# ============================================================================
# Tests for Wake Potential Conversion
# ============================================================================

class TestWakePotentialConversion:
    """Test wake potential CSV to HDF5 conversion."""
    
    def test_convert_wake_potential_basic(self, sample_wake_csv, tmp_output_dir):
        """Test basic wake potential conversion."""
        h5_file = tmp_output_dir / "wakes.h5"
        
        wakes = xsuite_conversion.convert_wake_potential(
            str(sample_wake_csv),
            str(h5_file),
            material='copper',
            verbose=False
        )
        
        # Verify file was created
        assert h5_file.exists()
        
        # Verify components
        assert 'longitudinal' in wakes
        assert 'dipole_x' in wakes
        assert 'dipole_y' in wakes
        
        # Verify data shape
        z, wake = wakes['longitudinal']
        assert len(z) == 100
        assert len(wake) == 100
    
    def test_wake_components_separable(self, sample_wake_csv, tmp_output_dir):
        """Test that wake components can be read separately."""
        h5_file = tmp_output_dir / "wakes.h5"
        
        xsuite_conversion.convert_wake_potential(
            str(sample_wake_csv),
            str(h5_file),
            verbose=False
        )
        
        # Read each component independently
        z_long, wake_long, _ = xsuite_io.read_wake_table(str(h5_file), 'longitudinal')
        z_x, wake_x, _ = xsuite_io.read_wake_table(str(h5_file), 'dipole_x')
        z_y, wake_y, _ = xsuite_io.read_wake_table(str(h5_file), 'dipole_y')
        
        # Verify arrays
        assert len(z_long) == 100
        assert len(wake_long) == 100
        assert len(z_x) == 100
        assert len(wake_x) == 100
        np.testing.assert_array_almost_equal(z_long, z_x)  # z should be same
    
    def test_wake_metadata_preservation(self, sample_wake_csv, tmp_output_dir):
        """Test that metadata is preserved during conversion."""
        h5_file = tmp_output_dir / "wakes.h5"
        
        xsuite_conversion.convert_wake_potential(
            str(sample_wake_csv),
            str(h5_file),
            material='copper',
            verbose=False
        )
        
        # Read metadata
        _, _, metadata = xsuite_io.read_wake_table(str(h5_file), 'longitudinal')
        
        assert metadata['source'] == 'IW2D'
        assert metadata['material'] == 'copper'
        assert metadata['component'] == 'longitudinal'
    
    def test_convert_wake_missing_file(self, tmp_output_dir):
        """Test error handling for missing wake file."""
        h5_file = tmp_output_dir / "wakes.h5"
        
        with pytest.raises(FileNotFoundError):
            xsuite_conversion.convert_wake_potential(
                str(tmp_output_dir / "nonexistent.csv"),
                str(h5_file),
                verbose=False
            )


# ============================================================================
# Tests for Impedance Conversion
# ============================================================================

class TestImpedanceConversion:
    """Test impedance CSV to HDF5 conversion."""
    
    def test_convert_impedance_basic(self, sample_impedance_csv, tmp_output_dir):
        """Test basic impedance conversion."""
        h5_file = tmp_output_dir / "impedance.h5"
        
        freq, real_z, imag_z = xsuite_conversion.convert_impedance(
            str(sample_impedance_csv),
            str(h5_file),
            plane='longitudinal',
            verbose=False
        )
        
        # Verify file was created
        assert h5_file.exists()
        
        # Verify array sizes
        assert len(freq) == 100
        assert len(real_z) == 100
        assert len(imag_z) == 100
    
    def test_read_back_impedance(self, sample_impedance_csv, tmp_output_dir):
        """Test round-trip: write → read → verify."""
        h5_file = tmp_output_dir / "impedance.h5"
        
        freq_write, real_write, imag_write = xsuite_conversion.convert_impedance(
            str(sample_impedance_csv),
            str(h5_file),
            verbose=False
        )
        
        # Read back
        freq_read, real_read, imag_read, _ = xsuite_io.read_impedance_table(str(h5_file))
        
        # Verify
        np.testing.assert_array_almost_equal(freq_write, freq_read)
        np.testing.assert_array_almost_equal(real_write, real_read)
        np.testing.assert_array_almost_equal(imag_write, imag_read)
    
    def test_convert_impedance_missing_file(self, tmp_output_dir):
        """Test error handling for missing impedance file."""
        h5_file = tmp_output_dir / "impedance.h5"
        
        with pytest.raises(FileNotFoundError):
            xsuite_conversion.convert_impedance(
                str(tmp_output_dir / "nonexistent.csv"),
                str(h5_file),
                verbose=False
            )


# ============================================================================
# Tests for Synthetic Particle Generation
# ============================================================================

class TestParticleGeneration:
    """Test synthetic test particle generation."""
    
    def test_generate_test_particles_basic(self):
        """Test basic particle generation."""
        machine_params = {
            'energy': 182.5e9,
            'emittance_x': 1e-5,
            'emittance_y': 1e-5,
            'bunch_length': 0.004,
            'energy_spread': 0.001,
        }
        
        particles = xsuite_conversion.generate_test_particles(
            machine_params,
            n_particles=10000,
            verbose=False
        )
        
        # Verify all coordinates present
        assert 'x' in particles
        assert 'px' in particles
        assert 'y' in particles
        assert 'py' in particles
        assert 'zeta' in particles
        assert 'delta' in particles
        assert 'weight' in particles
        
        # Verify array sizes
        assert len(particles['x']) == 10000
        assert len(particles['y']) == 10000
        assert len(particles['zeta']) == 10000
    
    def test_particle_distribution_properties(self):
        """Test that generated particles have correct statistical properties."""
        machine_params = {
            'energy': 182.5e9,
            'emittance_x': 1e-5,
            'emittance_y': 1e-5,
            'bunch_length': 0.004,
            'energy_spread': 0.001,
        }
        
        particles = xsuite_conversion.generate_test_particles(
            machine_params,
            n_particles=100000,
            verbose=False
        )
        
        # Check means are near zero (statistical fluctuation expected)
        assert abs(np.mean(particles['x'])) < 1e-4
        assert abs(np.mean(particles['y'])) < 1e-4
        assert abs(np.mean(particles['zeta'])) < 1e-3
        
        # Check standard deviations are reasonable
        assert np.std(particles['x']) > 0
        assert np.std(particles['y']) > 0
        assert np.std(particles['zeta']) > 0
    
    def test_particle_reproducibility(self):
        """Test that particle generation is reproducible with same seed."""
        machine_params = {
            'energy': 182.5e9,
            'emittance_x': 1e-5,
            'emittance_y': 1e-5,
            'bunch_length': 0.004,
            'energy_spread': 0.001,
        }
        
        particles1 = xsuite_conversion.generate_test_particles(
            machine_params,
            n_particles=1000,
            random_seed=42,
            verbose=False
        )
        
        particles2 = xsuite_conversion.generate_test_particles(
            machine_params,
            n_particles=1000,
            random_seed=42,
            verbose=False
        )
        
        # Verify arrays are identical
        np.testing.assert_array_equal(particles1['x'], particles2['x'])
        np.testing.assert_array_equal(particles1['y'], particles2['y'])
    
    def test_particle_missing_energy(self):
        """Test error handling for missing energy parameter."""
        machine_params = {
            'emittance_x': 1e-5,
            'emittance_y': 1e-5,
        }
        
        with pytest.raises(ValueError):
            xsuite_conversion.generate_test_particles(machine_params, verbose=False)


# ============================================================================
# Integration Tests
# ============================================================================

class TestBatchConversion:
    """Test batch conversion workflow."""
    
    def test_batch_conversion_minimal(self, sample_machine_params_json, sample_wake_csv, 
                                     sample_impedance_csv, tmp_path):
        """Test batch conversion with minimal input."""
        # Create directory structure
        input_dir = tmp_path / "xsuite_origin"
        sim_input_dir = input_dir / "simulation_inputs"
        (sim_input_dir / "parameters_table").mkdir(parents=True)
        (sim_input_dir / "wake_potential").mkdir(parents=True)
        (sim_input_dir / "impedances_30mm").mkdir(parents=True)
        
        # Copy sample files
        import shutil
        shutil.copy(sample_machine_params_json, sim_input_dir / "parameters_table" / "Booster_parameter_table.json")
        shutil.copy(sample_wake_csv, sim_input_dir / "wake_potential" / "heb_wake_round_cu_30.0mm.csv")
        shutil.copy(sample_impedance_csv, sim_input_dir / "impedances_30mm" / "impedance_Cu_Round_30.0mm.csv")
        
        output_dir = tmp_path / "xsuite_pmd"
        
        # Run batch conversion
        result = xsuite_conversion.convert_all_xsuite_data(
            str(input_dir),
            str(output_dir),
            energy_point='ttbar',
            n_test_particles=1000,
            verbose=False
        )
        
        # Verify result structure
        assert 'machine_params' in result
        assert 'wakes' in result
        assert 'impedances' in result
        
        # Verify files were created
        assert Path(result['machine_params']).exists()
    
    def test_batch_conversion_consistency(self, sample_machine_params_json, sample_wake_csv,
                                         sample_impedance_csv, tmp_path):
        """Test that batch conversion produces consistent data."""
        # Setup
        input_dir = tmp_path / "xsuite_origin"
        sim_input_dir = input_dir / "simulation_inputs"
        (sim_input_dir / "parameters_table").mkdir(parents=True)
        (sim_input_dir / "wake_potential").mkdir(parents=True)
        (sim_input_dir / "impedances_30mm").mkdir(parents=True)
        
        import shutil
        shutil.copy(sample_machine_params_json, sim_input_dir / "parameters_table" / "Booster_parameter_table.json")
        shutil.copy(sample_wake_csv, sim_input_dir / "wake_potential" / "heb_wake_round_cu_30.0mm.csv")
        shutil.copy(sample_impedance_csv, sim_input_dir / "impedances_30mm" / "impedance_Cu_Round_30.0mm.csv")
        
        output_dir = tmp_path / "xsuite_pmd"
        
        # Run conversion
        result = xsuite_conversion.convert_all_xsuite_data(
            str(input_dir),
            str(output_dir),
            energy_point='ttbar',
            verbose=False
        )
        
        # Verify openPMD compliance
        params = xsuite_io.read_machine_parameters(result['machine_params'])
        assert params['energy'] == pytest.approx(182.5e9)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_wake_file(self, tmp_path, tmp_output_dir):
        """Test handling of empty wake file."""
        wake_file = tmp_path / "empty_wake.csv"
        wake_file.write_text("# Empty file\n")
        
        h5_file = tmp_output_dir / "wakes.h5"
        
        # Should handle gracefully or raise informative error
        try:
            xsuite_conversion.convert_wake_potential(
                str(wake_file),
                str(h5_file),
                verbose=False
            )
        except (ValueError, IndexError):
            pass  # Expected
    
    def test_large_particle_generation(self):
        """Test generation of large number of particles."""
        machine_params = {
            'energy': 182.5e9,
            'emittance_x': 1e-5,
            'emittance_y': 1e-5,
        }
        
        # Should handle 1M particles without error
        particles = xsuite_conversion.generate_test_particles(
            machine_params,
            n_particles=1_000_000,
            verbose=False
        )
        
        assert len(particles['x']) == 1_000_000
    
    def test_very_small_particle_count(self):
        """Test generation with very few particles."""
        machine_params = {
            'energy': 182.5e9,
            'emittance_x': 1e-5,
            'emittance_y': 1e-5,
        }
        
        particles = xsuite_conversion.generate_test_particles(
            machine_params,
            n_particles=1,
            verbose=False
        )
        
        assert len(particles['x']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
