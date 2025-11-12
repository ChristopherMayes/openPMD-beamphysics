"""
XSuite input/output conversion utilities for openPMD-beamphysics.

This module provides conversion functions to transform XSuite simulation data
(machine parameters, wake potentials, impedance tables) into openPMD-compatible
HDF5 format for integration testing and interoperability.

Example
-------
>>> from pmd_beamphysics.interfaces.xsuite_conversion import convert_all_xsuite_data
>>> convert_all_xsuite_data(
...     xsuite_input_dir='./xsuite_origin/simulation_inputs/',
...     output_dir='./xsuite_pmd/',
...     energy_point='ttbar',
...     n_test_particles=100000
... )

"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

from . import xsuite_io


# ============================================================================
# Machine Parameter Conversion
# ============================================================================

def convert_machine_parameters(
    json_path: str,
    h5_output: str,
    energy_point: str = 'ttbar',
    author: str = "XSuite",
    date: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convert Booster_parameter_table.json to openPMD HDF5 format.
    
    Extracts machine parameters at a specified energy point and writes them
    to HDF5 using the standard openPMD format with xsuite_io functions.
    
    Parameters
    ----------
    json_path : str
        Path to Booster_parameter_table.json
    h5_output : str
        Output HDF5 file path
    energy_point : str, optional
        Energy point to extract: 'injection', 'z', 'w', 'zh', 'ttbar'
        Default is 'ttbar' (182.5 GeV)
    author : str
        Author name for metadata. Default is "XSuite".
    date : str, optional
        Creation date in ISO 8601 format. If None, uses input file modification time.
    verbose : bool, optional
        Print extraction progress. Default is True.
    
    Returns
    -------
    dict
        Dictionary of extracted machine parameters
        
    Raises
    ------
    FileNotFoundError
        If json_path does not exist
    KeyError
        If energy_point not found in JSON
    ValueError
        If JSON structure is unexpected
        
    Examples
    --------
    >>> params = convert_machine_parameters(
    ...     'Booster_parameter_table.json',
    ...     'machine_params.h5',
    ...     energy_point='ttbar',
    ...     author='FCC-ee Team'
    ... )
    >>> print(f"Beam energy: {params['energy'] / 1e9:.1f} GeV")
    Beam energy: 182.5 GeV
    """
    import os
    from datetime import datetime
    
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Machine parameters JSON not found: {json_path}")
    
    # Get creation date from file if not provided
    if date is None:
        mtime = os.path.getmtime(json_path)
        date = datetime.fromtimestamp(mtime).isoformat() + 'Z'
    
    if verbose:
        print(f"Loading machine parameters from {json_path.name}...")
    
    # Load JSON
    with open(json_path, 'r') as f:
        booster_params = json.load(f)
    
    # Extract energy
    try:
        energy_eV = booster_params['E'][energy_point]
    except KeyError:
        available = list(booster_params['E'].keys())
        raise KeyError(
            f"Energy point '{energy_point}' not found. "
            f"Available: {available}"
        )
    
    # Extract circumference
    circumference_m = booster_params['C']['value'] * 1e3  # km -> m
    
    # Extract bunch parameters
    bunch = booster_params['bunch']
    emittance_x = bunch['epsnx']['value']
    emittance_y = bunch['epsny']['value']
    bunch_length = bunch['sigmaz']['value']
    energy_spread = bunch['sigmae']['value']
    
    # Extract optics at this energy point
    optics = booster_params['optics']
    
    # Extract Twiss parameters (if stored as per-energy dicts)
    params_to_extract = {
        'energy': energy_eV,
        'circumference': circumference_m,
        'harmonic_number': booster_params.get('harmonic_number', None),
        'particles_per_bunch': booster_params['Np'][energy_point] 
            if isinstance(booster_params['Np'], dict) else booster_params['Np'],
        'emittance_x': emittance_x,
        'emittance_y': emittance_y,
        'bunch_length': bunch_length,
        'energy_spread': energy_spread,
        'tune_x': optics['Qx'][energy_point] 
            if isinstance(optics['Qx'], dict) else optics['Qx'],
        'tune_y': optics['Qy'][energy_point]
            if isinstance(optics['Qy'], dict) else optics['Qy'],
        'chromaticity_x': optics['chix'][energy_point]
            if isinstance(optics['chix'], dict) else optics['chix'],
        'chromaticity_y': optics['chiy'][energy_point]
            if isinstance(optics['chiy'], dict) else optics['chiy'],
        'momentum_compaction': optics['alpha'][energy_point]
            if isinstance(optics['alpha'], dict) else optics['alpha'],
    }
    
    # Add RF parameters
    rf = booster_params.get('RF', {})
    if 'RF_freq' in rf:
        rf_dict = rf['RF_freq']
        if isinstance(rf_dict, dict) and energy_point in rf_dict:
            params_to_extract['RF_frequency'] = rf_dict[energy_point]
        elif isinstance(rf_dict, dict) and 'value' in rf_dict:
            params_to_extract['RF_frequency'] = rf_dict['value']
        elif isinstance(rf_dict, (int, float)):
            params_to_extract['RF_frequency'] = rf_dict
    if 'Vtot' in rf:
        v_dict = rf['Vtot']
        if isinstance(v_dict, dict) and energy_point in v_dict:
            params_to_extract['RF_voltage'] = v_dict[energy_point]
        elif isinstance(v_dict, dict) and 'value' in v_dict:
            params_to_extract['RF_voltage'] = v_dict['value']
        elif isinstance(v_dict, (int, float)):
            params_to_extract['RF_voltage'] = v_dict
    
    # Add beam pipe info
    beam_pipe = booster_params.get('beam_pipe', {})
    if 'D' in beam_pipe:
        d_dict = beam_pipe['D']
        params_to_extract['beam_pipe_diameter'] = (
            d_dict['value'] if isinstance(d_dict, dict) else d_dict
        )
    params_to_extract['beam_pipe_material'] = beam_pipe.get('material', 'Copper')
    
    # Filter out None values (parameters not found in JSON)
    params_to_extract = {k: v for k, v in params_to_extract.items() if v is not None}
    
    # Write to HDF5
    xsuite_io.write_machine_parameters(h5_output, params_to_extract, author=author, date=date)
    
    if verbose:
        print(f"  ✓ Extracted {len(params_to_extract)} parameters")
        print(f"  ✓ Energy point: {energy_point} ({energy_eV/1e9:.1f} GeV)")
        print(f"  ✓ Author: {author}")
        print(f"  ✓ Date: {date}")
        print(f"  ✓ Wrote to {h5_output}")
    
    return params_to_extract


# ============================================================================
# Wake Potential Conversion
# ============================================================================

def convert_wake_potential(
    csv_path: str,
    h5_output: str,
    material: str = 'copper',
    author: str = "XSuite",
    date: str = None,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert wake potential CSV to openPMD HDF5 format.
    
    Reads a wake potential file in ECSV format (astropy-compatible) and
    converts it to openPMD HDF5 with separate groups for each component
    (longitudinal, dipole_x, dipole_y).
    
    Parameters
    ----------
    csv_path : str
        Path to wake potential CSV file (ECSV format)
    h5_output : str
        Output HDF5 file path
    material : str, optional
        Material identifier for metadata. Default is 'copper'.
    author : str
        Author name for metadata. Default is "XSuite".
    date : str, optional
        Creation date in ISO 8601 format. If None, uses input file modification time.
    verbose : bool, optional
        Print conversion progress. Default is True.
    
    Returns
    -------
    dict
        Dictionary with keys 'longitudinal', 'dipole_x', 'dipole_y'
        Each value is (z_array, wake_array) tuple
        
    Raises
    ------
    FileNotFoundError
        If csv_path does not exist
    ValueError
        If CSV format is not recognized
        
    Examples
    --------
    >>> wakes = convert_wake_potential(
    ...     'heb_wake_round_cu_30.0mm.csv',
    ...     'wakes.h5',
    ...     material='copper',
    ...     author='FCC-ee Team'
    ... )
    """
    import os
    from datetime import datetime
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Wake file not found: {csv_path}")
    
    # Get creation date from file if not provided
    if date is None:
        mtime = os.path.getmtime(csv_path)
        date = datetime.fromtimestamp(mtime).isoformat() + 'Z'
    
    if verbose:
        print(f"Converting wake potential from {csv_path.name}...")
    
    # Try to read with astropy (ECSV format)
    try:
        from astropy.io import ascii
        data = ascii.read(csv_path)
        time = np.array(data['time'])
        longitudinal = np.array(data['longitudinal'])
        dipole_x = np.array(data['dipole_x'])
        dipole_y = np.array(data['dipole_y'])
    except ImportError:
        warnings.warn("astropy not available, using numpy fallback for CSV parsing")
        # Fallback: read manually, skipping comments
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Find data start (skip comments and metadata)
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and not line.startswith('schema'):
                data_start = i
                break
        
        # Parse data
        data_lines = [l.split() for l in lines[data_start:] if l.strip()]
        time = np.array([float(d[0]) for d in data_lines])
        longitudinal = np.array([float(d[1]) for d in data_lines])
        dipole_x = np.array([float(d[2]) for d in data_lines])
        dipole_y = np.array([float(d[3]) for d in data_lines])
    
    # z is distance behind particle (negative time -> positive z)
    z = -time
    
    # Sort by z for physical ordering
    sort_idx = np.argsort(z)
    z = z[sort_idx]
    longitudinal = longitudinal[sort_idx]
    dipole_x = dipole_x[sort_idx]
    dipole_y = dipole_y[sort_idx]
    
    # Create metadata
    metadata = {
        'source': 'IW2D',
        'material': material,
        'pipe_diameter': '30mm',
        'gap': '0.2mm'
    }
    
    # Write each component
    components = {
        'longitudinal': (z, longitudinal),
        'dipole_x': (z, dipole_x),
        'dipole_y': (z, dipole_y)
    }
    
    for component, (z_data, wake_data) in components.items():
        xsuite_io.write_wake_table(
            h5_output,
            z_data,
            wake_data,
            component=component,
            metadata=metadata,
            author=author,
            date=date
        )
    
    if verbose:
        print(f"  ✓ Converted {len(z)} wake points")
        print(f"  ✓ Components: {list(components.keys())}")
        print(f"  ✓ Z range: {z.min():.6f} to {z.max():.6f} m")
        print(f"  ✓ Author: {author}")
        print(f"  ✓ Date: {date}")
        print(f"  ✓ Wrote to {h5_output}")
    
    return components


# ============================================================================
# Impedance Conversion
# ============================================================================

def convert_impedance(
    csv_path: str,
    h5_output: str,
    plane: str = 'longitudinal',
    author: str = "XSuite",
    date: str = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert impedance CSV to openPMD HDF5 format.
    
    Reads frequency-domain impedance data and writes it to HDF5 in
    openPMD format with real and imaginary parts.
    
    Parameters
    ----------
    csv_path : str
        Path to impedance CSV file
    h5_output : str
        Output HDF5 file path
    plane : str, optional
        Impedance plane: 'longitudinal', 'x', 'y'. Default is 'longitudinal'.
    author : str
        Author name for metadata. Default is "XSuite".
    date : str, optional
        Creation date in ISO 8601 format. If None, uses input file modification time.
    verbose : bool, optional
        Print conversion progress. Default is True.
    
    Returns
    -------
    tuple
        (frequency, real_impedance, imag_impedance) arrays
        
    Examples
    --------
    >>> freq, real_z, imag_z = convert_impedance(
    ...     'impedance_Cu_Round_30.0mm.csv',
    ...     'impedance.h5',
    ...     plane='longitudinal',
    ...     author='FCC-ee Team'
    ... )
    """
    import os
    from datetime import datetime
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Impedance file not found: {csv_path}")
    
    # Get creation date from file if not provided
    if date is None:
        mtime = os.path.getmtime(csv_path)
        date = datetime.fromtimestamp(mtime).isoformat() + 'Z'
    
    if verbose:
        print(f"Converting impedance from {csv_path.name}...")
    
    # Try astropy first
    try:
        from astropy.io import ascii
        data = ascii.read(csv_path)
        if hasattr(data, 'colnames') and len(data.colnames) >= 3:
            frequency = np.array(data[data.colnames[0]])
            real_imp = np.array(data[data.colnames[1]])
            imag_imp = np.array(data[data.colnames[2]])
        else:
            raise ValueError("Unexpected CSV structure")
    except (ImportError, KeyError, IndexError, ValueError):
        warnings.warn("astropy parsing failed, using numpy fallback")
        # Manual fallback - skip header lines starting with #
        data = np.genfromtxt(csv_path, skip_header=1, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        frequency = data[:, 0]
        real_imp = data[:, 1]
        imag_imp = data[:, 2]
    
    # Write to HDF5
    xsuite_io.write_impedance_table(
        h5_output,
        frequency,
        real_imp,
        imag_imp,
        plane=plane,
        metadata={'source': 'IW2D/2PHDT simulation'},
        author=author,
        date=date
    )
    
    if verbose:
        print(f"  ✓ Converted {len(frequency)} frequency points")
        print(f"  ✓ Plane: {plane}")
        print(f"  ✓ Frequency range: {frequency.min():.2e} to {frequency.max():.2e} Hz")
        print(f"  ✓ Author: {author}")
        print(f"  ✓ Date: {date}")
        print(f"  ✓ Wrote to {h5_output}")
    
    return frequency, real_imp, imag_imp


# ============================================================================
# Synthetic Test Particle Generation
# ============================================================================

def generate_test_particles(
    machine_params: Dict[str, Any],
    n_particles: int = 100000,
    output_h5: Optional[str] = None,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic Gaussian bunch for testing.
    
    Creates a test particle distribution matching the machine parameters
    with Gaussian profile in all 6D phase space dimensions.
    
    Parameters
    ----------
    machine_params : dict
        Machine parameters dict (e.g., from convert_machine_parameters)
        Must contain: energy, emittance_x, emittance_y, bunch_length
    n_particles : int, optional
        Number of macroparticles. Default is 100000.
    output_h5 : str, optional
        If provided, save particles to this HDF5 file in openPMD format
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.
    verbose : bool, optional
        Print generation progress. Default is True.
    
    Returns
    -------
    dict
        Dictionary with keys: x, px, y, py, zeta, delta, weight
        Each value is an array of shape (n_particles,)
        
    Examples
    --------
    >>> params = convert_machine_parameters('params.json', 'params.h5')
    >>> particles = generate_test_particles(params, n_particles=1e6)
    >>> print(f"Generated {len(particles['x'])} particles")
    Generated 1000000 particles
    """
    np.random.seed(random_seed)
    
    if verbose:
        print(f"Generating {n_particles} test particles...")
    
    # Extract parameters
    energy_eV = machine_params.get('energy')
    if energy_eV is None:
        raise ValueError("machine_params must contain 'energy'")
    
    emittance_x = machine_params.get('emittance_x', 1e-5)  # Default to 10 μm
    emittance_y = machine_params.get('emittance_y', 1e-5)
    bunch_length = machine_params.get('bunch_length', 0.004)  # 4 mm default
    energy_spread = machine_params.get('energy_spread', 0.001)  # 0.1% default
    
    # Reference momentum
    m_e = 9.109e-31  # kg
    c = 2.998e8      # m/s
    E_rest = 0.511e6  # eV (electron rest mass)
    
    E_kinetic = energy_eV - E_rest  # Subtract rest mass
    p_ref = np.sqrt(E_kinetic**2 + 2*E_rest*E_kinetic) / c  # kg·m/s (relativistic)
    gamma = energy_eV / E_rest
    beta = np.sqrt(1 - 1/gamma**2)
    
    # Twiss parameters (assume matched for simplicity)
    beta_x = machine_params.get('beta_x', 100)  # m, typical value
    beta_y = machine_params.get('beta_y', 100)  # m, typical value
    alpha_x = 0.0  # Assume zero for matched beam
    alpha_y = 0.0
    
    # Generate phase space coordinates
    # x, px coordinates (transverse horizontal)
    phi_x = np.random.uniform(0, 2*np.pi, n_particles)
    r_x = np.sqrt(np.random.uniform(0, 1, n_particles))  # sqrt for uniform in area
    
    x = r_x * np.sqrt(2 * emittance_x * beta_x) * np.cos(phi_x)
    px = r_x * np.sqrt(2 * emittance_x / beta_x) * (
        np.sin(phi_x) - alpha_x * np.cos(phi_x)
    )
    
    # y, py coordinates (transverse vertical)
    phi_y = np.random.uniform(0, 2*np.pi, n_particles)
    r_y = np.sqrt(np.random.uniform(0, 1, n_particles))
    
    y = r_y * np.sqrt(2 * emittance_y * beta_y) * np.cos(phi_y)
    py = r_y * np.sqrt(2 * emittance_y / beta_y) * (
        np.sin(phi_y) - alpha_y * np.cos(phi_y)
    )
    
    # Longitudinal coordinates
    zeta = np.random.normal(0, bunch_length/4, n_particles)  # σ = L/4
    delta = np.random.normal(0, energy_spread, n_particles)  # ΔE/E
    
    # Macroparticle weight (equal weighting)
    weight = np.ones(n_particles)
    
    particles = {
        'x': x,
        'px': px,
        'y': y,
        'py': py,
        'zeta': zeta,
        'delta': delta,
        'weight': weight,
    }
    
    if verbose:
        print(f"  ✓ Generated particles:")
        print(f"    - x: {x.std():.3e} ± {x.mean():.3e} m")
        print(f"    - y: {y.std():.3e} ± {y.mean():.3e} m")
        print(f"    - zeta: {zeta.std():.3e} ± {zeta.mean():.3e} m")
        print(f"    - delta: {delta.std():.3e} ± {delta.mean():.3e}")
    
    # Save to HDF5 if requested
    if output_h5:
        try:
            from pmd_beamphysics import ParticleGroup
            import h5py as h5_module
            
            # Create HDF5 file with openPMD structure manually
            # (ParticleGroup constructor may have different API)
            with h5_module.File(output_h5, 'w') as f:
                # Add openPMD required attributes
                f.attrs['openPMD'] = '1.1.0'
                f.attrs['openPMDextension'] = 'XSuite'
                
                # Create particles group
                particles_grp = f.create_group('/data/particles/')
                
                # Store particle data as datasets
                for key, value in particles.items():
                    particles_grp.create_dataset(key, data=value)
                
                # Add machine parameters as metadata
                for key, value in machine_params.items():
                    try:
                        particles_grp.attrs[f'machine_{key}'] = value
                    except (TypeError, ValueError):
                        pass  # Skip non-serializable attributes
            
            if verbose:
                print(f"  ✓ Saved to {output_h5}")
        except Exception as e:
            warnings.warn(f"Failed to save HDF5: {e}")
            if verbose:
                print(f"  ✗ Failed to save to {output_h5}: {e}")
    
    return particles


# ============================================================================
# Batch Conversion (All-in-One)
# ============================================================================

def convert_all_xsuite_data(
    xsuite_input_dir: str,
    output_dir: str,
    energy_point: str = 'ttbar',
    materials: Optional[List[str]] = None,
    n_test_particles: int = 100000,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Convert all XSuite data to openPMD format in one call.
    
    Processes all files in xsuite_input_dir and creates corresponding
    openPMD HDF5 files in output_dir.
    
    Parameters
    ----------
    xsuite_input_dir : str
        Input directory containing simulation_inputs/ subdirectory
    output_dir : str
        Output directory for HDF5 files
    energy_point : str, optional
        Energy point for machine parameters. Default is 'ttbar'.
    materials : list, optional
        Materials to process (e.g., ['copper', 'stainless_steel']).
        If None, auto-detect from file names.
    n_test_particles : int, optional
        Number of synthetic test particles. Default is 100000.
    verbose : bool, optional
        Print progress. Default is True.
    
    Returns
    -------
    dict
        Dictionary mapping data type to output file paths:
        {
            'machine_params': 'machine_parameters.h5',
            'wakes': {'copper': 'wakes_copper.h5', ...},
            'impedances': {'copper': 'impedances_copper.h5', ...},
            'test_particles': 'test_particles_gaussian.h5'
        }
        
    Examples
    --------
    >>> files = convert_all_xsuite_data(
    ...     xsuite_input_dir='./xsuite_origin/',
    ...     output_dir='./xsuite_pmd/',
    ...     energy_point='ttbar'
    ... )
    >>> print(files['machine_params'])
    ./xsuite_pmd/machine_parameters.h5
    """
    input_dir = Path(xsuite_input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'wakes': {},
        'impedances': {},
    }
    
    if verbose:
        print("=" * 70)
        print("XSuite → openPMD Batch Conversion")
        print("=" * 70)
    
    # 1. Convert machine parameters
    if verbose:
        print("\n[1/3] Converting machine parameters...")
    params_json = input_dir / 'simulation_inputs' / 'parameters_table' / 'Booster_parameter_table.json'
    if params_json.exists():
        params_h5 = output_dir / 'machine_parameters.h5'
        params = convert_machine_parameters(str(params_json), str(params_h5), energy_point)
        result['machine_params'] = str(params_h5)
    else:
        warnings.warn(f"Machine parameters JSON not found: {params_json}")
        params = {}
    
    # 2. Convert wake potentials
    if verbose:
        print("\n[2/3] Converting wake potentials...")
    wake_dir = input_dir / 'simulation_inputs' / 'wake_potential'
    if wake_dir.exists():
        wake_files = sorted(wake_dir.glob('*.csv'))
        for wake_file in wake_files:
            filename_stem = wake_file.stem.lower()
            
            # Determine material
            if 'cu' in filename_stem or 'copper' in filename_stem:
                material = 'copper'
            elif 'ss' in filename_stem or 'stainless' in filename_stem:
                material = 'stainless_steel'
            else:
                material = 'unknown'
            
            # Skip if materials filter applied
            if materials and material not in materials:
                continue
            
            try:
                wake_h5 = output_dir / f'wakes_{material}_{wake_file.stem.split("_")[-1]}.h5'
                convert_wake_potential(str(wake_file), str(wake_h5), material)
                result['wakes'][material] = str(wake_h5)
            except Exception as e:
                warnings.warn(f"Failed to convert {wake_file}: {e}")
    else:
        warnings.warn(f"Wake directory not found: {wake_dir}")
    
    # 3. Convert impedances
    if verbose:
        print("\n[3/3] Converting impedance tables...")
    imp_dir = input_dir / 'simulation_inputs' / 'impedances_30mm'
    if imp_dir.exists():
        imp_files = sorted(imp_dir.glob('impedance_*.csv'))
        for imp_file in imp_files:
            filename_stem = imp_file.stem.lower()
            
            # Determine material
            if 'cu' in filename_stem or 'copper' in filename_stem:
                material = 'copper'
            elif 'ss' in filename_stem or 'stainless' in filename_stem:
                material = 'stainless_steel'
            else:
                material = 'unknown'
            
            if materials and material not in materials:
                continue
            
            try:
                imp_h5 = output_dir / f'impedance_{material}_30mm.h5'
                convert_impedance(str(imp_file), str(imp_h5), plane='longitudinal')
                result['impedances'][material] = str(imp_h5)
            except Exception as e:
                warnings.warn(f"Failed to convert {imp_file}: {e}")
    else:
        warnings.warn(f"Impedance directory not found: {imp_dir}")
    
    # 4. Generate synthetic test particles
    if verbose:
        print("\n[4/4] Generating synthetic test particles...")
    if params:
        test_particles_h5 = output_dir / 'test_particles_gaussian.h5'
        try:
            generate_test_particles(
                params,
                n_particles=n_test_particles,
                output_h5=str(test_particles_h5),
                verbose=verbose
            )
            result['test_particles'] = str(test_particles_h5)
        except Exception as e:
            warnings.warn(f"Failed to generate test particles: {e}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("Conversion complete!")
        print("=" * 70)
    
    return result
