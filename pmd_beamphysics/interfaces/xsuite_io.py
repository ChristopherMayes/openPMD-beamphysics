"""
XSuite I/O utilities for openPMD-beamphysics.

This module provides functions to read and write XSuite data in openPMD format.
"""

import h5py
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from pathlib import Path


class XSuiteIOError(Exception):
    """Custom exception for XSuite I/O errors."""
    pass


def write_machine_parameters(
    h5_file: Union[str, h5py.File],
    params: Dict[str, Any],
    base_path: str = "/machineParameters/",
    author: str = "XSuite",
    date: Optional[str] = None
) -> None:
    """
    Write machine parameters to HDF5 file in openPMD format.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    params : dict
        Dictionary of machine parameters
    base_path : str
        Base path in HDF5 file for machine parameters
    author : str
        Author name for metadata
    date : str, optional
        Creation date in ISO 8601 format. If None, uses current date.
        
    Examples
    --------
    >>> params = {
    ...     'energy': 182.5e9,  # eV
    ...     'circumference': 97750.0,  # m
    ...     'harmonic_number': 132500,
    ...     'particles_per_bunch': 1.7e11
    ... }
    >>> write_machine_parameters('output.h5', params, author='MyTeam')
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'a')
        should_close = True
    
    try:
        # Add openPMD compliance attributes to root if not present
        if 'openPMD' not in h5_file.attrs:
            h5_file.attrs['openPMD'] = '1.1.0'
        if 'openPMDextension' not in h5_file.attrs:
            h5_file.attrs['openPMDextension'] = 'beamPhysics'
        if 'basePath' not in h5_file.attrs:
            h5_file.attrs['basePath'] = '/xsuite/'
        if 'meshesPath' not in h5_file.attrs:
            h5_file.attrs['meshesPath'] = 'simulationData/'
        if 'particlesPath' not in h5_file.attrs:
            h5_file.attrs['particlesPath'] = 'particleData/'
        if 'author' not in h5_file.attrs:
            h5_file.attrs['author'] = author
        if 'software' not in h5_file.attrs:
            h5_file.attrs['software'] = 'xsuite_io'
        if 'softwareVersion' not in h5_file.attrs:
            h5_file.attrs['softwareVersion'] = '1.0'
        if 'date' not in h5_file.attrs:
            h5_file.attrs['date'] = date or datetime.now().isoformat() + 'Z'
        if 'comment' not in h5_file.attrs:
            h5_file.attrs['comment'] = 'FCC-ee booster machine parameters in openPMD format'
        
        # Create group if it doesn't exist
        if base_path not in h5_file:
            grp = h5_file.create_group(base_path)
        else:
            grp = h5_file[base_path]
        
        # Write each parameter as attribute
        for key, value in params.items():
            grp.attrs[key] = value
            
    finally:
        if should_close:
            h5_file.close()


def read_machine_parameters(
    h5_file: Union[str, h5py.File],
    base_path: str = "/machineParameters/"
) -> Dict[str, Any]:
    """
    Read machine parameters from HDF5 file.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    base_path : str
        Base path in HDF5 file for machine parameters
        
    Returns
    -------
    dict
        Dictionary of machine parameters
        
    Examples
    --------
    >>> params = read_machine_parameters('output.h5')
    >>> print(params['energy'])
    182500000000.0
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'r')
        should_close = True
    
    try:
        if base_path not in h5_file:
            raise XSuiteIOError(f"Path {base_path} not found in HDF5 file")
        
        grp = h5_file[base_path]
        params = dict(grp.attrs)
        
        return params
        
    finally:
        if should_close:
            h5_file.close()


def write_wake_table(
    h5_file: Union[str, h5py.File],
    z: np.ndarray,
    wake: np.ndarray,
    component: str = "longitudinal",
    base_path: str = "/wakeData/",
    metadata: Optional[Dict[str, Any]] = None,
    author: str = "XSuite",
    date: Optional[str] = None
) -> None:
    """
    Write wake potential table to HDF5 file.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    z : np.ndarray
        Distance behind particle (m)
    wake : np.ndarray
        Wake potential values (V/C for longitudinal, V/C/m for transverse)
    component : str
        Wake component: 'longitudinal', 'dipole_x', 'dipole_y', 'quadrupole'
    base_path : str
        Base path in HDF5 file for wake data
    metadata : dict, optional
        Additional metadata (source, date, etc.)
    author : str
        Author name for metadata
    date : str, optional
        Creation date in ISO 8601 format. If None, uses current date.
        
    Examples
    --------
    >>> z = np.linspace(0, 1, 1000)  # m
    >>> wake = np.exp(-z/0.1) * 1e6  # V/C
    >>> write_wake_table('wakes.h5', z, wake, component='longitudinal', author='MyTeam')
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'a')
        should_close = True
    
    try:
        # Add openPMD compliance attributes to root if not present
        if 'openPMD' not in h5_file.attrs:
            h5_file.attrs['openPMD'] = '1.1.0'
        if 'openPMDextension' not in h5_file.attrs:
            h5_file.attrs['openPMDextension'] = 'beamPhysics'
        if 'basePath' not in h5_file.attrs:
            h5_file.attrs['basePath'] = '/xsuite/'
        if 'meshesPath' not in h5_file.attrs:
            h5_file.attrs['meshesPath'] = 'simulationData/'
        if 'particlesPath' not in h5_file.attrs:
            h5_file.attrs['particlesPath'] = 'particleData/'
        if 'author' not in h5_file.attrs:
            h5_file.attrs['author'] = author
        if 'software' not in h5_file.attrs:
            h5_file.attrs['software'] = 'xsuite_io'
        if 'softwareVersion' not in h5_file.attrs:
            h5_file.attrs['softwareVersion'] = '1.0'
        if 'date' not in h5_file.attrs:
            h5_file.attrs['date'] = date or datetime.now().isoformat() + 'Z'
        if 'comment' not in h5_file.attrs:
            h5_file.attrs['comment'] = 'FCC-ee booster wake potential functions in openPMD format'
        
        # Create wake data group
        wake_path = f"{base_path}{component}/"
        if wake_path in h5_file:
            del h5_file[wake_path]
        
        grp = h5_file.create_group(wake_path)
        
        # Write data
        grp.create_dataset('z', data=z)
        grp.create_dataset('wake', data=wake)
        
        # Add metadata
        grp.attrs['component'] = component
        grp.attrs['z_unit'] = 'm'
        
        if component == 'longitudinal':
            grp.attrs['wake_unit'] = 'V/C'
        else:
            grp.attrs['wake_unit'] = 'V/C/m'
        
        if metadata:
            for key, value in metadata.items():
                grp.attrs[key] = value
                
    finally:
        if should_close:
            h5_file.close()


def read_wake_table(
    h5_file: Union[str, h5py.File],
    component: str = "longitudinal",
    base_path: str = "/wakeData/"
) -> tuple:
    """
    Read wake potential table from HDF5 file.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    component : str
        Wake component to read
    base_path : str
        Base path in HDF5 file for wake data
        
    Returns
    -------
    z : np.ndarray
        Distance behind particle (m)
    wake : np.ndarray
        Wake potential values
    metadata : dict
        Metadata attributes
        
    Examples
    --------
    >>> z, wake, metadata = read_wake_table('wakes.h5', component='longitudinal')
    >>> print(f"Wake length: {z.max():.3f} m")
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'r')
        should_close = True
    
    try:
        wake_path = f"{base_path}{component}/"
        if wake_path not in h5_file:
            raise XSuiteIOError(f"Wake component {component} not found at {wake_path}")
        
        grp = h5_file[wake_path]
        
        z = grp['z'][:]
        wake = grp['wake'][:]
        metadata = dict(grp.attrs)
        
        return z, wake, metadata
        
    finally:
        if should_close:
            h5_file.close()


def write_impedance_table(
    h5_file: Union[str, h5py.File],
    frequency: np.ndarray,
    real_impedance: np.ndarray,
    imag_impedance: np.ndarray,
    plane: str = "longitudinal",
    base_path: str = "/impedanceData/",
    metadata: Optional[Dict[str, Any]] = None,
    author: str = "XSuite",
    date: Optional[str] = None
) -> None:
    """
    Write impedance table to HDF5 file.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    frequency : np.ndarray
        Frequency array (Hz)
    real_impedance : np.ndarray
        Real part of impedance (Ohm)
    imag_impedance : np.ndarray
        Imaginary part of impedance (Ohm)
    plane : str
        Impedance plane: 'longitudinal', 'x', 'y'
    base_path : str
        Base path in HDF5 file
    metadata : dict, optional
        Additional metadata
    author : str
        Author name for metadata
    date : str, optional
        Creation date in ISO 8601 format. If None, uses current date.
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'a')
        should_close = True
    
    try:
        # Add openPMD compliance attributes to root if not present
        if 'openPMD' not in h5_file.attrs:
            h5_file.attrs['openPMD'] = '1.1.0'
        if 'openPMDextension' not in h5_file.attrs:
            h5_file.attrs['openPMDextension'] = 'beamPhysics'
        if 'basePath' not in h5_file.attrs:
            h5_file.attrs['basePath'] = '/xsuite/'
        if 'meshesPath' not in h5_file.attrs:
            h5_file.attrs['meshesPath'] = 'simulationData/'
        if 'particlesPath' not in h5_file.attrs:
            h5_file.attrs['particlesPath'] = 'particleData/'
        if 'author' not in h5_file.attrs:
            h5_file.attrs['author'] = author
        if 'software' not in h5_file.attrs:
            h5_file.attrs['software'] = 'xsuite_io'
        if 'softwareVersion' not in h5_file.attrs:
            h5_file.attrs['softwareVersion'] = '1.0'
        if 'date' not in h5_file.attrs:
            h5_file.attrs['date'] = date or datetime.now().isoformat() + 'Z'
        if 'comment' not in h5_file.attrs:
            h5_file.attrs['comment'] = 'FCC-ee booster longitudinal impedance in openPMD format'
        
        imp_path = f"{base_path}{plane}/"
        if imp_path in h5_file:
            del h5_file[imp_path]
        
        grp = h5_file.create_group(imp_path)
        
        # Write data
        grp.create_dataset('frequency', data=frequency)
        grp.create_dataset('real', data=real_impedance)
        grp.create_dataset('imag', data=imag_impedance)
        
        # Add metadata
        grp.attrs['plane'] = plane
        grp.attrs['frequency_unit'] = 'Hz'
        grp.attrs['impedance_unit'] = 'Ohm'
        
        if metadata:
            for key, value in metadata.items():
                grp.attrs[key] = value
                
    finally:
        if should_close:
            h5_file.close()


def read_impedance_table(
    h5_file: Union[str, h5py.File],
    plane: str = "longitudinal",
    base_path: str = "/impedanceData/"
) -> tuple:
    """
    Read impedance table from HDF5 file.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    plane : str
        Impedance plane to read
    base_path : str
        Base path in HDF5 file
        
    Returns
    -------
    frequency : np.ndarray
        Frequency array (Hz)
    real_impedance : np.ndarray
        Real part of impedance
    imag_impedance : np.ndarray
        Imaginary part of impedance
    metadata : dict
        Metadata attributes
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'r')
        should_close = True
    
    try:
        imp_path = f"{base_path}{plane}/"
        if imp_path not in h5_file:
            raise XSuiteIOError(f"Impedance plane {plane} not found at {imp_path}")
        
        grp = h5_file[imp_path]
        
        frequency = grp['frequency'][:]
        real_impedance = grp['real'][:]
        imag_impedance = grp['imag'][:]
        metadata = dict(grp.attrs)
        
        return frequency, real_impedance, imag_impedance, metadata
        
    finally:
        if should_close:
            h5_file.close()


def validate_xsuite_file(h5_file: Union[str, h5py.File]) -> Dict[str, bool]:
    """
    Validate XSuite openPMD file structure.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
        
    Returns
    -------
    dict
        Validation results with boolean flags
        
    Examples
    --------
    >>> validation = validate_xsuite_file('output.h5')
    >>> if validation['has_machine_params']:
    ...     print("Machine parameters found")
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'r')
        should_close = True
    
    try:
        validation = {
            'has_machine_params': '/machineParameters/' in h5_file,
            'has_wake_data': '/wakeData/' in h5_file,
            'has_impedance_data': '/impedanceData/' in h5_file,
            'has_particle_data': False,
            'is_valid_openpmd': False
        }
        
        # Check for particle data groups
        for key in h5_file.keys():
            if 'data' in key.lower():
                validation['has_particle_data'] = True
                break
        
        # Check for openPMD required attributes
        if 'openPMD' in h5_file.attrs:
            validation['is_valid_openpmd'] = True
        
        return validation
        
    finally:
        if should_close:
            h5_file.close()


def list_components(
    h5_file: Union[str, h5py.File],
    data_type: str = "wake"
) -> List[str]:
    """
    List available wake or impedance components in file.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    data_type : str
        Type of data: 'wake' or 'impedance'
        
    Returns
    -------
    list
        List of available components/planes
        
    Examples
    --------
    >>> components = list_components('wakes.h5', data_type='wake')
    >>> print(components)
    ['longitudinal', 'dipole_x', 'dipole_y']
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'r')
        should_close = True
    
    try:
        if data_type == "wake":
            base_path = "/wakeData/"
        elif data_type == "impedance":
            base_path = "/impedanceData/"
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        if base_path not in h5_file:
            return []
        
        grp = h5_file[base_path]
        components = list(grp.keys())
        
        return components
        
    finally:
        if should_close:
            h5_file.close()


def compute_file_checksum(h5_file: Union[str, h5py.File], algorithm: str = "md5") -> str:
    """
    Compute checksum of HDF5 file for data integrity verification.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path
    algorithm : str
        Hash algorithm: 'md5', 'sha1', 'sha256'
        
    Returns
    -------
    str
        Hexadecimal checksum string
        
    Examples
    --------
    >>> checksum = compute_file_checksum('output.h5', algorithm='sha256')
    >>> print(f"File checksum: {checksum}")
    """
    import hashlib
    
    if not isinstance(h5_file, str):
        raise ValueError("compute_file_checksum requires a file path string")
    
    # Select hash algorithm
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Read file in chunks and compute hash
    with open(h5_file, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    
    return hasher.hexdigest()


# ============================================================================
# Function Aliases for Compatibility with __init__.py
# ============================================================================

# Machine parameters
load_machine_parameters = read_machine_parameters

# Wake potentials  
load_wake_potential = read_wake_table

# Impedance
load_impedance = read_impedance_table

# Validation (alias)
validate_openpmd_file = validate_xsuite_file


# Particle I/O functions (stubs - to be implemented in future)
def load_particles(h5_file: Union[str, h5py.File], iteration: int = 0, **kwargs):
    """
    Load particle data from HDF5 file.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    iteration : int
        Iteration number to load
    **kwargs
        Additional parameters
        
    Returns
    -------
    particles : dict
        Particle data dictionary
        
    Note
    ----
    This is a placeholder. Full implementation will be added in future updates.
    For now, use the main ParticleGroup class from pmd_beamphysics.
    """
    raise NotImplementedError(
        "load_particles is not yet implemented for XSuite I/O module. "
        "This function will be added in future updates. "
        "For now, please use the standard ParticleGroup interface."
    )


def save_particles(
    h5_file: Union[str, h5py.File],
    particles,
    iteration: int = 0,
    **kwargs
):
    """
    Save particle data to HDF5 file in XSuite format.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    particles : dict or ParticleGroup
        Particle data to save
    iteration : int
        Iteration number
    **kwargs
        Additional parameters
        
    Note
    ----
    This is a placeholder. Full implementation will be added in future updates.
    For now, use the main ParticleGroup class from pmd_beamphysics.
    """
    raise NotImplementedError(
        "save_particles is not yet implemented for XSuite I/O module. "
        "This function will be added in future updates. "
        "For now, please use the standard ParticleGroup interface."
    )


# ============================================================================
# Optics/Lattice I/O
# ============================================================================

def write_optics_data(
    h5_file: Union[str, h5py.File],
    optics_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    base_path: str = "/opticsData/",
    author: str = "XSuite",
    date: Optional[str] = None
) -> None:
    """
    Write optics/lattice data to HDF5 file in openPMD format.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    optics_data : dict
        Complete lattice definition (XSuite Line object as dict)
    metadata : dict, optional
        Lattice metadata (element counts, lengths, etc.)
    base_path : str
        Base path in HDF5 file for optics data
    author : str
        Author name for metadata
    date : str, optional
        Creation date in ISO 8601 format. If None, uses current date.
    """
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'w')
        should_close = True
    
    try:
        # Add openPMD compliance attributes to root if not present
        if 'openPMD' not in h5_file.attrs:
            h5_file.attrs['openPMD'] = '1.1.0'
        if 'openPMDextension' not in h5_file.attrs:
            h5_file.attrs['openPMDextension'] = 'beamPhysics'
        if 'basePath' not in h5_file.attrs:
            h5_file.attrs['basePath'] = '/xsuite/'
        if 'meshesPath' not in h5_file.attrs:
            h5_file.attrs['meshesPath'] = 'simulationData/'
        if 'particlesPath' not in h5_file.attrs:
            h5_file.attrs['particlesPath'] = 'particleData/'
        if 'author' not in h5_file.attrs:
            h5_file.attrs['author'] = author
        if 'software' not in h5_file.attrs:
            h5_file.attrs['software'] = 'xsuite_io'
        if 'softwareVersion' not in h5_file.attrs:
            h5_file.attrs['softwareVersion'] = '1.0'
        if 'date' not in h5_file.attrs:
            h5_file.attrs['date'] = date or datetime.now().isoformat() + 'Z'
        if 'comment' not in h5_file.attrs:
            h5_file.attrs['comment'] = 'FCC-ee booster optics/lattice definition in openPMD format'
        
        # Create optics data group
        if base_path not in h5_file:
            grp = h5_file.create_group(base_path)
        else:
            grp = h5_file[base_path]
        
        # Store lattice metadata
        if metadata:
            for key, value in metadata.items():
                try:
                    grp.attrs[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable attributes
                    pass
        
        # Store complete optics data as JSON string (for complex nested structures)
        import json as json_module
        optics_json = json_module.dumps(optics_data, indent=2)
        grp.attrs['lattice_json_length'] = len(optics_json)
        
        # Store JSON in compressed text dataset (HDF5 strings have limitations)
        json_dataset = grp.create_dataset(
            'lattice_definition',
            data=np.array(optics_json, dtype=h5py.string_dtype(encoding='utf-8'))
        )
        json_dataset.attrs['format'] = 'JSON'
        json_dataset.attrs['description'] = 'Complete XSuite lattice definition'
                
    finally:
        if should_close:
            h5_file.close()


def read_optics_data(
    h5_file: Union[str, h5py.File],
    base_path: str = "/opticsData/"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Read optics/lattice data from HDF5 file.
    
    Parameters
    ----------
    h5_file : str or h5py.File
        HDF5 file path or open file object
    base_path : str
        Base path in HDF5 file for optics data
        
    Returns
    -------
    optics_data : dict
        Lattice definition
    metadata : dict
        Optics metadata (element counts, lengths, etc.)
    """
    import json as json_module
    
    should_close = False
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'r')
        should_close = True
    
    try:
        if base_path not in h5_file:
            raise XSuiteIOError(f"Path {base_path} not found in HDF5 file")
        
        grp = h5_file[base_path]
        
        # Read metadata attributes
        metadata = dict(grp.attrs)
        
        # Read lattice definition
        if 'lattice_definition' in grp:
            json_str = grp['lattice_definition'][()]
            if isinstance(json_str, bytes):
                json_str = json_str.decode('utf-8')
            optics_data = json_module.loads(json_str)
        else:
            optics_data = {}
        
        return optics_data, metadata
        
    finally:
        if should_close:
            h5_file.close()


# ============================================================================
# Function Aliases for Compatibility
# ============================================================================
# These aliases match the expected names in pmd_beamphysics/interfaces/__init__.py

# Machine parameters
load_machine_parameters = read_machine_parameters
save_machine_parameters = write_machine_parameters

# Wake potentials
load_wake_potential = read_wake_table
save_wake_potential = write_wake_table

# Impedance
load_impedance = read_impedance_table
save_impedance = write_impedance_table

# Particle functions (stubs for now - to be implemented)
def load_particles(h5_file: Union[str, h5py.File], **kwargs):
    """
    Load particle data from HDF5 file.
    
    Note: This is a placeholder. Full implementation pending.
    """
    raise NotImplementedError("load_particles is not yet implemented for XSuite I/O. "
                            "This function will be added in future updates.")

def save_particles(h5_file: Union[str, h5py.File], particles, **kwargs):
    """
    Save particle data to HDF5 file.
    
    Note: This is a placeholder. Full implementation pending.
    """
    raise NotImplementedError("save_particles is not yet implemented for XSuite I/O. "
                            "This function will be added in future updates.")

# Alternative name for validation
validate_file = validate_xsuite_file
