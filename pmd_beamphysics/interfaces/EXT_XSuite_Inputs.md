# OpenPMD Extension for XSuite Collective Effects Input Data (EXT_XSuite_Inputs)

**Version:** 0.1.0  
**Date:** 2025-11-12  
**Authors:** XSuite Collaboration  
**Status:** Draft

## 1. Introduction

This extension defines OpenPMD-compliant formats for collective effects input data used in XSuite beam dynamics simulations. It complements the EXT_XSuite output specification by providing standardized, self-describing formats for wake potentials, impedance models, and other collective effects data.

### 1.1 Motivation

Collective effects simulations require well-documented input data with clear provenance. This extension ensures:
- **Reproducibility**: Complete metadata for regenerating or validating input data
- **Shareability**: Standardized format for data exchange between institutions
- **Traceability**: Full lineage from measurement/calculation to simulation
- **FAIR compliance**: Findable, Accessible, Interoperable, Reusable data

### 1.2 Scope

This extension covers input data for:
- **Collective effects**: Wake potentials (longitudinal and transverse), impedance models (frequency domain), beam-beam interaction data
- **Machine parameters**: Global configuration, beam properties, optics parameters, RF system, geometry
- **Particle distributions**: Initial coordinates for tracking simulations  
- **Lattice definitions**: XSuite Line objects, MADx sequence files, element placement and Twiss parameters
- **Operation modes**: Multi-mode machine configurations (e.g., different physics programs at various energies)

## 2. General Requirements

### 2.1 File Format

All collective effects input data MUST be stored in HDF5 format with `.h5` extension to support:
- Hierarchical data organization
- Embedded metadata
- Efficient numerical array storage
- OpenPMD attribute compliance

### 2.2 Required Root Attributes

All input files MUST include these OpenPMD root attributes:

```python
# OpenPMD standard attributes
attrs['openPMD'] = '2.0.0'
attrs['openPMDextension'] = 'XSuite_Inputs;1'
attrs['basePath'] = '/data/%T/'
attrs['dataType'] = 'collective_effects_input'
attrs['software'] = 'XSuite'
attrs['softwareVersion'] = '0.x.x'
attrs['date'] = '2025-11-12 14:30:00 +0000'

# Input-specific attributes
attrs['inputDataType'] = 'wake_potential' | 'impedance' | 'beam_beam' | 'space_charge'
attrs['author'] = 'Name <email@institution.org>'
attrs['institution'] = 'Institution name'
attrs['comment'] = 'Description of the input data'
```

### 2.3 Provenance Tracking

All input files MUST include a `provenance` group with:

```
/provenance/
    (attr) generationMethod = 'measurement' | 'analytical' | 'simulation' | 'semi-analytical'
    (attr) generationDate = ISO 8601 timestamp
    (attr) generationSoftware = Software name and version
    (attr) generationParameters = JSON string of parameters
    (attr) references = DOI or citation for methodology
    (attr) validityRange = JSON string of applicable ranges
    (attr) uncertainty = Estimated uncertainty description
    (attr) lastModified = ISO 8601 timestamp
    (attr) modificationHistory = JSON array of changes
```

### 2.4 Unit System

All quantities MUST specify units using OpenPMD conventions:

```python
dataset.attrs['unitSI'] = 1.0  # Conversion factor to SI
dataset.attrs['unitDimension'] = [0, 0, 0, 0, 0, 0, 0]  # [L, M, T, I, Θ, N, J]
```

Standard units:
- **Wake potential**: V/C (Volt per Coulomb) or V/C/m
- **Impedance**: Ω (Ohm) or Ω/m
- **Length/position**: m (meters)
- **Frequency**: Hz (Hertz)
- **Time**: s (seconds)

## 3. Wake Potential Input Data

Wake potentials describe the electromagnetic field left behind by a leading bunch that affects trailing bunches.

### 3.1 File Structure

```
/data/wake_potential/
    component = 'longitudinal' | 'dipolar_x' | 'dipolar_y' | 'quadrupolar' | 'constant_x' | 'constant_y'
    
    # Required datasets
    wake_table/
        (attr) description = "Wake potential values"
        (attr) unitSI = 1.0
        (attr) unitDimension = [2, 1, -3, -1, 0, 0, 0]  # V/C for longitudinal
        (attr) interpolation = 'linear' | 'cubic' | 'pchip'
        
    source_position/
        (attr) description = "Position coordinate for wake evaluation"
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]  # m
        (attr) coordinateSystem = 's' | 'z' | 'time'
        
    # Optional datasets
    wake_derivative/
        (attr) description = "Derivative of wake potential"
        (attr) order = 1
        
/geometry/
    (attr) elementType = 'resistive_wall' | 'geometric' | 'cavity' | 'bellows' | 'mask' | 'other'
    (attr) material = Material description (e.g., 'copper', 'stainless_steel')
    (attr) conductivity = Value in S/m (if applicable)
    (attr) permeability = Relative permeability
    (attr) length = Element length in m
    (attr) radius = Pipe radius in m (if applicable)
    (attr) shape = 'circular' | 'rectangular' | 'elliptical' | 'other'
    (attr) dimensions = JSON string with shape parameters
```

### 3.2 Coordinate System Conventions

- **Longitudinal wakes**: Source position behind leading particle (s > 0 or z < 0)
- **Transverse wakes**: Wake includes geometric factor, units V/C/m
- **Sign convention**: Wake force = -q * W(s) for trailing particle

### 3.3 Example: Resistive Wall Wake

```python
import h5py
import numpy as np

# Create wake file
with h5py.File('resistive_wall_copper_round.h5', 'w') as f:
    # Root attributes
    f.attrs['openPMD'] = '2.0.0'
    f.attrs['openPMDextension'] = 'XSuite_Inputs;1'
    f.attrs['inputDataType'] = 'wake_potential'
    f.attrs['author'] = 'John Doe <john.doe@cern.ch>'
    f.attrs['date'] = '2025-11-12T14:30:00Z'
    
    # Provenance
    prov = f.create_group('provenance')
    prov.attrs['generationMethod'] = 'analytical'
    prov.attrs['generationSoftware'] = 'ImpedanceWake2D v1.2'
    prov.attrs['references'] = '10.1103/PhysRevSTAB.3.104401'
    prov.attrs['generationParameters'] = '{"formula": "AC_resistive_wall", "regime": "classical"}'
    
    # Geometry
    geom = f.create_group('geometry')
    geom.attrs['elementType'] = 'resistive_wall'
    geom.attrs['material'] = 'copper'
    geom.attrs['conductivity'] = 5.96e7  # S/m
    geom.attrs['length'] = 1.0  # m
    geom.attrs['radius'] = 0.018  # m
    geom.attrs['shape'] = 'circular'
    
    # Wake data
    data_grp = f.create_group('data/wake_potential')
    data_grp.attrs['component'] = 'longitudinal'
    
    # Source positions (m)
    s = np.linspace(0, 1.0, 1000)
    ds_wake = data_grp.create_dataset('source_position', data=s)
    ds_wake.attrs['unitSI'] = 1.0
    ds_wake.attrs['unitDimension'] = [1, 0, 0, 0, 0, 0, 0]
    ds_wake.attrs['coordinateSystem'] = 's'
    
    # Wake potential (V/C)
    W = calculate_resistive_wall_wake(s)
    ds_table = data_grp.create_dataset('wake_table', data=W)
    ds_table.attrs['unitSI'] = 1.0
    ds_table.attrs['unitDimension'] = [2, 1, -3, -1, 0, 0, 0]
    ds_table.attrs['interpolation'] = 'linear'
```

## 4. Impedance Input Data

Impedance models in frequency domain, including resonator models and measured/calculated impedance tables.

### 4.1 File Structure

```
/data/impedance/
    component = 'longitudinal' | 'dipolar_x' | 'dipolar_y' | 'quadrupolar'
    representation = 'table' | 'resonator_model' | 'broadband_model'
    
    # For tabulated impedance
    frequency/
        (attr) unitSI = 1.0
        (attr) unitDimension = [0, 0, -1, 0, 0, 0, 0]  # Hz
        
    impedance_real/
        (attr) description = "Real part of impedance"
        (attr) unitSI = 1.0
        (attr) unitDimension = [2, 1, -3, -2, 0, 0, 0]  # Ω
        
    impedance_imag/
        (attr) description = "Imaginary part of impedance"
        (attr) unitSI = 1.0
        (attr) unitDimension = [2, 1, -3, -2, 0, 0, 0]  # Ω
        
    # For resonator models
    resonators/
        (group) Contains subgroups for each resonator
        resonator_001/
            (attr) resonanceFrequency = Value in Hz
            (attr) resonanceFrequency_unitSI = 1.0
            (attr) shuntImpedance = Value in Ω (longitudinal) or Ω/m (transverse)
            (attr) shuntImpedance_unitSI = 1.0
            (attr) qualityFactor = Dimensionless Q
            (attr) model = 'parallel_RLC' | 'series_RLC' | 'other'

/measurement/
    (attr) facility = Measurement location
    (attr) method = 'wire_measurement' | 'beam_measurement' | 'simulation' | 'analytical'
    (attr) measurementDate = ISO 8601 timestamp
    (attr) beamEnergy = Energy in eV (if applicable)
    (attr) temperature = Temperature in K (if relevant)
```

### 4.2 Example: Broadband Resonator Model

```python
with h5py.File('broadband_resonator_model.h5', 'w') as f:
    # Root attributes
    f.attrs['openPMD'] = '2.0.0'
    f.attrs['openPMDextension'] = 'XSuite_Inputs;1'
    f.attrs['inputDataType'] = 'impedance'
    
    # Data group
    data_grp = f.create_group('data/impedance')
    data_grp.attrs['component'] = 'longitudinal'
    data_grp.attrs['representation'] = 'resonator_model'
    
    # Define multiple resonators
    res_grp = data_grp.create_group('resonators')
    
    # Resonator 1: Broad resonance
    res1 = res_grp.create_group('resonator_001')
    res1.attrs['resonanceFrequency'] = 1.3e9  # Hz
    res1.attrs['resonanceFrequency_unitSI'] = 1.0
    res1.attrs['shuntImpedance'] = 50e3  # Ω
    res1.attrs['shuntImpedance_unitSI'] = 1.0
    res1.attrs['qualityFactor'] = 1.0
    res1.attrs['model'] = 'parallel_RLC'
    
    # Resonator 2: Narrow resonance
    res2 = res_grp.create_group('resonator_002')
    res2.attrs['resonanceFrequency'] = 2.1e9  # Hz
    res2.attrs['shuntImpedance'] = 10e3  # Ω
    res2.attrs['qualityFactor'] = 100.0
    res2.attrs['model'] = 'parallel_RLC'
```

## 5. Machine Parameters

Machine and beam parameters defining the accelerator configuration and operating conditions. Stored as separate, reusable files.

### 5.1 File Structure

```
/data/parameters/
    (attr) parameterSetName = "PA31-3.0" or similar identifier
    (attr) accelerator = "FCC-ee" or machine name
    (attr) date = ISO 8601 timestamp
    
    /global/
        circumference/
            (attr) value = Value in m
            (attr) unitSI = 1.0
            (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
            (attr) symbol = "C"
            (attr) description = "Machine circumference"
            
    /beam/
        particles_per_bunch/
            (attr) symbol = "Np"
            (attr) description = "Number of particles per bunch"
            (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]  # dimensionless
            # Mode-specific values stored as attributes or subgroups
            
        number_of_bunches/
            (attr) symbol = "Nb"
            (attr) description = "Number of bunches"
            (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
            
        normalized_emittance_x/
            (attr) value = Value in m
            (attr) unitSI = 1.0
            (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
            (attr) symbol = "εₙₓ"
            (attr) stage = "injection" | "collision"
            
        normalized_emittance_y/
        bunch_length/
            (attr) symbol = "σz"
            (attr) unitSI = 1.0
            (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
            
        energy_spread/
            (attr) symbol = "σE/E"
            (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
            
    /optics/
        tune_x/
            (attr) symbol = "Qx"
            (attr) description = "Horizontal tune"
            (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
            
        tune_y/
            (attr) symbol = "Qy"
            
        chromaticity_x/
            (attr) symbol = "ξx"
            
        chromaticity_y/
            (attr) symbol = "ξy"
            
        momentum_compaction/
            (attr) symbol = "αc"
            (attr) description = "Momentum compaction factor"
            
        synchrotron_integrals/
            I2/
                (attr) symbol = "I₂"
                (attr) description = "Second synchrotron radiation integral"
            I3/
                (attr) symbol = "I₃"
            I5/
                (attr) symbol = "I₅"
                
        damping_time_transverse/
            (attr) unitSI = 1.0
            (attr) unitDimension = [0, 0, 1, 0, 0, 0, 0]  # seconds
            (attr) symbol = "τxy"
            
        damping_time_longitudinal/
            (attr) symbol = "τs"
            (attr) unitSI = 1.0
            (attr) unitDimension = [0, 0, 1, 0, 0, 0, 0]
            
        coupling/
            (attr) description = "Horizontal/vertical coupling"
            (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
            
        energy_acceptance/
            (attr) symbol = "Δp/p"
            (attr) description = "Maximum energy acceptance"
            
    /rf/
        frequency/
            (attr) unitSI = 1.0
            (attr) unitDimension = [0, 0, -1, 0, 0, 0, 0]  # Hz
            (attr) symbol = "fRF"
            
        voltage/
            (attr) unitSI = 1.0
            (attr) unitDimension = [2, 1, -3, -1, 0, 0, 0]  # Volt
            (attr) symbol = "VRF"
            (attr) description = "Total RF voltage"
            
        harmonic_number/
            (attr) description = "RF harmonic number"
            (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
            
        synchronous_phase_injection/
            (attr) unitSI = 0.017453292519943295  # π/180 for degrees to radians
            (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
            (attr) symbol = "φs,inj"
            
        synchronous_phase_extraction/
            (attr) symbol = "φs,ext"
            
        synchrotron_tune_injection/
            (attr) symbol = "Qs,inj"
            (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
            
        synchrotron_tune_extraction/
            (attr) symbol = "Qs,ext"
            
    /geometry/
        beam_pipe_shape/
            (attr) value = "circular" | "rectangular" | "elliptical"
            
        beam_pipe_material/
            (attr) value = "copper" | "stainless_steel" | etc.
            
        beam_pipe_diameter/
            (attr) unitSI = 1.0
            (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
            (attr) description = "Inner diameter for circular pipe"
            
    /energy/
        injection/
            (attr) unitSI = 1.60217662e-19  # eV to Joules
            (attr) unitDimension = [2, 1, -2, 0, 0, 0, 0]  # Energy
            (attr) description = "Beam energy at injection"
            
        extraction/
            # Or store per mode if different
            
/operation_modes/
    (attr) available_modes = ['z', 'w', 'zh', 'ttbar']  # JSON array
    (attr) mode_descriptions = JSON dict with descriptions
    (attr) default_mode = 'z'
    
    # Mode-specific parameters stored as groups
    mode_z/
        (attr) physics_process = "Z-pole running"
        (attr) beam_energy = Value in eV
        (attr) particles_per_bunch = Value
        (attr) number_of_bunches = Value
        # ... other mode-specific overrides
        
    mode_w/
        (attr) physics_process = "W pair production"
        (attr) beam_energy = Value
        
    mode_zh/
        (attr) physics_process = "Higgs factory"
        
    mode_ttbar/
        (attr) physics_process = "Top quark production"

/lattice_reference/
    (attr) lattice_file = "relative/path/to/lattice.json" (optional)
    (attr) lattice_version = "v2.3" (optional)
    (attr) lattice_checksum = "SHA256:..." (optional)
```

### 5.2 Operation Modes

For parameters that vary by physics program (operation mode), use one of two approaches:

**Approach 1: Mode-specific attributes** (for simple cases)
```python
dset = f.create_dataset('beam/particles_per_bunch', data=0)  # dummy
dset.attrs['mode_z'] = 2.5e10
dset.attrs['mode_w'] = 2.5e10
dset.attrs['mode_zh'] = 1.0e10
dset.attrs['mode_ttbar'] = 1.0e10
```

**Approach 2: Mode groups** (for complex cases)
```python
# Store all mode parameters in separate groups
modes = f.create_group('operation_modes')
z_grp = modes.create_group('mode_z')
z_grp.attrs['beam_energy'] = 45.6e9  # eV
z_grp.attrs['particles_per_bunch'] = 2.5e10
```

### 5.3 Example: Complete Parameters File

```python
import h5py
import numpy as np
from datetime import datetime

def create_machine_parameters(output_file, param_dict):
    """Create OpenPMD-compliant machine parameters file from dict"""
    
    with h5py.File(output_file, 'w') as f:
        # Root attributes
        f.attrs['openPMD'] = '2.0.0'
        f.attrs['openPMDextension'] = 'XSuite_Inputs;1'
        f.attrs['inputDataType'] = 'machine_parameters'
        f.attrs['date'] = datetime.now().isoformat()
        
        # Parameters metadata
        data_grp = f.create_group('data/parameters')
        data_grp.attrs['parameterSetName'] = param_dict['version']
        data_grp.attrs['accelerator'] = 'FCC-ee'
        
        # Global parameters
        global_grp = data_grp.create_group('global')
        circ = global_grp.create_dataset('circumference', data=param_dict['C']['value'])
        circ.attrs['unitSI'] = 1.0
        circ.attrs['unitDimension'] = [1, 0, 0, 0, 0, 0, 0]
        circ.attrs['symbol'] = 'C'
        circ.attrs['description'] = param_dict['C']['comments']
        
        # Beam parameters with mode support
        beam_grp = data_grp.create_group('beam')
        
        # Particles per bunch (mode-dependent)
        npb = beam_grp.create_dataset('particles_per_bunch', data=0)
        npb.attrs['symbol'] = 'Np'
        npb.attrs['description'] = param_dict['Np']['comments']
        for mode in ['z', 'w', 'zh', 'ttbar']:
            npb.attrs[f'mode_{mode}'] = param_dict['Np'][mode]
        
        # Number of bunches (mode-dependent)
        nbb = beam_grp.create_dataset('number_of_bunches', data=0)
        nbb.attrs['symbol'] = 'Nb'
        for mode in ['z', 'w', 'zh', 'ttbar']:
            nbb.attrs[f'mode_{mode}'] = param_dict['Nb'][mode]
        
        # Emittances (mode-independent in this case)
        epsx = beam_grp.create_dataset('normalized_emittance_x', 
                                       data=param_dict['bunch']['epsnx']['value'])
        epsx.attrs['unitSI'] = 1.0
        epsx.attrs['unitDimension'] = [1, 0, 0, 0, 0, 0, 0]
        epsx.attrs['symbol'] = 'εₙₓ'
        epsx.attrs['stage'] = 'injection'
        
        # Optics parameters
        optics_grp = data_grp.create_group('optics')
        
        qx = optics_grp.create_dataset('tune_x', data=param_dict['optics']['Qx']['z'])
        qx.attrs['symbol'] = 'Qx'
        qx.attrs['description'] = param_dict['optics']['Qx']['comments']
        qx.attrs['unitDimension'] = [0, 0, 0, 0, 0, 0, 0]
        
        # RF parameters
        rf_grp = data_grp.create_group('rf')
        
        freq = rf_grp.create_dataset('frequency', data=param_dict['RF']['RF_freq']['z'])
        freq.attrs['unitSI'] = 1.0
        freq.attrs['unitDimension'] = [0, 0, -1, 0, 0, 0, 0]
        freq.attrs['symbol'] = 'fRF'
        
        # Geometry
        geom_grp = data_grp.create_group('geometry')
        geom_grp.attrs['beam_pipe_shape'] = param_dict['beam_pipe']['shape']
        geom_grp.attrs['beam_pipe_material'] = param_dict['beam_pipe']['material']
        
        diam = geom_grp.create_dataset('beam_pipe_diameter', 
                                       data=param_dict['beam_pipe']['D']['value'])
        diam.attrs['unitSI'] = 1.0
        diam.attrs['unitDimension'] = [1, 0, 0, 0, 0, 0, 0]
        
        # Operation modes
        modes_grp = f.create_group('operation_modes')
        modes_grp.attrs['available_modes'] = ['z', 'w', 'zh', 'ttbar']
        modes_grp.attrs['default_mode'] = 'z'
        
        mode_desc = {
            'z': 'Z-pole running',
            'w': 'W pair production',
            'zh': 'Higgs factory',
            'ttbar': 'Top quark production'
        }
        import json
        modes_grp.attrs['mode_descriptions'] = json.dumps(mode_desc)
        
        # Create detailed mode groups
        for mode in ['z', 'w', 'zh', 'ttbar']:
            mode_grp = modes_grp.create_group(f'mode_{mode}')
            mode_grp.attrs['physics_process'] = mode_desc[mode]
            mode_grp.attrs['beam_energy'] = param_dict['E'][mode]
            mode_grp.attrs['beam_energy_unitSI'] = 1.60217662e-19  # eV to J
            mode_grp.attrs['particles_per_bunch'] = param_dict['Np'][mode]
            mode_grp.attrs['number_of_bunches'] = param_dict['Nb'][mode]

# Usage
create_machine_parameters('fcc_ee_parameters.h5', param_dict)
```

### 5.4 Conversion from JSON Parameters

Utility to convert existing JSON parameter files:

```python
import json
import h5py

def json_to_openpmd_parameters(json_file, output_h5):
    """Convert JSON parameter file to OpenPMD format"""
    with open(json_file, 'r') as f:
        params = json.load(f)
    
    create_machine_parameters(output_h5, params)
    print(f"Converted {json_file} -> {output_h5}")
```

## 6. Initial Particle Distributions

Initial particle coordinates for tracking simulations.

### 6.1 File Structure

Follows openPMD-beamphysics particle record format:

```
/data/particles/
    (attr) speciesType = "electron" | "positron" | "proton" | etc.
    (attr) numParticles = Total number of particles
    (attr) chargeState = Charge in units of e
    (attr) totalCharge = Total charge in C
    (attr) distribution_type = "matched" | "measured" | "generated"
    
    x/
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]  # m
        (dataset) Array of x positions
        
    px/
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 1, -1, 0, 0, 0, 0]  # momentum (kg⋅m/s)
        (dataset) Array of px momenta
        
    y/
        (dataset) Array of y positions
        
    py/
        (dataset) Array of py momenta
        
    z/  # or 'zeta' for XSuite convention
        (attr) coordinateSystem = "s" | "z" | "zeta"
        (dataset) Array of longitudinal positions
        
    pz/  # or 'delta' for XSuite convention
        (attr) definition = "delta" | "pt" | "pz"
        (dataset) Array of longitudinal momenta
        
    particleId/
        (attr) description = "Unique particle identifier"
        (dataset) Integer IDs for tracking
        
    weight/
        (attr) description = "Macroparticle weight"
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        (dataset) Statistical weights (optional)

/generation/
    (attr) method = "gaussian_matched" | "tracking" | "measurement" | "custom"
    (attr) software = "XSuite" | "MADx" | etc.
    (attr) timestamp = ISO 8601 timestamp
    (attr) seed = Random seed (if applicable)
    
    # For matched distributions
    twiss_parameters/
        beta_x/
            (attr) value = Value in m
            (attr) unitSI = 1.0
        alpha_x/
        beta_y/
        alpha_y/
        dispersion_x/
        dispersion_y/
        
    # Reference to parameters file
    parameters_file/
        (attr) file = "relative/path/to/parameters.h5"
        (attr) operation_mode = "z"
        (attr) checksum = "SHA256:..."

/statistics/
    (attr) mean_x = Value in m
    (attr) mean_px = Value
    (attr) rms_x = Value in m
    (attr) rms_px = Value
    (attr) emittance_x = Geometric emittance in m
    (attr) emittance_y = Geometric emittance in m
    # ... other statistical moments
```

### 6.2 XSuite Coordinate Mapping

XSuite uses (x, px, y, py, zeta, delta) coordinates:

- **zeta**: Longitudinal position relative to reference = z - beta0 * c * t
- **delta**: Fractional momentum deviation = (p - p0) / p0

Store with proper metadata:

```python
# Zeta coordinate
zeta_dset = particles.create_dataset('zeta', data=zeta_array)
zeta_dset.attrs['unitSI'] = 1.0
zeta_dset.attrs['unitDimension'] = [1, 0, 0, 0, 0, 0, 0]
zeta_dset.attrs['coordinateSystem'] = 'zeta'
zeta_dset.attrs['definition'] = 'z - beta0*c*t (XSuite convention)'

# Delta coordinate
delta_dset = particles.create_dataset('delta', data=delta_array)
delta_dset.attrs['unitDimension'] = [0, 0, 0, 0, 0, 0, 0]
delta_dset.attrs['definition'] = '(p - p0)/p0 (XSuite convention)'
```

### 6.3 Example: Gaussian Matched Distribution

```python
import xsuite as xs
import h5py
import numpy as np

def save_xsuite_particles_to_openpmd(particles, output_file, metadata):
    """Save XSuite particles to OpenPMD format"""
    
    with h5py.File(output_file, 'w') as f:
        # Root attributes
        f.attrs['openPMD'] = '2.0.0'
        f.attrs['openPMDextension'] = 'XSuite_Inputs;1'
        f.attrs['inputDataType'] = 'particle_distribution'
        
        # Particle data
        part_grp = f.create_group('data/particles')
        part_grp.attrs['speciesType'] = metadata.get('species', 'electron')
        part_grp.attrs['numParticles'] = len(particles.x)
        part_grp.attrs['distribution_type'] = metadata.get('type', 'matched')
        
        # Save coordinates
        coords = ['x', 'px', 'y', 'py', 'zeta', 'delta']
        units = [
            ([1, 0, 0, 0, 0, 0, 0], 1.0),  # x in m
            ([1, 1, -1, 0, 0, 0, 0], 1.0),  # px in kg⋅m/s (or normalized)
            ([1, 0, 0, 0, 0, 0, 0], 1.0),  # y in m
            ([1, 1, -1, 0, 0, 0, 0], 1.0),  # py
            ([1, 0, 0, 0, 0, 0, 0], 1.0),  # zeta in m
            ([0, 0, 0, 0, 0, 0, 0], 1.0),  # delta dimensionless
        ]
        
        for coord, (unit_dim, unit_si) in zip(coords, units):
            data = getattr(particles, coord)
            dset = part_grp.create_dataset(coord, data=data)
            dset.attrs['unitDimension'] = unit_dim
            dset.attrs['unitSI'] = unit_si
            
        # Particle IDs
        if hasattr(particles, 'particle_id'):
            pid = part_grp.create_dataset('particleId', data=particles.particle_id)
        else:
            pid = part_grp.create_dataset('particleId', data=np.arange(len(particles.x)))
        
        # Generation metadata
        gen_grp = f.create_group('generation')
        gen_grp.attrs['method'] = metadata.get('method', 'gaussian_matched')
        gen_grp.attrs['software'] = f"XSuite {xs.__version__}"
        gen_grp.attrs['timestamp'] = datetime.now().isoformat()
        
        # Statistics
        stats = f.create_group('statistics')
        stats.attrs['mean_x'] = float(np.mean(particles.x))
        stats.attrs['rms_x'] = float(np.std(particles.x))
        stats.attrs['mean_y'] = float(np.mean(particles.y))
        stats.attrs['rms_y'] = float(np.std(particles.y))
        # Calculate emittances
        # ...

# Usage
particles = xs.Particles(...)
save_xsuite_particles_to_openpmd(
    particles, 
    'initial_distribution.h5',
    metadata={'species': 'electron', 'method': 'gaussian_matched'}
)
```

### 6.4 Loading Particles from OpenPMD

```python
def load_particles_from_openpmd(input_file):
    """Load XSuite particles from OpenPMD format"""
    
    with h5py.File(input_file, 'r') as f:
        part_grp = f['data/particles']
        
        particles = xs.Particles(
            x=part_grp['x'][:],
            px=part_grp['px'][:],
            y=part_grp['y'][:],
            py=part_grp['py'][:],
            zeta=part_grp['zeta'][:],
            delta=part_grp['delta'][:],
        )
        
        # Load metadata
        metadata = {
            'species': part_grp.attrs.get('speciesType'),
            'numParticles': part_grp.attrs.get('numParticles'),
        }
        
    return particles, metadata
```

### 6.5 Migration from Pickle Files

```python
import pickle

def convert_pickle_to_openpmd(pickle_file, output_h5):
    """Convert legacy pickle particle file to OpenPMD"""
    
    with open(pickle_file, 'rb') as f:
        particles = pickle.load(f)
    
    # Assuming particles is XSuite Particles object
    save_xsuite_particles_to_openpmd(
        particles,
        output_h5,
        metadata={'method': 'converted_from_pickle'}
    )
```

## 7. Lattice and Line Files

Accelerator lattice definitions in various formats.

### 7.1 Native XSuite Line Format

XSuite Line objects can be exported to JSON with embedded metadata:

```
/data/line/
    (attr) lineFormat = "xsuite_json" | "xsuite_xml"
    (attr) xsuiteVersion = Version string
    (attr) latticeVersion = User-defined version
    
    line_json/
        (attr) description = "JSON representation of Line object"
        (dataset) String dataset containing JSON
        
    element_names/
        (dataset) Ordered list of element names
        
    s_positions/
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
        (dataset) S-coordinate of each element

/twiss/
    (attr) method = "4D" | "6D"
    (attr) computed_at = ISO 8601 timestamp
    
    beta_x/
        (dataset) Beta function at each element
    alpha_x/
    # ... other Twiss parameters
    
/parameters_reference/
    (attr) parameters_file = "path/to/parameters.h5"
    (attr) operation_mode = "z"
    (attr) checksum = "SHA256:..."
```

### 7.2 MADx Input Files

For MADx sequence (.seq) and strength (.str) files:

```
/data/madx/
    (attr) madxVersion = MADx version used
    (attr) sequenceFile = Original .seq filename
    (attr) strengthFile = Original .str filename
    
    sequence/
        (dataset) String dataset with .seq file contents
        
    strength/
        (dataset) String dataset with .str file contents
        
/conversion/
    (attr) converted_to_xsuite = true/false
    (attr) conversion_date = ISO 8601 timestamp
    (attr) xsuite_line_file = "path/to/converted/line.json"
```

### 7.3 Example: Complete Lattice File

```python
import json

def save_line_to_openpmd(line, output_file, metadata=None):
    """Save XSuite Line to OpenPMD format"""
    
    with h5py.File(output_file, 'w') as f:
        # Root attributes
        f.attrs['openPMD'] = '2.0.0'
        f.attrs['openPMDextension'] = 'XSuite_Inputs;1'
        f.attrs['inputDataType'] = 'lattice'
        
        # Line data
        line_grp = f.create_group('data/line')
        line_grp.attrs['lineFormat'] = 'xsuite_json'
        line_grp.attrs['xsuiteVersion'] = xs.__version__
        
        # Serialize line to JSON
        line_dict = line.to_dict()
        line_json_str = json.dumps(line_dict, indent=2)
        line_grp.create_dataset('line_json', data=line_json_str)
        
        # Element names and positions
        line_grp.create_dataset('element_names', 
                               data=[e.name for e in line.elements])
        
        s_pos = line.get_s_elements()
        s_dset = line_grp.create_dataset('s_positions', data=s_pos)
        s_dset.attrs['unitSI'] = 1.0
        s_dset.attrs['unitDimension'] = [1, 0, 0, 0, 0, 0, 0]
        
        # Compute and store Twiss
        twiss = line.twiss()
        twiss_grp = f.create_group('twiss')
        twiss_grp.attrs['method'] = '4D'
        twiss_grp.attrs['computed_at'] = datetime.now().isoformat()
        
        for param in ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dy']:
            if hasattr(twiss, param):
                dset = twiss_grp.create_dataset(param, data=getattr(twiss, param))
                dset.attrs['unitSI'] = 1.0
                if 'bet' in param or 'd' in param:
                    dset.attrs['unitDimension'] = [1, 0, 0, 0, 0, 0, 0]  # m
                else:
                    dset.attrs['unitDimension'] = [0, 0, 0, 0, 0, 0, 0]
        
        # Reference to parameters
        if metadata and 'parameters_file' in metadata:
            ref_grp = f.create_group('parameters_reference')
            ref_grp.attrs['parameters_file'] = metadata['parameters_file']
            ref_grp.attrs['operation_mode'] = metadata.get('operation_mode', 'z')
```

## 8. Beam-Beam Input Data

Opposing beam distribution for beam-beam simulations.

### 5.1 File Structure

```
/data/beam_beam/
    representation = 'gaussian' | 'particle_distribution' | 'moments'
    
    # For Gaussian beams
    sigma_x/
        (attr) value = RMS beam size in m
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
    sigma_y/
    sigma_z/
    mean_x/  # Orbit offset
    mean_y/
    mean_z/
    intensity/
        (attr) value = Number of particles
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
    
    # For particle distributions
    particles/
        (follows openPMD-beamphysics particle record structure)
        x/
        px/
        y/
        py/
        z/
        pz/
        
/collision/
    (attr) crossingAngle = Crossing angle in radians
    (attr) separationX = Horizontal separation in m
    (attr) separationY = Vertical separation in m
    (attr) ipLocation = Interaction point s-coordinate in m
```

## 9. Reference from Simulation Files

Simulation output files (following EXT_XSuite.md) MUST reference ALL input files using:

```
/inputs/
    collective_effects/
        (attr) wakeFile_001 = "relative/path/to/wake_file.h5"
        (attr) wakeFile_001_checksum = "SHA256:..."
        (attr) impedanceFile_001 = "relative/path/to/impedance_file.h5"
        (attr) impedanceFile_001_checksum = "SHA256:..."
        
    machine_parameters/
        (attr) parametersFile = "relative/path/to/parameters.h5"
        (attr) parametersFile_checksum = "SHA256:..."
        (attr) operation_mode = "z"  # Which mode was used
        
    lattice/
        (attr) latticeFile = "relative/path/to/line.h5"
        (attr) latticeFile_checksum = "SHA256:..."
        (attr) latticeVersion = "v2.3"
        
    initial_distribution/
        (attr) particleFile = "relative/path/to/particles.h5"
        (attr) particleFile_checksum = "SHA256:..."
        (attr) numParticles = 1000000
```

This creates bidirectional traceability:
- **Input → Output**: Which simulations used this input data
- **Output → Input**: Which input data produced these results

### 9.1 Complete Simulation Provenance

The combination of all input references creates a complete simulation lineage:

```
Parameters (fcc_ee_params.h5, mode_z)
    ↓
Lattice (fcc_ee_lattice_v2.3.h5)
    ↓
Initial Distribution (matched_gaussian.h5)
    ↓
Collective Effects (resistive_wall.h5, broadband_impedance.h5)
    ↓
SIMULATION
    ↓
Output Data (tracking_output.h5)
```

## 10. Validation and Verification

### 10.1 Required Validation Checks

Input files SHOULD include validation data:

```
/validation/
    (attr) testedWith = "XSuite version that validated this file"
    (attr) testDate = ISO 8601 timestamp
    (attr) physicalityChecks = JSON string describing checks performed
    (attr) unitTests = JSON array of unit test results
```

### 10.2 Recommended Validation Tests

- **Wake potentials**: Causality check (W(s) = 0 for s < 0)
- **Impedance**: Kramers-Kronig relations
- **Beam-beam**: Positive-definite covariance matrix
- **Units**: Dimensional analysis consistency
- **Machine parameters**: 
  - Physical range checks (positive emittances, realistic energies)
  - Consistency checks (RF frequency matches circumference/harmonic)
  - Mode completeness (all modes have required parameters)
- **Particle distributions**:
  - Coordinate bounds (finite values, no NaNs)
  - Statistical consistency (emittances match expected values)
  - Particle count matches metadata
- **Lattices**:
  - Periodic closure (Twiss at s=0 matches s=C)
  - Stability (tunes in stable region)
  - Element order consistency

### 10.3 Validation Utility Example

```python
def validate_input_file(filename):
    """Comprehensive validation of OpenPMD input files"""
    
    with h5py.File(filename, 'r') as f:
        # Check OpenPMD compliance
        assert 'openPMD' in f.attrs, "Missing openPMD attribute"
        assert 'openPMDextension' in f.attrs, "Missing extension attribute"
        assert 'XSuite_Inputs' in f.attrs['openPMDextension']
        
        input_type = f.attrs.get('inputDataType')
        
        if input_type == 'wake_potential':
            validate_wake(f)
        elif input_type == 'impedance':
            validate_impedance(f)
        elif input_type == 'machine_parameters':
            validate_parameters(f)
        elif input_type == 'particle_distribution':
            validate_particles(f)
        elif input_type == 'lattice':
            validate_lattice(f)
            
    print(f"✓ {filename} passed validation")

def validate_parameters(f):
    """Validate machine parameters file"""
    params = f['data/parameters']
    
    # Check for required groups
    assert 'global' in params, "Missing global parameters"
    assert 'beam' in params, "Missing beam parameters"
    assert 'optics' in params, "Missing optics parameters"
    
    # Physical checks
    if 'beam/normalized_emittance_x' in params:
        eps_x = params['beam/normalized_emittance_x'][()]
        assert eps_x > 0, f"Invalid emittance: {eps_x}"
        assert eps_x < 1e-3, f"Unrealistically large emittance: {eps_x}"
    
    # Mode consistency
    if 'operation_modes' in f:
        modes = f['operation_modes'].attrs['available_modes']
        for mode in modes:
            mode_grp = f[f'operation_modes/mode_{mode}']
            assert 'beam_energy' in mode_grp.attrs, f"Mode {mode} missing energy"
            
    print("  Parameters validation passed")
```

## 11. Migration from Legacy Formats

### 11.1 From Text Files

Common pattern for converting text-based wake/impedance files:

```python
import numpy as np
import h5py

def convert_legacy_wake_to_openpmd(txt_file, output_h5, metadata):
    """Convert legacy wake text file to OpenPMD format"""
    # Read legacy format
    data = np.loadtxt(txt_file)
    s = data[:, 0]
    W = data[:, 1]
    
    # Create OpenPMD file
    with h5py.File(output_h5, 'w') as f:
        # Add root attributes from metadata dict
        for key, val in metadata['root_attrs'].items():
            f.attrs[key] = val
        
        # Add provenance
        prov = f.create_group('provenance')
        prov.attrs['generationMethod'] = 'converted_from_legacy'
        prov.attrs['originalFile'] = txt_file
        prov.attrs['conversionDate'] = datetime.now().isoformat()
        
        # Add data
        data_grp = f.create_group('data/wake_potential')
        # ... (add datasets with proper attributes)
```

### 11.2 From ECSV with Embedded Metadata

ECSV files with metadata can be parsed to extract both data and provenance:

```python
from astropy.table import Table

def convert_ecsv_to_openpmd(ecsv_file, output_h5):
    """Convert ECSV with metadata to OpenPMD format"""
    # Read ECSV (preserves metadata)
    table = Table.read(ecsv_file, format='ascii.ecsv')
    
    # Extract metadata from ECSV header
    metadata = table.meta
    
    # Create OpenPMD file with metadata preserved
    # ... (implementation)
```

### 11.3 From JSON Machine Parameters

Convert existing JSON parameter files to OpenPMD:

```python
import json
import h5py
from datetime import datetime

def json_parameters_to_openpmd(json_file, output_h5):
    """Convert JSON parameter file (like FCC-ee params) to OpenPMD"""
    
    with open(json_file, 'r') as f:
        params = json.load(f)
    
    # Use the create_machine_parameters function from Section 5.3
    create_machine_parameters(output_h5, params)
    
    print(f"✓ Converted {json_file} → {output_h5}")
    print(f"  Version: {params.get('version', 'unknown')}")
    print(f"  Modes: {list(params.get('Np', {}).keys())}")

# Usage
json_parameters_to_openpmd('fcc_ee_params.json', 'fcc_ee_params.h5')
```

## 12. Examples and Use Cases

### 12.1 Complete Workflow Example

```python
import xsuite as xs
import h5py

# 1. Load wake input file
wake_file = 'collective_effects_inputs/wakes/resistive_wall.h5'

# 2. Create XSuite wake element
with h5py.File(wake_file, 'r') as f:
    s = f['data/wake_potential/source_position'][:]
    W = f['data/wake_potential/wake_table'][:]
    component = f['data/wake_potential'].attrs['component']
    
    wake = xs.WakeTable(
        source_position=s,
        wake_table=W,
        component=component
    )

# 3. Track with provenance
line = xs.Line(elements=[..., wake, ...])
line.track(particles)

# 4. Save output with input file reference
line.to_openpmd(
    'output.h5',
    input_files={'wake': wake_file}
)
```

### 12.2 Sharing Between Institutions

Input files following this standard can be exchanged via:
- **Zenodo/CERN Open Data**: Assign DOIs to input datasets
- **Institutional repositories**: Version-controlled storage
- **Direct exchange**: Self-contained files with complete metadata

### 12.3 End-to-End Simulation Setup

```python
import xsuite as xs
import h5py

# 1. Load machine parameters
params = load_machine_parameters('fcc_ee_params.h5', mode='z')

# 2. Load or create lattice
line = xs.Line.from_json('fcc_ee_lattice.json')

# 3. Load initial distribution
particles = load_particles_from_openpmd('initial_distribution.h5')

# 4. Load collective effects
with h5py.File('resistive_wall.h5', 'r') as f:
    wake = xs.WakeTable.from_openpmd(f)
    
# 5. Setup simulation
line.insert_element(name='wake', element=wake, at_s=1000.0)

# 6. Track
line.track(particles, num_turns=1000)

# 7. Save with complete provenance
line.to_openpmd(
    'simulation_output.h5',
    input_files={
        'parameters': 'fcc_ee_params.h5',
        'lattice': 'fcc_ee_lattice.json',
        'particles': 'initial_distribution.h5',
        'wake': 'resistive_wall.h5'
    },
    metadata={'operation_mode': 'z'}
)
```

## 13. Future Extensions

Planned additions to this specification:
- Electron cloud density maps
- Non-linear space charge field maps  
- Synchrotron radiation damping parameters
- IBS (Intra-Beam Scattering) lookup tables
- Machine imperfection data (alignment, field errors)

## Appendix A: Complete File Examples

See companion repository for complete working examples:

### Collective Effects Inputs
- `examples/wake_resistive_wall.h5` - Circular pipe resistive wall wake
- `examples/wake_geometric_cavity.h5` - RF cavity geometric wake
- `examples/impedance_kicker.h5` - Kicker impedance (resonator model)
- `examples/impedance_broadband.h5` - Broadband impedance table
- `examples/beam_beam_gaussian.h5` - Gaussian beam-beam interaction

### Machine Configuration
- `examples/fcc_ee_parameters.h5` - Complete FCC-ee machine parameters with all operation modes
- `examples/lhc_parameters.h5` - LHC parameters example
- `examples/simple_ring_parameters.h5` - Minimal parameters for testing

### Lattices and Lines
- `examples/fcc_ee_lattice.h5` - FCC-ee lattice with Twiss
- `examples/simple_fodo.h5` - FODO cell for testing
- `examples/madx_converted_lattice.h5` - Example MADx → OpenPMD conversion

### Particle Distributions
- `examples/gaussian_matched.h5` - Matched Gaussian distribution
- `examples/measured_injection.h5` - Distribution from measurement
- `examples/macro_particles_1M.h5` - 1 million macroparticles

### Conversion Examples
- `examples/convert_json_params.py` - JSON → HDF5 parameters
- `examples/convert_madx_lattice.py` - MADx → HDF5 lattice
- `examples/convert_wake_tables.py` - Text → HDF5 wakes

### Complete Workflows
- `examples/full_simulation_setup.py` - Load all inputs and run simulation
- `examples/validate_all_inputs.py` - Validation workflow
- `examples/provenance_tracking.py` - Track input → output lineage

## Appendix B: Compatibility Matrix

### Input Data Types and XSuite Components

| Input Type | File Extension | XSuite Component | Required Fields |
|------------|---------------|------------------|-----------------|
| Wake potential | .h5 | WakeTable | wake_table, source_position |
| Impedance (resonator) | .h5 | WakeResonator | resonators/* |
| Impedance (table) | .h5 | ImpedanceTable | frequency, impedance_real, impedance_imag |
| Beam-beam | .h5 | BeamBeamBiGaussian2D/3D | sigma_x, sigma_y, intensity |
| Machine parameters | .h5 | Line configuration | global/*, beam/*, optics/*, rf/* |
| Particle distribution | .h5 | Particles | x, px, y, py, zeta, delta |
| Lattice (XSuite) | .h5, .json | Line | line_json, element_names |
| Lattice (MADx) | .h5 | Line (via import) | sequence, strength |

### File Format Decision Tree

```
What are you storing?
├─ Collective effects data
│  ├─ Time domain → wake_potential
│  └─ Frequency domain → impedance
├─ Machine configuration
│  ├─ Global parameters → machine_parameters
│  └─ Lattice structure → lattice
└─ Particle data
   ├─ Initial state → particle_distribution
   └─ Tracked particles → (see EXT_XSuite.md output spec)
```

## Appendix C: Validation Tools

Reference implementation of validation utilities available in `xsuite.io.openpmd`:

### Core Validation Functions
- `validate_input_file(filename)`: Check OpenPMD compliance for any input type
- `validate_wake(file_handle)`: Verify wake causality and physical constraints
- `validate_impedance(file_handle)`: Check Kramers-Kronig relations
- `validate_parameters(file_handle)`: Validate machine parameters consistency
- `validate_particles(file_handle)`: Check particle distribution statistics
- `validate_lattice(file_handle)`: Verify lattice stability and closure
- `verify_units(dataset)`: Validate unit attributes and dimensional consistency

### Utility Functions
- `compute_checksum(filename)`: Calculate SHA256 checksum for file referencing
- `load_with_provenance(filename)`: Load input with full metadata extraction
- `compare_input_versions(file1, file2)`: Diff two versions of same input type

### Conversion Tools
- `json_to_openpmd_parameters(json_file, h5_file)`: Convert JSON params to HDF5
- `madx_to_openpmd_lattice(seq_file, str_file, h5_file)`: Convert MADx to HDF5
- `pickle_to_openpmd_particles(pkl_file, h5_file)`: Convert pickle to HDF5
- `text_to_openpmd_wake(txt_file, h5_file, metadata)`: Convert text wake data
- `ecsv_to_openpmd(ecsv_file, h5_file)`: Preserve ECSV metadata in HDF5

### Example Usage

```python
from xsuite.io.openpmd import validate_input_file, compute_checksum

# Validate all inputs before simulation
inputs = {
    'parameters': 'fcc_ee_params.h5',
    'lattice': 'fcc_ee_line.h5',
    'particles': 'initial_dist.h5',
    'wake': 'resistive_wall.h5'
}

for name, filename in inputs.items():
    try:
        validate_input_file(filename)
        checksum = compute_checksum(filename)
        print(f"✓ {name}: {filename} [SHA256: {checksum[:16]}...]")
    except Exception as e:
        print(f"✗ {name}: {filename} - {e}")
```

### Command-Line Tools

```bash
# Validate any OpenPMD input file
$ xsuite-validate input.h5

# Convert legacy formats
$ xsuite-convert params.json params.h5
$ xsuite-convert wake.txt wake.h5 --metadata wake_metadata.json
$ xsuite-convert particles.pkl particles.h5

# Inspect file contents
$ xsuite-inspect input.h5
$ xsuite-inspect input.h5 --show-provenance
$ xsuite-inspect input.h5 --verify-references
```
