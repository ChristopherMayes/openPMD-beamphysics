# OpenPMD Extension for XSuite Simulation Output (EXT_XSuite)

**Version:** 1.0.0  
**Date:** 2025-11-12  
**Authors:** XSuite Collaboration  
**Status:** Specification

## 1. Introduction

This extension defines the OpenPMD format for XSuite beam dynamics simulation output data. It extends the openPMD-beamphysics particle record standard to support:
- Turn-by-turn tracking data
- Collective effects simulations
- Statistical beam moments
- Multi-turn storage strategies
- TMCI and instability analysis data

### 1.1 Relationship to openPMD-beamphysics

This extension builds upon the [openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics) standard, which defines:
- Particle coordinate systems
- Statistical quantities
- Beam rigidity and reference particle

XSuite uses the **6D phase space coordinates**: `(x, px, y, py, zeta, delta)` where:
- `zeta`: Longitudinal position relative to reference particle
- `delta`: Fractional momentum deviation `(p - p0)/p0`

### 1.2 Scope

This specification covers XSuite simulation output for:
- Single-turn tracking snapshots
- Multi-turn evolution data
- Collective effects (space charge, wakefields, beam-beam)
- Statistical beam analysis
- Parameter scan results (see EXT_XSuite_Scans.md)

## 2. File Structure

### 2.1 Root Attributes

All XSuite output files MUST include:

```python
# OpenPMD standard attributes
attrs['openPMD'] = '2.0.0'
attrs['openPMDextension'] = 'XSuite;1'
attrs['basePath'] = '/data/%T/'
attrs['dataType'] = 'simulation_output'

# XSuite-specific attributes
attrs['software'] = 'XSuite'
attrs['softwareVersion'] = 'X.Y.Z'
attrs['date'] = ISO 8601 timestamp
attrs['author'] = 'User Name <email@institution.org>'
```

### 2.2 Hierarchy

```
/
├── data/                          # OpenPMD standard
│   ├── 0000/                     # First iteration (turn 0)
│   │   └── particles/
│   ├── 0001/                     # Second iteration (turn 1)
│   │   └── particles/
│   └── .../
│
├── simulation_parameters/         # Simulation configuration
├── inputs/                        # References to input files
├── collective_effects/            # Collective effects metadata
├── statistics/                    # Beam statistical moments
└── provenance/                    # Simulation provenance
```

## 3. Particle Data

### 3.1 XSuite Coordinates

Particle data follows openPMD-beamphysics particle record with XSuite conventions:

```
/data/{iteration}/particles/
    
    x/
        (dataset) Horizontal position [m]
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
        
    px/
        (dataset) Horizontal momentum [dimensionless: px/p0]
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        (attr) definition = "Normalized horizontal momentum px/p0"
        
    y/
        (dataset) Vertical position [m]
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
        
    py/
        (dataset) Vertical momentum [dimensionless: py/p0]
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        (attr) definition = "Normalized vertical momentum py/p0"
        
    zeta/
        (dataset) Longitudinal position [m]
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
        (attr) coordinateSystem = "zeta"
        (attr) definition = "z - beta0*c*t (XSuite convention)"
        
    delta/
        (dataset) Fractional momentum deviation [dimensionless]
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        (attr) definition = "(p - p0)/p0"
        
    particleId/
        (dataset) Unique particle identifier [integer]
        (attr) description = "Persistent ID for particle tracking"
        
    weight/
        (dataset) Macroparticle weight [dimensionless]
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        (attr) description = "Number of real particles per macroparticle"
        
    state/
        (dataset) Particle state [integer]
        (attr) description = "1=active, 0=lost, <0=lost at specific element"
```

### 3.2 Reference Particle

Following openPMD-beamphysics convention:

```
/data/{iteration}/particles/
    (attr) speciesType = "electron" | "positron" | "proton"
    (attr) chargeState = Charge in units of e
    (attr) numParticles = Total number of particles
    
    # Reference particle (p0)
    (attr) p0c = Reference momentum [eV/c]
    (attr) p0c_unitSI = 1.60217662e-19
    (attr) energy = Total reference energy [eV]
    (attr) energy_unitSI = 1.60217662e-19
```

## 4. Multi-Turn Storage Strategies

XSuite simulations often track over thousands of turns. This section defines storage strategies.

### 4.1 Strategy Selection

```
/simulation_parameters/
    (attr) storage_strategy = "selective" | "monitors" | "statistics" | "full"
    (attr) num_turns = Total number of turns tracked
    (attr) save_frequency = Save every N turns (if applicable)
```

### 4.2 Selective Turn Storage

Store only specific turns:

```
/data/
    (attr) storage_strategy = "selective"
    (attr) stored_turns = [0, 100, 200, ..., 10000]
    
    0000/particles/    # Turn 0
    0001/particles/    # Turn 100
    0002/particles/    # Turn 200
    ...
```

### 4.3 Turn-by-Turn Monitors

Store statistical quantities at every turn:

```
/statistics/turn_by_turn/
    (attr) storage_strategy = "monitors"
    (attr) num_turns = 10000
    
    mean_x/
        (dataset) [num_turns]
        (attr) unitSI = 1.0
        (attr) unitDimension = [1, 0, 0, 0, 0, 0, 0]
        
    mean_px/
        (dataset) [num_turns]
        
    rms_x/
        (dataset) [num_turns]
        (attr) description = "RMS beam size"
        
    emittance_x/
        (dataset) [num_turns]
        (attr) description = "Geometric emittance"
        
    # Similarly for y, zeta, delta
    
    total_intensity/
        (dataset) [num_turns]
        (attr) description = "Number of surviving particles"
```

### 4.4 Statistical Moments Only

Store only beam moments (no individual particles):

```
/statistics/
    (attr) storage_strategy = "statistics"
    
    final_turn/
        mean/
            x/
            px/
            y/
            py/
            zeta/
            delta/
            
        covariance/
            (dataset) [6, 6] covariance matrix
            (attr) coordinates = ["x", "px", "y", "py", "zeta", "delta"]
```

### 4.5 Full Turn-by-Turn Storage

For TMCI analysis - store momentum at every turn:

```
/data/particles/
    (attr) storage_strategy = "turn_by_turn_momentum"
    (attr) stored_coordinates = ["px", "py", "delta"]
    (attr) reason = "Required for TMCI FFT analysis"
    
    turn_by_turn/
        px/
            (dataset) [num_particles, num_turns]
            (attr) description = "Horizontal momentum at each turn"
            
        py/
            (dataset) [num_particles, num_turns]
            
        delta/
            (dataset) [num_particles, num_turns]
```

**Storage size:** With compression, ~10 GB per 100k particles × 10k turns

## 5. Collective Effects Metadata

### 5.1 Space Charge

```
/collective_effects/space_charge/
    (attr) enabled = true
    (attr) solver = "PIC" | "frozen" | "quasi_frozen"
    (attr) pic_grid = [nx, ny, nz]
    (attr) pic_grid_extent = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
```

### 5.2 Wakefields

```
/collective_effects/wakefields/
    (attr) num_wake_elements = N
    
    wake_001/
        (attr) element_name = "resistive_wall_01"
        (attr) wake_type = "longitudinal" | "dipolar_x" | "dipolar_y"
        (attr) input_file = "path/to/wake.h5"
        (attr) input_file_checksum = "SHA256:..."
        (attr) s_location = Position in lattice [m]
```

### 5.3 Beam-Beam

```
/collective_effects/beam_beam/
    (attr) num_ip = Number of interaction points
    
    ip_001/
        (attr) s_location = Position in lattice [m]
        (attr) crossing_angle = Angle [rad]
        (attr) opposing_beam_file = "path/to/beam.h5"
```

## 6. Simulation Parameters

```
/simulation_parameters/
    (attr) num_turns = 10000
    (attr) num_particles = 1000000
    (attr) tracking_method = "xtrack" | "xpart"
    (attr) integrator = "drift-kick-drift" | "exact"
    
    # Time information
    (attr) turn_time = Revolution period [s]
    (attr) turn_time_unitSI = 1.0
    (attr) turn_time_unitDimension = [0, 0, 1, 0, 0, 0, 0]
    
    # Beam parameters
    (attr) bunch_intensity = Particles per bunch
    (attr) bunch_length = RMS bunch length [m]
    (attr) energy_spread = RMS energy spread (relative)
```

## 7. Input File References

Link to all input files used:

```
/inputs/
    (attr) parameter_file = "path/to/parameters.h5"
    (attr) parameter_file_checksum = "SHA256:..."
    (attr) parameter_operation_mode = "z"  # Which mode was used
    
    (attr) lattice_file = "path/to/lattice.h5"
    (attr) lattice_file_checksum = "SHA256:..."
    
    (attr) initial_distribution = "path/to/particles.h5"
    (attr) initial_distribution_checksum = "SHA256:..."
    
    collective_effects/
        (attr) wake_001 = "path/to/wake.h5"
        (attr) wake_001_checksum = "SHA256:..."
```

## 8. Provenance

```
/provenance/
    (attr) simulation_id = Unique identifier (UUID)
    (attr) start_time = ISO 8601 timestamp
    (attr) end_time = ISO 8601 timestamp
    (attr) computation_time = Wall clock time [s]
    (attr) hostname = Compute node hostname
    (attr) num_cores = Number of CPU cores used
    (attr) num_gpus = Number of GPUs used (if applicable)
    
    # Software versions
    (attr) xsuite_version = "X.Y.Z"
    (attr) xtrack_version = "X.Y.Z"
    (attr) xpart_version = "X.Y.Z"
    (attr) python_version = "3.X.Y"
    
    # Reproducibility
    (attr) random_seed = Seed value (if applicable)
    (attr) git_commit = Git commit hash
```

## 9. Statistics Group

```
/statistics/
    # Single-turn statistics
    final_turn/
        (attr) turn_number = Last turn number
        
        mean/
            x/
                (attr) value = Mean x position [m]
                (attr) unitSI = 1.0
            # ... for all coordinates
            
        std/
            x/
                (attr) value = RMS x [m]
            # ... for all coordinates
            
        emittance/
            geometric_x/
                (attr) value = Geometric emittance [m]
                (attr) definition = "sqrt(det(Sigma_x))"
            geometric_y/
            geometric_z/
            
            normalized_x/
                (attr) value = Normalized emittance [m]
                (attr) definition = "beta*gamma*emittance_geometric"
        
        covariance/
            (dataset) [6, 6] covariance matrix
            (attr) coordinates = ["x", "px", "y", "py", "zeta", "delta"]
    
    # Turn-by-turn statistics (optional)
    turn_by_turn/
        # See Section 4.3
```

## 10. Integration with openPMD-beamphysics

### 10.1 ParticleGroup Compatibility

XSuite output files are designed to be readable by openPMD-beamphysics ParticleGroup:

```python
from pmd_beamphysics import ParticleGroup

# Read XSuite output
pg = ParticleGroup('xsuite_output.h5')

# Access standard quantities
print(pg.x, pg.px, pg.y, pg.py)
print(pg['mean_x'], pg['sigma_x'])

# XSuite-specific coordinates
print(pg.zeta, pg.delta)  # Instead of z, pz
```

### 10.2 Coordinate Conversion

For codes expecting `(z, pz)` instead of `(zeta, delta)`:

```python
# zeta = z - beta0*c*t
# delta = (p - p0)/p0

# Convert to z, pz
z = pg.zeta + pg.beta * c * pg.t
pz = pg.p0 * (1 + pg.delta) * pg.beta_z
```

## 11. Example Files

### 11.1 Single-Turn Output

```python
import h5py
import numpy as np

with h5py.File('single_turn.h5', 'w') as f:
    # Root attributes
    f.attrs['openPMD'] = '2.0.0'
    f.attrs['openPMDextension'] = 'XSuite;1'
    f.attrs['software'] = 'XSuite'
    f.attrs['dataType'] = 'simulation_output'
    
    # Single iteration
    particles = f.create_group('data/0000/particles')
    particles.attrs['speciesType'] = 'electron'
    particles.attrs['numParticles'] = 1000000
    particles.attrs['p0c'] = 45.6e9  # eV/c
    
    # Particle coordinates
    for coord in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
        ds = particles.create_dataset(coord, data=...)
        ds.attrs['unitSI'] = 1.0
        ds.attrs['unitDimension'] = [...]
```

### 11.2 Multi-Turn with Monitors

```python
with h5py.File('multi_turn_monitors.h5', 'w') as f:
    # ... root attributes ...
    
    # Simulation parameters
    sim = f.create_group('simulation_parameters')
    sim.attrs['num_turns'] = 10000
    sim.attrs['storage_strategy'] = 'monitors'
    
    # Turn-by-turn statistics
    tbt = f.create_group('statistics/turn_by_turn')
    tbt.attrs['num_turns'] = 10000
    
    # Store mean and RMS at each turn
    for coord in ['x', 'px', 'y', 'py']:
        mean_ds = tbt.create_dataset(f'mean_{coord}', data=mean_array)
        rms_ds = tbt.create_dataset(f'rms_{coord}', data=rms_array)
```

## 12. Best Practices

### 12.1 Storage Optimization

- Use compression: `h5py.create_dataset(..., compression='gzip')`
- Store only needed coordinates for TMCI analysis
- Use `float32` instead of `float64` when precision allows
- Store statistical moments instead of full distributions when possible

### 12.2 Provenance

- Always include input file references with checksums
- Store git commit hash for reproducibility
- Include random seed if stochastic processes used

### 12.3 Metadata

- Use descriptive `(attr) description` for all datasets
- Include units and dimensions for all physical quantities
- Reference standards and definitions used

## 13. Validation

Files can be validated using:

```python
from pmd_beamphysics.interfaces import xsuite_io

# Validate XSuite output file
xsuite_io.validate_output_file('output.h5')
# Checks:
# - OpenPMD compliance
# - Required attributes present
# - Unit consistency
# - XSuite coordinate conventions
```

## Appendix A: Coordinate System Reference

| XSuite | openPMD-beamphysics | Description |
|--------|---------------------|-------------|
| x | x | Horizontal position [m] |
| px | px/p0 | Normalized horizontal momentum |
| y | y | Vertical position [m] |
| py | py/p0 | Normalized vertical momentum |
| zeta | z - β₀ct | Longitudinal position [m] |
| delta | (p-p₀)/p₀ | Fractional momentum deviation |

## Appendix B: Storage Size Estimates

| Strategy | Particles | Turns | Coordinates | Compressed Size |
|----------|-----------|-------|-------------|-----------------|
| Single turn | 1M | 1 | 6 | 20 MB |
| Selective (100 turns) | 1M | 100 | 6 | 2 GB |
| Monitors | 1M | 10k | stats only | 5 MB |
| Full momentum | 100k | 10k | 3 (px,py,δ) | 10 GB |

## Appendix C: Compatibility Matrix

| Code | Input Format | Output Format | Notes |
|------|--------------|---------------|-------|
| XSuite | EXT_XSuite_Inputs | EXT_XSuite | Native support |
| openPMD-beamphysics | openPMD-beamphysics | openPMD-beamphysics | Coordinate conversion needed |
| Elegant | SDDS | openPMD-beamphysics | Via conversion tools |
| MADx | TFS | EXT_XSuite_Inputs | Via lattice import |

---

**Version History:**
- v1.0.0 (2025-11-12): Initial specification
- Replaces previous EXT_XSuite versions

**Related Specifications:**
- EXT_XSuite_Inputs.md - Input data format
- EXT_XSuite_Scans.md - Parameter scan organization
