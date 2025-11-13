# OpenPMD Extension for XSuite Parameter Scans (EXT_XSuite_Scans)

**Version:** 0.1.0  
**Date:** 2025-11-12  
**Status:** Draft

## 1. Introduction

This extension defines OpenPMD-compliant formats for organizing multi-dimensional parameter scan simulations. Common use cases include:
- TMCI threshold scans (bunch intensity vs. chromaticity)
- Instability studies (tune scans, impedance scans)
- Optimization studies (multi-parameter beam dynamics)
- Machine commissioning scenarios

## 2. Architecture Overview

Parameter scans use a **hybrid storage model**:

```
scan_name/
├── scan_manifest.h5              # Scan metadata + lightweight results
├── scan_config.h5                # Shared configuration (base parameters)
├── inputs/                       # Shared input files
│   ├── base_parameters.h5
│   ├── lattice.h5
│   └── wake_files/
└── outputs/                      # Individual simulation outputs
    ├── point_0000.h5
    ├── point_0001.h5
    └── ...
```

**Benefits:**
- Manifest enables quick analysis without loading full data
- Individual outputs support parallel execution and fault tolerance
- Shared inputs avoid duplication
- Full data available for detailed analysis of specific points

## 3. Scan Manifest File

The manifest file (`scan_manifest.h5`) stores scan metadata and summary results.

### 3.1 Root Structure

```
/
├── @openPMD = "2.0.0"
├── @openPMDextension = "XSuite_Scans;1"
├── @dataType = "parameter_scan"
├── @scanName = "tmci_intensity_scan"
├── @scanType = "grid" | "adaptive" | "random" | "optimization"
├── @scanPurpose = "TMCI threshold study"
├── @date = ISO 8601 timestamp
├── @author = "Name <email>"
│
├── scan_parameters/              # What parameters were varied
├── parameter_space/              # The actual parameter values
├── scan_results/                 # Lightweight results from all points
├── simulation_status/            # Which points completed successfully
├── references/                   # Links to input files and outputs
└── analysis/                     # Derived quantities (thresholds, etc.)
```

### 3.2 Scan Parameters Definition

```
/scan_parameters/
    (attr) numParameters = 2  # Number of varied parameters
    (attr) totalPoints = 100  # Total scan points
    (attr) completedPoints = 98  # Successfully completed
    (attr) scanType = "grid" | "adaptive" | "random"
    
    parameter_001/
        (attr) name = "bunch_intensity"
        (attr) symbol = "Np"
        (attr) description = "Particles per bunch"
        (attr) unitSI = 1.0
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        (attr) min_value = 1e10
        (attr) max_value = 5e10
        (attr) num_points = 10
        (attr) scale = "linear" | "log"
        (attr) reference_file = "inputs/base_parameters.h5"
        (attr) reference_path = "/data/parameters/beam/particles_per_bunch"
        
    parameter_002/
        (attr) name = "chromaticity_x"
        (attr) symbol = "ξx"
        (attr) description = "Horizontal chromaticity"
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        (attr) min_value = -10.0
        (attr) max_value = 10.0
        (attr) num_points = 10
        (attr) scale = "linear"
        (attr) reference_file = "inputs/base_parameters.h5"
        (attr) reference_path = "/data/parameters/optics/chromaticity_x"
```

### 3.3 Parameter Space

Stores the actual parameter values for each scan point:

```
/parameter_space/
    (attr) shape = [num_param_1, num_param_2, ...]  # Grid shape
    (attr) encoding = "grid" | "list"
    
    # For grid scans:
    parameter_001_values/
        (dataset) 1D array of parameter 1 values [10]
        (attr) name = "bunch_intensity"
        (attr) unitSI = 1.0
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        
    parameter_002_values/
        (dataset) 1D array of parameter 2 values [10]
        (attr) name = "chromaticity_x"
        
    # For non-grid scans (adaptive, random):
    point_parameters/
        (dataset) 2D array [num_points, num_parameters]
        (attr) column_names = ["bunch_intensity", "chromaticity_x"]
```

### 3.4 Scan Results (Lightweight)

Summary results for quick analysis:

```
/scan_results/
    (attr) result_type = "summary"  # Only lightweight metrics
    (attr) full_data_location = "outputs/"
    
    # Results organized by scan point
    # For grid scans: N-dimensional arrays
    growth_rate/
        (dataset) 2D array [10, 10] or N-D depending on parameters
        (attr) description = "Instability growth rate"
        (attr) unitSI = 1.0
        (attr) unitDimension = [0, 0, -1, 0, 0, 0, 0]  # 1/s
        (attr) nan_on_failure = true
        
    tune_shift_x/
        (dataset) 2D array [10, 10]
        (attr) description = "Horizontal tune shift"
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        
    tune_shift_y/
        (dataset) 2D array [10, 10]
        
    emittance_growth/
        (dataset) 2D array [10, 10]
        (attr) description = "Emittance growth over simulation"
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]  # Fractional
        
    # Optional: Time series statistics
    final_turn/
        (dataset) 2D array [10, 10]
        (attr) description = "Last stable turn"
        
    convergence_status/
        (dataset) 2D array of integers [10, 10]
        (attr) description = "0=converged, 1=unstable, 2=failed"
```

### 3.5 Simulation Status

Track which simulations completed:

```
/simulation_status/
    completed/
        (dataset) Boolean array matching parameter space shape
        
    timestamps/
        start_time/
            (dataset) ISO 8601 timestamp array
        end_time/
            (dataset) ISO 8601 timestamp array
            
    computation_time/
        (dataset) Computation time in seconds per point
        (attr) unitSI = 1.0
        (attr) unitDimension = [0, 0, 1, 0, 0, 0, 0]
        
    error_messages/
        (dataset) String array with error messages (empty if successful)
```

### 3.6 References

Link to all input/output files:

```
/references/
    (attr) config_file = "scan_config.h5"
    (attr) base_parameters = "inputs/base_parameters.h5"
    (attr) lattice_file = "inputs/lattice.h5"
    
    output_files/
        (dataset) String array of output file paths
        # e.g., ["outputs/point_0000.h5", "outputs/point_0001.h5", ...]
        
    point_to_file_map/
        (dataset) Integer array mapping parameter space indices to file indices
        
    checksums/
        (dataset) SHA256 checksums for all output files
```

### 3.7 Analysis Results

Derived quantities like instability thresholds:

```
/analysis/
    (attr) analysis_software = "XSuite Analysis v1.0"
    (attr) analysis_date = ISO 8601 timestamp
    
    tmci_threshold/
        (attr) method = "growth_rate_interpolation"
        (attr) threshold_criterion = "growth_rate = 0"
        (attr) value = 3.2e10  # Threshold intensity
        (attr) unitSI = 1.0
        (attr) unitDimension = [0, 0, 0, 0, 0, 0, 0]
        (attr) uncertainty = 0.1e10
        
    threshold_curve/
        # For multi-parameter scans
        (dataset) Threshold as function of other parameters
        (attr) independent_parameter = "chromaticity_x"
```

## 4. Scan Configuration File

Shared configuration (`scan_config.h5`) stores fixed parameters:

```
/
├── @openPMD = "2.0.0"
├── @openPMDextension = "XSuite_Scans;1"
├── @dataType = "scan_configuration"
│
├── simulation_config/
│   (attr) num_turns = 10000
│   (attr) num_particles = 1000000
│   (attr) tracking_method = "xtrack"
│   (attr) collective_effects = true
│   
├── fixed_parameters/
│   # Parameters that DON'T vary in scan
│   # Copy of relevant sections from base_parameters.h5
│   
└── input_file_references/
    (attr) base_parameters = "inputs/base_parameters.h5"
    (attr) base_parameters_checksum = "SHA256:..."
    (attr) lattice = "inputs/lattice.h5"
    (attr) wake_files = ["inputs/wake_resistive_wall.h5", ...]
```

## 5. Individual Output Files

Each scan point has its own output file following EXT_XSuite.md:

```
outputs/point_0037.h5:
/
├── @openPMD = "2.0.0"
├── @openPMDextension = "XSuite;1"
├── @scanManifest = "../scan_manifest.h5"  # Link back to manifest
├── @scanPoint = 37  # Linear index
├── @scanCoordinates = [4, 7]  # Grid coordinates [i, j]
│
├── data/
│   # Standard XSuite output (see EXT_XSuite.md)
│   
├── inputs/
│   (attr) scan_parameters
│       (attr) bunch_intensity = 3.5e10
│       (attr) chromaticity_x = -2.0
│   # References to shared input files
│   
└── scan_metadata/
    (attr) parameter_point = 37
    (attr) parameter_values = [3.5e10, -2.0]
    (attr) parameter_names = ["bunch_intensity", "chromaticity_x"]
```

## 6. Example Use Cases

### 6.1 TMCI Intensity Threshold Scan

**Setup:**
- Vary: Bunch intensity (1e10 to 5e10, 20 points)
- Vary: Horizontal chromaticity (-10 to 10, 21 points)
- Fixed: Machine optics, wake functions
- Result: 2D instability diagram

**Files:**
```
tmci_scan/
├── scan_manifest.h5          # 2D arrays [20, 21]
├── scan_config.h5
├── inputs/
│   ├── fcc_ee_base.h5
│   └── resistive_wall_wake.h5
└── outputs/
    ├── point_0000.h5         # Np=1e10, ξx=-10
    ├── point_0001.h5         # Np=1e10, ξx=-9
    └── ...
    └── point_0419.h5         # Np=5e10, ξx=+10
```

**Analysis:**
```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load scan results
with h5py.File('tmci_scan/scan_manifest.h5', 'r') as f:
    # Get parameter values
    intensity = f['parameter_space/parameter_001_values'][:]
    chroma = f['parameter_space/parameter_002_values'][:]
    
    # Get growth rates
    growth = f['scan_results/growth_rate'][:]
    
    # Get TMCI threshold
    threshold = f['analysis/tmci_threshold'].attrs['value']

# Plot instability diagram
plt.contourf(chroma, intensity, growth, levels=20)
plt.axhline(threshold, color='red', label='TMCI threshold')
plt.xlabel('Chromaticity ξx')
plt.ylabel('Bunch Intensity')
plt.title('TMCI Stability Diagram')
```

### 6.2 Adaptive Scan (Threshold Refinement)

**Setup:**
- Initial coarse grid
- Adaptive refinement near instability boundary
- Non-uniform parameter distribution

**Parameter space encoding:**
```python
/parameter_space/
    encoding = "list"  # Not a regular grid
    
    point_parameters/
        (dataset) [N, 2] array
        # Each row: [bunch_intensity, chromaticity_x]
```

### 6.3 Multi-Objective Optimization

**Setup:**
- Vary: 4-5 machine parameters
- Objective: Minimize emittance growth + maximize intensity
- Method: Genetic algorithm or Bayesian optimization

```python
/scan_parameters/
    numParameters = 5
    scanType = "optimization"
    
    optimization_config/
        (attr) algorithm = "genetic_algorithm"
        (attr) population_size = 100
        (attr) generations = 50
        (attr) objective_function = "weighted_sum"
        
/scan_results/
    objective_values/
        (dataset) [num_points] array
        
    pareto_front/
        (dataset) Indices of Pareto-optimal points
```

## 7. Scan Execution Workflow

### 7.1 Setup Phase

```python
import xsuite_scans as xscans

# Define parameter scan
scan = xscans.ParameterScan(
    name="tmci_intensity_scan",
    base_config="inputs/fcc_ee_base.h5"
)

# Add scan parameters
scan.add_parameter(
    name="bunch_intensity",
    path="/data/parameters/beam/particles_per_bunch",
    values=np.linspace(1e10, 5e10, 20)
)

scan.add_parameter(
    name="chromaticity_x",
    path="/data/parameters/optics/chromaticity_x",
    values=np.linspace(-10, 10, 21)
)

# Define output metrics
scan.add_output_metric("growth_rate", compute_growth_rate)
scan.add_output_metric("tune_shift_x", compute_tune_shift)

# Create scan manifest
scan.create_manifest("scan_manifest.h5")
```

### 7.2 Execution Phase (Parallel)

```python
# Run scan (parallelized)
scan.run(
    num_workers=40,
    output_dir="outputs/",
    save_full_data=True
)

# Or submit to batch system
scan.submit_to_batch(
    system="slurm",
    nodes=10,
    time="2:00:00"
)
```

### 7.3 Analysis Phase

```python
# Load scan results
scan = xscans.load_scan("scan_manifest.h5")

# Quick analysis (from manifest only)
threshold = scan.find_threshold("growth_rate", criterion=0)
print(f"TMCI threshold: {threshold:.2e} particles/bunch")

# Detailed analysis (load specific points)
interesting_points = scan.find_points(
    "growth_rate", 
    condition=lambda x: 0 < x < 0.1
)

for point in interesting_points:
    data = scan.load_point(point)
    # Detailed analysis of near-threshold behavior
```

## 8. Best Practices

### 8.1 Storage Efficiency

- Store only lightweight results in manifest (< 1 MB per point)
- Keep full tracking data in individual files
- Use compression for large datasets
- Prune unnecessary turn-by-turn data

### 8.2 Fault Tolerance

- Mark failed points in simulation_status
- Store error messages for debugging
- Enable restart from partial results
- Use checksums to detect corrupted files

### 8.3 Reproducibility

- Reference all input files with checksums
- Store scan execution metadata (software versions, timestamps)
- Include analysis code/scripts in repository
- Version scan manifests

### 8.4 Scalability

**Small scans** (< 100 points):
- Store all results in manifest
- Keep full data in individual files

**Medium scans** (100-10,000 points):
- Use hybrid approach
- Compress individual outputs

**Large scans** (> 10,000 points):
- Consider database backend for manifest
- Use hierarchical output organization
- Implement streaming analysis

## 9. File Naming Conventions

```
scan_name/
├── scan_manifest.h5
├── scan_config.h5
├── inputs/
│   └── [descriptive names]
└── outputs/
    ├── point_0000.h5          # Zero-padded linear index
    ├── point_0001.h5
    └── ...
    
# Alternative: Encode parameters in filename
└── outputs/
    ├── Np_1.0e10_xi_-10.0.h5
    ├── Np_1.0e10_xi_-9.0.h5
    └── ...
```

## 10. Compatibility with EXT_XSuite

Individual scan point outputs MUST follow EXT_XSuite.md specification with additions:

```
point_0037.h5:
/
├── [Standard EXT_XSuite structure]
│
└── scan_metadata/             # Additional scan-specific metadata
    (attr) scanManifest = "../scan_manifest.h5"
    (attr) scanPoint = 37
    (attr) scanCoordinates = [4, 7]
    (attr) parameterValues = [3.5e10, -2.0]
    (attr) parameterNames = ["bunch_intensity", "chromaticity_x"]
```

This ensures scan outputs can be analyzed both:
- **Collectively**: Via scan manifest
- **Individually**: As standalone simulations

## Appendix A: Complete Example Files

See companion repository:
- `examples/scans/tmci_intensity_scan/` - Full TMCI scan example
- `examples/scans/optimization_study/` - Multi-parameter optimization
- `examples/scripts/create_scan.py` - Scan setup script
- `examples/scripts/analyze_scan.py` - Analysis utilities

## Appendix B: Scan Types

| Scan Type | Use Case | Parameter Space | Storage |
|-----------|----------|-----------------|---------|
| Grid | TMCI diagrams, stability maps | Regular N-D grid | N-D arrays |
| Adaptive | Threshold refinement | Irregular, focused | List of points |
| Random | Monte Carlo, uncertainty | Random sampling | List of points |
| Optimization | Machine commissioning | Guided search | Generation-based |
| Line | 1D slices through parameter space | 1D array | 1D arrays |

## Appendix C: Performance Considerations

**Manifest file size estimation:**
- Metadata: ~10 KB
- Parameter space: ~1 KB per parameter
- Results: ~8 bytes × num_points × num_metrics
- For 10,000 points × 10 metrics: ~800 KB

**Individual output size:**
- Lightweight (summary only): ~100 KB
- With turn-by-turn monitors: ~10 MB
- Full particle distributions: ~1 GB per point

**Recommendation:** 
- Keep manifest < 100 MB for fast loading
- Use selective turn storage in individual outputs
- Implement on-demand loading for large scans
