# XSuite Input Conversion - Quick Reference

## Summary

Converted XSuite FCC-ee HEB simulation input files to openPMD-compliant HDF5 format.

### What Was Converted

✅ **4 Machine Parameter Files**
- `machine/machine_z.h5` (45.6 GeV)
- `machine/machine_w.h5` (80.0 GeV)
- `machine/machine_zh.h5` (120.0 GeV)
- `machine/machine_ttbar.h5` (182.5 GeV)

✅ **1 Wake Potential File**
- `wakes/wake_copper.h5` (15,599 points, 3 components)

✅ **2 Impedance Files**
- `impedance/impedance_copper_longitudinal.h5` (585 frequency points)
- `impedance/impedance_stainless_longitudinal.h5` (585 frequency points)

### File Locations

| Type | Location | Count |
|------|----------|-------|
| Machine | `tests/tests_xsuite/xsuite_pmd/machine/` | 4 |
| Wakes | `tests/tests_xsuite/xsuite_pmd/wakes/` | 1 |
| Impedance | `tests/tests_xsuite/xsuite_pmd/impedance/` | 2 |
| Manifest | `tests/tests_xsuite/xsuite_pmd/conversion_manifest.json` | 1 |

### How to Use

#### List all converted files
```bash
ls -lh tests/tests_xsuite/xsuite_pmd/**/*.h5
```

#### Load machine parameters in Python
```python
import h5py

with h5py.File('tests/tests_xsuite/xsuite_pmd/machine/machine_ttbar.h5', 'r') as f:
    print("Energy (GeV):", f.attrs['energy_GeV'])
    print("Circumference (m):", f['circumference'][()])
    print("epsn_x (m):", f['epsn_x'][()])
```

#### Load wake potential in Python
```python
import h5py

with h5py.File('tests/tests_xsuite/xsuite_pmd/wakes/wake_copper.h5', 'r') as f:
    # Longitudinal wake
    z_long = f['longitudinal/z'][:]
    wake_long = f['longitudinal/wake'][:]
    
    # Dipole wakes
    z_dipx = f['dipole_x/z'][:]
    wake_dipx = f['dipole_x/wake'][:]
```

#### Load impedance in Python
```python
import h5py

with h5py.File('tests/tests_xsuite/xsuite_pmd/impedance/impedance_copper_longitudinal.h5', 'r') as f:
    freq = f['frequency'][:]
    z_real = f['impedance_real'][:]
    z_imag = f['impedance_imag'][:]
```

### Re-run Conversion

```bash
# Convert all files (default)
python convert_xsuite_inputs.py --verbose

# Convert specific energy points only
python convert_xsuite_inputs.py --energy-points z w --verbose

# Use custom input/output paths
python convert_xsuite_inputs.py \
    --xsuite-input /path/to/inputs \
    --output /path/to/output \
    --verbose
```

### Conversion Script Location

📍 `convert_xsuite_inputs.py` (root of openPMD-beamphysics project)

### Parameters Extracted per Energy Point

- Circumference (C)
- Number of particles (Np)
- Number of bunches (Nb)
- Energy (E)
- Emittance X (epsn_x)
- Emittance Y (epsn_y)
- Bunch length (sigma_z)
- Energy spread (sigma_e)
- Horizontal tune (Qx)
- Vertical tune (Qy)
- Horizontal chromaticity (chi_x)
- Vertical chromaticity (chi_y)
- Momentum compaction (alpha)
- Dispersion invariants: I2, I3, I5, I6

### Wake Components

1. **Longitudinal** - V/C (voltage per coulomb)
2. **Dipole X** - V/C/m (voltage per coulomb per meter)
3. **Dipole Y** - V/C/m (voltage per coulomb per meter)

### Impedance Data

- **Frequency range**: 1.0e-5 Hz to 1.0e+15 Hz
- **Points per dataset**: 585 frequency points
- **Components**: Real and imaginary impedance

### Verification

Check the conversion manifest:
```bash
cat tests/tests_xsuite/xsuite_pmd/conversion_manifest.json
```

This JSON file lists all converted files and their paths.

### Data Source

All input data sourced from: `tests/tests_xsuite/xsuite_origin/simulation_inputs/`

### Next Steps

- Use converted HDF5 files in openPMD-beamphysics analysis workflows
- Convert simulation outputs when available
- Validate data in openPMD visualization tools

---

**Total Size**: ~45 MB  
**Format**: HDF5 with openPMD metadata  
**Status**: ✅ Ready for use
