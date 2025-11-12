# Quick Reference: Updated XSuite Conversion Tools

## Overview

The XSuite input conversion utilities now produce **100% openPMD-compliant** output with automatic metadata generation.

---

## Key Features

✅ **Author Metadata** - Configurable author name for all outputs  
✅ **Date Extraction** - Automatic ISO 8601 timestamps from source files  
✅ **Full Compliance** - All 10 openPMD attributes (5 required + 5 recommended)  
✅ **Backward Compatible** - Existing code works without changes  
✅ **Production Ready** - 7/7 files verified compliant

---

## Quick Start

### Basic Usage

```bash
cd openPMD-beamphysics
python convert_xsuite_inputs.py --author "Your Organization" --verbose
```

### With Custom Paths

```bash
python convert_xsuite_inputs.py \
    --xsuite-input /path/to/xsuite/input_data \
    --output ./openpmd_data \
    --author "FCC-ee Team" \
    --verbose
```

### Selective Conversion

```bash
python convert_xsuite_inputs.py \
    --energy-points ttbar zh \
    --materials copper \
    --author "My Team"
```

---

## CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--xsuite-input` | Path | `tests/tests_xsuite/xsuite_origin/simulation_inputs` | Input directory |
| `--output` | Path | `tests/tests_xsuite/xsuite_pmd` | Output directory |
| `--author` | String | `"XSuite"` | **NEW:** Author metadata |
| `--energy-points` | List | `z w zh ttbar` | Energy points to convert |
| `--materials` | List | `copper` | Materials to convert |
| `--verbose` / `-v` | Flag | False | Verbose output |

---

## Conversion Output

### Files Generated (7 total)

```
xsuite_pmd/
├── machine/
│   ├── machine_z.h5          (45.6 GeV)    [7.8 KB]
│   ├── machine_w.h5          (80.0 GeV)    [7.8 KB]
│   ├── machine_zh.h5         (120.0 GeV)   [7.8 KB]
│   └── machine_ttbar.h5      (182.5 GeV)   [7.8 KB]
├── wakes/
│   └── wake_copper.h5        (15,599 pts)  [754 KB]
├── impedance/
│   ├── impedance_copper_longitudinal.h5     [24 KB]
│   └── impedance_stainless_longitudinal.h5  [24 KB]
└── conversion_manifest.json   (metadata)
```

### Metadata Included in Each File

```
File-Level Attributes (10 total):
  ✓ openPMD = "1.1.0"
  ✓ openPMDextension = "beamPhysics"
  ✓ basePath = "/xsuite/"
  ✓ meshesPath = "simulationData/"
  ✓ particlesPath = "particleData/"
  ✓ author = "FCC-ee Collective Effects Team"
  ✓ software = "xsuite_io"
  ✓ softwareVersion = "1.0"
  ✓ date = "2024-08-26T11:41:32Z"  (ISO 8601, UTC)
  ✓ comment = "FCC-ee booster machine parameters in openPMD format"
```

---

## Date Extraction

### How It Works

1. **Source file modification time** → Extracted from input file
2. **Convert to ISO 8601** → Format: `YYYY-MM-DDTHH:MM:SSZ`
3. **Store in HDF5** → File-level `date` attribute
4. **Example dates:**
   - Machine params: `2024-08-26T11:41:32Z`
   - Wake data: `2024-09-04T13:02:47Z`
   - Impedance: `2024-04-15T13:20:52Z`

### Override with Custom Date

```python
from pmd_beamphysics.interfaces.xsuite_conversion import convert_machine_parameters

convert_machine_parameters(
    'params.json',
    'output.h5',
    author='My Team',
    date='2025-01-15T10:30:00Z'  # Override with custom date
)
```

---

## Compliance Verification

### Check File Compliance

```bash
python inspect_outputs.py
```

**Output Example:**
```
================================================================================
Files checked: 7
Fully compliant: 7/7
Missing required attributes (total): 0
Missing recommended attributes (total): 0

✅ ALL FILES FULLY COMPLIANT!
```

### Manual Inspection

```python
import h5py

with h5py.File('machine_ttbar.h5', 'r') as f:
    print("Author:", f.attrs['author'])
    print("Date:", f.attrs['date'])
    print("Software:", f.attrs['software'])
    print("Comment:", f.attrs['comment'])
```

---

## API Changes

### Updated Function Signatures

**Before:**
```python
convert_machine_parameters(json_path, h5_output, energy_point='ttbar')
convert_wake_potential(csv_path, h5_output, material='copper')
convert_impedance(csv_path, h5_output, plane='longitudinal')
```

**After (with new optional parameters):**
```python
convert_machine_parameters(
    json_path, 
    h5_output, 
    energy_point='ttbar',
    author='XSuite',              # NEW
    date=None,                     # NEW (auto-extracted if None)
    verbose=True
)

convert_wake_potential(
    csv_path,
    h5_output,
    material='copper',
    author='XSuite',              # NEW
    date=None,                     # NEW
    verbose=True
)

convert_impedance(
    csv_path,
    h5_output,
    plane='longitudinal',
    author='XSuite',              # NEW
    date=None,                     # NEW
    verbose=True
)
```

---

## Example Workflow

### Step 1: Convert with Your Organization

```bash
python convert_xsuite_inputs.py \
    --author "CERN Accelerator School" \
    --verbose
```

### Step 2: Verify Compliance

```bash
python inspect_outputs.py
```

### Step 3: Use in Your Analysis

```python
import h5py

# Open any converted file
with h5py.File('tests/tests_xsuite/xsuite_pmd/machine/machine_ttbar.h5', 'r') as f:
    print("Metadata:")
    print(f"  Author: {f.attrs['author']}")
    print(f"  Date: {f.attrs['date']}")
    print(f"  Software: {f.attrs['software']}")
    print(f"  openPMD: {f.attrs['openPMD']}")
    
    # Access machine parameters
    params = dict(f['/machineParameters/'].attrs)
    print(f"\nMachine Parameters ({len(params)} total):")
    for key, value in params.items():
        print(f"  {key}: {value}")
```

---

## Conversion Manifest

The `conversion_manifest.json` includes:

```json
{
  "conversion_timestamp": "2025-11-12T18:33:53.985700",
  "xsuite_input_dir": "...",
  "output_dir": "...",
  "author": "FCC-ee Collective Effects Team",
  "machine_parameters": {
    "z": "...",
    "w": "...",
    "zh": "...",
    "ttbar": "..."
  },
  "wake_potentials": {
    "copper": "..."
  },
  "impedances": 2
}
```

---

## Troubleshooting

### Issue: Author not appearing in files

**Solution:** Check CLI parameter spelling:
```bash
python convert_xsuite_inputs.py --author "My Organization"
                                         ^^^^^^^^ (not --Author or -a)
```

### Issue: Dates show wrong timezone

**Solution:** Dates are stored in UTC (Z suffix). This is correct per openPMD standard.

### Issue: Files show as "not compliant"

**Solution:** Run conversion again and verify output:
```bash
python convert_xsuite_inputs.py --author "Your Name" --verbose
python inspect_outputs.py
```

---

## Files Modified/Created

| File | Status | Purpose |
|------|--------|---------|
| `pmd_beamphysics/interfaces/xsuite_io.py` | ✏️ UPDATED | Added author/date parameters |
| `pmd_beamphysics/interfaces/xsuite_conversion.py` | ✏️ UPDATED | Added author/date support |
| `convert_xsuite_inputs.py` | ✏️ UPDATED | Added `--author` CLI parameter |
| `inspect_outputs.py` | ✨ NEW | Compliance verification tool |
| `CONVERSION_UPDATE_COMPLETE.md` | ✨ NEW | Detailed documentation |

---

## Status

```
✅ Production Ready
✅ 100% Compliance (7/7 files)
✅ All Metadata Present
✅ Author Support Implemented
✅ Date Extraction Working
✅ Verified & Tested
```

---

## Next Steps

1. **Use converted files** in your openPMD workflows
2. **Re-run conversion** if source data updates (dates auto-extract)
3. **Verify compliance** anytime with `inspect_outputs.py`
4. **Check manifest** for conversion history and metadata

---

**Documentation Updated:** November 12, 2025  
**Status:** ✅ Complete and Tested  
**Compliance Level:** 100% openPMD v1.1.0
