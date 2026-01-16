# XSuite Test Data Analysis & Conversion Strategy

## Overview

The `xsuite_origin` folder contains simulation input data (not output) for the FCC-ee HEB (High Energy Beam) project. These inputs need to be converted to **openPMD-compatible format** for testing the xsuite-openPMD-beamphysics integration.

---

## Folder Structure Analysis

### `xsuite_origin/`
```
xsuite_origin/
├── simulation_inputs/          # ← Contains actual test data
│   ├── impedances_30mm/        # Impedance frequency-domain data
│   ├── optics/                 # Machine lattice definitions
│   ├── parameters_table/       # Machine parameters (JSON)
│   └── wake_potential/         # Wake function time-domain data
└── simulation_outputs/         # ← Currently EMPTY (target for outputs)
```

---

## Input Data Details

### 1. Machine Parameters (`parameters_table/`)

**File:** `Booster_parameter_table.json`

**Content:** Complete FCC-ee booster machine specification with nested structure:

```json
{
  "version": "PA31-3.0",
  "C": { "value": 90658.74531999999, "unit": "km", "comments": "circumference" },
  "Np": { 
    "z": 25000000000.0,
    "w": 25000000000.0,
    "zh": 10000000000,
    "ttbar": 10000000000,
    "unit": "",
    "comments": "Number of particles per bunch"
  },
  "E": {
    "injection": 20000000000.0,      # 20 GeV
    "z": 45600000000.0,               # 45.6 GeV (z-boson)
    "w": 80000000000.0,               # 80 GeV (W-boson)
    "zh": 120000000000.0,             # 120 GeV
    "ttbar": 182500000000.0,          # 182.5 GeV (top-antitop)
    "unit": "eV"
  },
  "bunch": {
    "epsnx": { "value": 1e-5, "unit": "m" },     # Normalized horizontal emittance
    "epsny": { "value": 1e-5, "unit": "m" },     # Normalized vertical emittance
    "sigmaz": { "value": 0.004, "unit": "m" },   # Bunch length
    "sigmae": { "value": 0.001, "unit": "" }     # Energy spread
  },
  "optics": {
    "Qx": { "z": 414.225, ... },                 # Horizontal tune
    "Qy": { "w": 410.29, ... },                  # Vertical tune
    "chix": { ... },                              # Horizontal chromaticity
    "chiy": { ... },                              # Vertical chromaticity
    "alpha": { ... },                             # Momentum compaction
    "I2", "I3", "I5": { ... },                   # Synchrotron integrals
    "damp_xy", "damp_s": { ... },                # Damping times
    "coupling": { ... }                           # Horizontal-vertical coupling
  },
  "RF": {
    "RF_freq": { "value": 800000000.0, "unit": "Hz" },
    "Vtot": { "value": 50084569.67, "unit": "Volt" },
    "phis_inj": { "value": 178.47, "unit": "degree" },
    "phis_ext": { ... },
    "Qs_inj", "Qs_ext": { ... }                  # Synchrotron tunes
  },
  "beam_pipe": {
    "shape": "Circular",
    "material": "Copper",
    "D": { "value": 0.06, "unit": "m" }
  }
}
```

**Key Features:**
- Multi-energy specification (injection, z, w, zh, ttbar)
- Separated optics parameters per energy point
- Detailed RF parameters
- Beam pipe geometry

---

### 2. Optics Files (`optics/`)

**Files:**
- `PA31_2023_02_09_10_23.json` — Machine lattice model (JSON-based, xsuite format)
- `PA31_2024_07_08_10_22.json` — Updated lattice model
- `heb_ring_withcav.json` — Ring with cavities
- `heb_ring_withcav.seq` — Sequence format (legacy)

**Format:** XSuite lattice description (likely `Twiss`-compatible JSON or `.seq` format)

---

### 3. Wake Potential Data (`wake_potential/`)

**Files:**
- `heb_wake_round_cu_30.0mm.csv` — Primary wake function (15,615 lines)
- `wake_long_copper_PyHT.txt` — Longitudinal copper wake
- `wake_long_stainless_PyHT.txt` — Longitudinal stainless steel wake
- `wake_tr_copper_PyHT.txt` — Transverse copper wake
- `wake_tr_stainless_PyHT.txt` — Transverse stainless steel wake

**Example Data (heb_wake_round_cu_30.0mm.csv):**

```
# ECSV 1.0 format (Extensible Character Separated Values)
# Origin: IW2D simulation
# Author: Ali Rajabi
# Date: 2024/06/01
# Comments: Resistive wall, copper, round pipe, 30mm diameter

time           longitudinal  dipole_x         dipole_y
(s)            (V/C)         (V/C/m)          (V/C/m)
-3.3356        -0.0771       0.00420          0.00420
-0.1701        -7.8861       0.01024          0.01024
-0.1701        -7.8053       0.00970          0.00970
...            ...           ...              ...
```

**Structure:**
- **Time-domain** representation of wake functions
- **Frequency-independent** (raw CST/PyHT simulation output)
- **3 components:** longitudinal + dipole_x + dipole_y
- **Units:** V/C (longitudinal), V/C/m (transverse)
- **Format:** ECSV (standardized ASCII table format used in astropy)

---

### 4. Impedance Tables (`impedances_30mm/`)

**Files:**
- `impedance_Cu_Round_30.0mm.csv` — Copper impedance table
- `impedance_SS_Round_30.0mm.csv` — Stainless steel impedance table
- `Wake_Cu_Round_30.0mm.csv` — Wake data (copper)
- `Wake_SS_Round_30.0mm.csv` — Wake data (stainless steel)
- `Wake_Bunch_Cu_Round_30.0mm_0.2mm.csv` — Bunch-wide wake
- `Wake_Bunch_SS_Round_30.0mm_0.2mm.csv` — Bunch-wide wake
- `output_long_2phdt.txt` — Longitudinal impedance (frequency domain)
- `output_tr_2phdt.txt` — Transverse impedance (frequency domain)

**Format:** Frequency-domain impedance (Re + Im parts)

---

## Conversion Strategy for openPMD Compatibility

### Phase 1: Convert Machine Parameters
**Input:** `Booster_parameter_table.json`
**Output:** `xsuite_pmd/machine_parameters.h5`

```python
# Function: convert_machine_parameters()
# Logic:
#   1. Parse JSON -> extract energy levels for target state
#   2. Extract optics (Twiss: βx, βy, αx, αy)
#   3. Write to HDF5 using xsuite_io.write_machine_parameters()
#   4. Add metadata (beam pipe, RF, damping times)
```

### Phase 2: Convert Wake Functions
**Input:** `wake_potential/*.csv`
**Output:** `xsuite_pmd/wakes_*.h5`

```python
# Function: convert_wake_potential()
# Logic:
#   1. Read CSV (ECSV format) using astropy.io.ascii
#   2. Extract z, longitudinal, dipole_x, dipole_y arrays
#   3. Write 3 components using xsuite_io.write_wake_table():
#      - component='longitudinal', wake_unit='V/C'
#      - component='dipole_x', wake_unit='V/C/m'
#      - component='dipole_y', wake_unit='V/C/m'
#   4. Add metadata: source='IW2D', date, author
```

### Phase 3: Convert Impedance Tables
**Input:** `impedances_30mm/*.csv`
**Output:** `xsuite_pmd/impedance_*.h5`

```python
# Function: convert_impedance()
# Logic:
#   1. Read frequency-domain data
#   2. Separate into real/imag components
#   3. Write using xsuite_io.write_impedance_table():
#      - plane='longitudinal' or plane='x' / plane='y'
#   4. Add source metadata
```

### Phase 4: Create Particle Test Data (Synthetic)
**Output:** `xsuite_pmd/test_particles.h5`

```python
# Function: generate_test_particles()
# Logic:
#   1. Use machine parameters to initialize beam
#   2. Generate Gaussian distribution:
#      - N_particles = 1e5 (for fast testing)
#      - Emit_x = 1e-5 m (from parameters)
#      - Emit_y = 1e-5 m
#      - σ_z = 0.004 m (bunch length)
#      - σ_e / E = 0.001 (energy spread)
#   3. Write as ParticleGroup → openPMD HDF5
#   4. Include Twiss parameters in file attributes
```

---

## Implementation Plan

### Step 1: Create Conversion Utilities Module

**File:** `pmd_beamphysics/interfaces/xsuite_conversion.py`

```python
"""
Convert XSuite simulation inputs/outputs to openPMD format.
"""

def convert_machine_parameters(json_path, h5_output, energy_key='ttbar'):
    """Convert Booster_parameter_table.json → machine_parameters.h5"""
    pass

def convert_wake_potential(csv_path, h5_output, material='copper'):
    """Convert wake CSV → openPMD wake HDF5"""
    pass

def convert_impedance(csv_path, h5_output, plane='longitudinal'):
    """Convert impedance CSV → openPMD impedance HDF5"""
    pass

def generate_test_particles(machine_params, n_particles=1e5, h5_output='test_particles.h5'):
    """Generate synthetic Gaussian bunch for testing"""
    pass
```

### Step 2: Add Test Cases

**File:** `tests/tests_xsuite/test_conversion.py`

```python
"""
Test conversion of XSuite inputs to openPMD format.
"""

class TestConversionUtilities:
    def test_convert_machine_parameters(self, xsuite_origin_path):
        """Test machine parameter conversion"""
        pass
    
    def test_convert_wake_potential(self, xsuite_origin_path):
        """Test wake function conversion"""
        pass
    
    def test_convert_impedance(self, xsuite_origin_path):
        """Test impedance table conversion"""
        pass
    
    def test_round_trip_consistency(self, xsuite_origin_path):
        """Test: convert → read back → verify"""
        pass
```

### Step 3: Document Data Flow

**File:** `tests/tests_xsuite/README.md`

```markdown
# XSuite ↔ OpenPMD Test Suite

## Input Data (`xsuite_origin/`)
- Machine parameters: Booster_parameter_table.json
- Wake functions: wake_potential/*.csv (ECSV format)
- Impedances: impedances_30mm/*.csv (frequency domain)
- Lattice: optics/*.json (xsuite format)

## Test Workflow
1. Load machine parameters → extract energy, optics
2. Load wake functions → format for openPMD
3. Load impedances → format for openPMD
4. Generate synthetic particles → Gaussian bunch
5. Round-trip test: write → read → verify
6. Statistical invariance test: emittance, Twiss unchanged

## Output Data (`xsuite_pmd/`)
- machine_parameters.h5 (openPMD format)
- wakes_*.h5 (openPMD format, 3 components)
- impedance_*.h5 (openPMD format)
- test_particles.h5 (synthetic Gaussian bunch)
```

---

## Data Type Mapping: XSuite → openPMD

| XSuite Quantity | XSuite Format | openPMD Mapping | Unit |
|-----------------|---------------|-----------------|------|
| Beam energy | JSON (eV) | `machineParameters/energy` | eV |
| Emittance | JSON (m) | `machineParameters/emittance_*` | m (normalized) |
| Bunch length | JSON (m) | `machineParameters/bunch_length` | m |
| Circumference | JSON (km) | `machineParameters/circumference` | m |
| Tune (Qx, Qy) | JSON (unitless) | `machineParameters/tune_x/y` | – |
| Beta function | JSON (m) | `machineParameters/beta_x/y` | m |
| Wake (long) | CSV (V/C) | `wakeData/longitudinal/wake` | V/C |
| Wake (dipole) | CSV (V/C/m) | `wakeData/dipole_x/y/wake` | V/C/m |
| Impedance (Re) | CSV (Ω) | `impedanceData/*/real` | Ω |
| Impedance (Im) | CSV (Ω) | `impedanceData/*/imag` | Ω |
| Particle x, px, y, py, zeta, delta | Synthetic | `particles/x, px, y, py, zeta, delta` | SI units |

---

## Expected Output Structure

```
xsuite_pmd/
├── machine_parameters.h5
│   └── /machineParameters/
│       ├── @energy = 182.5e9 eV
│       ├── @circumference = 97750 m
│       ├── @emittance_x = 1e-5 m
│       ├── @emittance_y = 1e-5 m
│       ├── @beta_x = 0.125 m
│       ├── @beta_y = 0.068 m
│       └── [other Twiss parameters]
│
├── wakes_copper_30mm.h5
│   ├── /wakeData/longitudinal/
│   │   ├── @component = 'longitudinal'
│   │   ├── @wake_unit = 'V/C'
│   │   ├── z[15615]
│   │   └── wake[15615]
│   ├── /wakeData/dipole_x/
│   │   ├── @component = 'dipole_x'
│   │   ├── @wake_unit = 'V/C/m'
│   │   ├── z[15615]
│   │   └── wake[15615]
│   └── /wakeData/dipole_y/
│       └── [same as dipole_x]
│
├── impedance_copper_30mm.h5
│   ├── /impedanceData/longitudinal/
│   │   ├── @plane = 'longitudinal'
│   │   ├── frequency[...]
│   │   ├── real[...]
│   │   └── imag[...]
│   ├── /impedanceData/x/
│   └── /impedanceData/y/
│
└── test_particles_gaussian.h5
    ├── @openPMD = '1.1.0'
    ├── @openPMDextension = 'XSuite'
    └── /data/particles/
        ├── x, y, z (position) [m]
        ├── px, py, pz (momentum) [kg·m/s]
        ├── weight (macroparticle weight)
        ├── species (electron/positron)
        └── [metadata: Twiss, emittance, etc.]
```

---

## Testing Checklist

- [ ] Load Booster_parameter_table.json → extract all energy states
- [ ] Convert wake CSV → validate 3 components read back
- [ ] Convert impedance CSV → validate complex impedance preserved
- [ ] Round-trip test: write → read → compare arrays (< 1e-12 tolerance)
- [ ] Metadata preservation: source, date, author fields survive
- [ ] Generate synthetic particles → check statistical moments
- [ ] Run test suite with pytest
- [ ] Validate openPMD compliance (check with openPMD validator if available)

---

## References

- **XSuite I/O Module:** `/pmd_beamphysics/interfaces/xsuite_io.py`
- **Test Framework:** `/tests/tests_xsuite/test_xsuite_io.py`
- **Input Data:** `/tests/tests_xsuite/xsuite_origin/`
- **ECSV Format:** https://github.com/astropy/astropy/blob/main/astropy/io/ascii/ecsv.py
- **openPMD Standard:** https://github.com/openPMD/openPMD-standard

