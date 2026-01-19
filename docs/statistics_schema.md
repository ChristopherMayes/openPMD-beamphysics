# Statistics Standard Schema

This document describes the YAML schema used to define particle beam statistics in openPMD-beamphysics. The schema provides a structured, maintainable format for documenting statistical quantities with their mathematical definitions, units, and references.

## Schema Version

The current schema version is **1.0**.

The schema version is specified at the top level of the YAML file:

```yaml
schema_version: "1.0"
```

### Versioning Policy

- **Major version** (X.0): Breaking changes that require updates to existing entries
- **Minor version** (1.X): Additive changes (new optional fields, new categories)

## File Location

The statistics standard is defined in:

```
pmd_beamphysics/statistics_standard/statistics_standard.yaml
```

## Top-Level Structure

```yaml
schema_version: "1.0"

categories:
  - id: <category_id>
    name: <Category Name>
    description: <Category description>

statistics:
  - label: <statistic_label>
    mathlabel: <LaTeX string>
    units: <unit string>
    description: <verbose description>
    reference: <reference string>
    category: <category_id>
    # Optional fields:
    formula: <LaTeX formula>
    reference_url: <URL>
    aliases: [<alias1>, <alias2>]
```

## Categories

Categories group related statistics for organized documentation. Each category has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier used to reference from statistics |
| `name` | string | Yes | Human-readable category name |
| `description` | string | Yes | Brief description of the category |

### Standard Categories

| ID | Name | Description |
|----|------|-------------|
| `coordinates` | Phase Space Coordinates | Basic position and momentum coordinates |
| `time` | Time Coordinate | Time-related quantities |
| `relativistic` | Relativistic Quantities | Energy, momentum, velocity factors |
| `slopes` | Transverse Slopes | Angular divergence quantities |
| `polar` | Polar/Cylindrical Coordinates | Cylindrical coordinate representations |
| `normalized` | Normalized Coordinates | Courant-Snyder normalized coordinates |
| `emittance` | Emittance | Phase space volume measures |
| `twiss` | Twiss Parameters | Courant-Snyder lattice functions |
| `beam` | Beam Properties | Integrated beam properties |
| `particle` | Particle Properties | Individual particle properties |

## Statistics Entry Fields

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `label` | string | Code key used to access the statistic in ParticleGroup | `"norm_emit_x"` |
| `mathlabel` | string | LaTeX string for mathematical symbol (without `$` delimiters) | `"\epsilon_{n,x}"` |
| `units` | string | Unit string compatible with `pmd_unit` | `"m"`, `"eV/c"`, `"1"` |
| `description` | string | Verbose human-readable description | `"Normalized RMS emittance..."` |
| `reference` | string | Citation or reference for the definition | `"Courant & Snyder, Ann. Phys. 3 (1958)"` |
| `category` | string | Category ID this statistic belongs to | `"emittance"` |

### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `formula` | string | LaTeX formula for derived quantities | `"\epsilon = \sqrt{\det(\Sigma)}"` |
| `reference_url` | string | URL/DOI link to the reference | `"https://doi.org/10.1016/..."` |
| `aliases` | list[string] | Alternative labels for the same quantity | `["emit_x", "emittance_x"]` |

## Field Validation Rules

### `label`

- Must be a valid Python identifier or contain only alphanumeric characters, underscores, and `/`
- Must be unique across all statistics entries
- Should match the attribute/key name used in `ParticleGroup`
- Examples: `"x"`, `"norm_emit_x"`, `"z/c"`, `"twiss_beta_x"`

### `mathlabel`

- Must be valid LaTeX (will be wrapped in `$...$` for rendering)
- Use raw strings or escape backslashes properly in YAML
- Common patterns:
  - Greek letters: `\alpha`, `\beta`, `\gamma`, `\epsilon`
  - Subscripts: `x_0`, `p_x`, `\beta_x`
  - Overlines: `\overline{x}`
  - Text in math: `\text{kinetic}`, `\mathrm{avg}`

### `units`

- Must be a unit string recognized by `pmd_unit` or a compound expression
- Use `"1"` for dimensionless quantities
- Standard units:
  - Length: `"m"`
  - Time: `"s"`
  - Energy: `"eV"`
  - Momentum: `"eV/c"`
  - Charge: `"C"`
  - Current: `"A"`
  - Angle: `"rad"`
- Compound units: `"m*eV/c"`, `"m^2"`, `"1/m"`
- Special: `"sqrt(m)"` for normalized coordinates

### `description`

- Should be a complete sentence or phrase
- Include physical meaning and context
- Use YAML multiline syntax for long descriptions:

```yaml
description: >-
  Normalized RMS emittance in the horizontal plane.
  This quantity is invariant under linear symplectic transport.
```

### `formula`

- LaTeX math expression (without delimiters)
- Use standard notation consistent with references
- Include defining equations for derived quantities

### `reference`

- Free-form string for citation
- Preferred format: `"Author, Journal Volume, Pages (Year)"`
- For internal standards: `"openPMD-beamphysics standard"`

### `reference_url`

- Full URL (preferably DOI link)
- Format: `"https://doi.org/10.xxxx/xxxxx"`

## Example Entry

```yaml
- label: norm_emit_x
  mathlabel: \epsilon_{n,x}
  units: m
  description: >-
    Normalized RMS emittance in the horizontal plane.
    Invariant under linear transport for a relativistic beam.
  formula: \epsilon_{n,x} = \frac{1}{mc} \sqrt{\langle x^2 \rangle \langle p_x^2 \rangle - \langle x p_x \rangle^2}
  reference: "Wiedemann, Particle Accelerator Physics (Springer)"
  reference_url: https://doi.org/10.1007/978-3-319-18317-6
  category: emittance
```

## Adding New Statistics

To add a new statistic:

1. Identify the appropriate category (or create a new one if needed)
2. Add an entry to the `statistics` list with all required fields
3. Include `formula` if the quantity is derived from other quantities
4. Add `reference_url` if a DOI or stable URL is available
5. Run the validation script to check for errors:
   ```bash
   python scripts/generate_statistics_docs.py --validate
   ```
6. Regenerate the documentation:
   ```bash
   python scripts/generate_statistics_docs.py
   ```

## Computed Statistics

The `ParticleGroup` class supports computed statistics via bracket access with prefixes:

| Prefix | Example | Computes |
|--------|---------|----------|
| `sigma_` | `P['sigma_x']` | Standard deviation |
| `mean_` | `P['mean_x']` | Weighted average |
| `min_` | `P['min_x']` | Minimum value |
| `max_` | `P['max_x']` | Maximum value |
| `ptp_` | `P['ptp_x']` | Peak-to-peak (max - min) |
| `delta_` | `P['delta_x']` | Value minus mean |
| `cov_X__Y` | `P['cov_x__px']` | Covariance element |

These derived quantities do not need individual entries in the YAML file, as their units and labels are computed automatically from the base quantity.

## Integration with Code

The statistics standard YAML file can be loaded and used programmatically:

```python
import yaml
from pathlib import Path

# Load the standard
yaml_path = Path(__file__).parent / "statistics_standard.yaml"
with open(yaml_path) as f:
    standard = yaml.safe_load(f)

# Access statistics
for stat in standard['statistics']:
    print(f"{stat['label']}: {stat['units']}")
```
