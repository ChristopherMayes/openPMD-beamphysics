"""
Statistics Standard module for openPMD-beamphysics.

This module provides tools for loading, validating, and generating documentation
from the particle beam statistics standard defined in statistics_standard.yaml.

Example usage:

    from pmd_beamphysics.statistics_standard import load_standard, validate_standard

    standard = load_standard()
    errors = validate_standard(standard)
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from pmd_beamphysics.units import pmd_unit

__all__ = [
    "YAML_PATH",
    "load_standard",
    "validate_standard",
    "validate_against_particlegroup",
    "generate_markdown",
    "generate_computed_markdown",
    "load_computed_statistics",
    "get_computed_statistic",
    "export_computed_statistics",
    "ARRAY_KEYS",
    "OPERATORS",
]

# Path to the YAML file (in the same directory as this module)
YAML_PATH = Path(__file__).parent / "statistics.yaml"

# Required fields for validation
REQUIRED_STAT_FIELDS = [
    "label",
    "mathlabel",
    "units",
    "description",
    "reference",
    "category",
]

# ---------------------------------------------------------------------------
# Computed Statistics Configuration
# ---------------------------------------------------------------------------

# Base array keys that can have operators applied
ARRAY_KEYS = """
x y z px py pz t status weight id
z/c
p energy kinetic_energy xp yp higher_order_energy
r theta pr ptheta
Lz
gamma beta beta_x beta_y beta_z
x_bar px_bar Jx Jy
""".split()

# Operators and their properties for computed statistics
OPERATORS = {
    "mean_": {
        "name": "Mean",
        "description_template": "Weighted mean of {base_desc}",
        "mathlabel_template": r"\langle {base_mathlabel} \rangle",
        "reference": "Standard weighted average",
    },
    "sigma_": {
        "name": "Standard Deviation",
        "description_template": "Weighted standard deviation of {base_desc}",
        "mathlabel_template": r"\sigma_{{{base_mathlabel}}}",
        "reference": "Standard weighted standard deviation",
    },
    "min_": {
        "name": "Minimum",
        "description_template": "Minimum value of {base_desc}",
        "mathlabel_template": r"\min({base_mathlabel})",
        "reference": "NumPy min operation",
    },
    "max_": {
        "name": "Maximum",
        "description_template": "Maximum value of {base_desc}",
        "mathlabel_template": r"\max({base_mathlabel})",
        "reference": "NumPy max operation",
    },
    "ptp_": {
        "name": "Peak-to-Peak",
        "description_template": "Peak-to-peak range (max - min) of {base_desc}",
        "mathlabel_template": r"\Delta {base_mathlabel}",
        "reference": "NumPy ptp (peak-to-peak) operation",
    },
    "delta_": {
        "name": "Delta",
        "description_template": "Deviation from mean of {base_desc}",
        "mathlabel_template": r"\Delta {base_mathlabel}",
        "reference": "Particle value minus weighted mean",
    },
}

# Categories for computed statistics
COMPUTED_CATEGORIES = [
    {
        "id": "computed_operators",
        "name": "Operator Statistics",
        "description": "Statistics computed by applying operators (mean, sigma, min, max, ptp, delta) to base quantities.",
    },
    {
        "id": "computed_covariance",
        "name": "Covariance Statistics",
        "description": "Weighted covariance between pairs of base quantities.",
    },
]
REQUIRED_CATEGORY_FIELDS = ["id", "name", "description"]


def load_standard(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and parse the statistics standard YAML file.

    Parameters
    ----------
    path : Path, optional
        Path to the YAML file. If None, uses the bundled statistics_standard.yaml.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    if path is None:
        path = YAML_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_standard(standard: Dict[str, Any]) -> List[str]:
    """
    Validate the statistics standard YAML structure.

    Parameters
    ----------
    standard : dict
        Parsed YAML content from load_standard().

    Returns
    -------
    list of str
        List of error messages. Empty if validation passes.
    """
    errors = []

    # Check schema version
    if "schema_version" not in standard:
        errors.append("Missing 'schema_version' field")

    # Check categories
    if "categories" not in standard:
        errors.append("Missing 'categories' field")
    else:
        category_ids = set()
        for i, cat in enumerate(standard["categories"]):
            for field in REQUIRED_CATEGORY_FIELDS:
                if field not in cat:
                    errors.append(f"Category {i}: missing required field '{field}'")
            if "id" in cat:
                if cat["id"] in category_ids:
                    errors.append(f"Duplicate category id: '{cat['id']}'")
                category_ids.add(cat["id"])

    # Check statistics
    if "statistics" not in standard:
        errors.append("Missing 'statistics' field")
    else:
        labels = set()
        for i, stat in enumerate(standard["statistics"]):
            label = stat.get("label", f"<index {i}>")

            # Check required fields
            for field in REQUIRED_STAT_FIELDS:
                if field not in stat:
                    errors.append(
                        f"Statistic '{label}': missing required field '{field}'"
                    )

            # Check for duplicate labels
            if "label" in stat:
                if stat["label"] in labels:
                    errors.append(f"Duplicate statistic label: '{stat['label']}'")
                labels.add(stat["label"])

            # Check category reference
            if "category" in stat and "categories" in standard:
                if stat["category"] not in category_ids:
                    errors.append(
                        f"Statistic '{label}': unknown category '{stat['category']}'"
                    )

    return errors


def validate_against_particlegroup(standard: Dict[str, Any]) -> List[str]:
    """
    Validate that statistic labels are accessible in ParticleGroup.

    This creates a minimal ParticleGroup and attempts to access each label.

    Parameters
    ----------
    standard : dict
        Parsed YAML content from load_standard().

    Returns
    -------
    list of str
        List of warning messages for inaccessible labels.
    """
    warnings = []

    try:
        from pmd_beamphysics import ParticleGroup
        import numpy as np

        # Create a minimal ParticleGroup for introspection
        data = {
            "x": np.array([0.0]),
            "px": np.array([0.0]),
            "y": np.array([0.0]),
            "py": np.array([0.0]),
            "z": np.array([0.0]),
            "pz": np.array([1e6]),  # Need nonzero pz for some derived quantities
            "t": np.array([0.0]),
            "weight": np.array([1e-12]),
            "status": np.array([1]),
            "species": "electron",
        }
        P = ParticleGroup(data=data)

        for stat in standard.get("statistics", []):
            label = stat.get("label")
            if label is None:
                continue

            # Try to access the attribute
            try:
                _ = P[label]
            except (AttributeError, KeyError, Exception) as e:
                warnings.append(f"Label '{label}' not accessible in ParticleGroup: {e}")

    except ImportError:
        warnings.append("Could not import ParticleGroup for validation")

    return warnings


def _label_to_anchor(label: str) -> str:
    """Convert a label to a valid markdown anchor."""
    # Replace special characters that might interfere with anchors
    return label.replace("/", "").replace(" ", "-").lower()


def generate_markdown(standard: Dict[str, Any]) -> str:
    """
    Generate Markdown documentation from the statistics standard.

    Parameters
    ----------
    standard : dict
        Parsed YAML content from load_standard().

    Returns
    -------
    str
        Markdown-formatted documentation string.
    """
    lines = []

    # Header
    lines.append("# Particle Beam Statistics Standard")
    lines.append("")
    lines.append(
        "This document defines the standard statistics available in `ParticleGroup`."
    )
    lines.append(f"Schema version: **{standard.get('schema_version', 'unknown')}**")
    lines.append("")
    lines.append(
        "See [Statistics Schema](../statistics_schema.md) for documentation on how to extend this standard."
    )
    lines.append("")
    lines.append(
        "Unit dimensions follow the [openPMD unit system](https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/STANDARD.md#unit-systems-and-dimensionality): "
        "`(length, mass, time, current, temperature, amount, luminous intensity)`."
    )
    lines.append("")

    # Build category lookup
    categories = {cat["id"]: cat for cat in standard.get("categories", [])}

    # Group statistics by category
    stats_by_category: Dict[str, List[Dict]] = {cat_id: [] for cat_id in categories}

    for stat in standard.get("statistics", []):
        cat_id = stat.get("category")
        if cat_id in stats_by_category:
            stats_by_category[cat_id].append(stat)

    # Table of contents
    lines.append("## Contents")
    lines.append("")
    for cat_id, cat in categories.items():
        if stats_by_category[cat_id]:
            anchor = cat["name"].lower().replace(" ", "-").replace("/", "")
            lines.append(f"- [{cat['name']}](#{anchor})")
    lines.append("- [Detailed Definitions](#detailed-definitions)")
    lines.append("- [Computed Statistics](#computed-statistics)")
    lines.append("")

    # Generate summary tables for each category
    for cat_id, cat in categories.items():
        stats = stats_by_category[cat_id]
        if not stats:
            continue

        lines.append(f"## {cat['name']}")
        lines.append("")
        lines.append(cat["description"])
        lines.append("")

        # Table header
        lines.append("| Label | Symbol | Units | Description |")
        lines.append("|-------|--------|-------|-------------|")

        for stat in stats:
            label = stat.get("label", "")
            mathlabel = stat.get("mathlabel", "")
            units = stat.get("units", "")
            # Truncate description for table, remove newlines
            desc = stat.get("description", "").replace("\n", " ").strip()
            if len(desc) > 80:
                desc = desc[:77] + "..."

            # Format for table with link to details
            anchor = _label_to_anchor(label)
            label_fmt = f"[`{label}`](#{anchor})"
            math_fmt = f"${mathlabel}$" if mathlabel else ""
            units_fmt = f"`{units}`" if units else ""

            lines.append(f"| {label_fmt} | {math_fmt} | {units_fmt} | {desc} |")

        lines.append("")

    # Detailed definitions section (all categories together)
    lines.append("## Detailed Definitions")
    lines.append("")

    for cat_id, cat in categories.items():
        stats = stats_by_category[cat_id]
        if not stats:
            continue

        lines.append(f"### {cat['name']}")
        lines.append("")

        for stat in stats:
            label = stat.get("label", "")
            mathlabel = stat.get("mathlabel", "")
            units = stat.get("units", "")
            description = stat.get("description", "")
            formula = stat.get("formula")
            reference = stat.get("reference", "")
            reference_url = stat.get("reference_url")

            # Use anchor-friendly id with HTML anchor tag
            anchor = _label_to_anchor(label)
            lines.append(f"#### `{label}` {{: #{anchor} }}")
            lines.append("")
            if mathlabel:
                lines.append(f"**Symbol:** ${mathlabel}$")
                lines.append("")
            lines.append(f"**Units:** `{units}`")
            lines.append("")

            # Add unitSI and unitDimension from pmd_unit
            if units:
                try:
                    u = pmd_unit.from_symbol(units)
                    lines.append(f"**unitSI:** `{u.unitSI}`")
                    lines.append("")
                    lines.append(f"**unitDimension:** `{u.unitDimension}`")
                    lines.append("")
                except (ValueError, KeyError):
                    pass  # Skip if unit cannot be parsed

            lines.append(description)
            lines.append("")
            if formula:
                lines.append("**Formula:**")
                lines.append("")
                lines.append(f"$${formula}$$")
                lines.append("")
            if reference:
                if reference_url:
                    lines.append(f"**Reference:** [{reference}]({reference_url})")
                else:
                    lines.append(f"**Reference:** {reference}")
                lines.append("")
            lines.append("---")
            lines.append("")

    # Computed statistics section
    lines.append("## Computed Statistics")
    lines.append("")
    lines.append("The following prefixes can be applied to any base statistic label:")
    lines.append("")
    lines.append("| Prefix | Example | Description |")
    lines.append("|--------|---------|-------------|")
    lines.append("| `sigma_` | `sigma_x` | Standard deviation $\\sigma$ |")
    lines.append("| `mean_` | `mean_x` | Weighted average $\\langle x \\rangle$ |")
    lines.append("| `min_` | `min_x` | Minimum value |")
    lines.append("| `max_` | `max_x` | Maximum value |")
    lines.append("| `ptp_` | `ptp_x` | Peak-to-peak (max - min) |")
    lines.append(
        "| `delta_` | `delta_x` | Deviation from mean $x - \\langle x \\rangle$ |"
    )
    lines.append("| `cov_X__Y` | `cov_x__px` | Covariance $\\langle X, Y \\rangle$ |")
    lines.append("")
    lines.append(
        "See [Computed Statistics](computed_statistics.md) for a complete enumeration of all computed statistics."
    )
    lines.append("")

    return "\n".join(lines)


def get_statistic(
    label: str, standard: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Look up a statistic by label.

    Parameters
    ----------
    label : str
        The label to look up.
    standard : dict, optional
        Parsed YAML content. If None, loads the bundled standard.

    Returns
    -------
    dict or None
        The statistic entry, or None if not found.
    """
    if standard is None:
        standard = load_standard()

    for stat in standard.get("statistics", []):
        if stat.get("label") == label:
            return stat
    return None


def get_category(
    category_id: str, standard: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Look up a category by ID.

    Parameters
    ----------
    category_id : str
        The category ID to look up.
    standard : dict, optional
        Parsed YAML content. If None, loads the bundled standard.

    Returns
    -------
    dict or None
        The category entry, or None if not found.
    """
    if standard is None:
        standard = load_standard()

    for cat in standard.get("categories", []):
        if cat.get("id") == category_id:
            return cat
    return None


# ---------------------------------------------------------------------------
# Computed Statistics Functions
# ---------------------------------------------------------------------------


def _multiply_units(unit1: str, unit2: str) -> str:
    """Multiply two unit strings together."""
    if unit1 == "1" and unit2 == "1":
        return "1"
    if unit1 == "1":
        return unit2
    if unit2 == "1":
        return unit1

    # Handle common simplifications
    if unit1 == unit2:
        return f"{unit1}^2"

    return f"{unit1}*{unit2}"


def _generate_computed_statistics_list(base_stats: Dict[str, Dict]) -> List[Dict]:
    """
    Generate all computed statistics from operators and base keys.

    Parameters
    ----------
    base_stats : dict
        Dictionary mapping label -> statistic dict from the base standard.

    Returns
    -------
    list of dict
        List of computed statistic entries.
    """
    computed = []

    # Generate operator + key combinations
    for op_prefix, op_info in OPERATORS.items():
        for key in ARRAY_KEYS:
            base = base_stats.get(key)
            if not base:
                continue

            label = f"{op_prefix}{key}"
            base_mathlabel = base.get("mathlabel", key)
            base_desc = base.get("description", key).lower()

            # Handle units (same as base for all current operators)
            units = base.get("units", "1")

            stat = {
                "label": label,
                "mathlabel": op_info["mathlabel_template"].format(
                    base_mathlabel=base_mathlabel
                ),
                "units": units,
                "description": op_info["description_template"].format(
                    base_desc=base_desc
                ),
                "reference": op_info["reference"],
                "category": "computed_operators",
                "base_statistic": key,
                "operator": op_prefix.rstrip("_"),
            }
            computed.append(stat)

    # Generate covariance combinations (cov_{key1}__{key2})
    for key1 in ARRAY_KEYS:
        base1 = base_stats.get(key1)
        if not base1:
            continue

        for key2 in ARRAY_KEYS:
            base2 = base_stats.get(key2)
            if not base2:
                continue

            label = f"cov_{key1}__{key2}"
            mathlabel1 = base1.get("mathlabel", key1)
            mathlabel2 = base2.get("mathlabel", key2)

            # Covariance units are product of the two base units
            units1 = base1.get("units", "1")
            units2 = base2.get("units", "1")
            units = _multiply_units(units1, units2)

            desc1 = base1.get("description", key1).lower().rstrip(".")
            desc2 = base2.get("description", key2).lower().rstrip(".")

            stat = {
                "label": label,
                "mathlabel": rf"\langle {mathlabel1} \cdot {mathlabel2} \rangle - \langle {mathlabel1} \rangle \langle {mathlabel2} \rangle",
                "units": units,
                "description": f"Weighted covariance between {desc1} and {desc2}.",
                "reference": "Standard weighted covariance",
                "category": "computed_covariance",
                "base_statistics": [key1, key2],
            }
            computed.append(stat)

    return computed


@lru_cache(maxsize=1)
def load_computed_statistics() -> Dict[str, Any]:
    """
    Generate and return computed statistics derived from base statistics.

    This function generates statistics for all combinations of:
    - Operators (mean_, sigma_, min_, max_, ptp_, delta_) with array keys
    - Covariance combinations (cov_{key1}__{key2})

    The result is cached, so subsequent calls return the same dictionary.

    Returns
    -------
    dict
        Dictionary with 'schema_version', 'description', 'categories', and 'statistics' keys,
        following the same format as the base statistics standard.

    Example
    -------
    >>> computed = load_computed_statistics()
    >>> len(computed['statistics'])  # Number of computed statistics
    1147
    >>> computed['statistics'][0]['label']  # First computed statistic
    'mean_x'
    """
    # Load base statistics
    base_standard = load_standard()
    base_stats = {stat["label"]: stat for stat in base_standard.get("statistics", [])}

    # Generate computed statistics
    computed_list = _generate_computed_statistics_list(base_stats)

    return {
        "schema_version": "1.0",
        "description": "Auto-generated computed statistics derived from base statistics and operators.",
        "categories": COMPUTED_CATEGORIES,
        "statistics": computed_list,
    }


def get_computed_statistic(label: str) -> Optional[Dict]:
    """
    Look up a computed statistic by its label.

    Parameters
    ----------
    label : str
        The statistic label to look up (e.g., 'sigma_x', 'cov_x__px').

    Returns
    -------
    dict or None
        The statistic entry, or None if not found.

    Example
    -------
    >>> stat = get_computed_statistic('sigma_x')
    >>> stat['mathlabel']
    '\\\\sigma_{x}'
    >>> stat['units']
    'm'
    """
    computed = load_computed_statistics()
    for stat in computed.get("statistics", []):
        if stat.get("label") == label:
            return stat
    return None


def export_computed_statistics(path: Union[Path, str]) -> None:
    """
    Export computed statistics to a YAML file.

    This is useful for debugging or inspection of the generated statistics.

    Parameters
    ----------
    path : Path or str
        Path to write the YAML file.

    Example
    -------
    >>> export_computed_statistics('computed_statistics.yaml')
    """
    computed = load_computed_statistics()
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            computed,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )


def generate_computed_markdown() -> str:
    """
    Generate Markdown documentation for computed statistics.

    Returns
    -------
    str
        Markdown-formatted documentation string.
    """
    computed = load_computed_statistics()
    lines = []

    # Header
    lines.append("# Computed Statistics Reference")
    lines.append("")
    lines.append(
        "This document lists all computed statistics available in `ParticleGroup`."
    )
    lines.append("")
    lines.append(
        "These are derived from [base statistics](statistics_standard.md) by applying operators."
    )
    lines.append("")

    # Summary
    stats = computed.get("statistics", [])
    n_operators = sum(1 for s in stats if s["category"] == "computed_operators")
    n_covariance = sum(1 for s in stats if s["category"] == "computed_covariance")
    lines.append(f"**Total:** {len(stats)} computed statistics")
    lines.append("")
    lines.append(
        f"- {n_operators} operator statistics (mean, sigma, min, max, ptp, delta)"
    )
    lines.append(f"- {n_covariance} covariance statistics")
    lines.append("")

    # Table of contents
    lines.append("## Contents")
    lines.append("")
    lines.append("- [Operators](#operators)")
    for op_prefix, op_info in OPERATORS.items():
        op_name = op_info["name"]
        anchor = op_name.lower().replace(" ", "-").replace("-", "-")
        lines.append(f"    - [{op_name}](#{anchor})")
    lines.append("- [Covariances](#covariances)")
    lines.append("")

    # Operators section
    lines.append("## Operators")
    lines.append("")
    lines.append("The following operators can be applied to any base array statistic:")
    lines.append("")

    # Table of operators
    lines.append("| Operator | Symbol Pattern | Description |")
    lines.append("|----------|----------------|-------------|")
    for op_prefix, op_info in OPERATORS.items():
        mathlabel = op_info["mathlabel_template"].format(base_mathlabel="X")
        desc = op_info["description_template"].format(base_desc="X")
        lines.append(f"| `{op_prefix}` | ${mathlabel}$ | {desc} |")
    lines.append("")

    # Generate tables for each operator
    for op_prefix, op_info in OPERATORS.items():
        op_name = op_info["name"]
        lines.append(f"### {op_name}")
        lines.append("")
        lines.append(
            op_info["description_template"].format(base_desc="base quantities") + "."
        )
        lines.append("")

        # Filter stats for this operator
        op_stats = [s for s in stats if s.get("operator") == op_prefix.rstrip("_")]

        lines.append("| Label | Symbol | Units | Base |")
        lines.append("|-------|--------|-------|------|")
        for stat in op_stats:
            label = stat.get("label", "")
            mathlabel = stat.get("mathlabel", "")
            units = stat.get("units", "")
            base = stat.get("base_statistic", "")

            math_fmt = f"${mathlabel}$" if mathlabel else ""
            units_fmt = f"`{units}`" if units else ""
            base_link = f"[`{base}`](statistics_standard.md#{_label_to_anchor(base)})"

            lines.append(f"| `{label}` | {math_fmt} | {units_fmt} | {base_link} |")

        lines.append("")

    # Covariances section
    lines.append("## Covariances")
    lines.append("")
    lines.append("Covariances are computed between all pairs of base array statistics.")
    lines.append("")
    lines.append("**Format:** `cov_{X}__{Y}` where X and Y are base statistic labels.")
    lines.append("")
    lines.append(
        "**Symbol:** $\\langle X Y \\rangle - \\langle X \\rangle \\langle Y \\rangle$"
    )
    lines.append("")
    lines.append("**Units:** Product of the units of X and Y.")
    lines.append("")

    # Create a summary table of base keys
    lines.append("### Available Base Keys")
    lines.append("")
    lines.append(
        "Covariances can be computed between any pair of these base statistics:"
    )
    lines.append("")
    lines.append("| Key | Symbol | Units |")
    lines.append("|-----|--------|-------|")

    base_standard = load_standard()
    base_stats = {s["label"]: s for s in base_standard.get("statistics", [])}
    for key in ARRAY_KEYS:
        if key in base_stats:
            stat = base_stats[key]
            mathlabel = stat.get("mathlabel", key)
            units = stat.get("units", "")
            key_link = f"[`{key}`](statistics_standard.md#{_label_to_anchor(key)})"
            lines.append(f"| {key_link} | ${mathlabel}$ | `{units}` |")

    lines.append("")

    # Full enumeration of all covariances
    lines.append("### All Covariances")
    lines.append("")
    lines.append("| Label | Symbol | Units |")
    lines.append("|-------|--------|-------|")

    cov_stats = [s for s in stats if s.get("category") == "computed_covariance"]
    for stat in cov_stats:
        label = stat.get("label", "")
        mathlabel = stat.get("mathlabel", "")
        units = stat.get("units", "")
        math_fmt = f"${mathlabel}$" if mathlabel else ""
        units_fmt = f"`{units}`" if units else ""
        lines.append(f"| `{label}` | {math_fmt} | {units_fmt} |")

    lines.append("")

    return "\n".join(lines)
