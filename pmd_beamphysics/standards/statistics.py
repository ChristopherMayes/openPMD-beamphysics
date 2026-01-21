"""
Statistics Standard module for openPMD-beamphysics.

This module provides tools for loading, validating, and generating documentation
from the particle beam statistics standard defined in statistics_standard.yaml.

Example usage:

    from pmd_beamphysics.statistics_standard import load_standard, validate_standard

    standard = load_standard()
    errors = validate_standard(standard)
"""

from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "YAML_PATH",
    "load_standard",
    "validate_standard",
    "validate_against_particlegroup",
    "generate_markdown",
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
REQUIRED_CATEGORY_FIELDS = ["id", "name", "description"]


def load_standard(path: Path | None = None) -> dict[str, Any]:
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


def validate_standard(standard: dict[str, Any]) -> list[str]:
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


def validate_against_particlegroup(standard: dict[str, Any]) -> list[str]:
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


def generate_markdown(standard: dict[str, Any]) -> str:
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

    # Build category lookup
    categories = {cat["id"]: cat for cat in standard.get("categories", [])}

    # Group statistics by category
    stats_by_category: dict[str, list[dict]] = {cat_id: [] for cat_id in categories}

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

    return "\n".join(lines)


def get_statistic(
    label: str, standard: dict[str, Any] | None = None
) -> dict[str, Any] | None:
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
    category_id: str, standard: dict[str, Any] | None = None
) -> dict[str, Any] | None:
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
