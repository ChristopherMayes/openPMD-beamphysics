#!/usr/bin/env python3
"""
Generate statistics documentation from the YAML standard.

This is a convenience wrapper around the pmd_beamphysics.statistics_standard module.

Usage:
    python scripts/generate_statistics_docs.py [--validate] [--validate-code] [--output PATH]

Options:
    --validate       Only validate the YAML file without generating docs
    --validate-code  Also validate labels against ParticleGroup
    --output         Output path for generated markdown (default: docs/api/statistics_standard.md)
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from pmd_beamphysics.standards.statistics import (
    YAML_PATH,
    load_standard,
    validate_standard,
    validate_against_particlegroup,
    generate_markdown,
    generate_computed_markdown,
)

# Default output paths
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "api" / "statistics_standard.md"
DEFAULT_COMPUTED_OUTPUT = REPO_ROOT / "docs" / "api" / "computed_statistics.md"


def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics documentation from YAML standard"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate the YAML file without generating docs",
    )
    parser.add_argument(
        "--validate-code",
        action="store_true",
        help="Also validate labels against ParticleGroup",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for generated markdown",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=YAML_PATH,
        help="Path to statistics_standard.yaml",
    )

    args = parser.parse_args()

    # Load YAML
    print(f"Loading {args.yaml}...")
    try:
        standard = load_standard(args.yaml)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        sys.exit(1)

    # Validate
    print("Validating schema...")
    errors = validate_standard(standard)

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("Schema validation passed.")

    # Optional code validation
    if args.validate_code:
        print("Validating against ParticleGroup...")
        warnings = validate_against_particlegroup(standard)
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("Code validation passed.")

    if args.validate:
        print("Validation complete.")
        sys.exit(0)

    # Generate markdown for base statistics
    print("Generating base statistics documentation...")
    markdown = generate_markdown(standard)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"  Written to {args.output}")

    # Generate markdown for computed statistics
    print("Generating computed statistics documentation...")
    computed_markdown = generate_computed_markdown()

    with open(DEFAULT_COMPUTED_OUTPUT, "w", encoding="utf-8") as f:
        f.write(computed_markdown)
    print(f"  Written to {DEFAULT_COMPUTED_OUTPUT}")

    print("Done!")


if __name__ == "__main__":
    main()
