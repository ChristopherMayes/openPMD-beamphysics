#!/usr/bin/env python
"""
Export computed statistics to a YAML file for inspection/debugging.

This script uses the runtime-generated computed statistics from the
pmd_beamphysics.standards.statistics module and exports them to a YAML file.

Usage:
    python scripts/generate_computed_statistics.py [output_path]

If no output path is specified, writes to pmd_beamphysics/standards/computed_statistics.yaml
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pmd_beamphysics.standards.statistics import (  # noqa: E402
    export_computed_statistics,
    load_computed_statistics,
)


def main():
    # Determine output path
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = (
            project_root / "pmd_beamphysics" / "standards" / "computed_statistics.yaml"
        )

    print("Generating computed statistics...")
    computed = load_computed_statistics()

    n_operators = sum(
        1 for s in computed["statistics"] if s["category"] == "computed_operators"
    )
    n_covariance = sum(
        1 for s in computed["statistics"] if s["category"] == "computed_covariance"
    )

    print(f"  Generated {len(computed['statistics'])} computed statistics")
    print(f"    - {n_operators} operator statistics")
    print(f"    - {n_covariance} covariance statistics")

    print(f"Writing to {output_path}...")
    export_computed_statistics(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
