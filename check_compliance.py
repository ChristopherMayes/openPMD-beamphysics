#!/usr/bin/env python3
"""
openPMD and openPMD Beam Physics Compliance Checker

Validates converted XSuite HDF5 files against openPMD standards and
openPMD beam physics extension requirements.

Usage:
    python check_compliance.py
    python check_compliance.py --fix  # Auto-add missing metadata
    python check_compliance.py --verbose
"""

import h5py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys


# openPMD Required Metadata
OPENPMD_REQUIRED = {
    "openPMD": {"type": str, "description": "openPMD version (e.g., '2.0.0')"},
    "openPMDextension": {"type": str, "description": "Extensions used (e.g., 'BeamPhysics;E-10-0-0')"},
    "basePath": {"type": str, "description": "Path to base group (e.g., '/data/iteration/%T/')"},
    "meshesPath": {"type": str, "description": "Path to mesh groups"},
    "particlesPath": {"type": str, "description": "Path to particle groups"},
}

# openPMD Recommended Metadata
OPENPMD_RECOMMENDED = {
    "author": {"type": str, "description": "Author name"},
    "software": {"type": str, "description": "Software that created the file"},
    "softwareVersion": {"type": str, "description": "Software version"},
    "date": {"type": str, "description": "Creation date (ISO 8601)"},
    "comment": {"type": str, "description": "File description/comment"},
}

# openPMD Beam Physics Extension Requirements
BEAM_PHYSICS_REQUIRED = {
    "species": {"type": str, "description": "Particle species (e.g., 'electron')"},
    "beamStatus": {"type": int, "description": "Beam status code"},
}

# Custom metadata for simulation sources
SIMULATION_METADATA = {
    "sourceFile": {"type": str, "description": "Original source filename"},
    "sourceFormat": {"type": str, "description": "Original format (e.g., 'XSuite JSON')"},
    "conversionDate": {"type": str, "description": "Conversion timestamp"},
    "conversionTool": {"type": str, "description": "Tool used for conversion"},
    "conversionToolVersion": {"type": str, "description": "Version of conversion tool"},
}


class ComplianceChecker:
    """Check openPMD and beam physics compliance."""

    def __init__(self, verbose=False, fix=False):
        self.verbose = verbose
        self.fix = fix
        self.results = {}
        self.xsuite_pmd_dir = Path(__file__).parent / "tests" / "tests_xsuite" / "xsuite_pmd"

    def check_file(self, filepath: Path) -> Dict:
        """Check a single HDF5 file for compliance."""
        result = {
            "file": str(filepath),
            "exists": filepath.exists(),
            "size_mb": filepath.stat().st_size / 1024 / 1024 if filepath.exists() else 0,
            "compliance": {},
            "missing_metadata": [],
            "extra_metadata": [],
            "issues": [],
        }

        if not filepath.exists():
            result["issues"].append(f"File not found: {filepath}")
            return result

        try:
            with h5py.File(filepath, "r+" if self.fix else "r") as f:
                # Check file-level attributes
                result["compliance"]["file_metadata"] = self._check_metadata(
                    f, OPENPMD_REQUIRED, OPENPMD_RECOMMENDED
                )
                result["missing_metadata"].extend(
                    result["compliance"]["file_metadata"].get("missing_required", [])
                )
                result["missing_metadata"].extend(
                    result["compliance"]["file_metadata"].get("missing_recommended", [])
                )

                # Check dataset attributes
                result["compliance"]["datasets"] = self._check_datasets(f)

                # Check structure
                result["compliance"]["structure"] = self._check_structure(f)

        except Exception as e:
            result["issues"].append(f"Error reading file: {str(e)}")

        return result

    def _check_metadata(self, obj, required: Dict, recommended: Dict = None) -> Dict:
        """Check metadata attributes against requirements."""
        result = {
            "present": {},
            "missing_required": [],
            "missing_recommended": [],
            "type_errors": [],
        }

        if recommended is None:
            recommended = {}

        # Check required attributes
        for attr_name, attr_info in required.items():
            if attr_name in obj.attrs:
                value = obj.attrs[attr_name]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                result["present"][attr_name] = {
                    "value": str(value)[:100],  # Truncate for display
                    "type": type(value).__name__,
                }
            else:
                result["missing_required"].append(attr_name)

        # Check recommended attributes
        for attr_name, attr_info in recommended.items():
            if attr_name in obj.attrs:
                value = obj.attrs[attr_name]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                result["present"][attr_name] = {
                    "value": str(value)[:100],
                    "type": type(value).__name__,
                }
            else:
                result["missing_recommended"].append(attr_name)

        # Check for extra attributes not in spec
        for attr_name in obj.attrs:
            if attr_name not in required and attr_name not in recommended:
                result["extra"] = result.get("extra", [])
                result["extra"].append(attr_name)

        return result

    def _check_datasets(self, f: h5py.File) -> Dict:
        """Check all datasets for required attributes."""
        result = {
            "count": 0,
            "with_units": 0,
            "without_units": [],
            "missing_metadata": [],
        }

        def check_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                result["count"] += 1
                if "unit" in obj.attrs or "unitSI" in obj.attrs or "unitDimension" in obj.attrs:
                    result["with_units"] += 1
                else:
                    result["without_units"].append(name)

        f.visititems(check_dataset)
        return result

    def _check_structure(self, f: h5py.File) -> Dict:
        """Check HDF5 structure."""
        result = {
            "root_groups": [],
            "datasets": [],
            "has_groups": len(list(f.keys())) > 0,
        }

        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                result["root_groups"].append(key)
            else:
                result["datasets"].append(key)

        return result

    def add_metadata(self, filepath: Path, metadata: Dict) -> bool:
        """Add metadata to HDF5 file."""
        try:
            with h5py.File(filepath, "r+") as f:
                for key, value in metadata.items():
                    if key not in f.attrs:
                        f.attrs[key] = value
                        if self.verbose:
                            print(f"  Added: {key} = {value}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"  Error adding metadata: {e}")
            return False

    def check_all(self):
        """Check all converted files."""
        print("=" * 80)
        print("openPMD COMPLIANCE CHECKER")
        print("=" * 80)
        print()

        # Find all HDF5 files
        h5_files = list(self.xsuite_pmd_dir.rglob("*.h5"))

        if not h5_files:
            print(f"❌ No HDF5 files found in {self.xsuite_pmd_dir}")
            return

        print(f"Found {len(h5_files)} HDF5 files\n")

        # Check each file
        for filepath in sorted(h5_files):
            relative_path = filepath.relative_to(self.xsuite_pmd_dir)
            print(f"📄 {relative_path}")
            print("-" * 80)

            result = self.check_file(filepath)
            self.results[str(filepath)] = result

            # Display results
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"  ⚠️  {issue}")
            else:
                print(f"  ✓ Size: {result['size_mb']:.2f} MB")

                # File metadata
                file_meta = result["compliance"].get("file_metadata", {})
                present = file_meta.get("present", {})
                missing_req = file_meta.get("missing_required", [])
                missing_rec = file_meta.get("missing_recommended", [])

                if present:
                    print(f"  ✓ Metadata present: {len(present)}")
                    for key, info in present.items():
                        print(f"    - {key}: {info['value']}")

                if missing_req:
                    print(f"  ❌ Missing REQUIRED metadata: {len(missing_req)}")
                    for key in missing_req:
                        print(f"    - {key}")

                if missing_rec:
                    print(f"  ⚠️  Missing RECOMMENDED metadata: {len(missing_rec)}")
                    for key in missing_rec:
                        print(f"    - {key}")

                # Dataset info
                datasets = result["compliance"].get("datasets", {})
                if datasets["count"] > 0:
                    print(f"  ✓ Datasets: {datasets['count']}")
                    print(f"    - With units: {datasets['with_units']}")
                    if datasets["without_units"]:
                        print(f"    - Without units: {len(datasets['without_units'])}")
                        for ds in datasets["without_units"][:3]:
                            print(f"      * {ds}")

                # Structure
                structure = result["compliance"].get("structure", {})
                if structure["root_groups"]:
                    print(f"  ✓ Groups: {', '.join(structure['root_groups'])}")

            print()

        # Summary
        self._print_summary()

        # Generate report
        self._generate_report()

    def _print_summary(self):
        """Print compliance summary."""
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total_files = len(self.results)
        compliant_files = 0
        missing_required_count = 0
        missing_recommended_count = 0

        for result in self.results.values():
            file_meta = result["compliance"].get("file_metadata", {})
            missing_req = file_meta.get("missing_required", [])
            missing_rec = file_meta.get("missing_recommended", [])

            missing_required_count += len(missing_req)
            missing_recommended_count += len(missing_rec)

            if not missing_req:
                compliant_files += 1

        print(f"\nFiles checked: {total_files}")
        print(f"✓ Fully compliant (all required metadata): {compliant_files}/{total_files}")
        print(f"❌ Missing REQUIRED metadata instances: {missing_required_count}")
        print(f"⚠️  Missing RECOMMENDED metadata instances: {missing_recommended_count}")
        print()

        if missing_required_count == 0 and missing_recommended_count == 0:
            print("✅ ALL FILES FULLY COMPLIANT")
        elif missing_required_count == 0:
            print("⚠️  All required metadata present, but missing some recommended metadata")
        else:
            print("❌ Some files missing REQUIRED metadata")

        print()

    def _generate_report(self):
        """Generate compliance report."""
        report_file = self.xsuite_pmd_dir / "compliance_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "checker_version": "1.0",
            "files_checked": len(self.results),
            "results": {},
        }

        for filepath, result in self.results.items():
            report["results"][filepath] = {
                "size_mb": result["size_mb"],
                "compliance": result["compliance"],
                "missing_metadata": result["missing_metadata"],
                "issues": result["issues"],
            }

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ Report saved: {report_file}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check openPMD compliance of converted XSuite files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Automatically add missing metadata"
    )

    args = parser.parse_args()

    checker = ComplianceChecker(verbose=args.verbose, fix=args.fix)
    checker.check_all()


if __name__ == "__main__":
    main()
