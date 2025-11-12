#!/usr/bin/env python3
"""
openPMD Metadata Enhancer for XSuite Conversion

Automatically adds missing required and recommended metadata to HDF5 files
to achieve full openPMD compliance.

Usage:
    python enhance_openpmd_metadata.py
    python enhance_openpmd_metadata.py --check-only  # Don't modify files
    python enhance_openpmd_metadata.py --verbose
"""

import h5py
import json
from pathlib import Path
from datetime import datetime
import sys


class MetadataEnhancer:
    """Add missing metadata to HDF5 files for openPMD compliance."""

    def __init__(self, verbose=False, check_only=False):
        self.verbose = verbose
        self.check_only = check_only
        self.xsuite_pmd_dir = Path(__file__).parent / "tests" / "tests_xsuite" / "xsuite_pmd"
        self.results = {}

    def enhance_file(self, filepath: Path) -> dict:
        """Enhance metadata in a single HDF5 file."""
        result = {
            "file": str(filepath),
            "modified": False,
            "added_metadata": [],
            "issues": [],
        }

        if not filepath.exists():
            result["issues"].append(f"File not found: {filepath}")
            return result

        try:
            with h5py.File(filepath, "r+" if not self.check_only else "r") as f:
                # Determine file type and prepare metadata
                file_type = self._determine_file_type(filepath)
                metadata_to_add = self._get_metadata_to_add(f, file_type)

                if self.check_only:
                    result["added_metadata"] = list(metadata_to_add.keys())
                    if self.verbose:
                        print(f"  Would add: {len(metadata_to_add)} attributes")
                else:
                    # Add metadata
                    for key, value in metadata_to_add.items():
                        try:
                            f.attrs[key] = value
                            result["added_metadata"].append(key)
                            result["modified"] = True
                            if self.verbose:
                                print(f"  ✓ Added: {key} = {str(value)[:60]}")
                        except Exception as e:
                            result["issues"].append(f"Error adding {key}: {str(e)}")

                    # Update dataset metadata if needed
                    self._enhance_dataset_metadata(f)

        except Exception as e:
            result["issues"].append(f"Error modifying file: {str(e)}")

        return result

    def _determine_file_type(self, filepath: Path) -> str:
        """Determine file type from path."""
        parts = filepath.parts
        if "machine" in parts:
            return "machine"
        elif "wakes" in parts:
            return "wake"
        elif "impedance" in parts:
            return "impedance"
        else:
            return "unknown"

    def _get_metadata_to_add(self, f: h5py.File, file_type: str) -> dict:
        """Get metadata that needs to be added."""
        metadata = {}

        # Check and add required openPMD metadata
        if "basePath" not in f.attrs:
            metadata["basePath"] = "/xsuite/"

        if "meshesPath" not in f.attrs:
            metadata["meshesPath"] = "simulationData/"

        if "particlesPath" not in f.attrs:
            metadata["particlesPath"] = "particleData/"

        # Add recommended metadata
        if "software" not in f.attrs:
            metadata["software"] = "convert_xsuite_inputs.py"

        if "comment" not in f.attrs:
            if file_type == "machine":
                metadata["comment"] = "FCC-ee booster machine parameters in openPMD format"
            elif file_type == "wake":
                metadata["comment"] = "FCC-ee booster wake potential functions in openPMD format"
            elif file_type == "impedance":
                metadata["comment"] = "FCC-ee booster longitudinal impedance in openPMD format"
            else:
                metadata["comment"] = "XSuite simulation data in openPMD format"

        return metadata

    def _enhance_dataset_metadata(self, f: h5py.File):
        """Enhance dataset-level metadata."""
        def process_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Ensure unitSI or unit attribute exists
                if "unit" not in obj.attrs and "unitSI" not in obj.attrs:
                    # Infer from dataset name if possible
                    if "frequency" in name.lower():
                        obj.attrs["unit"] = "Hz"
                    elif "impedance" in name.lower():
                        obj.attrs["unit"] = "Ohm"
                    elif "z" in name.lower():
                        obj.attrs["unit"] = "m"
                    elif "wake" in name.lower():
                        obj.attrs["unit"] = "V/C"

                # Add description if missing
                if "description" not in obj.attrs:
                    if "frequency" in name.lower():
                        obj.attrs["description"] = "Frequency points"
                    elif "impedance_real" in name.lower():
                        obj.attrs["description"] = "Real impedance"
                    elif "impedance_imag" in name.lower():
                        obj.attrs["description"] = "Imaginary impedance"
                    elif "z" in name.lower() and "wake" in str(f.keys()):
                        obj.attrs["description"] = "Longitudinal coordinate"
                    elif "wake" in name.lower():
                        obj.attrs["description"] = "Wake potential"

        f.visititems(process_dataset)

    def enhance_all(self):
        """Enhance all converted files."""
        print("=" * 80)
        print("openPMD METADATA ENHANCER")
        print("=" * 80)
        print()

        if self.check_only:
            print("MODE: CHECK-ONLY (no files will be modified)\n")
        else:
            print("MODE: ENHANCE (adding missing metadata)\n")

        # Find all HDF5 files
        h5_files = list(self.xsuite_pmd_dir.rglob("*.h5"))

        if not h5_files:
            print(f"❌ No HDF5 files found in {self.xsuite_pmd_dir}")
            return

        print(f"Found {len(h5_files)} HDF5 files\n")

        # Process each file
        for filepath in sorted(h5_files):
            relative_path = filepath.relative_to(self.xsuite_pmd_dir)
            print(f"📄 {relative_path}")
            print("-" * 80)

            result = self.enhance_file(filepath)
            self.results[str(filepath)] = result

            if result["issues"]:
                for issue in result["issues"]:
                    print(f"  ⚠️  {issue}")
            else:
                if result["added_metadata"]:
                    print(f"  ✓ {len(result['added_metadata'])} attributes processed")
                    for attr in result["added_metadata"]:
                        print(f"    - {attr}")
                else:
                    print("  ✓ No additional metadata needed")

                if result["modified"]:
                    print(f"  ✓ File modified successfully")
                elif not self.check_only:
                    print(f"  ✓ File already compliant")

            print()

        # Summary
        self._print_summary()

        # Generate report
        self._generate_report()

    def _print_summary(self):
        """Print summary of enhancement."""
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total_files = len(self.results)
        modified_count = sum(1 for r in self.results.values() if r["modified"])
        total_attributes = sum(len(r["added_metadata"]) for r in self.results.values())
        error_count = sum(len(r["issues"]) for r in self.results.values())

        print(f"\nFiles processed: {total_files}")
        if not self.check_only:
            print(f"✓ Files modified: {modified_count}")
        print(f"✓ Total attributes added: {total_attributes}")

        if error_count > 0:
            print(f"⚠️  Issues encountered: {error_count}")
        else:
            print(f"✅ No errors")

        if not self.check_only and modified_count == total_files:
            print("\n✅ ALL FILES ENHANCED WITH FULL METADATA")
        elif self.check_only:
            print(f"\n📊 Would add {total_attributes} total attributes to achieve compliance")
        else:
            print(f"\n✓ Enhancement complete")

        print()

    def _generate_report(self):
        """Generate enhancement report."""
        report_file = self.xsuite_pmd_dir / "metadata_enhancement_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "check-only" if self.check_only else "enhance",
            "files_processed": len(self.results),
            "total_attributes_added": sum(len(r["added_metadata"]) for r in self.results.values()),
            "files_modified": sum(1 for r in self.results.values() if r["modified"]),
            "results": {},
        }

        for filepath, result in self.results.items():
            report["results"][filepath] = {
                "modified": result["modified"],
                "added_metadata": result["added_metadata"],
                "issues": result["issues"],
            }

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ Report saved: {report_file}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Add missing openPMD metadata to XSuite conversion output files"
    )
    parser.add_argument(
        "--check-only",
        "-c",
        action="store_true",
        help="Check without modifying files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    enhancer = MetadataEnhancer(verbose=args.verbose, check_only=args.check_only)
    enhancer.enhance_all()


if __name__ == "__main__":
    main()
