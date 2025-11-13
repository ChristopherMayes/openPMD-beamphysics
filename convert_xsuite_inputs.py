#!/usr/bin/env python3
"""
Convert XSuite simulation inputs to openPMD HDF5 format.

This script converts FCC-ee simulation data (machine parameters, wake functions,
impedances) from XSuite format to openPMD-compliant HDF5 files in the xsuite_pmd/
directory hierarchy.

Usage:
    python convert_xsuite_inputs.py
    
    or with custom paths:
    
    python convert_xsuite_inputs.py \
        --xsuite-input /path/to/xsuite_heb/input_data \
        --output ./xsuite_pmd
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pmd_beamphysics.interfaces.xsuite_conversion import (
    convert_machine_parameters,
    convert_wake_potential,
    convert_impedance,
    convert_optics,
)


def find_impedance_files(impedance_dir):
    """Find all impedance CSV files in the impedances_30mm directory."""
    impedance_dir = Path(impedance_dir)
    if not impedance_dir.exists():
        return {}
    
    files = {}
    for csv_file in impedance_dir.glob("*.csv"):
        # Parse filename to extract material and plane info
        name = csv_file.stem
        if "impedance" not in name.lower():
            continue  # Skip wake files, only get impedance files
            
        if "Cu" in name or "copper" in name.lower():
            material = "copper"
        elif "SS" in name or "stainless" in name.lower():
            material = "stainless"
        else:
            material = "unknown"
        
        # All impedance files in impedances_30mm are longitudinal by default
        key = f"{material}_longitudinal"
        files[key] = csv_file
    
    return files


def main():
    """Main conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert XSuite simulation inputs to openPMD format"
    )
    parser.add_argument(
        "--xsuite-input",
        type=Path,
        default=Path(__file__).parent / "tests" / "tests_xsuite" / "xsuite_origin" / "simulation_inputs",
        help="Path to XSuite input_data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "tests" / "tests_xsuite" / "xsuite_pmd",
        help="Output directory for openPMD files",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="XSuite",
        help="Author name for metadata (default: 'XSuite')",
    )
    parser.add_argument(
        "--energy-points",
        nargs="+",
        default=["z", "w", "zh", "ttbar"],
        help="Energy points to convert",
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        default=["copper"],
        help="Materials to convert (for wakes and impedance)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    xsuite_input = Path(args.xsuite_input).expanduser().resolve()
    if not xsuite_input.exists():
        print(f"❌ Error: XSuite input directory not found: {xsuite_input}")
        return 1
    
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"📂 XSuite input directory: {xsuite_input}")
        print(f"📁 Output directory: {output_dir}")
        print(f"👤 Author: {args.author}")
        print()
    
    # ========== CONVERT MACHINE PARAMETERS ==========
    print("🔄 Converting machine parameters...")
    machine_dir = output_dir / "machine"
    machine_dir.mkdir(exist_ok=True)
    
    # For xsuite_origin, we have Booster_parameter_table.json with multiple energy points
    # Available energy points: z, w, zh, ttbar
    energy_files = {
        "z": "parameters_table/Booster_parameter_table.json",
        "w": "parameters_table/Booster_parameter_table.json",
        "zh": "parameters_table/Booster_parameter_table.json",
        "ttbar": "parameters_table/Booster_parameter_table.json",
    }
    
    converted_energies = {}
    for energy_point in args.energy_points:
        if energy_point not in energy_files:
            continue
            
        json_file = xsuite_input / energy_files[energy_point]
        if not json_file.exists():
            if args.verbose:
                print(f"  ⚠️  Skipping {energy_point}: file not found")
            continue
        
        try:
            output_file = machine_dir / f"machine_{energy_point}.h5"
            convert_machine_parameters(
                str(json_file),
                str(output_file),
                energy_point=energy_point,
                author=args.author,
            )
            converted_energies[energy_point] = str(output_file)
            if args.verbose:
                print(f"  ✅ {energy_point}: {output_file.name}")
        except Exception as e:
            if args.verbose:
                print(f"  ❌ Error converting {energy_point}: {e}")
    
    print(f"✅ Machine parameters: {len(converted_energies)}/{len(args.energy_points)} converted\n")
    
    # ========== CONVERT WAKE POTENTIALS ==========
    print("🔄 Converting wake potentials...")
    wakes_dir = output_dir / "wakes"
    wakes_dir.mkdir(exist_ok=True)
    
    wake_files = {
        "copper": "wake_potential/heb_wake_round_cu_30.0mm.csv",
        "stainless": "wake_potential/wake_long_stainless_PyHT.txt",
    }
    
    converted_wakes = {}
    for material, filename in wake_files.items():
        if material not in args.materials:
            continue
        
        wake_file = xsuite_input / filename
        if not wake_file.exists():
            if args.verbose:
                print(f"  ⚠️  Skipping {material}: {filename} not found")
            continue
        
        try:
            output_file = wakes_dir / f"wake_{material}.h5"
            convert_wake_potential(
                str(wake_file),
                str(output_file),
                material=material,
                author=args.author,
            )
            converted_wakes[material] = str(output_file)
            if args.verbose:
                print(f"  ✅ {material}: {output_file.name}")
        except Exception as e:
            if args.verbose:
                print(f"  ❌ Error converting {material} wakes: {e}")
    
    print(f"✅ Wake potentials: {len(converted_wakes)}/{len(args.materials)} converted\n")
    
    # ========== CONVERT IMPEDANCES ==========
    print("🔄 Converting impedance data...")
    impedance_dir = output_dir / "impedance"
    impedance_dir.mkdir(exist_ok=True)
    
    impedance_input_dir = xsuite_input / "impedances_30mm"
    impedance_files = find_impedance_files(impedance_input_dir)
    
    converted_impedances = 0
    for key, impedance_file in impedance_files.items():
        try:
            output_file = impedance_dir / f"impedance_{key}.h5"
            convert_impedance(
                str(impedance_file),
                str(output_file),
                plane="longitudinal",  # All impedance files in xsuite_origin are longitudinal
                author=args.author,
            )
            converted_impedances += 1
            if args.verbose:
                print(f"  ✅ {key}: {output_file.name}")
        except Exception as e:
            if args.verbose:
                print(f"  ❌ Error converting {key}: {e}")
    
    print(f"✅ Impedances: {converted_impedances} files converted\n")
    
    # ========== CONVERT OPTICS ==========
    print("🔄 Converting optics/lattice data...")
    optics_dir = output_dir / "optics"
    optics_dir.mkdir(exist_ok=True)
    
    optics_input_dir = xsuite_input / "optics"
    converted_optics = {}
    
    if optics_input_dir.exists():
        optics_files = sorted(optics_input_dir.glob("*.json"))
        for optics_file in optics_files:
            try:
                # Use filename stem as optics identifier
                optics_name = optics_file.stem
                output_file = optics_dir / f"optics_{optics_name}.h5"
                optics_info = convert_optics(
                    str(optics_file),
                    str(output_file),
                    author=args.author,
                )
                converted_optics[optics_name] = {
                    'path': str(output_file),
                    'n_elements': optics_info.get('n_elements', 0),
                    'total_length': optics_info.get('total_length', 0),
                }
                if args.verbose:
                    print(f"  ✅ {optics_name}: {output_file.name}")
            except Exception as e:
                if args.verbose:
                    print(f"  ❌ Error converting {optics_file.name}: {e}")
    else:
        if args.verbose:
            print(f"  ⚠️  Optics directory not found: {optics_input_dir}")
    
    print(f"✅ Optics: {len(converted_optics)} files converted\n")
    
    # ========== SUMMARY ==========
    print("=" * 70)
    print("📊 CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Machine parameters:  {len(converted_energies):>3} energy points")
    print(f"Wake potentials:     {len(converted_wakes):>3} materials")
    print(f"Impedances:          {converted_impedances:>3} files")
    print(f"Optics/Lattices:     {len(converted_optics):>3} files")
    print(f"Author:              {args.author}")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    
    # Create conversion manifest
    manifest = {
        "conversion_timestamp": datetime.now().isoformat(),
        "xsuite_input_dir": str(xsuite_input),
        "output_dir": str(output_dir),
        "author": args.author,
        "machine_parameters": converted_energies,
        "wake_potentials": converted_wakes,
        "impedances": converted_impedances,
        "optics": converted_optics,
    }
    
    manifest_file = output_dir / "conversion_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Manifest saved: {manifest_file.name}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
