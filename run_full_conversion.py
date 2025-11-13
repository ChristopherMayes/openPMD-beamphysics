#!/usr/bin/env python3
"""
Convert all new XSuite files (bunch, ecloud) to openPMD format.
"""
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from pmd_beamphysics.interfaces.xsuite_conversion import convert_all_xsuite_data

# Conversion parameters
xsuite_origin = 'tests/tests_xsuite/xsuite_origin'
output_pmd = 'tests/tests_xsuite/xsuite_pmd'

print("\n" + "=" * 80)
print(" XSuite to openPMD: FULL BATCH CONVERSION")
print("=" * 80)
print(f"\nInput:  {xsuite_origin}")
print(f"Output: {output_pmd}")

# Run conversion
result = convert_all_xsuite_data(
    xsuite_input_dir=xsuite_origin,
    output_dir=output_pmd,
    energy_point='ttbar',
    n_test_particles=100000,
    verbose=True
)

print("\n" + "=" * 80)
print(" CONVERSION RESULTS")
print("=" * 80)

# Display results
if result.get('machine_params'):
    print(f"\n✓ Machine Parameters:")
    print(f"  {result['machine_params']}")

if result.get('wakes'):
    print(f"\n✓ Wake Potentials:")
    for material, path in result['wakes'].items():
        print(f"  {material}: {path}")

if result.get('impedances'):
    print(f"\n✓ Impedance Tables:")
    for material, path in result['impedances'].items():
        print(f"  {material}: {path}")

if result.get('bunch'):
    print(f"\n✓ Bunch Data:")
    for name, path in result['bunch'].items():
        print(f"  {name}: {path}")

if result.get('ecloud'):
    print(f"\n✓ Electron Cloud Data:")
    for name, path in result['ecloud'].items():
        print(f"  {name}: {path}")

if result.get('test_particles'):
    print(f"\n✓ Test Particles:")
    print(f"  {result['test_particles']}")

print("\n" + "=" * 80)
print(" Conversion completed successfully!")
print("=" * 80 + "\n")
