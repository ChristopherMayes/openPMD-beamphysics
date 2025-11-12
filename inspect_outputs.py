#!/usr/bin/env python3
"""
Inspect converted openPMD HDF5 files for metadata and compliance.
"""

import h5py
import json
from pathlib import Path

def inspect_file(filepath):
    """Inspect a single HDF5 file."""
    filepath = Path(filepath)
    
    with h5py.File(filepath, 'r') as f:
        # File-level attributes
        file_attrs = dict(f.attrs)
        
        # Convert bytes to strings for display
        for key in file_attrs:
            if isinstance(file_attrs[key], bytes):
                file_attrs[key] = file_attrs[key].decode('utf-8')
        
        # Count datasets and groups
        num_datasets = sum(1 for item in f.values() if isinstance(item, h5py.Dataset))
        num_groups = sum(1 for item in f.values() if isinstance(item, h5py.Group))
        
        # File size
        file_size = filepath.stat().st_size
        
        return {
            'file': filepath.name,
            'size_kb': file_size / 1024,
            'datasets': num_datasets,
            'groups': num_groups,
            'attributes': file_attrs
        }

def check_compliance(file_attrs):
    """Check if file has required openPMD attributes."""
    required = ['openPMD', 'openPMDextension', 'basePath', 'meshesPath', 'particlesPath']
    recommended = ['author', 'software', 'softwareVersion', 'date', 'comment']
    
    missing_required = [attr for attr in required if attr not in file_attrs]
    missing_recommended = [attr for attr in recommended if attr not in file_attrs]
    
    return {
        'required_present': len(required) - len(missing_required),
        'required_total': len(required),
        'recommended_present': len(recommended) - len(missing_recommended),
        'recommended_total': len(recommended),
        'missing_required': missing_required,
        'missing_recommended': missing_recommended,
    }

def main():
    # Find all HDF5 files
    output_dir = Path(__file__).parent / "tests" / "tests_xsuite" / "xsuite_pmd"
    h5_files = sorted(output_dir.rglob("*.h5"))
    
    if not h5_files:
        print(f"❌ No HDF5 files found in {output_dir}")
        return 1
    
    print(f"\n{'='*80}")
    print(f"INSPECTING CONVERTED OPENPMD FILES")
    print(f"{'='*80}")
    print(f"Location: {output_dir}\n")
    
    # Inspect each file
    results = []
    compliance_summary = {
        'total_files': 0,
        'compliant_files': 0,
        'total_required_missing': 0,
        'total_recommended_missing': 0,
    }
    
    for h5_file in h5_files:
        print(f"\n📄 {h5_file.relative_to(output_dir)}")
        print("-" * 80)
        
        info = inspect_file(h5_file)
        compliance = check_compliance(info['attributes'])
        
        # Display file info
        print(f"  Size: {info['size_kb']:.1f} KB")
        print(f"  Datasets: {info['datasets']}, Groups: {info['groups']}")
        print()
        
        # Display attributes
        print("  📋 File-level Attributes:")
        attr_table = []
        for key, value in sorted(info['attributes'].items()):
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            attr_table.append([f"    {key}", str(value)[:60]])
        
        if attr_table:
            for row in attr_table:
                print(f"{row[0]:<30} = {row[1]}")
        
        # Compliance check
        print()
        print("  ✅ Compliance Check:")
        print(f"    Required attributes: {compliance['required_present']}/{compliance['required_total']}")
        if compliance['missing_required']:
            print(f"      ❌ Missing: {', '.join(compliance['missing_required'])}")
        print(f"    Recommended attributes: {compliance['recommended_present']}/{compliance['recommended_total']}")
        if compliance['missing_recommended']:
            print(f"      ⚠️  Missing: {', '.join(compliance['missing_recommended'])}")
        
        # Update summary
        compliance_summary['total_files'] += 1
        if not compliance['missing_required'] and not compliance['missing_recommended']:
            compliance_summary['compliant_files'] += 1
            print(f"    🎉 FULLY COMPLIANT!")
        
        compliance_summary['total_required_missing'] += len(compliance['missing_required'])
        compliance_summary['total_recommended_missing'] += len(compliance['missing_recommended'])
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Files checked: {compliance_summary['total_files']}")
    print(f"Fully compliant: {compliance_summary['compliant_files']}/{compliance_summary['total_files']}")
    print(f"Missing required attributes (total): {compliance_summary['total_required_missing']}")
    print(f"Missing recommended attributes (total): {compliance_summary['total_recommended_missing']}")
    
    if compliance_summary['compliant_files'] == compliance_summary['total_files']:
        print(f"\n✅ ALL FILES FULLY COMPLIANT!")
        return 0
    else:
        print(f"\n⚠️  Some files have missing attributes")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
