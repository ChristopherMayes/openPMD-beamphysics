#!/usr/bin/env python3
"""
Detailed inspection of the new fccee_h_thick.h5 optics file.
"""

import h5py
import json
from pathlib import Path

h5_file = Path(__file__).parent / 'tests/tests_xsuite/xsuite_pmd/optics/optics_fccee_h_thick.h5'

print("=" * 80)
print("DETAILED INSPECTION: optics_fccee_h_thick.h5")
print("=" * 80)

with h5py.File(h5_file, 'r') as f:
    print("\n📄 File-Level Attributes (openPMD Required & Recommended):")
    print("-" * 80)
    for key in sorted(f.attrs.keys()):
        val = f.attrs[key]
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        
        # Mark required vs recommended
        required = ['openPMD', 'openPMDextension', 'basePath', 'meshesPath', 'particlesPath']
        marker = "✓ REQ" if key in required else "✓ REC"
        
        print(f"  {marker} {key:.<25} {str(val)[:60]}")
    
    print("\n📊 File Structure:")
    print("-" * 80)
    print(f"  Datasets at root: {len([k for k in f.keys() if isinstance(f[k], h5py.Dataset)])}")
    print(f"  Groups at root: {len([k for k in f.keys() if isinstance(f[k], h5py.Group)])}")
    
    print("\n🔍 Content of /opticsData/ group:")
    print("-" * 80)
    if '/opticsData/' in f:
        grp = f['/opticsData/']
        
        print(f"  Metadata Attributes:")
        for key in sorted(grp.attrs.keys()):
            val = grp.attrs[key]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            if key == 'lattice_json_length':
                print(f"    {key:.<30} {val:>10} bytes")
            else:
                val_str = str(val)[:50]
                print(f"    {key:.<30} {val_str}")
        
        print(f"\n  Datasets in group:")
        for key in grp.keys():
            item = grp[key]
            if isinstance(item, h5py.Dataset):
                print(f"    • {key}")
                print(f"      Shape: {item.shape}")
                print(f"      Type: {item.dtype}")
                
                if 'lattice' in key.lower():
                    data = item[()]
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    
                    # Try to parse JSON
                    try:
                        lattice_dict = json.loads(data)
                        print(f"      Format: JSON (valid)")
                        print(f"      Keys: {list(lattice_dict.keys())}")
                        
                        # Count elements
                        elements = lattice_dict.get('elements', {})
                        print(f"      Elements: {len(elements)}")
                        
                        # Get xtrack version
                        version = lattice_dict.get('xtrack_version', 'unknown')
                        print(f"      XSuite Version: {version}")
                        
                    except json.JSONDecodeError:
                        print(f"      Format: Text (not JSON)")
                        print(f"      Content (first 100 chars): {str(data)[:100]}...")

print("\n" + "=" * 80)
print("COMPLIANCE VERIFICATION")
print("=" * 80)

with h5py.File(h5_file, 'r') as f:
    required = ['openPMD', 'openPMDextension', 'basePath', 'meshesPath', 'particlesPath']
    recommended = ['author', 'software', 'softwareVersion', 'date', 'comment']
    
    required_present = sum(1 for attr in required if attr in f.attrs)
    recommended_present = sum(1 for attr in recommended if attr in f.attrs)
    
    print(f"\n✅ Required Attributes: {required_present}/{len(required)}")
    for attr in required:
        status = "✓" if attr in f.attrs else "✗"
        print(f"  {status} {attr}")
    
    print(f"\n✅ Recommended Attributes: {recommended_present}/{len(recommended)}")
    for attr in recommended:
        status = "✓" if attr in f.attrs else "✗"
        print(f"  {status} {attr}")
    
    if required_present == len(required) and recommended_present == len(recommended):
        print(f"\n🎉 FILE IS 100% OPENPMD COMPLIANT!")
    
print("\n" + "=" * 80)
