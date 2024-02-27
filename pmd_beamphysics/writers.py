import numpy as np

from .units import unit, pg_units
from .readers import component_from_alias, load_field_attrs
from .tools import fstr, encode_attrs


def pmd_init(h5, basePath='/data/%T/', particlesPath='./' ):
    """
    Root attribute initialization.
    
    h5 should be the root of the file.
    """
    d = {
        'basePath':basePath,
        'dataType':'openPMD',
        'openPMD':'2.0.0',
        'openPMDextension':'BeamPhysics;SpeciesType',
        'particlesPath':particlesPath    
    }
    for k,v in d.items():
        h5.attrs[k] = fstr(v)
        
    
def pmd_field_init(h5, externalFieldPath='/ExternalFieldPath/%T/'):
    """
    Root attribute initialization for an openPMD-beamphysics External Field Mesh
    
    h5 should be the root of the file.
    """
    d = {
        'dataType':'openPMD',
        'openPMD':'2.0.0',
        'openPMDextension':'BeamPhysics',
        'externalFieldPath': externalFieldPath
    }
    for k,v in d.items():
        h5.attrs[k] = fstr(v)    
        
    
        
def write_pmd_bunch(h5, data, name=None):
    """
    Data is a dict with:
        np.array: 'x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight'
        str: 'species'
        int: n_particle

    Optional data:
        np.array: 'id'
        
    See inverse routine:
        .particles.load_bunch_data
    
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    # Attributes
    g.attrs['speciesType'] = fstr( data['species'] )
    g.attrs['numParticles'] = data['n_particle']
    g.attrs['totalCharge'] = data['charge']
    g.attrs['chargeUnitSI'] = 1.0
    
    # Required Datasets
    for key in ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight']:
        # Get full name, write data
        g2_name = component_from_alias[key]
        
        # Units
        u = pg_units(key)        
        
        # Write
        write_component_data(g, g2_name, data[key], unit=u)
    
        
    # Optional id. This does not have any units.
    if 'id' in data:
        g['id'] = data['id']

        
def write_pmd_field(h5, data, name=None):
    """
    Data is a dict with:
        attrs: flat dict of attributes. 
        components: flat dict of components
        
    See inverse routine:
        .readers.load_field_data
    
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    # Validate attrs
    attrs, other = load_field_attrs(data['attrs'])

    # Encode and write required and optional
    attrs = encode_attrs(attrs)
    for k, v in attrs.items():
        g.attrs[k] = v
    
    # All other attributes (don't change them)
    for k, v in other.items():
        g.attrs[k] = v
    
    # write components (datasets)
    for key, val in data['components'].items():
        
        # Units
        u = pg_units(key)   
        
        # Ensure complex
        val = val.astype(complex)

        # Write
        write_component_data(g, key, val, unit=u)

    
def write_component_data(h5, name, data, unit=None): 
    """
    Writes data to a dataset h5[name]
    
    If data is a constant array, a group is created with the constant value and shape
    
    If unit is given, this will be used 
    
    """
    # Check for constant component
    dat0 = data[0]
    if np.all(data == dat0):
        g = h5.create_group(name)
        g.attrs['value'] = dat0
        g.attrs['shape'] = data.shape
    else:
        h5[name] = data
        g = h5[name]
        if len(data.shape) > 1:
            g.attrs['gridDataOrder'] = fstr('C') # C order for numpy/h5py
    
    if unit:
        g.attrs['unitSI'] = unit.unitSI
        g.attrs['unitDimension'] = unit.unitDimension
        g.attrs['unitSymbol'] = fstr(unit.unitSymbol)
    
    return g         


