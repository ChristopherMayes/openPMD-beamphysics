import numpy as np

from .units import unit, pg_units
from .readers import component_alias


def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)



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
        g2_name = component_alias[key]
        
        # Units
        u = pg_units(key)        
        
        # Write
        g2 = write_component_data(g, g2_name, data[key], unit=u)
    
        
    # Optional id. This does not have any units.
    if 'id' in data or hasattr(data, 'id'):
        g['id'] = data['id']

            
    
    
def write_component_data(h5, name, data, unit=None): 
    """
    Writes data to a dataset h5[name]
    
    If data is a constant array, a group is created with the constant value and shape
    
    If unit is given, this will be used 
    
    """
    # Check for constant component
    if len(np.unique(data)) == 1:
        g = h5.create_group(name)
        g.attrs['value'] = data[0]
        g.attrs['shape'] = data.shape
    else:
        h5[name] = data
        g = h5[name]
    
    if unit:
        g.attrs['unitSI'] = unit.unitSI
        g.attrs['unitDimension'] = unit.unitDimension
        g.attrs['unitSymbol'] = unit.unitSymbol
    
    return g         


def write_complex_component_data(h5, name, data, unit=None):
    """
    Writes complex component data, with real and imaginary parts put into groups:
        r
        i
        
    In the future, this will be an HDF5 struct, and won't need a custom routine.
    
    """
    
    g = h5.create_group(name)
    
    write_component_data(g, 'r', np.real(data), unit=unit)
    write_component_data(g, 'i', np.imag(data), unit=unit )    
    
    return g