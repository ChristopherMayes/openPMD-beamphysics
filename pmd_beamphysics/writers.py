import numpy as np

from .units import unit, pg_units
from .readers import component_alias


def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)



def pmd_init(h5, basePath='/screen/%T/', particlesPath='/' ):
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
    
    # Datasets
    for key in ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight']:
        # Get full name, write data
        g2_name = component_alias[key]
        g2 = write_component_data(g, g2_name, data[key])
        
        # Units
        unit_name = pg_units(key)
        u = unit[unit_name]
        #print(u.unitSymbol, u.unitSI, u.unitDimension)
        g2.attrs['unitSI'] = u.unitSI
        g2.attrs['unitDimension'] = u.unitDimension
        g2.attrs['unitSymbol'] = u.unitSymbol
    
    
    
def write_component_data(h5, name, data): 
    """
    Writes data to a dataset h5[name]
    
    If data is a constant array, a group is created with the constant value and shape
    
    """
    # Check for constant component
    if len(np.unique(data)) == 1:
        g = h5.create_group(name)
        g.attrs['value'] = data[0]
        g.attrs['shape'] = data.shape
    else:
        h5[name] = data
        g = h5[name]
    
    return g         
