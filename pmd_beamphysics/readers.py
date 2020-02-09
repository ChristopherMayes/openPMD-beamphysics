from .units import dimension, dimension_name, SI_symbol, pg_units, c_light, e_charge

import numpy as np


#-----------------------------------------
# Records, components, units

particle_record_components = {
    'branchIndex':None,
    'chargeState':None,
    'electricField':['x', 'y', 'z'],
    'elementIndex':None,
    'magneticField':['x', 'y', 'z'],
    'locationInElement': None,
    'momentum':['x', 'y', 'z'],
    'momentumOffset':['x', 'y', 'z'],
    'photonPolarizationAmplitude':['x', 'y'],
    'photonPolarizationPhase':['x', 'y'],
    'sPosition':None,
    'totalMomentum':None,
    'totalMomentumOffset':None,
    #'particleCoordinatesToGlobalTransformation': ??
    'particleStatus':None,
    'pathLength':None,
    'position':['x', 'y', 'z'],
    'positionOffset':['x', 'y', 'z'],
    'spin':['x', 'y', 'z', 'theta', 'phi', 'psi'],
    'time':None,
    'timeOffset':None,
    'velocity':['x', 'y', 'z'],
    'weight':None
}
"""
Expected unit dimensions for paricle records
"""
particle_record_unit_dimension = {
    'branchIndex':dimension('1'),
    'chargeState':dimension('1'),
    'electricField':dimension('electric_field'),
    'elementIndex':dimension('1'),
    'magneticField':dimension('tesla'),
    'locationInElement': dimension('1'),
    'momentum':dimension('momentum'),
    'momentumOffset':dimension('momentum'),
    'photonPolarizationAmplitude':dimension('electric_field'),
    'photonPolarizationPhase':dimension('1'),
    'sPosition':dimension('length'),
    'totalMomentum':dimension('momentum'),
    'totalMomentumOffset':dimension('momentum'),
    #'particleCoordinatesToGlobalTransformation': ??
    'particleStatus':dimension('1'),
    'pathLength':dimension('length'),
    'position':dimension('length'),
    'positionOffset':dimension('length'),
    'spin':dimension('1'),
    'time':dimension('time'),
    'timeOffset':dimension('time'),
    'velocity':dimension('velocity'),
    'weight':dimension('charge')
}







"""
Convenient aliases for components
"""
component_alias = {
    'x':'position/x',
    'y':'position/y',
    'z':'position/z',
    'px':'momentum/x',
    'py':'momentum/y',
    'pz':'momentum/z',
    't':'time',
    'weight':'weight',
    'status':'particleStatus'
}


def particle_paths(h5):
    """
    Uses the basePath and particlesPath to find where openPMD particles should be
    
    """
    basePath = h5.attrs['basePath'].decode('utf-8')
    particlesPath = h5.attrs['particlesPath'].decode('utf-8')
    
    if '%T' not in basePath:
        return [basePath+particlesPath]
    path1, path2 = basePath.split('%T')
    tlist = list(h5[path1])
    paths =  [path1+t+path2+particlesPath for t in tlist]
    return paths



def is_constant_component(h5):
    """
    Constant record component should have 'value' and 'shape'
    """
    return 'value' and 'shape' in h5.attrs

def constant_component_value(h5):
    """
    Constant record component should have 'value' and 'shape'
    """
    unitSI = h5.attrs['unitSI']
    val = h5.attrs['value']
    if  unitSI == 1.0:
        return val
    else:
        return val*unitSI

def component_unit_dimension(h5):
    """
    Return the unit dimension tuple
    """
    return tuple(h5.attrs['unitDimension'])
    
def component_data(h5, slice = slice(None), unit_factor=1.0):
    """
    Returns a numpy array from an h5 component.
    
    Determines wheter a component has constant data, or array data, and returns that. 
    
    An optional slice allows parts of the array to be retrieved. 
    
    Unit factor is an addition factor to convert from SI units to output units. 
    
    """

    # look for unitSI factor. 
    if 'unitSI' in h5.attrs:
        factor = h5.attrs['unitSI']
    else:
        factor = 1.0   
    
    # Additional conversion factor
    if unit_factor:
        factor *= unit_factor
        
    if is_constant_component(h5):
        dat = np.full(h5.attrs['shape'], h5.attrs['value'])
    else:
        # Retrieve dataset
        dat = h5[slice]

    if factor != 1.0:
        dat *= factor
        
    return dat


def offset_component_name(component_name):
    """
    Many components can also have an offset, as in:
    
        position/x
        positionOffset/c

    Return the appropriate name.
    """
    x = component_name.split('/')
    if len(x) == 1:
        return x[0]+'Offset'
    else:
        return x[0]+'Offset/'+x[1]


def particle_array(h5, component, slice=slice(None), include_offset=True):
    """
    Main routine to return particle arrays in fixed units.
    All units are SI except momentum, which will be in eV/c. 
    
    Example:
        particle_array(h5['data/00001/particles/'], 'px')
        Will return the momentum/x + momentumOffset/x in eV/c. 
        
        
    """

    if component in component_alias:
        component = component_alias[component]

    if component in ['momentum/x', 'momentum/y', 'momentum/z']:
        unit_factor = (c_light / e_charge  ) # convert J/(m/s) to eV/c
    else:
        unit_factor = 1.0

    # Get data
    dat = component_data(h5[component], slice = slice, unit_factor=unit_factor)
        
        
    # Look for offset component
    ocomponent = offset_component_name(component)
    if include_offset and ocomponent in h5 :
        offset =  component_data(h5[ocomponent], slice = slice, unit_factor=unit_factor)
        dat += offset
        
        
    return dat
        
    


    
    
def all_components(h5):
    """
    Look for possible components in a particle group
    """
    components = []
    for record_name in h5:
        if record_name not in particle_record_components:
            continue
    
        # Look for components
        possible_components = particle_record_components[record_name]
        
        if not possible_components:
            # Record is a component
            components.append(record_name)
        else:
            g = h5[record_name]
            for cname in possible_components:
                if cname in g:
                    components.append(record_name+'/'+cname)
    
    return components



def component_str(particle_group, name):
    """
    Informational string from a component in a particle group (h5)
    """
    
    g = particle_group[name]
    record_name = name.split('/')[0]
    expected_dimension = particle_record_unit_dimension[record_name]
    this_dimension =  component_unit_dimension(g)
    dname = dimension_name(this_dimension)
    symbol = SI_symbol[dname]
  
    s = name+' '
    
    if is_constant_component(g):
        val = constant_component_value(g)
        shape = g.attrs['shape']
        s += f'[constant {val} with shape {shape}]'
    else:
        s += '['+str(len(g))+' items]'
        
    if symbol != '1':
        s += f' is a {dname} with units: {symbol}'
        
    if expected_dimension != this_dimension:   
        
        s +=', but expected units: '+ SI_symbol[dimension_name(expected_dimension)]
    
    return s