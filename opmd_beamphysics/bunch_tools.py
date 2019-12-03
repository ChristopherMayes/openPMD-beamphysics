import numpy as np

# Translate useful keys to openPMD paths
phase_space_key = {
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

# Legacy version
legacy_phase_space_key = {
    'px':'momentum/px',
    'py':'momentum/py',
    'pz':'momentum/pz',
}

def unit_factor(h5, key):
     # Convert to m, eV/c
    
    type = key.split('/')[0]
    # Basic conversion to SI
    if 'unitSI' in h5[key].attrs:
        # Check component
        factor = h5[key].attrs['unitSI']
    else:
        # Check group
        factor = h5[type].attrs['unitSI'] 
    if  type == 'position':
        pass
    elif type == 'momentum':
        factor /= (1.60217662e-19/299792458.) # convert J/(m/s) to eV/c
    elif type == 'weight':
        factor = 1
    elif type == 'time':
        factor = 1        
    elif type == 'particleStatus':
        factor = 1            
    else:
        print('unknown type:', type)
    return factor

# Convenience factors
nice_phase_space_factor = {
    't':1e12, # s -> ps
    'x':1e6, # m -> um 
    'y':1e6, # m -> um 
    'z':1e15 /299792458, # z -> z/c in fs
    'z_abs': 1000, # m -> mm 
    'px':1e-6, # eV/c  -> MeV/c
    'py':1e-6, # eV/c  -> MeV/c
    'pz':1e-9, # eV/c  -> GeV/c   
    'pz_abs': 1e-6 # eV/c  -> MeV/c
}
nice_phase_space_unit = {
    't':'ps',
    'x':'um',
    'y':'um',
    'z':'fs',
    'z_abs':'mm',
    'px':'MeV/c',
    'py':'MeV/c',
    'pz':'GeV/c',
    'pz_abs':'MeV/c'
}
nice_phase_space_label = {
    't':'t (ps)',
    'x':'x (um)',
    'y':'y (um)',
    'z':'z/c (fs)',
    'z_abs': ('z (mm)'),
    'px':'px (MeV/c)',
    'py':'py (MeV/c)',
    'pz':'pz (GeV/c)',
    'pz_abs': ('pz (MeV/c)')
}

def is_constant_component(h5):
    """
    Constant record component should have 'value' and 'shape'
    """
    return 'value' and 'shape' in h5.attrs

def component_data(h5, use_unitSI=True):
    """
    Determines wheter a component has constant data, or array data, and returns that. 
    """
    if is_constant_component(h5):
        dat = np.array(h5.attrs['value'])
    else:
        dat = h5[:]
    
    # look for unitSI factor. 
    if use_unitSI and ('unitSI' in h5.attrs):
        if h5.attrs['unitSI'] != 1.0:
            return dat * h5.attrs['unitSI']
        else: 
            return dat
    else:
        return dat

def particle_array(h5, component, liveOnly=False, liveStatus=1):
    
    # Special cases, add offsets
    if component == 'z_abs':
        offset = component_data(h5['positionOffset/z']) 
        component = 'z'
    elif component == 'pz_abs':
        offset = component_data(h5['momentumOffset/z']) 
        offset /= (1.60217662e-19/299792458.) # convert J/(m/s) to eV/c
        component = 'pz'
    else: 
        offset = 0
        
    key = phase_space_key[component]
    # Legacy syntax
    if component in ['px', 'py', 'pz']:
        if key not in h5:
            # Try legacy version
            key = legacy_phase_space_key[component]
    
    
    if is_constant_component(h5[key]):  
        dat = h5[key].attrs['value']*unit_factor(h5, key)
        dat = np.array([dat]) # Cast to array
        return dat
    else:
        factor = unit_factor(h5, key)
        dat = h5[key][:]*factor + offset

    if liveOnly:
        if 'particleStatus' not in h5:
            print('Warning: bunch does not have particleStatus. Assuming all particles are live.')
            return dat
        status = component_data(h5['particleStatus'], use_unitSI=False)

        if len(status) == 1:
            # Constant component
            if status[0] == liveStatus:
                return dat
            else:
                return np.array([])
            
        live = np.where(status == liveStatus) #== goodStatus )
        
        if len(live[0]) == 0:
            print('Warning: no particles')
        
        return dat[live]
    return dat

def bin_particles2d_h5(h5, component1, component2, bins=20, liveOnly=False, liveStatus=1,
    x_range=None, y_range=None):
   
    x = particle_array(h5, component1, liveOnly=liveOnly, liveStatus=liveStatus)
    y = particle_array(h5, component2, liveOnly=liveOnly, liveStatus=liveStatus)
    if not x_range:
        x_range = [np.min(x), np.max(x)]
    if not y_range:
        y_range = [np.min(y), np.max(y)]
    
    H, xedges, yedges = np.histogram2d(x,y, range = [x_range, y_range], bins=bins)
    return H, xedges, yedges



def load_bunch_h5(h5_bunch, liveOnly=False, liveStatus=1, use_time_as_z=False):
    """
    Load particles into structured numpy array
    """
    n = len(h5_bunch['position/x'])
    
    
    names = ['x', 'px', 'y', 'py', 'z', 'pz', 'weight']
    formats =  7*[np.float]
    
    data = np.empty(n, dtype=np.dtype({'names': names, 'formats':formats})) 
   
    for key in names:
        if key == 'z' and use_time_as_z:
            component = 't'
            factor = -299792458.
        else:
            component = key
            factor = 1
        print(key, component)    
        data[key] = factor*particle_array(h5_bunch, component, liveOnly=False)
        print(data[key])
    status = component_data(h5_bunch['particleStatus'], use_unitSI=False)
    print(status)
            
    if len(status) == 1 and liveOnly and status[0]==liveStatus:
        # Constant component
        return data
        
    if liveOnly:
        return data[:][np.where(status == liveStatus)]
    else:
        return data






