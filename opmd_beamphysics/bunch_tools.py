import numpy as np

# Translate useful keys to openPMD paths
phase_space_key = {
    'x':'position/x',
    'y':'position/y',
    'z':'position/z',
    'px':'momentum/px',
    'py':'momentum/py',
    'pz':'momentum/pz',
}

def unit_factor(h5, key):
     # Convert to m, eV/c
    
    type = key.split('/')[0]
    # Basic conversion to SI
    factor = h5[type].attrs['unitSI'] 
    if  type == 'position':
        pass
    elif type == 'momentum':
        factor /= (1.60217662e-19/299792458.) # convert J/(m/s) to eV/c
    elif type == 'weight':
        pass
    else:
        print('unknown type:', type)
    return factor

# Convenience factors
nice_phase_space_factor = {
    'x':1e6, # m -> um 
    'y':1e6, # m -> um 
    'z':1e15 /299792458, # z -> z/c in fs
    'z_abs': 1000, # m -> mm 
    'px':1e-6, # eV/c  -> MeV/c
    'py':1e-6, # eV/c  -> MeV/c
    'pz':1e-9, # eV/c  -> GeV/c   
    'pz_abs': 1e-6 # eV/c  -> MeV/c
}
nice_phase_space_label = {
    'x':'x (um)',
    'y':'y (um)',
    'z':'z/c (fs)',
    'z_abs': ('z (mm)'),
    'px':'px (MeV/c)',
    'py':'py (MeV/c)',
    'pz':'pz (GeV/c)',
    'pz_abs': ('pz (MeV/c)')
}


def particle_array(h5, component, liveOnly=False):
    
    # Special cases, add offsets
    if component == 'z_abs':
        offset = h5['positionOffset/z'].attrs['value'] * h5['positionOffset/z'].attrs['unitSI']
        component = 'z'
    elif component == 'pz_abs':
        offset = h5['momentumOffset/pz'].attrs['value'] * h5['momentumOffset/pz'].attrs['unitSI']
        offset /= (1.60217662e-19/299792458.) # convert J/(m/s) to eV/c
        component = 'pz'
    else: 
        offset = 0
        
    key = phase_space_key[component]
    dat = h5[key]*unit_factor(h5, key)  + offset
    
    
    
    if liveOnly:
        live = np.where(np.array(h5['particleStatus']) > 0 ) #== goodStatus )
        return dat[live]
    return dat

def bin_particles2d_h5(h5, component1, component2, bins=20, liveOnly=False):
   
    x = particle_array(h5, component1, liveOnly=liveOnly)
    y = particle_array(h5, component2, liveOnly=liveOnly)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    
    H, xedges, yedges = np.histogram2d(x,y, range = [[xmin, xmax], [ymin,ymax]], bins=bins)
    return H, xedges, yedges





