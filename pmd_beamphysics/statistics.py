import numpy as np



def norm_emit_calc(particle_group, planes=['x']):
    """
    
    2d, 4d, 6d normalized emittance calc
    
    planes = ['x', 'y'] is the 4d emittance
    
    planes = ['x', 'y', 'z'] is the 6d emittance
    
    Momenta for each plane are takes as p+plane, e.g. 'px' for plane='x'
    
    The normalization factor is (1/mc)^n_planes, so that the units are meters^n_planes
    
    """
    
    dim = len(planes)
    vars = []
    for k in planes:
        vars.append(k)
        vars.append('p'+k)
    
    S = particle_group.cov(*vars)
    
    mc2 = particle_group.mass
    
    norm_emit = np.sqrt(np.linalg.det(S)) / mc2**dim
    
    return norm_emit



def slice_statistics(particle_group,  keys=['mean_z'], n_slice=40, slice_key='z'):
    """
    Slices a particle group into n slices and returns statistics from each sliced defined in keys. 
    
    These statistics should be scalar floats for now.
    
    Any key can be used to slice on. 
    
    """
    sdat = {}
    for k in keys:
        sdat[k] = np.empty(n_slice)
    for i, pg in enumerate(particle_group.split(n_slice, key=slice_key)):
        for k in keys:
            sdat[k][i] = pg[k]
            
    return sdat