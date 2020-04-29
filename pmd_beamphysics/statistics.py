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