from h5py import File

def write_genesis4_distribution(particle_group,
                             h5file,
                             verbose=False):
    """
    
    
    Cooresponds to the `importdistribution` section in the Genesis4 manual. 
    
    Writes datesets to an h5 file:
    
    h5file: str or open h5 handle
    
    Datasets
        x is the horizontal coordinate in meters
        y is the vertical coordinate in meters
        xp = px/pz is the dimensionless trace space horizontal momentum
        yp = py/pz is the dimensionless trace space vertical momentum
        t is the time in seconds
        p  = relativistic gamma*beta is the total momentum divided by mc

    
        These should be the same as in .interfaces.elegant.write_elegant
        
        
    If particles are at different z, they will be drifted to the same z, 
    because the output should have different times. 
    
    """


    if isinstance(h5file, str):
        h5 = File(h5file, 'w')
    else:
        h5 = h5file
        
    if len(set(particle_group.z)) > 1:
        if verbose:
            print('Drifting particles to the same z')
        # Work on a copy, because we will drift
        P = particle_group.copy()
        # Drift to z. 
        P.drift_to_z()        
        
    
    else:
        P = particle_group

        
    for k in ['x', 'xp', 'y', 'yp', 't']:
        h5[k] = P[k]
        
    # p is really beta*gamma    
    h5['p'] = P['p']/P.mass
    
    
    if verbose:
        print(f'Datasets x, xp, y, yp, t, p written to: {h5file}')
        

   
