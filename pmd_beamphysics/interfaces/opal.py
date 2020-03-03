import numpy as np
   
def write_opal(particle_group,           
               outfile,
               dist_type = 'emitted',
               verbose=False): 

    """
    OPAL's ASCII format is described in:
    
        https://gitlab.psi.ch/OPAL/Manual-2.2/wikis/distribution
    
    outfile is the name out the ASCII file to be written.
    
    dist_type is one of:
    
        'emitted' : The longitudinal dimension is 
                    described by time, and the particles are emitted 
                    from the cathode plane over a set number of 'emission steps'.
                    
        'injected': the longiudinal dimension is 
                    described by z, and the particles are born instaneously 
                    in the simulation at time step 0.

    """

    n = particle_group.n_particle            
    x = particle_group.x
    y = particle_group.y
    
    # Get longitudinal coordinate
    if dist_type == 'emitted':
        
        # Check that z are all the same
        unique_z = np.unique(particle_group['z'])
        assert len(unique_z) == 1, 'All particles must be a the same z position'
        
        z = particle_group.t
        
    elif dist_type == 'injected':
        
        # Check that t are all the same
        unique_t = np.unique(particle_group['t'])
        assert len(unique_t) == 1, 'All particles must be a the same time'
        
        z = particle_group.z

    else:
        raise ValueError(f'unknown dist_type: {dist_type}')
    
    gamma = particle_group.gamma
    GBx = gamma*particle_group.beta_x
    GBy = gamma*particle_group.beta_y
    GBz = gamma*particle_group.beta_z
    
    header=str(n)
    dat = np.array([x, GBx, y, GBy, z, GBz]).T
    
    if verbose:
        print(f'writing {dist_type} {n} particles to {outfile}')
    np.savetxt(outfile, dat, header=header, comments='', fmt = '%20.12e')
        
