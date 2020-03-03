import numpy as np
   
def write_opal(ParticleGroup,           
               outfile,
               dist_type = 'emitted',
               verbose=False): 

    """
    Writes Astra style particles from a beam. 
    For now, the species must be electrons.   

    There are two types of distributions OPAL can read 
        from a file, Emitted or Injected. 
        https://gitlab.psi.ch/OPAL/Manual-2.2/wikis/distribution
    
    The difference between the two is how the 
        longitudinal dimension is defined and treated:
        
        In emitted distributions, the longitudinal dimension is 
        described by time, and the particles are emitted 
        from the cathode plane over a set number of 'emission steps'.
        
        For injected distributions, the longiudinal dimension is 
        described by z, and the particles are born instaneously 
        in the simulation at time step 0.

    """
    def vprint(*a, **k):
        if verbose:
            print(*a, **k) 

    # Format particles
    if dist_type == 'emitted': 
        # Column names are listed here for the code readers benefit
        # They are not used to make the file
        emitted_names = ['x', 'GBx','y', 'GBy','t','GBz']
        emitted_data  = np.column_stack([ParticleGroup.x, ParticleGroup.gamma*ParticleGroup.beta_x, 
                                         ParticleGroup.y, ParticleGroup.gamma*ParticleGroup.beta_y, 
                                         ParticleGroup.t, ParticleGroup.gamma*ParticleGroup.beta_z])
        vprint(f'writing emitted {ParticleGroup.n_particle} particles to {dist_type}-{outfile}')

        # Save distribution to text file
        np.savetxt(dist_type+'-'+outfile, emitted_data, header=str(ParticleGroup.n_particle), comments='', fmt = '%20.12e')

    elif dist_type == 'injected':
        injected_names = ['x', 'GBx','y', 'GBy','z','GBz']
        injected_data  = np.column_stack([ParticleGroup.x, ParticleGroup.gamma*ParticleGroup.beta_x, 
                                          ParticleGroup.y, ParticleGroup.gamma*ParticleGroup.beta_y, 
                                          ParticleGroup.z, ParticleGroup.gamma*ParticleGroup.beta_z])
        vprint(f'writing injected {ParticleGroup.n_particle} particles to {dist_type}-{outfile}')
        
        # Save distribution to text file
        np.savetxt(dist_type+'-'+outfile, injected_data, header=str(ParticleGroup.n_particle), comments='', fmt = '%20.12e')
    
    else:
        print('Invalid input argument for dist_type. No file made.')

    return None


