import numpy as np
   
def write_opal(beam,           
               outfile,        
               emitted=True,   
               injected=False, 
               verbose=False,  
               species='electron'): 

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

    assert species == 'electron' # TODO: add more species

    # Format particles
    if emitted:
        eoutfile      = 'emitted-'+outfile
        # Column names are listed here for the code readers benefit
        # They are not used to make the file
        emitted_names = ['x', 'GBx','y', 'GBy','t','GBz']
        emitted_data  = np.column_stack([beam.x, beam.gamma*beam.beta_x , beam.y, beam.gamma*beam.beta_y, beam.t, beam.gamma*beam.beta_z])
        vprint(f'writing emitted {beam.n_particle} particles to {eoutfile}')
        # Save in the 'high_res = T' format
        np.savetxt(eoutfile, emitted_data, header=str(beam.n_particle), comments='', fmt = '%20.12e')

    if injected:
        ioutfile       = 'injected-'+outfile
        # Column names are listed here for the code readers benefit
        # They are not used to make the file
        injected_names = ['x', 'GBx','y', 'GBy','z','GBz']
        injected_data  = np.colunm_stack([beam.x, beam.gamma*beam.beta_x , beam.y, beam.gamma*beam.beta_y, beam.z, beam.gamma*beam.beta_z])
        vprint(f'writing injected {beam.n_particle} particles to {ioutfile}')

        # Save in the 'high_res = T' format
        np.savetxt(ioutfile, injected_data, header=str(beam.n_particle), comments='', fmt = '%20.12e')

