import numpy as np

c_light = 299792458.




def parse_impact_particles(filePath, 
                           names=('x', 'GBx', 'y', 'GBy', 'z', 'GBz'),
                           skiprows=0):
    """
    Parse Impact-T input and output particle data.
    Typical filenames: 'partcl.data', 'fort.40', 'fort.50'.
    
    Note that partcl.data has the number of particles in the first line, so skiprows=1 should be used.
    
    Returns a structured numpy array
    
    Impact-T input/output particles distribions are ASCII files with columns:
    x (m)
    GBy = gamma*beta_x (dimensionless)
    y (m)
    GBy = gamma*beta_y (dimensionless)
    z (m)
    GBz = gamma*beta_z (dimensionless)
    
    Routine from lume-impact: 
        https://github.com/ChristopherMayes/lume-impact
    
    """
    
    dtype={'names': names,
           'formats': 6*[np.float]}
    pdat = np.loadtxt(filePath, skiprows=skiprows, dtype=dtype,
                     ndmin=1) # to make sure that 1 particle is parsed the same as many.

    return pdat    
    



def impact_particles_to_particle_data(tout, mc2=0, species=None, time=0, macrocharge=0, cathode_kinetic_energy_ref=None, verbose=False):
    """
    Convert impact particles to data for ParticleGroup
    
    particle_charge is the charge in units of |e|
    
    At the cathode, Impact-T translates z to t = z / (beta*c) for emission,
    where (beta*c) is the velocity calculated from kinetic energy:
        header['Bkenergy'] in eV.
    This is purely a conversion factor. 
    
    If cathode_kinetic_energy_ref is given, z will be parsed appropriately to t, and z will be set to 0. 
    
    Otherwise, particles will be set to the same time.
    
    """
    
    #mc2 = SPECIES_MASS[species]
    assert mc2 >0, 'mc2 must be specified'
    assert species, 'species must be specified'
       
    data = {}
    
    n_particle = len(tout['x'])
    
    data['x'] = tout['x']
    data['y'] = tout['y']
    #data['z'] = tout['z'] will be handled below

    data['px'] = tout['GBx']*mc2
    data['py'] = tout['GBy']*mc2
    data['pz'] = tout['GBz']*mc2
    
    # Handle z
    if cathode_kinetic_energy_ref:
        # Cathode start
        z = np.full(n_particle, 0.0)
        
        # Note that this is purely a conversion factor. 
        gamma = 1.0 + cathode_kinetic_energy_ref/mc2
        betac = np.sqrt(1-1/gamma**2)*c_light
        
        t = tout['z']/betac
        if verbose:
            print(f'Converting z to t according to cathode_kinetic_energy_ref = {cathode_kinetic_energy_ref} eV')
        
        
    else:
        # Free space start
        z = tout['z']
        t = np.full(n_particle, time)
    
    data['z'] = z
    data['t'] = t
    
    data['status'] = np.full(n_particle, 1)
    if macrocharge == 0:
        weight = 1/n_particle
    else:
        weight = abs(macrocharge)
    data['weight'] =  np.full(n_particle, weight) 
    
    data['species'] = species
    data['n_particle'] = n_particle
    return data



def write_impact(particle_group,
                outfile,
                cathode_kinetic_energy_ref=None,
                include_header=True,
                verbose=False):
    """
    Writes Impact-T style particles from particle_group type data.
    
    outfile should ultimately be named 'partcl.data' for Impact-T
    
    For now, the species must be electrons. 
    
    If cathode_kinetic_energy_ref is given, t will be used to compute z for cathode emission.
    
    If include_header, the number of particles will be written as the first line. Default is True. 
    
    Otherwise, particles must have the same time, and should be started in free space.
    
    A dict is returned with info about emission, for use in Impact-T
    
    """

    def vprint(*a, **k):
        if verbose:
            print(*a, **k)
    
    n_particle = particle_group.n_particle
    
    vprint(f'writing {n_particle} particles to {outfile}')
    
    
    mc2 = particle_group.mass
    
    # Dict output for use in Impact-T
    output = {'input_particle_file':outfile}
    output['Np'] = n_particle
    
    # Handle z
    if cathode_kinetic_energy_ref:
        # Cathode start
    
        vprint(f'Cathode start with cathode_kinetic_energy_ref = {cathode_kinetic_energy_ref} eV')
        
        # Impact-T conversion factor in eV
        output['Bkenergy'] = cathode_kinetic_energy_ref
        
        # Note that this is purely a conversion factor. 
        gamma = 1.0 + cathode_kinetic_energy_ref/mc2
        betac = np.sqrt(1-1/gamma**2)*c_light
  
        # z equivalent
        z = -betac*particle_group['t']  

        # Get z span
        z_ptp = z.ptp()
        # Add tiny padding
        z_pad = 1e-20 # Tiny pad 
        
        # Shift all particles, so that z < 0
        z_shift = -(z.max() + z_pad)
        z += z_shift
        
        # Starting clock shift
        t_shift = z_shift/betac
        
        # Suggest an emission time
        output['Temission'] = (z_ptp + 2*z_pad)/betac
        
        # Change actual initial time to this shift (just set)
        output['Tini'] = t_shift
    
        # Informational
        #output['Temission_mean'] = tout.mean()
        
        # pz
        pz = particle_group['pz']
        #check for zero pz
        assert np.all(pz > 0), 'pz must be positive'
        
        # Make sure there as at least some small pz momentum. Simply shift.
        pz_small = 10 # eV/c
        small_pz = pz < pz_small
        pz[small_pz] += pz_small
         
        gamma_beta_z = pz/mc2
        
    else:
        # Free space start
        z = particle_group['z']
        
        t = np.unique(particle_group['t'])
        assert len(t) == 1, 'All particles must be a the same time'
        t = t[0]
        output['Tini'] = t
        output['Flagimg'] = 0 # Turn off Cathode start
        gamma_beta_z = particle_group['pz']/mc2
        
        vprint(f'Normal start with at time {t} s')
    
    # Form data table
    dat = np.array([
        particle_group['x'],
        particle_group['px']/mc2,
        particle_group['y'],
        particle_group['py']/mc2,
        z,
        gamma_beta_z
    ])
    
    # Save to ASCII
    if include_header:
        header=str(n_particle)
    else:
        header=''
        
    np.savetxt(outfile, dat.T, header=header, comments='')
    
    # Return info dict
    return output
