import numpy as np

c_light = 299792458.



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
                verbose=False,
                cathode_kinetic_energy_ref=None):
    """
    Writes Impact-T style particles from particle_group type data.
    
    outfile should ultimately be named 'partcl.data' for Impact-T
    
    For now, the species must be electrons. 
    
    If cathode_kinetic_energy_ref is given, t will be used to compute z for cathode emission.
    
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
        
        t = particle_group['t']
        
        # All t must be negative. 
        t_ptp = t.ptp()
        # Allow for some padding
        if t_ptp == 0:
            t_ptp = 1e-15  # s
        t_pad = t_ptp*1e-3 # 0.1% pad 
        t_shift = -t.max() -t_pad
        
        # Suggest an emission time
        output['Temission'] = t_ptp + 2*t_pad
        
        # This is the time that particles are shifted
        #output['Temission_shift'] = t_shift
        # Change actual initial time to this shift
        output['Tini'] = t_shift
        
        tout = t+t_shift
        
        output['Temission_mean'] = tout.mean()
        
        z = (tout)*betac       
        
        # pz
        pz = particle_group['pz']
        #check for zero pz
        assert np.all(pz > 0), 'pz must be positive'
        
        gamma_beta_z = pz/mc2
        
    else:
        # Free space start
        z = particle_group['z']
        
        t = np.unique(particle_group['t'])
        assert len(t) == 1, 'All particles must be a the same time'
        t = t[0]
        output['Tini'] = t
        output['Flagimg'] = 0 # Turn off Cathose start
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
    np.savetxt(outfile, dat.T, header=str(n_particle), comments='')
    
    # Return info dict
    return output
