import numpy as np


def write_astra(particle_group,
                outfile,
                verbose=False,
                species='electron',
                probe=False):
    """
    Writes Astra style particles from particle_group type data.
    
    For now, the species must be electrons. 
    
    If probe, the six standard probe particles will be written. 
    """

    def vprint(*a, **k):
        if verbose:
            print(*a, **k)
    
    vprint(f'writing {particle_group.n_particle} particles to {outfile}')

    assert species == 'electron' # TODO: add more species
    
    # number of lines in file
    size = particle_group.n_particle + 1 # Allow one for reference particle
    i_start = 1 # Start for data particles
    if probe:
        # Add six probe particles, according to the manual
        size += 6
        i_start += 6
    

    # Astra units and types
    #units = ['m', 'm', 'm', 'eV/c', 'eV/c', 'eV/c', 'ns', 'nC']
    names = ['x', 'y', 'z', 'px', 'py', 'pz', 't', 'q', 'index', 'status']
    types = 8*[np.float] + 2*[np.int8]

    
    # Reference particle
    ref_particle = {'q':0}
    sigma = {}
    for k in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
        ref_particle[k] = particle_group.avg(k)
        sigma[k] =  particle_group.std(k)
    ref_particle['t'] *= 1e9 # s -> nS
        
    # Make structured array
    dtype = np.dtype(list(zip(names, types)))
    data = np.zeros(size, dtype=dtype)
    for k in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
        data[k][i_start:] = getattr(particle_group, k)
    data['t'] *= 1e9 # s -> nS
    data['q'][i_start:] = particle_group.weight*1e9 # C -> nC
    
    # Set these to be the same
    data['index'] = 1    # electron
    data['status'] = -1  # Particle at cathode
    
    # Subtract off reference z, pz, t
    for k in ['z', 'pz', 't']:
        data[k] -= ref_particle[k]
        
    # Put ref particle in first position
    for k in ref_particle:
        data[k][0] = ref_particle[k]
    
    # Optional: probes, according to the manual
    if probe:
        data[1]['x'] = 0.5*sigma['x'];data[1]['t'] =  0.5*sigma['t']
        data[2]['y'] = 0.5*sigma['y'];data[2]['t'] = -0.5*sigma['t']
        data[3]['x'] = 1.0*sigma['x'];data[3]['t'] =  sigma['t']
        data[4]['y'] = 1.0*sigma['y'];data[4]['t'] = -sigma['t']
        data[5]['x'] = 1.5*sigma['x'];data[5]['t'] =  1.5*sigma['t']
        data[6]['y'] = 1.5*sigma['y'];data[6]['t'] = -1.5*sigma['t']        
        data[1:7]['status'] = -3
        data[1:7]['pz'] = 0 #? This is what the Astra Generator does
    
    # Save in the 'high_res = T' format
    np.savetxt(outfile, data, fmt = ' '.join(8*['%20.12e']+2*['%4i']))

