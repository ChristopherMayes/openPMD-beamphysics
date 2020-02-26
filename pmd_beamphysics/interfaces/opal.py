import numpy as np
   
def write_opal(beam,
               outfile,
               verbose=0,
               species='electron'): 

    def vprint(*a, **k):
        if verbose:
            print(*a, **k)

    vprint(f'writing {beam.n_particle} particles to {outfile}')
    assert species == 'electron' # TODO: add more species

    # Format particles
    names = ['x', 'px','y', 'py','t','pz']
    types = 6*[np.float]     
    
    # Make structured array
    dtype = np.dtype(list(zip(names, types)))
    data = np.zeros(beam.n_particle, dtype=dtype)
    for k in names:
        data[k][:] = getattr(beam, k)

    # Save in the 'high_res = T' format
    np.savetxt(outfile, data, header=str(beam.n_particle), comments='', fmt = ' '.join(6*['%20.12e']))


