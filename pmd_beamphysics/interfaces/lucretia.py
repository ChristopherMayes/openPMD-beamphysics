import scipy.io as sio
import numpy as np

def lucretia_to_data(filename, ele_name='BEGINNING', t_ref=0, exclude_dead_particles=True, verbose=False):

    """
    Load one beam in a Lucretia beam file as data for openPMD-beamphysics
    
    Parameters:
    ----------
    filename : str
               Lucretia '.mat' file name.
    ele_name : str
               name of the element at which the beam is located.
               An invalid name results in an error.
               If the beam file has one element, this only one beam is read. 
               Default: 'BEGINNING'
    t_ref : float, optional
            reference time of the beam in seconds. Default: 0.
    exclude_dead_particles : bool, optional
                             if True, excludes dead particles.  Default: True.

    
    Returns:
    ----------
    data: dict
        data for openPMD-beamphysics
    
    ----------

    
    Lucretia's format is described in:
    
        https://www.slac.stanford.edu/accel/ilc/codes/Lucretia/web/beam.html
        
    One Lucretia ".mat" file can include beams at multiple lattice elements.
    To find the beam at one element, one has to follow down this order of "fields":
    
        bstore >> ele_name >> Bunch >> x,
    
    in which x is a 6-to-Np array with:
    
        Lucretia x  = x in m
        Lucretia px = px/p in radian 
        Lucretia y  = y in m
        Lucretia py = py/p in radian 
        Lucretia z  = (t - t_ref)*c in m
        Lucretia p  = p in GeV/c
        
    Note that p is the total, not reference, momentum.
    
    To access valid element names in a Lucretia beam file, 
    use the helper function list_element_names(filename).
   
        dat = sio.loadmat('filename.mat')
        print(dat['bstore'].dtype)
        
    """

    ele_list = list_element_names(filename)
    if verbose:
        print(len(ele_list),'elements found in the file!')
    
    # Check if the element exists
    if (ele_name not in ele_list):
        raise ValueError('The provided element name '+ str(ele_name) +' does not exist in the file!')
    elif (len(ele_list) == 1):
        ele_name = ele_list[0]
        
    mdat = sio.loadmat(filename)
    
    coords = mdat['bstore'][ele_name][0,0]['Bunch'][0,0]['x'][0,0]
    charges = mdat['bstore'][ele_name][0,0]['Bunch'][0,0]['Q'][0,0][0]

    Np = coords.shape[1]

    x = coords[0]
    px_luc = coords[1] # normalized by total momentum 
    y = coords[2]
    py_luc = coords[3] # normalized by total momentum 
    z_luc = coords[4]
    ptot = coords[5]  # total momentum in GeV/c

    px = px_luc * ptot * 1e9 # in eV/c
    py = py_luc * ptot * 1e9
    pz = np.sqrt( (ptot * 1e9)**2 - px**2 - py**2 )

    t = z_luc / 299792458 + t_ref
       
    status = np.full(Np, 1)

    ix = np.where(ptot==0)
    status[ix] = 0
    n_dead = len(ix[0])

    if verbose:
        print(Np,'particles detected,', n_dead, 'found dead!')

    data = {
        'x':x,
        'px':px,
        'y':y,
        'py':py,
        'z':np.zeros(Np),
        'pz':pz,
        't':t, 
        'status':status, 
        'weight':charges, 
        'species':'electron'}
        
    if exclude_dead_particles and n_dead > 0:

        good = np.where(ptot>0)
        
        if verbose:
            print(f'Excluding {n_dead} dead particles')
            
        for k, v in data.items():
            if k == 'species':
                continue
            data[k] = data[k][good]
        
    return data


def write_lucretia(P, filePath, ele_name='BEGINNING', t_ref=0, stop_ix=None, verbose=True):
    """
    Write a ParticleGroup beam into a Lucretia beam file.
    
    Parameters:
    ----------
    P: ParticleGroup
               Particles to write. 
    
    filePath : str
               Lucretia '.mat' file name to be written to.
    ele_name : str
               name of the element at which the beam is located.
               Default: 'BEGINNING'
    t_ref : float, optional
            reference time of the beam in seconds. Default: 0.
    stop_ix : list of int, optional
              If provided, the length must equal to the number of particles.
              See Lucretia website for details.
              Default: None
    ----------
    
    Lucretia's format is described in:
    
        https://www.slac.stanford.edu/accel/ilc/codes/Lucretia/web/beam.html
      
    A general Lucretia beam file can include beams at multiple lattice elements.
    
    This routine only saves one beam at one element (name to be specified).
    Contents in the branches of the upper field structures are NOT defined.
    
    Lucretia beam follows:
    
        Lucretia x  = x in m
        Lucretia px = px/p in radian 
        Lucretia y  = y in m
        Lucretia py = py/p in radian 
        Lucretia z  = (t - t_ref)*c in m
        Lucretia p  = p in GeV/c
        
    Note that p is the total, not reference, momentum.
        
    """
    
    Np = P.n_particle

    ptot_nonzero = np.where( P.p==0, 1, P.p) # this prevents division by zero later
                                             # The replacement of "1" is dummy   
    x_luc = P.x
    px_luc = np.where( P.p==0, 0, P.px / ptot_nonzero )
    y_luc = P.y
    py_luc = np.where( P.p==0, 0, P.py / ptot_nonzero )
    z_luc = (P.t - t_ref) * 299792458
    ptot = P.p / 1E9 # total momentum in GeV/c

    if (np.all(stop_ix == None)):
        stop_ix = np.zeros(Np)
    else:
        if(len(stop_ix) != Np):
            print('Error: Length of stop_ix must equal to # particles !!')
        return
    
    # Form the lowest level of field structure
    l1 = np.array([x_luc,px_luc,y_luc,py_luc,z_luc,ptot])
    l2 = np.array([P.weight])
    l3 = np.array([stop_ix])
    
    # Wrapping upward in level fields 
    w1 = np.array([np.array([(l1, l2, l3)], dtype=[('x', 'O'), ('Q', 'O'), ('stop', 'O')])])
    w2 = np.array([np.array([(w1,)], dtype=[('Bunch', 'O')])])
    w3 = np.array([np.array([(w2,)], dtype=[(ele_name, 'O')])])
    w4 = {'bstore':w3}
    
    if verbose:
        print(f'writing {len(P)} particles in the Lucretia format to {filePath}')    
    
    sio.savemat(filePath, w4)
    
def list_element_names(filePath):
    """
    Find the element names in a Lucretia beam file
    
    Parameters:
    ----------
    filePath : str
               Lucretia '.mat' file name.

    
    Returns:
    ----------
    A list with the element names

    """
    dat = sio.loadmat(filePath)
    return list(dat['bstore'].dtype.fields)
