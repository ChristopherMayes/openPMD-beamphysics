import scipy.io as sio
import numpy as np
from pmd_beamphysics import ParticleGroup

def read_lucretia(filename, ele_name='', t_ref=0, kill_dead_particles=True, verbose=False):
    """
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
    t_ref is zero by default.
    
    To access valid element names in a Lucretia beam file, do:

        dat = sio.loadmat('filename.mat')
        print(dat['bstore'].dtype)
        
    """
    mdat = sio.loadmat(filename);
    coords = mdat['bstore'][ele_name][0,0]['Bunch'][0,0]['x'][0,0]
    charges = mdat['bstore'][ele_name][0,0]['Bunch'][0,0]['Q'][0,0]
    
    Np = coords.shape[1]

    x = coords[0]
    px_luc = coords[1] # normalized by total momentum 
    y = coords[2]
    py_luc = coords[3] # normalized by total momentum 
    z_luc = coords[4]
    ptot = coords[5]  # total momentum in GeV/c

    px = px_luc * ptot * 1E9 # in eV/c
    py = py_luc * ptot * 1E9
    pz = np.sqrt( (ptot * 1E9)**2 - px**2 - py**2 )

    t = z_luc / 299792458 + t_ref
       
    status = np.ones(Np)

    ix = np.where(ptot==0)
    status[ix] = 0
    n_dead = len(ix[0])

    if verbose:
        print(Np,'particles detected,', n_dead, 'found dead!')
    
    data = {'x':x,
        'px':px,
        'y':y,
        'py':py,
        'z':np.zeros(Np),
        'pz':pz,
        't':t, 
        'status':status, 
        'weight':charges, 
        'species':'electron'}
    
    P = ParticleGroup(data=data)
    
    if (kill_dead_particles):
        if verbose:
            print('Excluding dead particles (if any)...')
        P = P.where(P.p>0)
        
    return P


def write_lucretia(filename, P, ele_name='BEGINNING', t_ref=0, stop_ix=None):
    """
    Lucretia's format is described in:
    
        https://www.slac.stanford.edu/accel/ilc/codes/Lucretia/web/beam.html
      
    A general Lucretia beam file can include beams at multiple lattice elements.
    
    This routine only saves one beam at one element (name to be specified).
    Contents in the branches of the upper field structures are NOT filled.
    
    Lucretia beam follows:
    
        Lucretia x  = x in m
        Lucretia px = px/p in radian 
        Lucretia y  = y in m
        Lucretia py = py/p in radian 
        Lucretia z  = (t - t_ref)*c in m
        Lucretia p  = p in GeV/c
        
    Note that p is the total, not reference, momentum.
    t_ref is zero by default.
    stop_ix can be provided to indicate the indices of the dead particles.
        
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
    
    sio.savemat(filename, w4)