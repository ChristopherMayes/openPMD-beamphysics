import numpy as np



SCALAR_KEYS = ['avg_delta',
 'avg_px',
 'avg_py',
 'avg_x',
 'avg_y',
 'avg_z',
 'total_charge',
 'current',
 'max_delta',
 'max_px',
 'max_py',
 'max_x',
 'max_y',
 'max_z',
 'min_delta',
 'min_px',
 'min_py',
 'min_x',
 'min_y',
 'min_z',
 'norm_emit_x',
 'norm_emit_y',
 'p0c',
 'sigma_delta',
 'sigma_px',
 'sigma_py',
 'sigma_x',
 'sigma_y',
 'sigma_z',
 'twiss_alpha_x',
 'twiss_alpha_y',
 'twiss_beta_x',
 'twiss_beta_y',
 'twiss_emit_x',
 'twiss_emit_y',
 'twiss_gamma_x',
 'twiss_gamma_y']



def bunch_statistics(pdata, species='electron', p0c=None):
    """
    Calculate bunch statistics from raw particle data. 
    
    Converts to Bmad-style units:
        ['x', 'px', 'y', 'py', 'z', 'delta'],
        
        where px = m c^2 beta_x gamma / p0c     # Similar to x'
              py = m c^2 beta_x gamma / p0c     # Similar to y'
              delta = m c^2 beta gamma / p0c -1 # Relative momentum
              z = -beta*c(t-tref)
    
    Returns dict with many scalar keys (listed in bunch_stats.SCALAR_KEYS)
        plus the 6x6 'sigma_mat'
    
    
    TODO: 
        Better Twiss analysis
        coupling
        dispersion
        specied masses
    
    """
    
    x, px, y, py, z, pz, w =  pdata['x'], pdata['px'], pdata['y'], pdata['py'], pdata['z'], pdata['pz'], pdata['weight']
    
    ptot = np.sqrt(px**2+py**2+pz**2)
    # Reference momentum
    if not p0c:
        p0c = np.average(ptot, weights=w) 
   
    mass_of = {'electron': 0.51099895000e6 # eV/c
              }
    beta_gamma0 = p0c/mass_of[species]
    
    # Bmad-style units for analysis
    particles = np.array([ x, px/p0c, y, py/p0c, z, ptot/p0c - 1.0 ])
    
    # Statistical calcs
    centroid = np.average(particles.T, axis=0, weights=w)
    sigma_mat = np.cov(particles, aweights = w) 
    
    # Analysis
    d = {}
    d['p0c'] = p0c
    
    # Weights should sum to total charge
    d['total_charge'] = np.sum(w)
    
    names = ['x', 'px', 'y', 'py', 'z', 'delta']
    for i in range(6) :
        key = names[i]
        d['min_'+key] = particles[i].min()
        d['max_'+key] = particles[i].max()
        d['avg_'+key] = centroid[i]
        d['sigma_'+key] = np.sqrt(sigma_mat[i,i])
    
    # Average current
    d['current'] = d['total_charge']  * 299792458. / (d['max_z'] - d['min_z'])

    twiss_x = twiss_calc(sigma_mat[0:2,0:2])
    twiss_y = twiss_calc(sigma_mat[2:4,2:4])
    for key,item in twiss_x.items():
        d['twiss_'+key+'_x'] = item
    for key,item in twiss_y.items():
        d['twiss_'+key+'_y'] = item       
    
    d['norm_emit_x'] = d['twiss_emit_x']*beta_gamma0
    d['norm_emit_y'] = d['twiss_emit_y']*beta_gamma0
        
    d['sigma_mat'] = sigma_mat    
    
    return d


def twiss_calc(sigma_mat2):
    """
    Calculate Twiss parameters from the 2D sigma matrix. 
    
    Simple calculation. Makes no assumptions about units. 
    
    
    """
    assert sigma_mat2.shape == (2,2) # safety check
    twiss={}
    emit = np.sqrt(np.linalg.det(sigma_mat2)) 
    twiss['alpha'] = -sigma_mat2[0,1]/emit
    twiss['beta']  = sigma_mat2[0,0]/emit
    twiss['gamma'] = sigma_mat2[1,1]/emit
    twiss['emit']  = emit
    
    return twiss




def slice_statistics(particles, n_chunks=10, p0c=None, slice_key='z', scalar_keys=SCALAR_KEYS):
    """
    Calaculates slice statistics of particles. 
    Particles are sorted by slice_key (Default = z) and divided into even chunks. 
    
    """
    
    # Set up numpy structured arrays
    names = scalar_keys
    formats =  len(scalar_keys)*[np.float]
    scalar_data = np.empty(n_chunks, dtype=np.dtype({'names': names, 'formats':formats})) 
    sigma_mats = np.empty((n_chunks, 6, 6))
    
    # Sorting
    iz = np.argsort(particles[slice_key])

    # Split particles into chunks
    ii = -1
    for chunk in np.array_split(iz, n_chunks):
        ii+=1
        p = particles[chunk]
        stats = bunch_statistics(p, p0c=p0c)
        for k in scalar_keys:
            scalar_data[k][ii] = stats[k]
        sigma_mats[ii] = stats['sigma_mat']
    d = {}
    d['slice_scalars'] = scalar_data
    d['slice_sigma_mats'] = sigma_mats
    return d