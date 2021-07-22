import numpy as np



def norm_emit_calc(particle_group, planes=['x']):
    """
    
    2d, 4d, 6d normalized emittance calc
    
    planes = ['x', 'y'] is the 4d emittance
    
    planes = ['x', 'y', 'z'] is the 6d emittance
    
    Momenta for each plane are takes as p+plane, e.g. 'px' for plane='x'
    
    The normalization factor is (1/mc)^n_planes, so that the units are meters^n_planes
    
    """
    
    dim = len(planes)
    vars = []
    for k in planes:
        vars.append(k)
        vars.append('p'+k)
    
    S = particle_group.cov(*vars)
    
    mc2 = particle_group.mass
    
    norm_emit = np.sqrt(np.linalg.det(S)) / mc2**dim
    
    return norm_emit


def twiss_calc(sigma_mat2):
    """
    Calculate Twiss parameters from the 2D sigma matrix (covariance matrix):
    sigma_mat = <x,x>   <x, p>
                <p, x>  <p, p>

    This is a simple calculation. Makes no assumptions about units. 
    
    alpha = -<x, p>/emit
    beta  =  <x, x>/emit
    gamma =  <p, p>/emit
    emit = det(sigma_mat)
    
    """
    assert sigma_mat2.shape == (2,2), f'Bad shape: {sigma_mat2.shape}. This should be (2,2)' # safety check
    twiss={}
    emit = np.sqrt(np.linalg.det(sigma_mat2)) 
    twiss['alpha'] = -sigma_mat2[0,1]/emit
    twiss['beta']  = sigma_mat2[0,0]/emit
    twiss['gamma'] = sigma_mat2[1,1]/emit
    twiss['emit']  = emit
    
    return twiss



def twiss_match(x, p, beta0=1, alpha0=0, beta1=1, alpha1=0):
    """
    Simple Twiss matching. 
    
    Takes positions x and momenta p, and transforms them according to 
    initial Twiss parameters:
        beta0, alpha0 
    into final  Twiss parameters:
        beta1, alpha1
        
    This is simply the matrix ransformation:    
        xnew  = (   sqrt(beta1/beta0)                  0                 ) . ( x )
        pnew    (  (alpha0-alpha1)/sqrt(beta0*beta1)   sqrt(beta0/beta1) )   ( p ) 
        

    Returns new x, p
    
    """
    m11 = np.sqrt(beta1/beta0)
    m21 = (alpha0-alpha1)/np.sqrt(beta0*beta1)
    
    xnew = x * m11
    pnew = x * m21 + p / m11
    
    return xnew, pnew

def matched_particles(particle_group, beta=None, alpha=None, plane='x', p0c=None, inplace=False):
    """
    Perfoms simple Twiss 'matching' by applying a linear transformation to
        x, px if plane == 'x', or x, py if plane == 'y'
    
    Returns a new ParticleGroup
    
    If inplace, a copy will not be made, and changes will be done in place. 
    
    """
    
    assert plane in ('x', 'y'), f'Invalid plane: {plane}'
    
    if inplace:
        P = particle_group
    else:
        P = particle_group.copy()
    
    if not p0c:
        p0c = P['mean_p']
    

    # Use Bmad-style coordinates.
    # Get plane. 
    if plane == 'x':
        x = P.x
        p = P.px/p0c
    else:
        x = P.y
        p = P.py/p0c
        
    # Get current Twiss
    tx = twiss_calc(np.cov(x, p, aweights=P.weight))
    
    # If not specified, just fill in the current value.
    if alpha is None:
        alpha = tx['alpha']
    if beta is None:
        beta = tx['beta']
    
    # New coordinates
    xnew, pnew = twiss_match(x, p, beta0=tx['beta'], alpha0=tx['alpha'], beta1=beta, alpha1=alpha)
    
    # Set
    if plane == 'x':
        P.x = xnew
        P.px = pnew*p0c
    else:
        P.y = xnew
        P.py = pnew*p0c
        
    return P





def twiss_dispersion_calc(sigma3):
    """
    Twiss and Dispersion calculation from a 3x3 sigma (covariance) matrix from particles
    x, p, delta
    
    Formulas from: 
        https://uspas.fnal.gov/materials/19Knoxville/g-2/creation-and-analysis-of-beam-distributions.html
    
    Returns a dict with:
        alpha
        beta
        gamma
        emit
        eta      
        etap
    
    """
    
    # Collect terms
    
    delta2 = sigma3[2,2]
    xd = sigma3[0,2]
    pd = sigma3[1,2]
    
    eb =  sigma3[0,0] - xd**2 / delta2
    eg =  sigma3[1,1] - pd**2 / delta2
    ea = -sigma3[0,1] + xd * pd / delta2
    
    emit = np.sqrt(eb*eg - ea**2)
    
    # Form the output dict
    d = {}

    d['alpha'] = ea/emit
    d['beta']  = eb/emit
    d['gamma'] = eg/emit
    d['emit'] = emit
    d['eta']  = xd/delta2
    d['etap'] = pd/delta2

    return d


def particle_twiss_dispersion(particle_group, plane='x', fraction=1, p0c=None):
    """
    Twiss and Dispersion calc for a ParticleGroup. 
    
    Plane muse be:
        'x' or 'y'
    
    p0c is the reference momentum. If not give, the mean p will be used.
    
    Returns the same output dict as twiss_dispersion_calc, but with keys suffixed with the plane, i.e.:
     
        alpha_x
        beta_x
        gamma_x
        emit_x
        eta_x      
        etap_x
        norm_emit_x

    """
    
    assert plane in ['x', 'y']
    
    P = particle_group # convenience
    
    if fraction < 1:
        P = P[np.argsort(P[f'J{plane}'])][0:int(fraction*len(P))]
    
    if not p0c:
        p0c = P['mean_p'] 
    
    x = P[plane]
    xp = P['p'+plane]/p0c
    delta = P['p']/P['mean_p'] # - 1
    
    # Form covariance matrix
    np.cov([x, xp, delta], aweights=P.weight)
    
    # Actual calc
    twiss = twiss_dispersion_calc(np.cov([x, xp, delta]))
    
    # Add norm
    twiss['norm_emit'] = twiss['emit'] * P['mean_p']/P.mass
    
    # Add suffix
    out = {}
    for k in twiss:
        out[k+f'_{plane}'] = twiss[k]
    
    return out










# Linear Normal Form in 1 phase space plane.
# TODO: more advanced analysis e.g. Forest or Wolski or Sagan and Rubin or Ehrlichman. 

def A_mat_calc(beta, alpha, inverse=False):
    """
    Returns the 1D normal form matrix from twiss parameters beta and alpha
    
        A =   sqrt(beta)         0 
             -alpha/sqrt(beta)   1/sqrt(beta) 
    
    If inverse, the inverse will be returned:

        A^-1 =  1/sqrt(beta)     0 
                alpha/sqrt(beta) sqrt(beta)     
    
    This corresponds to the linear normal form decomposition:
    
        M = A . Rot(theta) . A^-1
    
    with a clockwise rotation matrix:
    
        Rot(theta) =  cos(theta) sin(theta)
                     -sin(theta) cos(theta)
    
    In the Bmad manual, G_q (Bmad) = A (here) in the Linear Optics chapter.
    
    A^-1 can be used to form normalized coordinates: 
        x_bar, px_bar   = A^-1 . (x, px)
    
    """
    a11 = np.sqrt(beta)  
    a22 = 1/a11
    a21 = -alpha/a11
    
    if inverse:
        return np.array([
                        [a22, 0],
                        [-a21, a11]])        
    else:
        return np.array([
                    [a11, 0],
                    [a21, a22]])
    

def amplitude_calc(x, p, beta=1, alpha=0):
    """
    Simple amplitude calculation of position and momentum coordinates
    relative to twiss beta and alpha. 

    J = (gamma x^2 + 2 alpha x p + beta p^2)/2
        
      = (x_bar^2 + px_bar^2)/ 2
      
    where gamma = (1+alpha^2)/beta
    
    """
    return (1+alpha**2)/beta/2 * x**2 + alpha*x*p + beta/2*p**2


def particle_amplitude(particle_group, plane='x', twiss=None, mass_normalize=True):
    """
    Returns the normalized amplitude array from a ParticleGroup for a given plane.
    
    Plane should be:
        'x' for the x, px plane
        'y' for the y, py plane 
    Other planes will work, but please check that the units make sense. 
        
    If mass_normalize (default=True), the momentum will be divided by the mass, so that the units are sqrt(m).        
    
    See: normalized_particle_coordinate
    """
    x = particle_group[plane]
    key2 = 'p'+plane
    
    if mass_normalize:
        # Note: do not do /=, because this will replace the ParticleGroup's internal array!
        p = particle_group[key2]/particle_group.mass
    else:
        p = particle_group[key2] 
        
    # User could supply twiss    
    if not twiss:
        sigma_mat2 = np.cov(x, p, aweights=particle_group.weight)
        twiss = twiss_calc(sigma_mat2)           
    
    J = amplitude_calc(x, p, beta=twiss['beta'], alpha=twiss['alpha'])
    
    return J 
    
    
    
def normalized_particle_coordinate(particle_group, key, twiss=None, mass_normalize=True):
    """
    Returns a single normalized coordinate array from a ParticleGroup
    
    Position or momentum is determined by the key. 
    If the key starts with 'p', it is a momentum, else it is a position,
    and the
    
    Intended use is for key to be one of:
        x, px, y py
        
    and the corresponding normalized coordinates are named with suffix _bar, i.e.:
        x_bar, px_bar, y_bar, py_bar
    
    If mass_normalize (default=True), the momentum will be divided by the mass, so that the units are sqrt(m).
    
    These are related to action-angle coordinates
        J: amplitude
        phi: phase
        
        x_bar =  sqrt(2 J) cos(phi)
        px_bar = sqrt(2 J) sin(phi)
        
    So therefore:
        J = (x_bar^2 + px_bar^2)/2
        phi = arctan(px_bar/x_bar)
    and:        
        <J> = norm_emit_x
     
     Note that the center may need to be subtracted in this case.
     
    """
    
    # Parse key for position or momentum coordinate
    if key.startswith('p'):
        momentum = True
        key1 = key[1:]
        key2 = key
        
    else:
        momentum = False
        key1 = key
        key2 = 'p'+key
            
    x = particle_group[key1]
    
    if mass_normalize:
        # Note: do not do /=, because this will replace the ParticleGroup's internal array!
        p = particle_group[key2]/particle_group.mass
    else:
        p = particle_group[key2]
    
     
    # User could supply twiss    
    if not twiss:
        sigma_mat2 = np.cov(x, p, aweights=particle_group.weight)
        twiss = twiss_calc(sigma_mat2)        
    
    A_inv = A_mat_calc(twiss['beta'], twiss['alpha'], inverse=True)

    if momentum:
        return A_inv[1,0]*x + A_inv[1,1]*p
    else:
        return A_inv[0,0]*x    
    
    
# ---------------
# Other utilities
    
def slice_statistics(particle_group,  keys=['mean_z'], n_slice=40, slice_key='z'):
    """
    Slices a particle group into n slices and returns statistics from each sliced defined in keys. 
    
    These statistics should be scalar floats for now.
    
    Any key can be used to slice on. 
    
    """
    sdat = {}
    for k in keys:
        sdat[k] = np.empty(n_slice)
    for i, pg in enumerate(particle_group.split(n_slice, key=slice_key)):
        for k in keys:
            sdat[k][i] = pg[k]
            
    return sdat



