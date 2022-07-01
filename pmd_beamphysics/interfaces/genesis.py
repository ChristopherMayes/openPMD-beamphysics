
import numpy as np
from h5py import File

from pmd_beamphysics.statistics import twiss_calc
from pmd_beamphysics.units import mec2, c_light



# Genesis 1.3 
#-------------
# Version 2 routines


def genesis2_beam_data1(pg):
    """
    
    Calculate statistics of a single particlegroup, 
    for use in a Genesis 1.3 v2 beam file
    
    Returns a dict of:
        zpos or tpos      : z or t of the mean in m or s
        curpeak           : current in A
        gamma0            : average relativistic gamma (dimensionless)
        emitx, emity      : Normalized emittances in m*rad
        rxbeam, rybeam    : sigma_x, sigma_y in m
        xbeam, ybeam      : <x>, <y> in m
        pxbeam, pybeam    : beta_x*gamma, beta_y*gamma (dimensionless)
        alpha_x, alpha_y  : Twiss alpha  (dinensionless)
    """
    
    d = {}
    
    # Handle z or t
    if len(set(pg.z)) == 1:
        # should be different t
        d['tpos'] = pg['mean_t']
    elif len(set(pg.t)) == 1:
         #should be different z
        d['zpos'] = pg['mean_z']
    else:
        raise ValueError(f'{pg} has mixed t and z coordinates.')

    d['curpeak'] = pg['average_current']     
    d['gamma0'] = pg['mean_gamma']    
    d['delgam'] = pg['sigma_gamma']
    d['emitx'] = pg['norm_emit_x']
    d['emity'] = pg['norm_emit_y']        
    d['rxbeam'] = pg['sigma_x']
    d['rybeam'] = pg['sigma_y']
    d['xbeam'] = pg['mean_x']
    d['ybeam'] = pg['mean_y']
    
    d['pxbeam'] = pg['mean_px']/pg.mass # beta_x*gamma
    d['pybeam'] = pg['mean_py']/pg.mass # beta_y*gamma
    
    # Twiss, for alpha
    twiss_x = twiss_calc(pg.cov('x', 'xp'))
    twiss_y = twiss_calc(pg.cov('y', 'yp'))
    d['alphax'] = twiss_x['alpha'] 
    d['alphay'] = twiss_y['alpha'] 
        
    return d


def genesis2_beam_data(pg, n_slice=None):
    """
    
    Slices a particlegroup into n_slice and forms the beam columns. 
    
    n_slice is the number of slices. If not given, the beam will be divided 
    so there are 100 particles in each slice. 
    
    Returns a dict of beam_columns, for use with write_genesis2_beam_file
    
    See: genesis2_beam_data1
    """
    
    
    # Handle z or t
    if len(set(pg.z)) == 1:
        # should be different t
        slice_key = 't'
    elif len(set(pg.t)) == 1:
         #should be different z
        slice_key = 'z'
    else:
        raise ValueError(f'{pg} has mixed t and z coordinates.')    
    
    # Automatic slicing
    if not n_slice:
        n_slice = len(pg)//100
    
    # Slice (split)
    pglist = pg.split(n_slice, key=slice_key)
    
    d = {}
    # Loop over the slices
    for pg in pglist:
        data1 = genesis2_beam_data1(pg)
        for k in data1:
            if k not in d:
                d[k] = []
            d[k].append(data1[k])
    for k in d:
        d[k] = np.array(d[k])
    
    return d
    

def write_genesis2_beam_file(fname, beam_columns, verbose=False):
    """
    Writes a Genesis 1.3 v2 beam file, using a dict beam_columns
    
    The header will be written as:
    ? VERSION=1.0
    ? SIZE=<length of the columns>
    ? COLUMNS <list of columns
    <data>

    This is a copy of the lume-genesis routine:
    
    genesis.writers.write_beam_file
    
    """
    
    # Get size
    names = list(beam_columns)
    size = len(beam_columns[names[0]])
    header=f"""? VERSION=1.0
? SIZE={size}
? COLUMNS {' '.join([n.upper() for n in names])}"""
    
    dat = np.array([beam_columns[name] for name in names]).T

    np.savetxt(fname, dat, header=header, comments='', fmt='%1.8e') # Genesis can't read format %1.16e - lines are too long?
    
    if verbose:
        print('Beam written:', fname)
    
    return header




def genesis2_dpa_to_data(dpa, *, xlamds, current, zsep=1, species='electron'):
    """
    Converts Genesis 1.3 v2 dpa data to ParticleGroup data.
    
    The Genesis 1.3 v2 dpa phase space coordinates and units are:
        gamma [1]
        phase [rad]
        x [m]
        y [m]
        px/mc [1]
        py/mc [1]
    The definition of the phase is different between the .par and .dpa files.
        .par file: phase = psi  = kw*z + field_phase
        .dpa file: phase = kw*z    
    
    Parameters
    ----------
    
    dpa: array 
        Parsed .dpa file as an array with shape (n_slice, 6, n_particles_per_slice)
    
    xlamds: float
        wavelength (m)
        
    zsep: int
        slice separation interval 
        
    current: array
        current array of length n_slice (A)
        
        
    species: str, required to be 'electron'
    
    Returns
    -------
    data: dict with keys: x, px, y, py, z, pz, weight, species, status
        in the units of the openPMD-beamphysics Python package:
        m, eV/c, m, eV/c, m, eV/c, C
        
        These are returned in z-coordinates, with z=0. 
    
    """
    
    assert species == 'electron'
    mc2 = mec2
    
    dz = xlamds*zsep 
    
    nslice, dims, n1 = dpa.shape  # n1 particles in a single slice
    assert dims == 6
    n_particle = n1 * nslice
    
    gamma = dpa[:,0,:].flatten()
    phase = dpa[:,1,:].flatten()
    x  =  dpa[:,2,:].flatten()
    y  =  dpa[:,3,:].flatten()
    px =  dpa[:,4,:].flatten() * mc2
    py =  dpa[:,5,:].flatten() * mc2
    pz =  np.sqrt( (gamma**2 - 1)*mc2**2 -px**2 - py*2 )
    
    i0 = np.arange(nslice)
    i_slice = np.repeat(i0[:, np.newaxis], n1, axis=1).flatten()
    
    # Spread particles out over zsep interval
    z  = dz * (i_slice + np.mod(phase/(2*np.pi * zsep), zsep))  # z = (zsep * xlamds) * (i_slice + mod(dpa_phase/2pi, 1)) 
    z = z.flatten()
    #t = np.full(n_particle, 0.0)
    
    # z-coordinates
    t = -z/c_light
    z = np.full(n_particle, 0.0)
    
    
    weight = np.repeat(current[:, np.newaxis], n1, axis=1).flatten() * dz / c_light / n1
    
    return {
        't': t,
        'x': x,
        'px':px,
        'y': y,
        'py': py,
        'z': z,
        'pz': pz,
        'species': species,
        'weight': weight,
        'status': np.full(n_particle, 1)
        }
    





#-------------
# Version 4 routines

def write_genesis4_distribution(particle_group,
                             h5file,
                             verbose=False):
    """
    
    
    Cooresponds to the `import distribution` section in the Genesis4 manual. 
    
    Writes datesets to an h5 file:
    
    h5file: str or open h5 handle
    
    Datasets
        x is the horizontal coordinate in meters
        y is the vertical coordinate in meters
        xp = px/pz is the dimensionless trace space horizontal momentum
        yp = py/pz is the dimensionless trace space vertical momentum
        t is the time in seconds
        p  = relativistic gamma*beta is the total momentum divided by mc

    
        These should be the same as in .interfaces.elegant.write_elegant
        
        
    If particles are at different z, they will be drifted to the same z, 
    because the output should have different times. 
    
    """


    if isinstance(h5file, str):
        h5 = File(h5file, 'w')
    else:
        h5 = h5file
        
    if len(set(particle_group.z)) > 1:
        if verbose:
            print('Drifting particles to the same z')
        # Work on a copy, because we will drift
        P = particle_group.copy()
        # Drift to z. 
        P.drift_to_z()        
        
    
    else:
        P = particle_group

        
    for k in ['x', 'xp', 'y', 'yp', 't']:
        h5[k] = P[k]
        
    # p is really beta*gamma    
    h5['p'] = P['p']/P.mass
    
    
    if verbose:
        print(f'Datasets x, xp, y, yp, t, p written to: {h5file}')
        

   
