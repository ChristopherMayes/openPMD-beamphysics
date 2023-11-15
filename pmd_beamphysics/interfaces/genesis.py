import os
import numpy as np
from h5py import File

from pmd_beamphysics.statistics import twiss_calc
from pmd_beamphysics.units import mec2, c_light, write_unit_h5, unit



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


def genesis4_beam_data(pg, n_slice=None):
    """
    Slices a particlegroup into n_slice and forms the sliced beam data. 
    
    n_slice is the number of slices. If not given, the beam will be divided 
    so there are 100 particles in each slice. 
    
    Returns a dict of beam_columns, for use with write_genesis2_beam_file
    
    This uses the same routines as genesis2, with some relabeling
    See: genesis2_beam_data1
    """
    
    # Re-use genesis2_beam_data
    g2data = genesis2_beam_data(pg, n_slice=n_slice)
    
    # Old, new, unit
    relabel = [
           ('tpos', 't', 's'),
           ('zpos', 's', 'm'),
           ('curpeak', 'current', 'A'),
           ('gamma0', 'gamma', '1'),
           ('delgam', 'delgam', '1'),
           ('emitx', 'ex', 'm'),
           ('emity', 'ey', 'm'),
           ('rxbeam', 'sigma_x', 'm'),
           ('rybeam', 'sigma_y', 'm'),
           ('xbeam', 'xcenter', 'm'),
           ('ybeam', 'ycenter', 'm'),
           ('pxbeam', 'pxcenter', '1'),
           ('pybeam', 'pycenter', '1'),
           ('alphax', 'alphax', '1'),
           ('alphay', 'alphay', '1'),
            ] 
    
    data = {}
    units = {}
    for g2key, g4key, u in relabel:  
        if g2key not in g2data:
            continue          
        data[g4key] = g2data[g2key]
        units[g4key] = unit(u)
        
    # Re-calculate these
    data['betax'] = data['gamma'] * data.pop('sigma_x')**2 / data['ex']
    data['betay'] = data['gamma'] * data.pop('sigma_y')**2 / data['ey']
    
    units['betax'] = unit('m')
    units['betay'] = unit('m')
    
    if 's' in data:
        data['s'] -= data['s'].min()
        
    return data, units

def write_genesis4_beam(particle_group, h5_fname, n_slice=None, verbose=False, return_input_str=False):
    """
    Writes sliced beam data to an HDF5 file
    
    """
    beam_data, units = genesis4_beam_data(particle_group, n_slice = n_slice)
    
    with File(h5_fname, 'w') as h5:
        for k in beam_data:
            h5[k] = beam_data[k]
            write_unit_h5(h5[k], units[k])
            
    if verbose:
        print('Genesis4 beam file written:', h5_fname) 


    if return_input_str:
        data_keys = list(beam_data)
        lines = genesis4_profile_file_input_str(data_keys, h5_fname)
        lines += genesis4_beam_input_str(data_keys)
        return lines
        
def _profile_file_lines(label, h5filename, xdata_key, ydata_key, isTime=False, reverse=False):
    lines = f"""&profile_file
  label = {label}
  xdata = {h5filename}/{xdata_key}
  ydata = {h5filename}/{ydata_key}"""
    if isTime:
        lines +=   "\n  isTime = T"
    if reverse:
        lines +=   "\n  reverse = T"       
    lines += "\n&end\n"
    return lines

def genesis4_profile_file_input_str(data_keys, h5filename):
    """
    Returns an input str suitable for the main Genesis4 input file
    for profile data.
    """

    h5filename = os.path.split(h5filename)[1] # Genesis4 does not understand paths
    
    if 's' in data_keys:
        xdata_key = 's'
        isTime = False
        reverse = False
    elif 't' in data_keys:
        xdata_key = 't'
        isTime = True
        reverse = True
    else:
        raise ValueError('no s or t found')

    lines = ''
    for ydata_key in data_keys:
        if ydata_key == xdata_key:
            continue
        lines += _profile_file_lines(ydata_key, h5filename, xdata_key, ydata_key, isTime, reverse)
            
    return lines

def genesis4_beam_input_str(data_keys):
    """
    Returns an input str suitable for the main Genesis4 input file
    for profile data.
    """    
    lines = ['&beam']
    for k in data_keys:
        if k in ('s', 't'):
            continue
        lines.append(f'  {k} = @{k}')
    lines.append('&end')
    return '\n'.join(lines)        



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
    
    If any of the weights are different, the bunch will be resampled
    to have equal weights.
    Note that this can be very slow for a large number of particles.
    
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
        
    if len(set(P.weight)) > 1:
        n = len(P)
        if verbose:
            print(f'Resampling {n} weighted particles')      
        P = P.resample(n, equal_weights=True)        
        
    for k in ['x', 'xp', 'y', 'yp', 't']:
        h5[k] = P[k]
        
    # p is really beta*gamma    
    h5['p'] = P['p']/P.mass
    
    
    if verbose:
        print(f'Datasets x, xp, y, yp, t, p written to: {h5file}')
        
        
def genesis4_par_to_data(h5, species='electron', smear=True):
    """
    Converts elegant data from an h5 handle or file
    to data for openPMD-beamphysics.
    
    Genesis4 datasets in the HDF5 file are named: 
    'x'
        x position in meters
    'px'
        = gamma * beta_x
    'y'
        y position in meters
    'py'
        = gamma * beta_y'
    'theta'
        angle within a slice in radians
    'gamma' 
        relativistic gamma
    'current' 
        Current in a single slice (scalar) in Amps
        
        
    Parameters
    ----------
    h5: open h5py handle or str
    
    smear: bool:
        Genesis4 often samples the beam by skipping slices
        in a step called 'sample'.
        This will smear out the theta coordinate over these slices, 
        preserving the modulus. 
    
    Returns
    -------
    data: dict for ParticleGroup
        
    
    """
    # Allow for opening a file
    if isinstance(h5, str):
        assert os.path.exists(h5), f'File does not exist: {h5}'
        h5 = File(h5, 'r')    
        
    if species != 'electron':
        raise ValueError('Only electrons supported for Genesis4')
    
    # Scalar arrays.
    # TODO: use refposition?
    scalars = ['beamletsize',
     'one4one',
     'refposition',
     'slicecount',
     'slicelength',
     'slicespacing'] 
    
    params = {}
    units = {}
    for k in scalars:
        assert len(h5[k]) == 1
        params[k] = h5[k][0]
        if 'unit' in h5[k].attrs:
            units[k] = h5[k].attrs['unit'].decode('utf-8')    
        
    # Useful local variables
    ds_slice = params['slicelength'] # single slice length
    s_spacing =  params['slicespacing'] # spacing between slices
    sample = int(s_spacing/ds_slice) # This should be an integer         
            
    x = []
    px = []
    y = []
    py = []
    z = []
    gamma = []
    weight = []
    
    i0 = 0
    for sname in sorted([g for g in h5 if g not in scalars]):
        g = h5[sname]

        current = g['current'][:] # I * s_spacing/c = Q 
        assert len(current) == 1
        # Skip zero current slices. These usually have nans in the particle data.
        if current == 0:
            i0 += sample
            continue        
        
        x.append( g['x'][:])
        px.append(g['px'][:]*mec2)  
        y.append( g['y'][:])
        py.append(g['py'][:]*mec2)  
        gamma.append(g['gamma'][:])
        
        # Smear theta over sample slices
        theta = g['theta'][:]
        irel  = (theta / (2*np.pi) % 1) # Relative bin position (0,1)
        n1 = len(theta)
        if smear:
            z1 = (irel + np.random.randint(0, high=sample, size=n1) + i0) * ds_slice # Random smear
        else: 
            z1 = (irel + i0 )* ds_slice
        z.append(z1)
           
        # Convert current to weight (C)
        # I * s_spacing/c = Q 
        q1 = np.full(n1, current) * s_spacing / c_light / n1
        weight.append(q1) 
        
        i0 += sample # skip samples
       
    # Collect
    x = np.hstack(x)
    px = np.hstack(px)
    y = np.hstack(y)
    py = np.hstack(py)
    gamma = np.hstack(gamma)
    z = np.hstack(z)
    weight = np.hstack(weight)  
    
    n = len(weight)
    p = np.sqrt(gamma**2 -1) * mec2
    pz = np.sqrt(p**2 - px**2 - py**2)  
    
    status=1
    data = {
        'x':x,
        'y':y,
        'z':z,
        'px':px,
        'py':py,
        'pz':pz,
        't': np.full(n, 0),
        'status': np.full(n, status),
        'species':species,
        'weight':weight,
    }    
    
    return data        
        

   
