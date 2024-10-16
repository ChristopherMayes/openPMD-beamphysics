import os
import numpy as np
from numpy import pi
from scipy import fft

from pmd_beamphysics.fields.expansion import spline_derivative_array

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
        z_ptp = np.ptp(z)
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






def riffle(a, b):
    return np.vstack((a,b)).reshape((-1,),order='F')

def create_fourier_coefficients(zdata, edata, n=None):
    """
    Literal transcription of Ji's routine RFcoeflcls.f90
    
    https://github.com/impact-lbl/IMPACT-T/blob/master/utilities/RFcoeflcls.f90
    
    Fixes bug with scaling the field by the max or min seen.
    
    Vectorized two loops
    
    Parameters
    ----------
    zdata: ndarray
        z-coordinates
    
    edata: ndarray
        field-coordinates
    
    n: int
        Number of Fourier coefficient to compute.
        None => n = len(edata) //2 + 1
        Default: None
    
    Returns
    -------    
    rfdata: ndarray of float
        Impact-T style Fourier coefficients
    
    """
    ndatareal=len(zdata)
    
    # Cast to np arrays for efficiency
    zdata = np.array(zdata)
    edata = np.array(edata)
    
    # Proper scaling
    scale = max(abs(edata.min()), abs(edata.max()))
    edata /=  scale
    
    if not n:
        n = len(edata) //2 + 1

    Fcoef = np.zeros(n)
    Fcoef2 = np.zeros(n)
    zlen = zdata[-1] - zdata[0]
    
    zhalf = zlen/2.0
    zmid = (zdata[-1]+zdata[0])/2
    h = zlen/(ndatareal-1)
    
    pi = np.pi
    # print("The RF data number is: ", ndatareal, zlen, zmid, h)
    
    jlist = np.arange(n)
    
    zz = zdata[0] - zmid
    Fcoef  = (-0.5*edata[0]*np.cos(jlist*2*pi*zz/zlen)*h)/zhalf
    Fcoef2 = (-0.5*edata[0]*np.sin(jlist*2*pi*zz/zlen)*h)/zhalf
    zz = zdata[-1] - zmid
    Fcoef  += -(0.5*edata[-1]*np.cos(jlist*2*pi*zz/zlen)*h)/zhalf          
    Fcoef2 += -(0.5*edata[-1]*np.sin(jlist*2*pi*zz/zlen)*h)/zhalf
        

    for i in range(ndatareal):
        zz = i*h+zdata[0]
        klo=0
        khi=ndatareal-1
        while (khi-klo > 1):
            k=(khi+klo)//2
            if(zdata[k] - zz > 1e-15):
                khi=k
            else:
                klo=k

        hstep=zdata[khi]-zdata[klo]
        slope=(edata[khi]-edata[klo])/hstep
        ez1 =edata[klo]+slope*(zz-zdata[klo])
        zz = zdata[0]+i*h - zmid

        Fcoef += (ez1*np.cos(jlist*2*pi*zz/zlen)*h)/zhalf
        Fcoef2 += (ez1*np.sin(jlist*2*pi*zz/zlen)*h)/zhalf

    return np.hstack([Fcoef[0], riffle(Fcoef[1:], Fcoef2[1:])]) 



def create_fourier_coefficients_via_fft(fz, n_coef=None):

    fz = np.real_if_close(fz)[:-1] # Skip last point, assumed to be periodic

    # Proper scaling
    fz = fz/ np.abs(fz).max()
    
    norm = len(fz)
     
    # Initial FFT
    y = fft.rfft(fz/norm)
  
    # Cuttoff 
    y = y[:n_coef] 
    
    # Shift to correspond with reconsruction
    sign =  np.resize([-1, 1], len(y)-1)   
    
    # Special signs and flatten
    fcoefs = 2*np.hstack([np.real(y[0]), riffle(  sign*np.real(y[1:]), -sign*np.imag(y[1:]))  ] )  

    return fcoefs






def fourier_field_reconsruction(z, fcoefs, z0=0, zlen=1.0, order=0):
    """
    Field reconsruction from Impact-T style Fourier coefficents. 
    
    See: pmd_beamphysics.interfaces.impact.create_fourier_coefficients
    
    Parameters
    ----------
    z: float
        z-position to reconstruct the field at
        
    fcoefs: ndarray
        Impact-T style Fourier coefficient array.
        
    
    z0: float, optional
        lower bound of the fieldmap z coordinate.
        Default: 0
        
    zlen: float, optional
        length of fieldmap
        Default: 1
        
    order: int, optional
        Order of the field derivative.
        Default: 0
        
    Returns
    -------
    field: float
        reconstructed field value at z
    
    """
    fcomplex = fcoefs[1::2] + 1j * fcoefs[2::2]
    n = np.arange(len(fcomplex)) + 1 # start at n=1
    phi = -2*pi*n*( (z-z0)/zlen - 1/2)
         
    if order == 0:
         fz =  fcoefs[0]/2 + np.sum( np.exp(1j * phi) * fcomplex )
    else:
         fz =  np.sum( (-1j * 2*pi*n/zlen)**order  *np.exp(1j * phi) * fcomplex )
         
    return np.real(fz)


def reconstruction_error(field, fcoefs, n_coef=None):
    """
    Calculate field reconstruction error for a given number of Fourier coefficients.
    Parameters
    ----------
    field: ndarray
        original field
    fcoefs: ndarray
        Impact-T-style Fourier coefficients
    n_coef: int
        Number of coefficients to use. 
        Default: None (use all)
        
    Returns
    -------
    error: float
    """
    if n_coef is None:
        n_pick = len(fcoefs)
    else:
        n_pick = n_coef*2 - 1
        
    z2 = np.linspace(0, 1, len(field))
    field2 = np.array([fourier_field_reconsruction(z, fcoefs[:n_pick]) for z in z2])
    error = np.sqrt(np.sum((field-field2)**2)) / len(field)
    return error



#--------------
# FieldMesh


def create_fourier_data(field_mesh, component, zmirror=False, n_coef=None):
    """
    Create Impact-T-style Fourier coefficients from an on-axis
    field component of a FieldMesh (e.g. "Ez")
    
    Parameters
    ----------
    field_mesh: FieldMesh
        FieldMesh to extract the on-axis field component from
    
    component: str
        Any computable FieldMesh component, 
        such as: 'Ez' or 'Bz'
    
    zmirror: bool, optional
        Mirror the field about z=0.
        Default: False
        
    
    Returns
    -------
    dict with:
        z: ndarray 
            z-coordinates
            
        field: ndarray 
            on-axis (r=0) field at z
            
        fcoefs: ndarray 
            Impact-T-style Fourier foefficients
    
    """
    
    ir = np.where(field_mesh.r == 0)
    if len(ir) != 1: 
        raise ValueError('No r=0 found')
    ir = ir[0][0] 
    
    
    z0 = field_mesh.coord_vec('z')
    
    field = field_mesh[component]
    field0 = np.real(field[ir, 0, :])
    
    if zmirror:
        assert z0[0] == 0, 'z0[0] must be 0 to mirror fields'
        field0 = np.hstack([field0[::-1], field0[1:]])
        z0 = np.hstack([-z0[::-1], z0[1:]])
        
    # Check for zero field
    if np.allclose(field0, 0):
        fcoefs = np.array([0.0])        
    else:    
        fcoefs = create_fourier_coefficients(z0, field0, n=n_coef) 
      
    return {'z': z0,
            'field': field0,
            'fcoefs': fcoefs,
           }






def create_impact_solrf_fieldmap_fourier(field_mesh, 
                               zmirror=False,
                               n_coef=None,
                               err_calc=False):
    """
    
    Parameters
    ----------
    field_mesh: FieldMesh
    
    zmirror: bool, optional
        Mirror the field about z=0.
        Default: False
    
    n_coef: int, optional
        Default: None => create the maximum number of coefficients

    err_calc: bool, optional
        If true, calculates an error by reconstructing the field at all of the original z points.
        Default: False
        
    Returns
    -------
    dict with:
        rfdata: ndarray
        
        zmin: float
        
        zmax: float
        
    """
    output = {}
    
    # Form rfdata according to the Impact-T manual
    rfdata = []
    
    info = {'format':'solrf'}
    field = {}
    
    for component in ('Ez', 'Bz'):
        dat = create_fourier_data(field_mesh, component, 
                                    zmirror=zmirror, n_coef=n_coef) 
        
        z = dat['z']
        if zmirror:
            zmin = z.min()
            zmax = z.max()
        else:
            zmin = 0
            zmax = np.ptp(z)
            
        fcoefs = dat['fcoefs']
        
        # Add to output
        scale = np.abs(dat['field']).max()
        
        info[component+'_scale'] = scale
        if err_calc:
            if scale == 0:
                err = 0
            else:
                err = reconstruction_error(dat['field']/scale, dat['fcoefs'])
            info[component+'_err'] = err

        field[component] = {
            'z0': zmin,
            'z1': zmax,  # = L * n periods
            'L': zmax-zmin, 
            'fourier_coefficients': fcoefs,
        }
        rfdata.append( [len(fcoefs), zmin, zmax, zmax-zmin])
        rfdata.append(fcoefs)

    # Add more info
    info['zmin'] = zmin
    info['zmax'] = zmax

    output = {
     'info': info,
     'data': np.hstack(rfdata),
     'field': field,

    }
    
    return output

def create_impact_solrf_fieldmap_derivatives(field_mesh,
                                    method = 'spline',
                                    spline_s=0, 
                                    spline_k=5):
    """
    
    Creates new-style rfdata consisting of the field and its first 
    three derivatives along the z axis for Ez and Bz. 
    
    FieldMesh.geometry must be 'cylindrical'
    
    Parameters
    ----------
    field_mesh: FieldMesh
    
    spline_s: float, default 0
        Spline smoothing factor.
        See: from scipy.interpolate.UnivariateSpline
       
    spline_k: int, default 5
        Degree of the smoothing spline
        See: from scipy.interpolate.UnivariateSpline       
        
        
    Returns
    -------
    dict with:
        rfdata: ndarray
        
        zmin: float
        
        zmax: float
        
    """
    assert field_mesh.geometry == 'cylindrical'
    if method != 'spline':
        raise NotImplementedError(f"Unknown method '{method}', must be 'spline'")
    
    
    output = {}
    
    # Form rfdata according to the Impact-T manual
    rfdata = []
    
    z = field_mesh.coord_vec('z')
    L = np.ptp(z)
    
    info = {'format':'solrf'}
    field = {}    
    for component in ('Ez', 'Bz'):
        
        fz = field_mesh[component][0,0,:]
        
        fz = np.real_if_close(fz)
        assert all(np.imag(fz) == 0)
        
        field[component] = {
                'z0': 0, # Force
                'z1': L, # = L * n periods
                'L': L}      
        
        if field_mesh.component_is_zero(component):
            # Dummy header
            rfdata.append( np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 0]
            ]))
            info[component+'_scale'] = 0
            field[component]['derivative_array'] = np.array([[0,0,0,0]])
                
        else:
            darray = spline_derivative_array(z, fz, s=spline_s, k=spline_k) 
    
            # Scale
            scale = np.abs(darray[:,0]).max()
            
            # Header        
            rfdata.append( np.array([[len(darray),  0, L , L]])) ## TEMP
            #rfdata.append( np.array([[len(darray),  zmin, zmax, L]]))            
                
            # output[component+'_darray'] = darray # debug
            info[component+'_scale'] = scale
            
            rfdata.append(darray/scale)
            
            #print('MAX: ', rfdata[-1][:,0].max())
            
            field[component]['derivative_array'] = darray/scale          
            
                  
    # Add more info
    info['zmin'] = 0
    info['zmax'] = L
                
    output = {
     'info': info,
     'data': np.vstack(rfdata),
     'field': field,

    }
    
    return output



def create_impact_solrf_ele(field_mesh,
                            *, 
                            zedge=0,
                            name=None,
                            scale=1,
                            phase=0,
                            style='fourier',
                            n_coef=30,
                            zmirror=None,
                            spline_s=1e-6,
                            spline_k = 5,
                            radius = 0.15,
                            x_offset = 0,
                            y_offset = 0,
                            file_id = 666,
                            output_path=None):
    """
    Creat Impact-T solrf element from a FieldMesh
    
    Parameters
    ----------
    name: str, default: None
    
    zedge: float
    
    scale: float, default: 1
    
    phase: float, default: 0
    
    style: str, default: 'fourier'
    
    zmirror: bool, default: None
        Mirror the field about z=0. This is necessary for non-periodic field such as electron guns.
        If None, will autmatically try to detect whether this is necessary.
        
    spline_s: float, default: 0
    
    spline_k: float, default: 0
    
    radius: float, default: 0
    
    x_offset: float, default: 0
    
    y_offset: float, default: 0
    
    file_id: int, default: 666
    
    output_path: str, default: None
        If given, the rfdata{file_id} file will be written to this path
    
    Returns
    -------
    dict with:
      line: str
          Impact-T style element line
          
      ele: dict
          LUME-Impact style element
          
      fmap: dict with:
            data: ndarray
            
            info: dict with
                Ez_scale: float
            
                Bz_scale: float
            
                Ez_err: float, optional
                
                Bz_err: float, optional
            
            field: dict with
                Bz: 
                    z0: float
                    z1: float
                    L: float
                    fourier_coefficients: ndarray
                        Only present when style = 'fourier'
                    derivative_array: ndarray
                        Only present when style = 'derivatives'
                Ez: 
                    z0: float
                    z1: float
                    L: float
                    fourier_coefficients: ndarray 
                        Only present when style = 'fourier'
                    derivative_array: ndarray
                        Only present when style = 'derivatives'
    
    """
    

    if style == 'fourier':     
        
        # Automatically detect need to mirror
        if zmirror is None and abs(field_mesh.Ez[0,0,0]) > 0.1:
            zmirror = True
        
        fmap = create_impact_solrf_fieldmap_fourier(field_mesh, n_coef = n_coef, zmirror=zmirror, err_calc=True)
        info = fmap['info']
        # print(f"{info['Ez_err']} {info['Bz_err']}")
        if info['Ez_err'] > 1e-3 or info['Bz_err'] > 1e-3:
            raise ValueError(f'Field reconstruction error too large with n_coef = {n_coef}!')
        
    elif style == 'derivatives':
        fmap = create_impact_solrf_fieldmap_derivatives(field_mesh, spline_s=spline_s, spline_k=spline_k)
        #print(fdat)
        #info = fmap['info']
        #print(' Scales: ', info['Bz_scale'], info['Ez_scale'])
        
    else:
        raise ValueError(f'Invalid style: {style}')
             
    freq = field_mesh.frequency
    L = field_mesh.dz * (field_mesh.shape[2]-1)

    
    # Scaling
    bscale = scale
    if field_mesh.component_is_zero('Bz'):
        bscale = 0
    bscale = np.round(bscale, 12)
    
    escale = scale
    if field_mesh.component_is_zero('Ez'):
        escale = 0
         
    # Round    
    s = np.round(zedge + L, 12)
    zedge = np.round(zedge, 12)      
    L = np.round(L, 12)
    bscale = np.round(bscale, 12)       
    escale = np.round(escale, 12)  
    theta0_deg = phase * 180/np.pi

    if name is None:
        name = f"solrf_{file_id}"
    
    fieldmap_filename = f'rfdata{file_id}'
    
    if output_path is not None:
        if not os.path.exists(output_path):
            raise ValueError(f'output_path does not exist: {output_path}')
        np.savetxt(os.path.join(output_path, fieldmap_filename), fmap['data'])
    
    ele = {
     'L': L,
     'type': 'solrf',
     'zedge': zedge,
     'rf_field_scale': escale,
     'rf_frequency': freq,
     'theta0_deg': theta0_deg,
     'filename': fieldmap_filename,
     'radius': radius,
     'x_offset': x_offset,
     'y_offset': y_offset,
     'x_rotation': 0.0,
     'y_rotation': 0.0,
     'z_rotation': 0.0, # This is tilt, but shouldn't affect anything because of the cylindrical symmetry.
     'solenoid_field_scale': bscale,
     'name': name,
     's': s, 
          }
    
    line = f"{L} 0 0 105 {zedge} {escale} {freq} {theta0_deg} {file_id} {radius} {x_offset} {y_offset} 0 0 0 {bscale} /name:{name}"
    
    
    
    
    return {'line': line,
            'rfdata': fmap['data'],
             'ele': ele,
             'fmap': fmap}
    
