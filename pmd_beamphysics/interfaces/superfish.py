import scipy.constants
mu_0 = scipy.constants.mu_0

import numpy as np








# ------------------
# FieldMesh write T7

def write_fish_t7(fm, filePath, fmt='%10.8e', verbose=False):
    """
    Writes a T7 file from FISH t7data dict. 
    
    Input:
        fm: FieldMesh object
        filePath: requested filePath to write
    
    See:
        superfish.parsers.parse_fish_t7
    """
    
    
    assert fm.geometry == 'cylindrical', 'TODO: cartesian.'
    assert fm.frequency != 0, 'Frequency must be non-zero.'
    
    rmin, _, zmin = fm.mins
    rmax, _, zmax = fm.maxs
    nr, _, nz = list(fm.shape)
    
    
    # Collect these. Units are cm, MHz
    xmin = zmin*100
    xmax = zmax*100
    nx   = nz
    ymin = rmin*100
    ymax = rmax*100
    ny   = nr
    freq = fm.frequency*1e-6
        
    # Get complex fields (helper function)
    Er, Ez, Btheta, _ =  fish_complex_to_real_fields(fm, verbose=verbose)

    # Scale to Superfish units
    Er *= 1e-6 # V/m -> MV/m
    Ez *= 1e-6
    Hphi = Btheta/mu_0  # convert to H field, and in Superfish phase convention. 
    E = np.hypot(Er, Ez)
 
    # Write T7 ASCII
    header = f"""{xmin} {xmax} {nx-1}
{freq}
{ymin} {ymax} {ny-1}"""
    
    # Unroll the arrays
    dat = np.array([field.reshape(nx*ny).T for field in [Ez, Er, E, Hphi]]).T
    
    np.savetxt(filePath, dat, header=header, comments='',  fmt = fmt)
    
    if verbose:
        print(f"Superfish T7 file '{filePath}' written for Fish problem.")    
    
    return filePath


def fish_complex_to_real_fields(fm, verbose=False):
    """
    Internal routine for .interfaces.superfish.write_fish_t7
    
    Takes complex Ez, Er, Btheta and determines the complex angle to rotate Ez onto the real axis.
    
    Returns rotated real fields:
        np.real(Er), np.real(Ez), -np.imag(Btheta), rot_angle
    
    """
    # Get complex fields
    Ez = fm.Ez[:,0,:]
    Er = fm.Er[:,0,:]
    Btheta = fm.Btheta[:,0,:]
    
    # Get imaginary argument (angle)
    iangle = np.unique(np.mod(np.angle(Ez), np.pi))

    # Make sure there is only one
    assert len(iangle) == 1, print(iangle)
    iangle = iangle[0]
    
    if iangle != 0:
        rot = np.exp(-1j*iangle)
        if verbose:
            print(f'Rotating complex field by {-iangle}')
    else:
        rot = 1
    
    Er = np.real(Er*rot)
    Ez = np.real(Ez*rot)
    Btheta = -np.imag(Btheta*rot)

    return Er, Ez, Btheta, -iangle
    

def write_poisson_t7(fm, filePath, fmt='%10.8e', verbose=False):
    """
    Writes a T7 file from POISSON t7data dict. 
    
    
    See:
        superfish.parsers.parse_poisson_t7
    """
    
    
    assert fm.geometry == 'cylindrical', 'TODO: cartesian.'
    assert fm.is_static, 'Static fields are required for Poisson T7'
    
    rmin, _, zmin = fm.mins
    rmax, _, zmax = fm.maxs
    nr, _, nz = list(fm.shape)
    
    
    # Collect these. Units are cm
    # Note: different from FISH!
    ymin = zmin*100
    ymax = zmax*100
    ny   = nz
    xmin = rmin*100
    xmax = rmax*100
    nx   = nr
    
    # Write T7 ASCII
    header = f"""{xmin} {xmax} {nx-1}
{ymin} {ymax} {ny-1}"""
    
    
    if fm.is_pure_electric:
        kr = 'Er'
        kz = 'Ez'
        ftype = 'electric'
        factor = 1
        
    else:
        kr = 'Br'
        kz = 'Bz'
        ftype = 'magnetic'
        factor = 1e4 # T->G
    fr = np.real(fm[kr][:,0,:])*factor
    fz = np.real(fm[kz][:,0,:])*factor
    
    # Unroll the arrays
    dat = np.array([field.reshape(nx*ny, order='F').T for field in [fr, fz]]).T
    
    np.savetxt(filePath, dat, header=header, comments='',  fmt = fmt)
    
    if verbose:
        print(f"Superfish T7 file '{filePath}' written for {ftype} Poisson problem.")
    
    return filePath


def write_superfish_t7(fm, filePath, fmt='%10.8e', verbose=False):
    """
    Writes a Superfish T7 file. This is a simple wrapper for:
        write_fish_t7
        write_poisson_t7
        
    If .is_static, a Poisson file is written. Otherwise a Fish file is written.
    
    Parameters
    ----------
    filePath: str
        File to write to
        
    fmt: str, default = %10.8e'
        Format to write numbers
    
    Returns
    -------
    filePath: str
        File written (same as input)
    
    
    """
    if fm.is_static:
        return write_poisson_t7(fm, filePath, fmt=fmt, verbose=verbose)
    else:
        return write_fish_t7(fm, filePath, fmt=fmt, verbose=verbose)



# ------------------
# Parsers for ASCII T7


def read_superfish_t7(filename,
                      type=None,
                      geometry='cylindrical'):
    """
    Parses a T7 file written by Posson/Superfish.
    
    Fish or Poisson T7 are automatically detected according to the second line.
    
    For Poisson problems, the type must be specified.
    
    Superfish fields oscillate as:
        Er, Ez ~ cos(wt)
        Hphi   ~ -sin(wt)
      
    For complex fields oscillating as e^-iwt
    
        Re(Ex*e^-iwt)   ~ cos
        Re(-iB*e^-iwt) ~ -sin        
    and therefore B = -i * mu_0 * H_phi is the complex magnetic field in Tesla
    

    Parameters:
    ----------
    filename: str
        T7 filename to read
    type: str, optional
        For Poisson files, required to be 'electric' or 'magnetic'. 
        Not used for Fish files
    geometry: str, optional
        field geometry, currently required to be the default: 'cylindrical'
    
    Returns:
    -------
    fieldmesh_data: dict of dicts:
        attrs
        components
        
        
    A FieldMesh object is instantiated from this as:
        FieldMesh(data=fieldmesh_data)
    
    """
    
    
    # ASCII parsing
    
    # Read header and data. 
    # zmin(cm), zmax(cm), nx-1
    # freq(MHz)
    # ymin(cm), ymax(cm), ny-1
    with open(filename, 'r') as f:
        line1 = f.readline().split()
        line2 = f.readline().split()
        # Read all lines and flatten the data. This accepts an old superfish format
        dat = f.read().replace('\n' , ' ').replace('\t', ' ').split()
        dat = np.loadtxt(dat)
        
    # The length of the second line gives a clue about the problem type
    if len(line2) == 1:
        problem = 'fish'
        line3 = dat[0:3] # Final header line
        dat = dat[3:] # body data
        n = len(dat)
        assert n % 4 == 0, f'{n} should be divisible by 4'
        dat = dat.reshape(n//4, 4)
        
    else:
        problem = 'poisson'
        n = len(dat)
        assert n % 2 == 0, f'{n} should be divisible by 2'
        dat = dat.reshape(len(dat)//2, 2)
    
    components = {}
    if problem=='fish':
        # zmin(cm), zmax(cm), nz-1
        # freq(MHz)
        # rmin(cm), rmax(cm), nr-1
        # 4 columns of data: Ez(MV/m), Er(MV/m), E(MV/m), Hphi(A/m)
        
        # FISH problem   
        zmin, zmax, nz =  float(line1[0])*1e-2, float(line1[1])*1e-2, int(line1[2])+1
        frequency = float(line2[0])*1e6 # MHz -> Hz
        _rmin, rmax, nr =  float(line3[0])*1e-2, float(line3[1])*1e-2, int(line3[2])+1  
        
        # Read and reshape
        # dat = np.loadtxt(filename, skiprows=3)
        #labels=['Ez', 'Er', 'E', 'Hphi']
        dat = dat.reshape(nr, 1, nz, 4)
        
        components['electricField/z'] = dat[:,:,:,0].astype(complex) * 1e6 # MV/m -> V/m
        components['electricField/r'] = dat[:,:,:,1].astype(complex) * 1e6 # MV/m -> V/m
        components['magneticField/theta'] = dat[:,:,:,3]  * -1j*mu_0 # A/m -> T
        
    else:
        # rmin(cm), rmax(cm), nx-1    # r in cylindrical geometry
        # zmin(cm), zmax(cm), ny-1    # z in cylindrical geometry
        
        # POISSON problem
        _rmin, rmax, nr =  float(line1[0])*1e-2, float(line1[1])*1e-2, int(line1[2])+1
        zmin, zmax, nz =  float(line2[0])*1e-2, float(line2[1])*1e-2, int(line2[2])+1        
        frequency=0
        
        # The structure here is different 
        # dat = np.loadtxt(t7file, skiprows=2)
        dat = dat.reshape(nz, 1, nr, 2)
        
        # type must be specified
        if type == 'electric':
            components['electricField/r'] = dat[:,:,:,0].T # V/m
            components['electricField/z'] = dat[:,:,:,1].T # V/m
        elif type == 'magnetic':
            components['magneticField/r'] = dat[:,:,:,0].T*1e-4 # G -> T
            components['magneticField/z'] = dat[:,:,:,1].T*1e-4 # G -> T
        else:
            raise ValueError("Poisson problems must specify type as 'electric' or 'magnetic")
    
    
    dz = (zmax-zmin)/(nz-1)
    dr = (rmax)/(nr-1)
    
    # Attributes
    attrs = {}
    attrs['eleAnchorPt'] = 'beginning'
    attrs['gridGeometry'] = 'cylindrical'
    attrs['axisLabels'] = ('r', 'theta', 'z')
    attrs['gridLowerBound'] = (0, 1, 0)
    attrs['gridSize'] = (nr, 1, nz)        
    attrs['gridSpacing'] = (dr, 0, dz)    
    
    # Set requested zmin
    attrs['gridOriginOffset'] = (0, 0, zmin) 
    
    attrs['fundamentalFrequency'] = frequency
    attrs['RFphase'] = 0
    if frequency == 0:
        attrs['harmonic'] = 0
    else:
        attrs['harmonic'] = 1
    
    
    return dict(attrs=attrs, components=components)


