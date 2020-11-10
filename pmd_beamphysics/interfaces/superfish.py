import scipy.constants
mu_0 = scipy.constants.mu_0

import numpy as np







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
    
    rmin, _, zmin = fm.min
    rmax, _, zmax = fm.max
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
    
    rmin, _, zmin = fm.min
    rmax, _, zmax = fm.max
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
        
        
    else:
        kr = 'Br'
        kz = 'Bz'
        ftype = 'magnetic'
    fr = np.real(fm[kr][:,0,:])
    fz = np.real(fm[kz][:,0,:])
    
    # Unroll the arrays
    dat = np.array([field.reshape(nx*ny, order='F').T for field in [fr, fz]]).T
    
    np.savetxt(filePath, dat, header=header, comments='',  fmt = fmt)
    
    if verbose:
        print(f"Superfish T7 file '{filePath}' written for {ftype} Poisson problem.")
    
    return filePath



