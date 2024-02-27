from scipy.interpolate import UnivariateSpline
from scipy import fft
import numpy as np
from pmd_beamphysics.units import c_light

def expand_1d_static_fieldmap(z0, fz0, spline_s=0):
    """
    Expands 1D static fieldmap z, fz into r, z using splines.
    
    Cylindrically symmetric geometry. 
    
    This is valid for both electric and magnetic fields. 
    
    Returns functions: 
        fr(r, z), fz(r, z)
    
    """
    
    # Make spline and derivatives
    S = UnivariateSpline(z0, fz0, k=5, s=spline_s)
    Sp = S.derivative()
    Sp2 = S.derivative(2)
    Sp3 = S.derivative(3)
    
    def fz(r, z):   
        return S(z) - r**2 * Sp2(z)/4
        
    def fr(r, z):   
        return -r*Sp(z)/2 + r**3 * Sp3(z)/16   
    
    return fr, fz



def expand_1d_dynamic_fieldmap(z, Ez0, frequency=0, spline_s=0):
    """
    Expands 1D dynamic elecric fieldmap z, Ez into r, z using splines.
    
    Cylindrically symmetric geometry. 
    
    Fields oscillate as:
        Er, Ez ~ sin(wt)  <=> ~  cos(wt)
        Btheta ~ cos(wt)      ~ -sin(wt)
    This is the Superfish convention. 
        
    
    Returns real functions: 
        Er(r, z), Ez(r, z), Btheta(r, z)
            
    """
    
    omega = 2*np.pi*frequency
    c_light=299792458.
    
    # Make spline and derivatives
    S = UnivariateSpline(z, Ez0, k=5, s=spline_s)
    Sp = S.derivative()
    Sp2 = S.derivative(2)
    Sp3 = S.derivative(3)
    
    def Ez(r, z):   
        return S(z) - r**2 /4 * (Sp2(z) + (omega/c_light)**2 * S(z))  # sin(omega*t)
        
    def Er(r, z):   
        return -r/2 * Sp(z) + r**3/16 * (Sp3(z) + (omega/c_light)**2 * Sp(z)) # sin(omega*t)
    
    def Btheta(r,z):
        return (r/2 * S(z) - r**3 /16 * (Sp2(z) + (omega/c_light)**2 *S(z) ))*omega/c_light**2. # cos(omega*t)
    
    
    return Er, Ez, Btheta


def spline_derivative_array(z, fz, s=0, k=5):
    
    # Make spline and derivatives
    S = UnivariateSpline(z, fz, k=k, s=s)
    fz1 = S.derivative()(z)
    fz2 = S.derivative(2)(z)
    fz3 = S.derivative(3)(z)
    
    a = np.array([fz, fz1, fz2, fz3]).T
    
    return a
    


def fft_derivative_array(fz, dz, ncoef=30, max_order=3):
    """
    Create derivatives of field data `fz` with regular spacing `dz`
    using an FFT method. 
    
    
    Parameters
    ----------
    fz: array
        Field values with regular spacing
        This is assumed to be periodic
    
    dz: float
        Array spacing
        
    ncoef: int, optional
        Number of Fourier coeffiecients to keep. 
        Default: 30
        
    max_order: int
        Maximum order of derivives to compute
    
        
    Returns
    -------
    array of shape ( len(fz), max_order+1 ) 
        representing the field values and derivatives
    
    
    """
    
    fz = np.real_if_close(fz)
    
    n = len(fz)
    L = (n)*dz 

    # Pad odd length
    if n % 2 == 1:
        odd = True
        fz = np.append(fz, 0)
        L = L + dz
    else:
        odd = False     
        
    # Initial FFT
    y = fft.rfft(fz)
    # Cuttoff 
    y[ncoef:] = 0     
    k = np.arange(len(y))

    derivs = []
    
    for order in range(max_order+1):
        a = fft.irfft(y)
        if odd:
            # Trim off to return the same length as fz
            a = a[:-1] 
        derivs.append(a)
        y *= 2*np.pi*1j*k / (L)

    out = np.array(derivs).T
    
    return out



def expand_radial(r, dfield, frequency=0):
    """
    Expand a field at r from its on-axis derivative array.
    
    See, for example, the Astra manual:
    https://www.desy.de/~mpyflo/Astra_manual/Astra-Manual_V3.2.pdf
    Appendix I: Field expansion formulas
    
    
    Parameters
    ----------
    r: float or array
    
    dfield: derivative array
    
    frequency: float, optional
        frequency in Hz. 
        Default: 0
    
    Returns
    -------
    fr: array 
        r field component
    
    fz: array
        z field component
    
    ftheta: array
        theta field component
    
    """

    f0 = dfield[:,0] # f
    f1 = dfield[:,1] # f'
    f2 = dfield[:,2] # f''
    f3 = dfield[:,3] # f'''
    
    omega = 2*np.pi*frequency
    ooc2 = (omega/c_light)**2  
    
    if frequency == 0:
        fz = f0 - r**2 / 4 * f2 
        fr = -r/2 * f1 + r**3 / 16 * f3
        ftheta = np.zeros_like(f0)                
    else:
        fz = f0 - r**2 / 4 * (f2 + ooc2 * f0) # cos(wt)
        fr = -r/2 * f1 + r**3 / 16 * (f3 + ooc2 * f0) # cos(wt)
        ftheta = (r/2 * f0 - r**3/16 * (f2 + ooc2 * f0) ) # sin(wt) * (-w/c^2 for electric, w for magnetic)
               
    return fr, fz, ftheta




def expand_fieldmesh_from_onaxis(fieldmesh, *,                                  
                                    dr=None,
                                    nr=10,
                                    inplace=False,
                                    method='spline',
                                    ncoef=None,
                                    spline_s = 0,
                                    zmirror = 'auto'
                                      ):
    """
    Create cylindrical FieldMesh data from 1-d on-axis field data.
    
    This uses an FFT method to compute up to third order derivaties,
    and then uses field expansion formulas to populate the component data.
    
    See, for example, the Astra manual
    
    
    Parameters
    ----------
    dr: float, optional
        radial coordinate spacing
        Default: same as z spacing
        
    nr: int, optional
        number of radial coordinates
        Default: 10
        
    frequency: float
        frequency in Hz
        
    method: str, default = 'spline'
        Expansion method to use, one of 'fft' or 'spline'. 
        
    ncoef: int, default None  => will use nz/4
        Number of Fourier coefficients to use for the expansion.
        Only used if method='fft'.
        
    spline_s: float, default 0
        Spline smoothing factor.
        See: from scipy.interpolate.UnivariateSpline
        
    zmirror: str or bool, default 'auto'
        Mirror the field about the minumum z before the expansion.
        This is necessary for non-periodic data
        'auto' will look at f[0]
        
    inplace: bool, default False
        If true, modify in-place. Otherwise a copy will be made.
    
    
    Returns
    -------
    fieldmesh: FieldMesh
    
    """
    
    if not inplace:
        fieldmesh = fieldmesh.copy()
    
    assert fieldmesh.geometry == 'cylindrical'
    
    has_Ez = ('electricField/z' in fieldmesh.components) 
    has_Bz = ('magneticField/z' in fieldmesh.components) 
    
    if has_Bz and has_Ez:
        raise NotImplementedError('Expanding both Ez and Bz not implemented')
        
    if has_Ez:
        fz = fieldmesh['electricField/z'][0,0,:]
    elif has_Bz:
        fz = fieldmesh['magneticField/z'][0,0,:]
    else:
        raise ValueError("Neither Ez nor Bz found")
    
    # Get attrs
    nz = fieldmesh.shape[2]
    dz = fieldmesh.dz
    frequency = fieldmesh.frequency

    # Get real field
    assert all(np.isreal(fz))
    fz = np.real(fz)
    
    zvec = fieldmesh.coord_vec('z')
    
    # Use the same as z
    if dr is None:
        dr = dz
        

    # Methods
    if method == 'fft':
        # Crude heuristic     
        if ncoef is None:
            ncoef = nz//4        

        # Heuristic to mirror field, 1% of field at 0. 
        if zmirror == 'auto':
            if abs(fz[0]/np.abs(fz).max()) > .01:
                zmirror = True
        if zmirror:
            fz = np.append(fz[::-1], fz[1:]) 
        
        dfield = fft_derivative_array(fz, dz, ncoef=ncoef)
        # Now strip off the part we want
        if zmirror:
            dfield = dfield[-nz:, :]
    
        # Collect field derivaties for each r, form large array.
        field = []
        for ir in range(nr):
            r = ir*dr
            field.append(expand_radial(r, dfield, frequency=frequency))
        field = np.array(field)
        
        # Extract
        field_r = field[:, 0,  :] #  cos(wt)
        field_z = field[:, 1,  :] #  cos(wt)
        field_theta = field[:, 2,  :] * -1j * 2*np.pi*frequency  / c_light**2 # -w/c^2 sin(wt) 
        
    
    elif method=='spline':
        rvec = np.linspace(0, dr*(nr-1), nr)
        RR, ZZ =  np.meshgrid(rvec, zvec, indexing='ij')
        
        if frequency == 0:
            Frf, Fzf = expand_1d_static_fieldmap(zvec, fz, spline_s=spline_s)
        else:
            Frf, Fzf, Fthetaf = expand_1d_dynamic_fieldmap(zvec, fz, frequency=frequency, spline_s=spline_s)
            field_theta = Fthetaf(RR, ZZ) * -1j
            
        field_r = Frf(RR, ZZ) #  cos(wt)
        field_z = Fzf(RR, ZZ) #  cos(wt)
        
    else:
        raise ValueError(f"Invalid method: {method}, must be one of ['fft', 'spline']")
    

    # Collect components
    attrs = fieldmesh.attrs
    components = fieldmesh.components
    if has_Ez:
        # TM mode
        components['electricField/r'] = np.expand_dims(field_r, 1) #  cos(wt)
        components['electricField/z'] = np.expand_dims(field_z, 1) #  cos(wt)
        
        if frequency != 0:
            components['magneticField/theta'] = np.expand_dims(field_theta, 1)    # -w/c^2 sin(wt)
            
    elif has_Bz:
        # TE mode
        components['magneticField/r'] = np.expand_dims(field_r, 1)
        components['magneticField/z'] = np.expand_dims(field_z, 1)        
        if frequency != 0:
            components['electricField/theta'] = np.expand_dims(field_theta, 1) * 1j * 2*np.pi*frequency    # w sin(wt)        
        
    
    # Update attrs
    attrs['gridSize'] = (nr, 1, nz)
    attrs['gridSpacing'] = (dr, 0, dz)

    return fieldmesh
