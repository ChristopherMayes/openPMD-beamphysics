from scipy.interpolate import UnivariateSpline
import numpy as np

def expand_1d_static_fieldmap(z0, fz0):
    """
    Expands 1D static fieldmap z, fz into r, z using splines.
    
    Cylindrically symmetric geometry. 
    
    This is valid for both electric and magnetic fields. 
    
    Returns functions: 
        fr(r, z), fz(r, z)
    
    """
    
    # Make spline and derivatives
    S = UnivariateSpline(z0, fz0, k=5, s=0)
    Sp = S.derivative()
    Sp2 = S.derivative(2)
    Sp3 = S.derivative(3)
    
    def fz(r, z):   
        return S(z) - r**2 * Sp2(z)/4
        
    def fr(r, z):   
        return -r*Sp(z)/2 + r**3 * Sp3(z)/16   
    
    return fr, fz



def expand_1d_dynamic_fieldmap(z, Ez0, frequency=0):
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
    ooc2 = (omega/c_light)**2 
    
    # Make spline and derivatives
    S = UnivariateSpline(z, Ez0, k=5, s=0)
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