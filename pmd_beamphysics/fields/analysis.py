from pmd_beamphysics.units import mec2, c_light
from pmd_beamphysics.species import charge_state, mass_of

from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.optimize import brent, brentq
import numpy as np


# Numpy migration per https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
if np.lib.NumpyVersion(np.__version__) >= '2.0.0':
    from numpy import trapezoid
else:
    # Support 'trapz' from numpy 1.0
    from numpy import trapz as trapezoid


#----------------------
# Analysis

def accelerating_voltage_and_phase(z, Ez, frequency):
    r"""
    Computes the accelerating voltage and phase for a v=c positively charged particle in an accelerating cavity field.
    
        Z = \int Ez * e^{-i k z} dz 
        
        where k = omega/c = 2*pi*frequency/c
        
        voltage = abs(Z)
        phase = arg(Z)
    
    Input:
        z  (float array):   z-coordinate array (m)
        Ez (complex array): On-axis complex Ez field array (V/m), oscillating as exp(-i omega t), with omega = 2*pi*frequency
        
    Output:
        voltage, phase in V, radian
    
    """
    c=299792458
    omega = 2*np.pi*frequency
    k = omega/c
    fz =Ez*np.exp(-1j*k*z)
    
    # Integrate
    Z = trapezoid(fz, z)
    
    # Max voltage at phase
    voltage = np.abs(Z)
    phase = np.angle(Z)
    
    return voltage, phase
    
    
    
def track_field_1d(z,
                   Ez,
                   frequency=0,
                   z0=0,
                   pz0=0,
                   t0=0,
                   mc2=mec2,  # electron
                   q0=-1,
                   debug=False,
                   max_step=None,
                  ):
    r"""
    Tracks a particle in a 1d complex electric field Ez, oscillating as Ez * exp(-i omega t)
    
    Uses scipy.integrate.solve_ivp to track the particle. 
    
    Equations of motion:
    
    $ \frac{dz}{dt} = \frac{pc}{\sqrt{(pc)^2 + m^2 c^4)}} c $ 

    $ \frac{dp}{dt} = q E_z $

    $ E_z = \Re f(z) \exp(-i \omega t) $
    
    
    Parameters
    ----------
    z : array_like      
        positions of the field Ez (m)
        
    Ez : array_like, complex
        On-axis longitudinal electric field (V/m)
        
    frequency : float
        RF frequency in Hz
        
    z0 :  float, optional = 0
        initial particle position (m)
        
    pz0 : float, optional = 0
        initial particle momentum (eV/c)
        
    t0 : float, optional = 0
        initial particle time (s)
        
    mc2 : float, optional = mec2
        initial particle mass (eV)
        
    q0 : float, optional = -1
        initial particle charge (e) (= -1 for electron)
        
    max_step: float, optional = None
        Maximum timestep for solve_ivp (s)
        None => max_step = 1/(2*frequency)    
        Fields typically have variation over a wavelength,
        so this should help avoid noisy solutions.
                
    debug, bool, optional = False
        If True, Will return the full solution to solve_ivp
    
    Returns
    -------
    z1 : float
        final z position in (m)
        
    pz1 : float:
        final particle momemtum (eV/c)
        
    t1 : float
        final time (s)
    
    
    """
    
    # Make interpolating function
    field = interpolate.interp1d(z, Ez * q0 * c_light, fill_value='extrapolate')
    zmax = z.max()
    tmax = 100/frequency
    omega = 2*np.pi*frequency

    # function to integrate
    def fun(t, y):
        z = y[0]
        p = y[1]  
        zdot = p/np.hypot(p, mc2) * c_light
        pdot =  np.real(field(z)*np.exp(-1j*omega*t))
        return np.array([zdot, pdot])
        
    # Events (stopping conditions)
    def went_backwards(t, y):
        return y[0]
    went_backwards.terminal = True
    went_backwards.direction = -1

    def went_max(t, y): 
        return y[0] - zmax
    went_max.terminal = True
    went_max.direction = 1

    if max_step is None:
        max_step = 1/(10*frequency)
    
    # Solve
    y0 =  np.array([z0, pz0])
    sol = solve_ivp(fun, (t0, tmax), y0,
                first_step = 1/frequency/1000,
                events=[went_backwards, went_max],
               # vectorized=True,   # Make it slower?
                method = 'RK45',
                    max_step=max_step)
           #      max_step=1/frequency/20)
    
    if debug:
        return sol
    
    # Final z, p, t
    zf = sol.y[0][-1]
    pf = sol.y[1][-1]
    tf = sol.t[-1]
    
    return zf, pf, tf 


def track_field_1df(Ez_f,
                   zstop=0,
                   tmax=0,
                   z0=0,
                   pz0=0,
                   t0=0,
                   mc2=mec2,  # electron
                   q0=-1,
                   debug=False,
                   max_step=None,
                    method='RK23'
                  ):
    r"""
    Similar to track_field_1d, execpt uses a function Ez_f
    
    Tracks a particle in a 1d electric field Ez(z, t)
    
    Uses scipy.integrate.solve_ivp to track the particle. 
    
    Equations of motion:
    
    $ \frac{dz}{dt} = \frac{pc}{\sqrt{(pc)^2 + m^2 c^4)}} c $ 

    $ \frac{dp}{dt} = q E_z $

    $ E_z = \Re f(z) \exp(-i \omega t) $
    
    
    Parameters
    ----------

        
    Ez_f : callable
        Ez_f(z, t) callable with two arguments z (m) and t (s)
        On-axis longitudinal electric field (V/m)
        
    zstop : float
        z stopping position (m)        
        
    tmax: float
        maximum timestep (s)
        
    z0 :  float, optional = 0
        initial particle position (m)
        
    pz0 : float, optional = 0
        initial particle momentum (eV/c)
        
    t0 : float, optional = 0
        initial particle time (s)
        
    mc2 : float, optional = mec2
        initial particle mass (eV)
        
    q0 : float, optional = -1
        initial particle charge (e) (= -1 for electron)
        
    max_step: float, optional = None
        Maximum timestep for solve_ivp (s)
        None => max_step = tmax/10
        Fields typically have variation over a wavelength,
        so this should help avoid noisy solutions.
                
    debug, bool, optional = False
        If True, Will return the full solution to solve_ivp
    
    Returns
    -------
    z1 : float
        final z position in (m)
        
    pz1 : float:
        final particle momemtum (eV/c)
        
    t1 : float
        final time (s)
    
    
    """
    
    # function to integrate
    def fun(t, y):
        z = y[0]
        p = y[1]  
        zdot = p/np.hypot(p, mc2) * c_light        
        pdot = Ez_f(z, t) * q0 * c_light
        return np.array([zdot, pdot])
        
    # Events (stopping conditions)
    def went_backwards(t, y):
        return y[0]
    went_backwards.terminal = True
    went_backwards.direction = -1

    def went_max(t, y): 
        return y[0] - zstop
    went_max.terminal = True
    went_max.direction = 1

    if max_step is None:
        max_step = tmax/10
    
    # Solve
    y0 =  np.array([z0, pz0])
    sol = solve_ivp(fun, (t0, tmax), y0,
                first_step = tmax/1000,
                events=[went_backwards, went_max],
               # vectorized=True,   # Make it slower?
                method = method,
                    max_step=max_step)    
    if debug:
        return sol
    
    # Final z, p, t
    zf = sol.y[0][-1]
    pf = sol.y[1][-1]
    tf = sol.t[-1]
    
    return zf, pf, tf 






def autophase_field(field_mesh, pz0=0, scale=1, species='electron', tol=1e-9, verbose=False, debug=False):
    """
    Finds the maximum accelerating of a FieldMesh by tracking a particle and using Brent's method from scipy.optimize.brent. 
    
    NOTE: Assumes that field_mesh.Ez[0,0,:] is the on-axis Ez field.
    TODO: generalize
    
    Parameters
    ----------
    fieldmesh : FieldMesh object
    
    pz0 : float, optional = 0 
        initial particle momentum in the z direction, in eV/c
        pz = 0 is a particle at rest. 
    scale : float, optional = 1 
        Additional field scale. 
    species : str, optional = 'electron'
        species to track.
    tol : float, optional = 1e-9
        Tolerence for brent: Stop if between iteration change is less than tol.
        
    debug : bool, optional = False
        If true, will return a function that tracks the field at a given phase in deg. 
    verbose : bool, optional = False
        If true, prints information about the v=c voltage and phase for the initial guess, and function call information.
        
    Returns
    -------
    phase : float
        Maximum accelerating phase in deg
    pz1 : float        
        Final particle momentum in the z direction, in eV/c
    
    """
    
    # Get field on-axis
    z = field_mesh.coord_vec('z')
    Ez = field_mesh.Ez[0,0,:]*scale
    frequency = field_mesh.frequency
    zmin = z.min()
    
    # Get mass and charge state
    mc2= mass_of(species)
    q0 = charge_state(species) # -1 for electrons
    
    # Function for use in brent
    def phase_f(phase_deg):
        zf, pf = track_field_1d(z,
                   Ez,
                   frequency=frequency,
                   z0=zmin,
                   pz0=pz0,
                   t0=phase_deg/360/frequency,
                   mc2=mc2,
                   max_step= 1/frequency/10,
                   q0=q0)
        return pf
    
    if debug:
        return phase_f
    
    # Get a quick estimate, to use in the bracket
    voltage0, phase0 = accelerating_voltage_and_phase(z, q0*Ez, frequency)
    phase0_deg = phase0*180/np.pi
    if verbose:
        print(f'v=c voltage: {voltage0} V, phase: {phase0_deg} deg')    
    
    alg_sign=-1
    phase_range = [phase0_deg-90, phase0_deg+90]
    phase1_deg, pz1, iter, funcalls = brent(lambda x: alg_sign*phase_f(x%360), brack=phase_range, maxiter=250, tol=tol, full_output=True)
    if verbose:
        print(f'    iterations: {iter}')
        print(f'    function calls: {funcalls}')
    
    return phase1_deg %360, alg_sign*pz1




def autophase_and_scale_field(field_mesh, voltage, pz0=0, species='electron', debug=False, verbose=False):
    """
    Finds the maximum accelerating of a FieldMesh.
    
    Uses two iterations of phasing, scaling.
    
    Parameters
    ----------
    fieldmesh : FieldMesh object
    
    voltage : float
        Desired on-crest voltage in V
        
    pz0 : float, optional = 0 
        initial particle momentum in the z direction, in eV/c
        pz = 0 is a particle at rest. 

    species : str, optional = 'electron'
        species to track.
        
    debug : bool, optional = False
        If true, will return a function that tracks the field at a given phase and scale.
        
    verbose : bool, optional = False
        If true, prints information about the v=c voltage and phase for the initial guess, and function call information.
        
    Returns
    -------
    phase : float
        Maximum accelerating phase in deg
    scale : float        
        scale factor for the field      
    
    """
    
    z = field_mesh.coord_vec('z')
    Ez = field_mesh.Ez[0,0,:]
    frequency = field_mesh.frequency
    zmin = z.min()
    
    # Get mass and charge
    mc2= mass_of(species)
    q0 = charge_state(species)
    energy0 = np.hypot(pz0, mc2)
    
    # Get and initial estimate
    voltage0, phase0 = accelerating_voltage_and_phase(z, q0*Ez, frequency)
    # convert to deg
    phase0 = phase0*180/np.pi    
    scale0 = voltage/voltage0
    if verbose:
        print(f'v=c voltage: {voltage0} V, phase: {phase0} deg')  
    
        
    def phase_scale_f(phase_deg, scale):
        zf, pf, _ = track_field_1d(z,
                   Ez*scale,
                   frequency=frequency,
                   z0=zmin,
                   pz0=pz0,
                   t0=phase_deg/360/frequency,
                   mc2=mc2,
                   max_step= 1/frequency/10,
                   q0=q0)
        
        delta_energy = np.hypot(pf, mc2) - energy0
        
        return delta_energy
    
    if debug:
        return phase_scale_f
    
    # Phase 1
    brack = [phase0-90, phase0+90]
    phase1 = brent(lambda x: -phase_scale_f(x, scale0), brack=brack, maxiter=250, tol=1e-6, full_output=False) %360
    
    # Scale 1
    s0 = scale0*0.9
    s1 = scale0*1.1
    scale1 = brentq(lambda x: phase_scale_f(phase1, x)/voltage - 1.0, s0, s1, maxiter=20, rtol=1e-6, full_output=False)
    
    if verbose:
        print(f'    Pass 1 delta energy: {phase_scale_f(phase1, scale1)} at phase  {phase1} deg')

    # Phase 2
    brack = [phase1-10, phase1+10]
    phase2 = brent(lambda x: -phase_scale_f(x, scale1), brack=brack, maxiter=250, tol=1e-9, full_output=False) %360    
    
    # Scale 2
    s0 = scale1*0.9
    s1 = scale1*1.1
    scale2 = brentq(lambda x: phase_scale_f(phase2, x)/voltage - 1.0, s0, s1, maxiter=20, rtol=1e-9, full_output=False)    
    
    if verbose:
        print(f'    Pass 2 delta energy: {phase_scale_f(phase2, scale2)} at phase  {phase2} deg')
    
    return phase2, scale2
