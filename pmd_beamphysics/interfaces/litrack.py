from pmd_beamphysics.units import c_light
import numpy as np

   
def write_litrack(particle_group,           
               outfile='litrack.zd',
               p0c=None,
               verbose=False): 

    """
    LiTrack is a Matlab code for the longitudinal phase space only.
    
   ASCII with columns:
   
   c*t (mm), relative energy deviation (%). 
   
   The head of the bunch has smaller ct
   
   
   bunch head is at smaller z.  the relative energy deviation unit is really %, which means I time 100 from relative energy spread.
    
    This routine makes ASCII particles, with column labels:
        'x', 'y', 'z', 'GBx', 'GBy', 'GBz', 't', 'q', 'nmacro'
    in SI units. 
    
    For now, only electrons are supported.

    """

    P = particle_group # convenience
    
    assert P.species == 'electron' # TODO: add more species
    
    assert np.all(P.weight > 0), 'ParticleGroup.weight must be > 0'
    
    n = particle_group.n_particle
    z = np.unique(P['z'])
    assert len(z) == 1, 'All particles must be a the same z. Please call .drift_to_z()'
    
    if p0c is None:
        p0c = P['mean_p']
        if verbose:
            print(f'Using mean_p as the reference momentum: {p0c} eV/c')
        
    
    ct = c_light*P.t
    delta = P.p/p0c -1.0 
    
    
    header = f"""% LiTrack particles
% 
% Created using the openPMD-beamphysics Python package
% https://github.com/ChristopherMayes/openPMD-beamphysics
%
% species: {P['species']}
% n_particle: {n}
% total charge: {P.charge} (C)
% reference momentum p0: {p0c} (eV/c)
%
% Columns: ct, delta = p/p0 -1
% Units: mm, percent"""

    outdat = np.array([ct*1000, delta*100]).T
    
    if verbose:
        print(f'writing {n} LiTrack particles to {outfile}')
    
    # Write ASCII
    np.savetxt(outfile, outdat, header=header, comments='', fmt = '%20.12e')
    
    return outfile
    
