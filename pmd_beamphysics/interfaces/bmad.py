import numpy as np
from pmd_beamphysics.units import c_light
from pmd_beamphysics.species import mass_of
 
# Remove from below, because this docstring is used directly in ParticleGroup
#     pg : ParticleGroup
#        The ParticleGroup instance to convert.
def particlegroup_to_bmad(pg, p0c=None, tref=None):
    """
    Convert a ParticleGroup instance to Bmad phase space coordinates.
    
    This function maps the properties of a ParticleGroup to their corresponding
    Bmad phase space coordinates as per the following mapping:
    
    Bmad      openPMD-beamphysics
    ----      -------------------
    Bmad x  = x
    Bmad px = px/p0c
    Bmad y  = y
    Bmad py = py/p0c   
    Bmad z = -beta * c * (t - tref)
    Bmad pz = p/p0c - 1
    Bmad t  = t
    
    Parameters
    ----------
    p0c : float, optional
        Reference momentum times the speed of light (in eV).
        Default is None, which uses pg['mean_p'].
    tref : float, optional
        Reference time (in seconds).
        Default is None, which uses pg['mean_t'].
    
    Returns
    -------
    dict
        A dictionary containing Bmad phase space coordinates.
    """
    
    if p0c is None:
        p0c = pg['mean_p']
    if tref is None:
        tref = pg['mean_t']
    
    # Conversion to Bmad units
    bmad_data = {
        'x': pg.x,
        'y': pg.y,
        'px': pg.px / p0c,
        'py': pg.py / p0c,
        'z': -pg.beta * c_light * (pg.t - tref),
        'pz': pg.p / p0c - 1,
        'charge': pg.weight,
        'species': pg.species,
        'p0c': p0c,
        'tref': tref,
        'state': pg.status,
    }
    
    return bmad_data


def bmad_to_particlegroup_data(bmad_data):
    """
    Convert Bmad particle data to a ParticleGroup data dictionary.
    
    This function reverses the conversion done by particlegroup_to_bmad, mapping
    Bmad phase space coordinates back to a ParticleGroup data format.
    
    Parameters
    ----------
    bmad_data : dict
        A dictionary containing Bmad phase space coordinates.
    
    Returns
    -------
    dict
        A dictionary of data suitable for instantiating a ParticleGroup.
    """
    
    # Conversion to ParticleGroup units
    species = bmad_data['species']
    mc2 = mass_of(species)
    
    p0c = bmad_data['p0c']
    tref = bmad_data.get('tref', 0)
    
    p = (1 + bmad_data['pz']) * p0c
    px = bmad_data['px'] * p0c
    py = bmad_data['py'] * p0c
    pz = np.sqrt(p**2 - px**2 - py**2)
    gamma2 = (p / mc2)**2 + 1
    beta = np.sqrt(1 - 1 / gamma2)
    
    pg_data = {
        'x': bmad_data['x'],
        'px': px,
        'y': bmad_data['y'],
        'py': py,
        'z': np.zeros(len(p)),  # Zero by definition in z-coordinates
        'pz': pz,
        't': tref - bmad_data['z'] / (beta * c_light),
        'species': species,
        'weight': bmad_data['charge'],
        'status': bmad_data['state'],
    }
    
    return pg_data

    
    
    
    
    
def write_bmad(particle_group,           
               outfile,
               p0c = None,
               t_ref = 0,
               verbose=False): 

    """
    Bmad's ASCII format is described in:
    
        https://www.classe.cornell.edu/bmad/manual.html
    
    Bmad normally uses s-based coordinates, with momenta:
        bmad px = px/p0
        bmad py = py/p0
        bmad pz = p/p0 - 1
    and longitudinal coordinate
        bmad z = -beta*c(t - t_ref)
    
    If p0c is given, this style of coordinates is written.
    
    Otherwise, Bmad's time based coordinates are written. 
    
    TODO: Spin
    
    """

    n = particle_group.n_particle            
    x = particle_group.x
    y = particle_group.y

    px = particle_group.px
    py = particle_group.py

    t = particle_group.t
    
    status  = particle_group.status 
    weight  = particle_group.weight
    
    if p0c:
        # s-based coordinates
        
        # Check that z are all the same
        unique_z = np.unique(particle_group.z)
        assert len(unique_z) == 1, 'All particles must be a the same z position'        
        
        px = px/p0c
        py = py/p0c
        
        z = -particle_group.beta*299792458*(t - t_ref)
        pz = particle_group.p/p0c -1.0
        
        
    else:
        # Time coordinates.
        z = t
        pz = particle_group.pz  
        
 
    header = f"""!ASCII::3
0 ! ix_ele, not used
1 ! n_bunch
{n} ! n_particle
BEGIN_BUNCH
{particle_group.species} 
{particle_group.charge}  ! bunch_charge
0 ! z_center
0 ! t_center"""
    
    
    # <x> <px> <y> <py> <z> <pz> <macro_charge> <state> <spin_x> <spin_y> <spin_z>
    
    
    fmt = 7*['%20.12e'] + 1*['%2i']
    
    dat = np.array([x, px, y, py, z, pz, weight, status]).T
    
    footer = """END_BUNCH"""

    
    
    if verbose:
        print(f'writing {n} particles in Bmad ASCII format to {outfile}')
        
    # Write to file        
    np.savetxt(outfile, dat, header=header, footer=footer, fmt=fmt, comments='')
        
