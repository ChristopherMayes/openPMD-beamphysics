import numpy as np

from pmd_beamphysics.species import mec2, e_charge

from scipy.constants import physical_constants

amu_to_rest_mass_energy = physical_constants['atomic mass constant energy equivalent in MeV'][0]*1e6


def identify_species(mass, charge):

    m = round(mec2)    
    q = round(charge*1e20)/1e20

    if m == 510999:
        if q == 1.6e-19:
            return 'positron'
        if q == -1.6e-19:
            return 'electron'
        
    raise Exception(f'Cannot identify species with mass {mass} and charge {charge}')

    
def flip_xaxis_to_zaxis(x, y, z):  
    return (-z, y, x)

def flip_zaxis_to_xaxis(x, y, z):
    return (z, y, -x)

def read_simion_ION_file(filename):
    """
    Read a SIMION *.ion file directly into a dictionary compatible dictionary

    *.ion files are used by SIMION to define input particles for tracking

    The keys of the dictionary are

    TOB: "time of birth" [microsec]
    MASS: ion mass [atomic mass units]
    CHARGE: ion charge [e], so an electron has CHARGE = -1
    X: x coordinate [mm]  NOTE: the main axis of cylindrical symmetry in SIMION is x, vs. z for particle group
    Y: y coordinate [mm]
    Z: z coordinate [mm]
    AZ: azimuthal angle of ions momentum vector [deg]  NOTE: SIMION's angles are not standard sphereical coordinatre angles 
    EL: elevation angle of ions momentum vector [deg]  NOTE: SIMION's angles are not standard sphereical coordinatre angles 
    KE: kinetic energy [eV]
    CWF: charge weighting factor (for non-uniform weighting in Space Charge calculations)
    COLOR: int [0, 1, 2...] - defines color of ion track in SIMION plotting

    """
    
    simion_vars = ['TOB', 'MASS', 'CHARGE', 'X', 'Y', 'Z', 'AZ', 'EL', 'KE', 'CWF', 'COLOR']
    
    data = np.loadtxt(filename, skiprows=1, delimiter=',')

    return {p: data[:, simion_vars.index(p)] for p in simion_vars}


def write_simion(particle_group, outfile, verbose=0, color=0, flip_z_to_x=True):

    """
    Write a particle group to SIMION *ion file

    NOTE: the main axis of cylindrical symmetry in SIMION is x, vs. z for particle group

    particle_group: ParticleGroup, object for writing
    outfile: str, filename or path to write particles to
    verbose: flag for printing
    color: int, 0, 1, 3,... defines color of ion track in SIMION (0 = black)
    flip_z_to_z: bool, rotate the particle group beam axis from z to x (default in SIMION)
    """
    
    header=';0'
    
    simion_params= ['TOB', 'MASS', 'CHARGE', 'X', 'Y', 'Z', 'AZ', 'EL', 'KE', 'CWF', 'COLOR']

    # simion_units = {
    #     "TOB": "usec",
    #     "MASS": "amu",
    #     "CHARGE": "e",
    #     "X": "mm",
    #     "Y": "mm",
    #     "Z": "mm",
    #     "AZ": "deg",
    #     "EL": "deg",
    #     "CWF": "",
    #     "COLOR": "",
    # }

    N = len(particle_group)
    
    data = np.zeros( (N, len(simion_params)) )
    data[:, simion_params.index('TOB')] = particle_group.t*1e6    # [P.t] = sec, convert to usec
    
    if(particle_group.species == 'electron'):
        data[:, simion_params.index('MASS')] = np.full(N, particle_group.mass/amu_to_rest_mass_energy)
        data[:, simion_params.index('CHARGE')] = np.full(N, -1)
    else:
        raise ValueError(f'Species {particle_group.species} is not supported')
    
    if(flip_z_to_x):
        x, y, z = flip_zaxis_to_xaxis(particle_group.x, particle_group.y, particle_group.z)
    else:
        x, y, z = particle_group.x, particle_group.y, particle_group.z
    
    data[:, simion_params.index('X')] = x*1e3
    data[:, simion_params.index('Y')] = y*1e3
    data[:, simion_params.index('Z')] = z*1e3
    
    if(flip_z_to_x):
        px, py, pz = flip_zaxis_to_xaxis(particle_group.px, particle_group.py, particle_group.pz)
    else:
        px, py, pz = particle_group.px, particle_group.py, particle_group.pz
    
    # Convert momenta to KE, azimuthal angle, elevation angle defined in SIMION manual
    data[:, simion_params.index('KE')] = particle_group.kinetic_energy                         # [eV] 
    data[:, simion_params.index('AZ')] = np.arctan2(-pz, px) * (180/np.pi)                     # [deg]
    data[:, simion_params.index('EL')] = np.arctan2(py, np.sqrt(px**2 + pz**2) ) * (180/np.pi) # [deg]
    
    # Charge Weighting Factor, derive from particle group weights
    data[:, simion_params.index('CWF')] = particle_group.charge / e_charge                
    data[:, simion_params.index('COLOR')] = np.full(N, color)

    np.savetxt(outfile, data, delimiter=',', header=header, comments='', fmt='  %.9e')


def simion_ion_file_particles_to_particle_data(filename, flip_x_to_z=True):

    """
    Read a SIMION *.ion file to particle data dictionary suitable for ParticleGroup

    NOTE: the main axis of cylindrical symmetry in SIMION is x, vs. z for particle group

    outfile: str, filename or path to write particles to
    flip_x_to_z: bool, rotate the SIMION beam axis from x to z (default in ParticleGroup)

    Returns: dict, defines data for a ParticleGroup
    """

    ions = read_simion_ION_file(filename)
    
    data={'n_particle': len(ions['X']),
          't': +ions['TOB']*1e-6,}
     
    charge = ions['CHARGE']*e_charge
    mass = ions['MASS']*amu_to_rest_mass_energy
    
    assert len(np.unique(mass))==1, 'Only one type of ion allowed'
    assert len(np.unique(charge))==1, 'Only one type of ion allowed'
    
    species = identify_species(mass[0], charge[0])
    data['species'] = species
    
    p = np.sqrt( (ions['KE'] + mec2)**2 - mec2**2) # total momentum in [eV/c]
    
    phi = ions['AZ']*(np.pi/180)
    theta = ions['EL']*(np.pi/180)
    
    x = ions['X']*1e-3
    y = ions['Y']*1e-3
    z = ions['Z']*1e-3
    
    px = p*np.cos(theta)*np.cos(phi)
    py = p*np.sin(theta)
    pz = p*np.cos(theta)*np.sin(phi)
    
    if(flip_x_to_z):
        x,   y,  z = flip_xaxis_to_zaxis( x,  y,  z)
        px, py, pz = flip_xaxis_to_zaxis(px, py, pz)

    data['x'], data['y'], data['z'] = x, y, z 
    data['px'], data['py'], data['pz'] = px, py, pz   
    
    data['id'] = np.arange(1, data['n_particle']+1)
    data['weight'] = np.abs(ions["CWF"])/data['n_particle']
    
    data['weight'] = data['weight']/np.sum(data['weight']) # norm
    data['status'] = np.full(data['n_particle'], 1)
    
    return data

def KE_AZ_EL_to_momentum(KE, AZ, EL):

    p = np.sqrt( (KE + mec2)**2 - mec2**2) # total momentum in [eV/c]
    phi = AZ*(np.pi/180)
    theta = EL*(np.pi/180)

    px = p*np.cos(theta)*np.cos(phi)
    py = p*np.sin(theta)
    pz = p*np.cos(theta)*np.sin(phi)

    return (px, py, pz)



