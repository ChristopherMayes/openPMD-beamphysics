import numpy as np
   
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
    
    zeros = np.full(n, 0)
    
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
        
