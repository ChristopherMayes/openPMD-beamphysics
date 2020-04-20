import numpy as np

def write_elegant(particle_group,           
               outfile,
               verbose=False): 

    """
    Elegant uses SDDS files. 
    
    Because elegant is an s-based code, particles are drifted to the center. 
    
    This routine writes an SDDS1 ASCII file, with a parameter
        Charge
   and columns
        't', 'x', 'xp', 'y', 'yp', 'p'        
    where 'p' is gamma*beta, in units: 
        elegant units are:
        s, m, 1, m, 1, 1
    
    All weights must be the same. 

    """


    # Work on a copy, because we will drift
    P = particle_group.copy()

    
    # Drift to z. 
    P.drift_to_z()
    
    # Form data
    keys = ['t', 'x', 'xp', 'y', 'yp', 'p']
    dat = {}
    for k in keys:
        dat[k] = P[k]
    # Correct p, this is really gamma*beta    
    dat['p'] /= P.mass
         
    if verbose:
        print(f'writing {len(P)} particles to {outfile}')

    # Note that the order of the columns matters below. 
    header = f"""SDDS1
! 
! Created using the openPMD-beamphysics Python package
! https://github.com/ChristopherMayes/openPMD-beamphysics
! species: {P['species']}
!
&parameter name=Charge, type=double, units=C, description="total charge in Coulombs" &end
&column name=t,  type=double, units=s, description="time in seconds" &end
&column name=x,  type=double, units=m, description="x in meters" &end
&column name=xp, type=double, description="px/pz" &end
&column name=y,  type=double, units=m, description="y in meters" &end
&column name=yp, type=double, description="py/pz" &end
&column name=p,  type=double, description="relativistic gamma*beta" &end
&data mode=ascii &end
{P['charge']}
{len(P)}"""
    
    # Write ASCII
    outdat = np.array([dat[k] for k in keys]).T        
    np.savetxt(outfile, outdat, header=header, comments='', fmt = '%20.12e')    
      
    
    return outfile