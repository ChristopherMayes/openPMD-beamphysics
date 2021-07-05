from pmd_beamphysics.units import mec2

import numpy as np
import subprocess
import os

from h5py import File

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
        s, m, 1, m, 1, 'mass*c'
    
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
        
    # Correct units for p, depending on the species
    species = particle_group.species
    if species in ['electron', 'positron']:
        p_units = 'units="m$be$nc", '
    elif species in ['proton']:
        p_units = 'units="m$bp$nc", '        
    else:
        p_units = ''

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
&column name=p,  type=double, {p_units}description="relativistic gamma*beta" &end
&data mode=ascii &end
{P['charge']}
{len(P)}"""
    
    # Write ASCII
    outdat = np.array([dat[k] for k in keys]).T        
    np.savetxt(outfile, outdat, header=header, comments='', fmt = '%20.12e')    
      
    
    return outfile



def elegant_h5_to_data(h5, group='page1', species='electron'):
    """
    Converts elegant data from an h5 handle or file to data for openPMD-beamphysics
    
    Elegant h5 has datasets:
    
    x, xp, y, xp, p=beta*gamma, t
    m,  1,   m, 1, 1, s
    
    Momentum are reconstructed as:
    
    pz = p / sqrt(1+xp^2 + yp^2)
    px = xp * pz
    py = yp * pz
    
    In s-based codes, z=0 by definition. 
    
    For now, only electrons are allowed.
    
    Units are checked. 
    
    All particles are assumed to be live (status = 1)

    TODO: More species. 
        
    """
    
    # Allow for opening a file
    if isinstance(h5, str):
        assert os.path.exists(h5), f'File does not exist: {h5}'
        h5 = File(h5, 'r')
        
    if group:
        g = h5[group]
    else:
        g = h5
        
    assert species=='electron', f'{species} not allowed yet. Only electron is implemented.'    
    mc2 = mec2
        
    # These should exist
    col = g['columns']
    par = g['parameters']
    
    p = col['p'][:]*mc2
    xp = col['xp'][:]
    yp = col['yp'][:]
    pz = p/np.sqrt(1 + xp**2 + yp**2)
    px = xp * pz
    py = yp * pz
    

    # Check charge unit
    charge = par['Charge'][0]
    charge_unit = par['Charge'].attrs['units']
    assert charge_unit == b'C', 'Expected C as unit for Charge'
    
    # Check dataset units
    expected_units = {
        'p':b'm$be$nc',
        'xp':b'',
        'yp':b'',
        'x':b'm',
        'y':b'm',
        't':b's'
    }    
    for c, v in expected_units.items():
        u = col[c].attrs['units']
        assert u==v, f'Dataset {c} units expected to have {v}, but have {u}'
    
    # number of particles
    n = len(p)
    
    status=1
    data = {
        'x':col['x'][:],
        'y':col['y'][:],
        'z':np.full(n, 0),
        'px':px,
        'py':py,
        'pz':pz,
        't': col['t'][:],
        'status': np.full(n, status),
        'species':species,
        'weight': np.full(n, abs(charge)/n),
        'id':  col['particleID'][:]
    }
    return data





def load_sdds(sddsfile, columns, sdds2plaindata_bin='sdds2plaindata', verbose=False):
    """
    Get tabular data from SDDS file
    
    """
    
    outfile = sddsfile+'_table'
    cmd0 = [sdds2plaindata_bin, sddsfile, outfile, '-noRowCount','-outputMode=ascii']
    
    cmd = cmd0 + [f'-col={c}' for c in columns] + ['-separator= ']
    
    if verbose:
        print('load_sdds command: ', ' '.join(cmd))
    
    output, error  = subprocess.Popen(
                    cmd, universal_newlines=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    if error:
        raise ValueError('load_sdds error: '+error)
    # Read table  
    rdat = np.loadtxt(outfile, ndmin=2)

    dat = {}
    for i, key in  enumerate(columns):
        dat[key] = rdat[:,i]
    
    # Cleanup
    os.remove(outfile)
    
    return dat


def elegant_to_data(sddsfile, charge=1.0, sdds2plaindata_bin='sdds2plaindata', species='electron', verbose=False):
    """
    Converts elegant SDDS data to data for openPMD-beamphysics.
    
    Similar to elegant_h5_to_data, which is better because it checks units and does not need the conversion to ASCII
    
    Requires the SDDS utility sdds2plaindata
    
    Columns in the file should be:
    
    x, xp, y, xp, p=beta*gamma, t
    m,  1,   m, 1, 1, s
    
    Momentum are reconstructed as:
    
    pz = p / sqrt(1+xp^2 + yp^2)
    px = xp * pz
    py = yp * pz
    
    In s-based codes, z=0 by definition. 
    
    For now, only electrons are allowed.
    
    
    All particles are assumed to be live (status = 1)

    TODO: More species. 
    
    Also see: elegant_h5_to_data
        
    """
    
    col = load_sdds(sddsfile, ['x', 'xp', 'y', 'yp', 't', 'p'], sdds2plaindata_bin=sdds2plaindata_bin, verbose=verbose)
    
        
    assert species=='electron', f'{species} not allowed yet. Only electron is implemented.'    
    mc2 = mec2
            
    p =  col['p']*mc2
    xp = col['xp']
    yp = col['yp']
    pz = p/np.sqrt(1 + xp**2 + yp**2)
    px = xp * pz
    py = yp * pz   
    
    # number of particles
    n = len(p)
    
    status=1
    data = {
        'x':col['x'],
        'y':col['y'],
        'z':np.full(n, 0),
        'px':px,
        'py':py,
        'pz':pz,
        't': col['t'],
        'status': np.full(n, status),
        'species':species,
        'weight': np.full(n, abs(charge)/n)
    }
    return data
