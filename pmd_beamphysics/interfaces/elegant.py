
import numpy as np
import subprocess
import os
   
def write_elegant(particle_group,           
               outfile,
               plaindata2sdds_bin=None,
               verbose=False): 

    """
    Elegant uses SDDS files. The particle file is made with:

    plaindata2sdds output.txt output.sdds -inputMode=ascii -col=t,double -col=x,double -col=xp,double -col=y,double -col=yp,double -col=p,double
    
    Because elegant is an s-based code, particles are drifted to the center. 
    
    This routine makes ASCII particles, with columns
        't', 'x', 'xp', 'y', 'yp', 'p'        
    where 'p' is gamma*beta
        
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
     
    # Write ASCII
    outdat = np.array([dat[k] for k in keys]).T        
    np.savetxt(outfile, outdat, comments='', fmt = '%20.12e')
    
    if verbose:
        print(f'writing {len(P)} particles to {outfile}')

    
    # Form run command
    args = 'output.txt output.sdds -inputMode=ascii'.split()
    args += [f'-col={k},double' for k in keys]
    
    if plaindata2sdds_bin:
        
        tempfile = outfile+'.txt'
        os.rename(outfile, tempfile)
        
        exe = os.path.expandvars(plaindata2sdds_bin)
        
        assert os.path.exists(exe), f'{exe} does not exist'
        cmd = [exe] + args
        if verbose:
            print(' '.join(cmd))
        subprocess.run(cmd)
        # Cleanup
        os.remove(tempfile)
        
    else: 
        runcmd = 'plaindata2sdds '+' '.join(args)
        print(f'ASCII particles written. Convert to SDDS using: {runcmd}')

        

    
        
