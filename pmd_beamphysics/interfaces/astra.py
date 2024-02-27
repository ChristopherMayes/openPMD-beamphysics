import numpy as np
from pmd_beamphysics.status import ParticleStatus
from pmd_beamphysics.readers import component_alias
import os



astra_species_name = {1:'electron', 2:'positron', 3:'proton', 4:'hydrogen'}
astra_species_index = {v:k for k, v in astra_species_name.items()} # Inverse mapping

astra_particle_status_names = {-1:'standard particle, at the cathode',
                        3:'trajectory probe particle',
                        5:'standard particle'}




def parse_astra_phase_file(filePath):
    """

    Parses astra particle dumps to data dict, that corresponds to the
    openpmd-beamphysics ParticeGroup data= input. 
    
    Units are in m, s, eV/c
    
    Live particles (status==5) are relabeled as status = 1.
    Original status == 2 are relabeled to status = 2 (previously unused by Astra)
    
    """
    
    #
    # Internal Astra Columns
    # x   y   z   px   py   pz   t macho_charge astra_index status_flag
    # m   m   m   eV/c eV/c eV/c ns    nC           1              1
    
    #  The first line is the reference particle in absolute corrdinate. Subsequent particles have:
    #  z pz t
    #  relative to the reference. 
    #
    #
    # astra_index represents the species: 1:electrons, 2:positrons, 3:protons, 4:hydroger, ...
    # There is a large table of status. Status_flag = 5 is a standard particle. 
    
    assert os.path.exists(filePath), f'particle file does not exist: {filePath}'
    
    data = np.loadtxt(filePath)
    ref = data[0,:] # Reference particle. 

    # position in m
    x = data[1:,0]
    y = data[1:,1]
    
    z_rel = data[1:,2]    
    z_ref = ref[2]
    #z = z_rel + z_ref
    
    # momenta in eV/c
    px = data[1:,3]
    py = data[1:,4]
    pz_rel = data[1:,5]
    
    pz_ref = ref[5]
    #pz = pz_rel + pz_ref
    
    # Time in seconds
    t_ref = ref[6]*1e-9
    t_rel = data[1:,6]*1e-9
    #t = t_rel + t_ref
    
    # macro charge in Coulomb. The sign doesn't matter, so make positive
    qmacro = np.abs(data[1:,7]*1e-9)
    
    species_index = data[1:,8].astype(int)
    status = data[1:,9].astype(int)  
    
    # Select particle by status 
    #probe_particles = np.where(status == 3) 
    #good_particles  = np.where(status == 5) 

    data = {}
    
    n_particle = len(x)
    
    data['x'] = x
    data['y'] = y
    data['z'] = z_rel + z_ref
    data['px'] = px
    data['py'] = py
    data['pz'] = pz_rel + pz_ref
    data['t_clock']  = t_rel + t_ref #np.full(n_particle, t_ref) # full array
    data['t'] =  t_ref
    
    # Status
    # The standard defines 1 as a live particle, but astra uses 1 as a 'passive' particle
    # and 5 as a 'standard' particle. 2 is not used. 
    # To preserve this information, make 1->2 and then 5->1
    where_1 = np.where(status==1)
    where_5 = np.where(status == 5)
    status[where_1] = 2
    status[where_5] = 1
    data['status'] = status 
    
    data['weight'] = qmacro
    
    unique_species = set(species_index)
    assert len(unique_species) == 1, 'All species must be the same'
    
    # Scalars
    data['species'] = astra_species_name[list(unique_species)[0]]
    data['n_particle'] = n_particle

    return data





def write_astra(particle_group,
                outfile,
                verbose=False,
                probe=False):
    """
    Writes Astra style particles from particle_group type data.
    
    For now, the species must be electrons. 
    
    If probe, the six standard probe particles will be written. 
    """

    def vprint(*a, **k):
        if verbose:
            print(*a, **k)
    
    vprint(f'writing {particle_group.n_particle} particles to {outfile}')
    
    # number of lines in file
    size = particle_group.n_particle + 1 # Allow one for reference particle
    i_start = 1 # Start for data particles
    if probe:
        # Add six probe particles, according to the manual
        size += 6
        i_start += 6
    

    # Astra units and types
    #units = ['m', 'm', 'm', 'eV/c', 'eV/c', 'eV/c', 'ns', 'nC']
    names = ['x', 'y', 'z', 'px', 'py', 'pz', 't', 'q', 'index', 'status']
    types = 8*[float] + 2*[np.int8]

    
    # Reference particle
    ref_particle = {'q':0}
    sigma = {}
    for k in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
        ref_particle[k] = particle_group.avg(k)
        std = particle_group.std(k)
        if std == 0:
            std = 1e-12 # Give some size
        sigma[k] = std
    ref_particle['t'] *= 1e9 # s -> nS
        
    # Make structured array
    dtype = np.dtype(list(zip(names, types)))
    data = np.zeros(size, dtype=dtype)
    for k in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
        data[k][i_start:] = getattr(particle_group, k)
    data['t'] *= 1e9 # s -> nS
    data['q'][i_start:] = particle_group.weight*1e9 # C -> nC
    
    # Set these to be the same
    data['index'] = astra_species_index[particle_group.species]
    
    # Status
    # The standard defines 1 as a live particle, but astra uses 1 as a 'passive' particle
    # and 5 as a 'standard' particle. 2 is not used. 
    # On parsing 1->2 and then 5->1    
    # Revese: 1->5, 2->1
    status = particle_group.status
    astra_status = status.copy()
    astra_status[ np.where(status==1) ] = 5 # Astra normal (alive)
    astra_status[ np.where(status==2) ] = 1 # Astra passive
    astra_status[ np.where(status==ParticleStatus.CATHODE)] = -1 # Astra cathode
    
    data['status'][i_start:] = astra_status
    
    # Handle reference particle. If any -1 are found, assume we are starting at the cathode
    if -1 in astra_status:
        ref_particle['status']= -1 # At cathode
    else:
        ref_particle['status']= 5 # standard particle
    
    # Subtract off reference z, pz, t
    for k in ['z', 'pz', 't']:
        data[k] -= ref_particle[k]
        
    # Put ref particle in first position
    for k in ref_particle:
        data[k][0] = ref_particle[k]
    
    # Optional: probes, according to the manual
    if probe:
        data[1]['x'] = 0.5*sigma['x'];data[1]['t'] =  0.5*sigma['t']
        data[2]['y'] = 0.5*sigma['y'];data[2]['t'] = -0.5*sigma['t']
        data[3]['x'] = 1.0*sigma['x'];data[3]['t'] =  sigma['t']
        data[4]['y'] = 1.0*sigma['y'];data[4]['t'] = -sigma['t']
        data[5]['x'] = 1.5*sigma['x'];data[5]['t'] =  1.5*sigma['t']
        data[6]['y'] = 1.5*sigma['y'];data[6]['t'] = -1.5*sigma['t']        
        data[1:7]['status'] = -3
        data[1:7]['q'] = 0.5e-5 # ? Seems to be required 
        data[1:7]['pz'] = 0 #? This is what the Astra Generator does
    
    # Save in the 'high_res = T' format
    np.savetxt(outfile, data, fmt = ' '.join(8*['%20.12e']+2*['%4i']))

    
    
    
def write_astra_1d_fieldmap(fm, filePath): 
    """
    Writes an Astra fieldmap file from a FieldMesh object.
    
    Requires cylindrical geometry for now.
    
    Parameters
    ----------
    filePath: str
        Filename to write to

    """
    
    z, fz = astra_1d_fieldmap_data(fm)
    
    # Flatten dat   
    dat = np.array([z, fz]).T    
    
    np.savetxt(filePath, dat, comments='') 
    
    
    
def astra_1d_fieldmap_data(fm): 
    """
    Astra fieldmap data.
    
    Requires cylindrical geometry for now.
    
    Returns
    -------
    z: array-like
        z coordinate in meters
    
    fz: array-like
        field amplitude corresponding to z
    
    """    
    
    
    assert fm.geometry == 'cylindrical', f'Geometry: {fm.geometry} not implemented'
    
    assert fm.shape[1] == 1, 'Cylindrical symmetry required'
    
    z = fm.coord_vec('z')
    
    if fm.is_static:
        if fm.is_pure_magnetic:
            fz = np.real(fm['Bz'][0,0,:])
        elif fm.is_pure_electric:
            fz = np.real(fm['Ez'][0,0,:])           
        else:
            raise ValueError('Mixed static field TODO')
            
    else:
        # Assume RF cavity, rotate onto the real axis
        # Get complex fields
        fz = fm.Ez[0,0,:]
        
        # Get imaginary argument (angle)
        iangle = np.unique(np.mod(np.angle(fz), np.pi))
    
        # Make sure there is only one
        assert len(iangle) == 1, print(iangle)
        iangle = iangle[0]
        
        if iangle != 0:
            rot = np.exp(-1j*iangle)
            # print(f'Rotating complex field by {-iangle}')
        else:
            rot = 1   
        fz = np.real(fz*rot)
            
    return z, fz  
    
    
    
    
def vec_spacing(vec):
    """
    Returns the spacing and minimum of a coordinate.
    Asserts that the spacing is uniform
    """
    vmin = vec.min()
    vmax = vec.max()
    n = len(vec)
    assert np.allclose(np.linspace(vmin, vmax, n), vec), 'Non-uniform spacing detected!'
    #return (vmax-vmin)/(n-1) 
    return np.mean(np.diff(vec))
    
    
def parse_astra_fieldmap_3d(filePath, frequency=0):
    """
    Parses a single Astra 3D fieldmap TXT file, described in the Astra manual.
    
    The format is:
        Nx x[1] x[2] ....... x[Nx-1] x[Nx]
        Ny y[1] y[2] ....... y[Ny-1] y[Ny]
        Nz z[1] z[2] ....... z[Nz-1] z[Nz]
        F[ 1, 1, 1] F[ 2, 1, 1] ... F[Nx, 1, 1] F[ 1, 2, 1] F[ 2, 2, 1]... F[Nx, 2, 1]
        F[ 1,Ny,Nz] F[ 2,Ny,Nz]................... F[Nx,Ny,Nz]
    where the items can be written in a free format, ignoring line breaks. 
    
    This routine should be used by:
        read_astra_3d_fieldmaps
        
        
    Parameters
    ----------
    filePath : str
        A single 3d fieldmap file, with extension in :
            'ex', 'ey', 'ez', 'bx', 'by', 'bz'

    Returns
    -------
    attrs : dict of attributes
    
    components: dict of one component, named from the extension.
    
    
    Notes
    -----
    
    Data to be written back to the file should be done as:
    components['Ex'].reshape(nx, ny*nz, order='F').T
    
    
    """
    # Get as a flat array. 
    txt = open(filePath).read().split()
    dat = np.asarray(txt, dtype=float)
    
    # Pick out Nx, Ny, Nz
    nx = int(dat[0])
    ny = int(dat[nx+1])
    nz = int(dat[nx+ny+2])
    
    # Pick out coordinate vectors. These can be irregular!
    xvec = dat[1:nx+1]
    yvec = dat[nx+2:nx+ny+2]
    zvec = dat[nx+ny+3:nx+ny+nz+3]
    
    # Get the grid data in 3D.
    # To write back to the file: grid.reshape(nx, ny*nz, order='F').T
    grid = dat[nx+ny+nz+3:].reshape(nx, ny, nz, order='F')

    
    # Debug
    #if raw:
    #    out = {}
    #    out['xvec'] = xvec
    #    out['yvec'] = yvec
    #    out['zvec'] = zvec    
    #    out['grid'] = grid    
    #    
    #    return out

    # Form proper attrs
    dx = vec_spacing(xvec)
    dy = vec_spacing(yvec)
    dz = vec_spacing(zvec)

    attrs = {}
    attrs['eleAnchorPt'] = 'beginning'
    attrs['gridGeometry'] = 'rectangular'
    attrs['axisLabels'] = ('x', 'y', 'z')
    attrs['gridLowerBound'] = (0, 0, 0)
    attrs['gridOriginOffset'] = (xvec.min(), yvec.min(), zvec.min())
    attrs['gridSpacing'] = (dx, dy, dz)
    attrs['gridSize'] = (nx, ny, nz)
    attrs['harmonic'] = 1
    attrs['fundamentalFrequency'] = frequency
    
    
    # Get a name (actually an alias Ex, By, ...)
    component_name = filePath.split('.')[-1].title()
    
    # There is only one component
    components = {component_name:grid}
    
    return attrs, components


def read_astra_3d_fieldmaps(common_filePath, frequency=0):
    """
    Reads multiple files from the common_filePath without the ex, 
    and returns a data dict to instantiate a FieldMesh object:
    
    Examples
    --------
    
        data = read_astra_3d_fieldmaps('3D_file_from_astra')
            # will parse 3D_file_from_astra.bx, .ey, etc.
        FieldMesh(data=data)
    
    
    
    Parameters
    ----------
    common_filePath: str
        File path of the common fieldmap without the extension.
        The actual field files should have extensions in:
            'ex', 'ey', 'ez', 'bx', 'by', 'bz'
    
    Returns
    -------
    data : dict
        dict of 'attrs' and 'components' to instantiate a FieldMesh object
        
        
        
    Notes
    -----
    This can only accept regular grids. Irregular grids are not in the openPMD standard.
    
    """
    attrs = {}
    components = {}
    for ext in ['ex', 'ey', 'ez', 'bx', 'by', 'bz']:
        file = f'{common_filePath}.{ext}'

        if not os.path.exists(file):
            continue
            
        # Actually parse
        attrs1, components1 = parse_astra_fieldmap_3d(file, frequency=frequency)
        
        # Assert that the attrs are the same
        if not attrs:
            attrs = attrs1
        else:
            assert attrs1 == attrs, 'attrs do not match'
        
        components.update(components1)
        
    data= dict(attrs=attrs, components=components)
    
    return data
    
    
    
def write_astra_3d_fieldmaps(fieldmesh_object, common_filePath):
    """
    
    Parameters
    ----------
    fieldmesh_object : FieldMesh
        .geometry must be 'rectangular'
    
    common_filePath : str
        common filename to write the .ex, .by etc. files to. 
        
    
    Returns
    -------
    flist : list of files written (str)
    """
    
    base = os.path.split(common_filePath)[1]
    assert base[0:2]=='3D' or base[3:5]=='3D', 'The base filename must begin with 3D or have 3D starting at the fourth character, according to Astra'
    
    assert fieldmesh_object.geometry == 'rectangular'
    
    nx, ny, nz = fieldmesh_object.shape
    
    flist = []
    for comp in fieldmesh_object.components:
        # This makes the correct extension
        ext = component_alias[comp].lower()
        fname = f'{common_filePath}.{ext}'
        flist.append(fname)
        
        # Form header
        header = ''
        for key in ['x', 'y', 'z']:
            vec = fieldmesh_object.coord_vec(key)
            vec =  np.around(vec, 15) # remove tiny irregularities
            vstr = ' '.join(vec.astype(str))
            header += f'{len(vec)} {vstr}\n'
            
        # 2D data to write
        dat = fieldmesh_object.components[comp].reshape(nx, ny*nz, order='F').T
        
        np.savetxt(fname, dat, header=header, comments='')
        
    return flist    
    
