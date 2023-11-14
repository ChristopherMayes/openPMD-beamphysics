from pmd_beamphysics.units import mu_0

def get_vec(x):
    sx = set(x)
    nx = len(sx)
    xlist = np.array(sorted(list(sx)))
    dx = np.diff(xlist)
    assert np.allclose(dx, dx[0])
    dx = dx[0]
    return min(x), max(x), dx, nx


def parse_cst_ascii_3d(filePath, n_header=2):
    """
    Parses a single 3d field file.
    
    The header format is:
    x [units] y [units] z [units] re_fx [units] im_fx [units] re_fy [units] im_fy [units] re_fz [units] im_fz [units]
    -------------
    ...
    
    in F order
    
    CST fields oscillate as:
        exp(i*omega*t)
    Which is the opposite of openPMD-beamphysics's convention:
        exp(-i*omega*t)
    """
    
    with open(filePath, 'r') as fid:
        header = fid.readline()

    headers = header.split()

    vars, units = headers[::2], headers[1::2]
    
    dat = np.loadtxt(filePath, skiprows=n_header)
    
    X = dat[:,0]*get_scale(units[0])
    Y = dat[:,1]*get_scale(units[1])
    Z = dat[:,2]*get_scale(units[2])
    
    xmin, xmax, dx, nx = get_vec(X)
    ymin, ymax, dy, ny = get_vec(Y)
    zmin, zmax, dz, nz = get_vec(Z)
    
    shape = (nx, ny, nz)
    
    # - sign to convert to exp(-i omega t)
    Fx = (dat[:,3] - 1j*dat[:,4]).reshape(shape, order='F')*get_scale(units[3])
    Fy = (dat[:,5] - 1j*dat[:,6]).reshape(shape, order='F')*get_scale(units[4])
    Fz = (dat[:,7] - 1j*dat[:,8]).reshape(shape, order='F')*get_scale(units[5])
    
    attrs = {}
    attrs['gridOriginOffset'] = (xmin, ymin, zmin)
    attrs['gridSpacing'] = (dx, dy, dz)
    attrs['gridSize'] = (nx, ny, nz)
    
    components = {'x':Fx, 'y':Fy, 'z':Fz}
    
    return attrs, components

def read_cst_ascii_3d_fields(efile, hfile, frequency=0):
    """
    Parse a complete 3d fieldmap from corresponding CST E and H field files
    
    """
    
    attrs1, components1 = parse_cst_ascii_3d(efile)
    attrs2, components2 = parse_cst_ascii_3d(hfile)
    
    for k in attrs1:
        v1 = attrs1[k]
        v2 = attrs2[k]
        if not v1 == v2:
            raise ValueError(f"Inconsistent values for {k}: {v1}, {v2}")
    
    components = {}
    for k in components1:
        components[f'electricField/{k}'] = components1[k]
    for k in components1:
        components[f'magneticField/{k}'] = components2[k] 
                   
    attrs = attrs1
    attrs['eleAnchorPt'] = 'center'
    attrs['gridGeometry'] = 'rectangular'
    attrs['axisLabels'] = ('x', 'y', 'z')
    attrs['gridLowerBound'] = (0, 0, 0)
    attrs['harmonic'] = 1
    attrs['fundamentalFrequency'] = frequency    
    
    data = dict(attrs=attrs, components=components)
                
    return data