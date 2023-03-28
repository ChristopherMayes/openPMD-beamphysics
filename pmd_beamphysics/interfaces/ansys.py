import numpy as np
from pmd_beamphysics.units import mu_0

def get_vec(x):
    sx = set(x)
    nx = len(sx)
    xlist = np.array(sorted(list(sx)))
    dx = np.diff(xlist)
    assert np.allclose(dx, dx[0])
    dx = dx[0]
    return min(x), max(x), dx, nx


def parse_ansys_ascii_3d(filePath, n_header=2):
    """
    Parses a single 3d field file.
    
    The format is:
    header1
    header2
    x y z re_fx im_fx re_fy im_fy re_fz im_fz 
    ...
    
    in C order
    
    
    Ansys fields oscillate as:
        exp(i*omega*t)
    Which is the opposite of openPMD-beamphysics's convention:
        exp(-i*omega*t)
    """
    
    dat = np.loadtxt(filePath, skiprows=n_header)
    X = dat[:,0]
    Y = dat[:,1]
    Z = dat[:,2]
    
    xmin, xmax, dx, nx = get_vec(X)
    ymin, ymax, dy, ny = get_vec(Y)
    zmin, zmax, dz, nz = get_vec(Z)
    
    shape = (nx, ny, nz)
    
    # - sign to convert to exp(-i omega t)
    Fx = (dat[:,3] - 1j*dat[:,4]).reshape(shape)
    Fy = (dat[:,5] - 1j*dat[:,6]).reshape(shape)
    Fz = (dat[:,7] - 1j*dat[:,8]).reshape(shape)
    
    attrs = {}
    attrs['gridOriginOffset'] = (xmin, ymin, zmin)
    attrs['gridSpacing'] = (dx, dy, dz)
    attrs['gridSize'] = (nx, ny, nz)
    
    components = {'x':Fx, 'y':Fy, 'z':Fz}
    
    return attrs, components

def read_ansys_ascii_3d_fields(efile, hfile, frequency=0):
    """
    
    
    """
    
    attrs1, components1 = parse_ansys_ascii_3d(efile)
    attrs2, components2 = parse_ansys_ascii_3d(hfile)
    
    for k in attrs1:
        v1 = attrs1[k]
        v2 = attrs2[k]
        if not v1 == v2:
            raise ValueError(f"Inconsistent values for {k}: {v1}, {v2}")
    
    components = {}
    for k in components1:
        components[f'electricField/{k}'] = components1[k]
    for k in components1:
        components[f'magneticField/{k}'] = components2[k]* mu_0     
                   
    attrs = attrs1
    attrs['eleAnchorPt'] = 'beginning'
    attrs['gridGeometry'] = 'rectangular'
    attrs['axisLabels'] = ('x', 'y', 'z')
    attrs['gridLowerBound'] = (0, 0, 0)
    attrs['harmonic'] = 1
    attrs['fundamentalFrequency'] = frequency    
    
    data = dict(attrs=attrs, components=components)
                
    return data

