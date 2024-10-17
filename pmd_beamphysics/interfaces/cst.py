import numpy as np
import os

from scipy.constants import mu_0 as mu0

def get_scale(unit):

    if(unit=='[mm]'):
        return 1e-3
    elif(unit=='[V/m]'):
        return 1
    elif(unit=='[A/m]'):
        return mu0

def get_vec(x):
    sx = set(x)
    nx = len(sx)
    xlist = np.array(sorted(list(sx)))
    dx = np.diff(xlist)
    assert np.allclose(dx, dx[0])
    dx = dx[0]
    return min(x), max(x), dx, nx


def read_cst_ascii_3d_field(filePath, n_header=2):
    """
    Parses a single 3d field file.
    
    The beginning of the header is:
    x [units] y [units] z [units]...
    -----------------------------...

    This is followed by the specification of the fields. For static fields is:

    ...Fx [units] Fy [units] Fz [units]
    ...--------------------------------

    where F is either E or H with corresponding MKS units.

    For time varying/complex fields, there will be a single file for E & H separately,
    and the remainder of the header will be of the form:

    ...FxRe [units] FyRe [units] FzRe [units] FxIm [units] FyRe [units] FzRe [units]
    ...-----------------------------------------------------------------------------

    
    Data is in F order
    """
    
    with open(filePath, 'r') as fid:
        header = fid.readline()

    headers = header.split()

    columns, units = headers[::2], headers[1::2]

    #print(columns, units)

    field_columns = list(set([c[:2] for c in columns if c.startswith('E') or c.startswith('H')]))

    if all([f.startswith('E') for f in field_columns]):
        field_type = 'electric'
    elif all([f.startswith('H') for f in field_columns]):
        field_type = 'magnetic'
    else:
        raise ValueError('Mixed CST mode not curretly supported.')
    
    dat = np.loadtxt(filePath, skiprows=n_header)
    
    X = dat[:,0]*get_scale(units[0])
    Y = dat[:,1]*get_scale(units[1])
    Z = dat[:,2]*get_scale(units[2])
    
    xmin, xmax, dx, nx = get_vec(X)
    ymin, ymax, dy, ny = get_vec(Y)
    zmin, zmax, dz, nz = get_vec(Z)
    
    shape = (nx, ny, nz)

    # Check if the field is complex:
    if( len(columns)==9 ):

        # - sign to convert to exp(-i omega t)
        Fx = (dat[:,3] - 1j*dat[:,4]).reshape(shape, order='F')*get_scale(units[3])
        Fy = (dat[:,5] - 1j*dat[:,6]).reshape(shape, order='F')*get_scale(units[4])
        Fz = (dat[:,7] - 1j*dat[:,8]).reshape(shape, order='F')*get_scale(units[5])       

    elif( len(columns)==6 ):

        Fx = dat[:,3].reshape(shape, order='F')*get_scale(units[3])
        Fy = dat[:,4].reshape(shape, order='F')*get_scale(units[4])
        Fz = dat[:,5].reshape(shape, order='F')*get_scale(units[5])
    
    attrs = {}
    attrs['gridOriginOffset'] = (xmin, ymin, zmin)
    attrs['gridSpacing'] = (dx, dy, dz)
    attrs['gridSize'] = (nx, ny, nz)
    
    components = {f'{field_type}Field/x':Fx, f'{field_type}Field/y':Fy, f'{field_type}Field/z':Fz}
    
    return attrs, components


def read_cst_ascii_3d_static_field(ffile):
    """
    Parse a complete 3d Real Electric or Magnetic field from corresponding CST E/H ASCII file
    
    """
    
    attrs, components = read_cst_ascii_3d_field(ffile)

    attrs['eleAnchorPt'] = 'center'
    attrs['gridGeometry'] = 'rectangular'
    attrs['axisLabels'] = ('x', 'y', 'z')
    attrs['gridLowerBound'] = (0, 0, 0)
    attrs['harmonic'] = 0
    attrs['fundamentalFrequency'] = 0
    
    data = dict(attrs=attrs, components=components)
                
    return data


def read_cst_ascii_3d_complex_fields(efile, hfile, frequency, harmonic=1):

    """
    Parse a complete 3d fieldmap from corresponding CST E and H field files

    efile: str
        Path to electric field file for full complex electromagnetic mode
        
    hfile: str
        Path to magnetic H-field file for full complex electromagnetic mode

    frequency: float
        Frequency of the mode in [Hz]

    harmonic: int, default = 1
        mode harmonic
    """

    assert os.path.exists(efile), "Could not find electric field file"
    assert os.path.exists(hfile), "Could not find magnetic field file"
    
    e_attrs, e_components = read_cst_ascii_3d_field(efile)
    b_attrs, b_components = read_cst_ascii_3d_field(hfile)

    assert e_attrs['gridOriginOffset'] == b_attrs['gridOriginOffset']
    assert e_attrs['gridSpacing'] == b_attrs['gridSpacing']
    assert e_attrs['gridSize'] == b_attrs['gridSize']
    
    components = {**e_components, **b_components}

    attrs = e_attrs
    attrs['eleAnchorPt'] = 'center'
    attrs['gridGeometry'] = 'rectangular'
    attrs['axisLabels'] = ('x', 'y', 'z')
    attrs['gridLowerBound'] = (0, 0, 0)
    attrs['harmonic'] = harmonic
    attrs['fundamentalFrequency'] = frequency    
    
    data = dict(attrs=attrs, components=components)
                
    return data
    





    