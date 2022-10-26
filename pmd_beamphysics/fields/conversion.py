import numpy as np

def fieldmesh_rectangular_to_cylindrically_symmetric_data(fieldmesh):
    """
    Returns a rectangular fieldmesh from a cartesion one, extracting the y=0 slice
    
    Parameters
    ----------
    fieldmesh: FieldMesh
    
    Returns
    -------
    new_fieldmesh: FieldMesh
    
    """
    
    assert fieldmesh.geometry == 'rectangular'
    
    # Find central slice x=0, y=0
    ix = np.where(fieldmesh.coord_vec('x') == 0)[0]
    assert len(ix) == 1
    ix = ix[0]
    
    iy = np.where(fieldmesh.coord_vec('y') == 0)[0]
    assert len(iy) == 1
    iy = iy[0]
    
    # Form components
    components = {'electricField/r':     fieldmesh.components['electricField/x'][ix:, iy:iy+1, :],
                  'magneticField/theta': fieldmesh.components['magneticField/y'][ix:, iy:iy+1, :],
                  'electricField/z':     fieldmesh.components['electricField/z'][ix:, iy:iy+1, :],
                 }
    
    # Update attrs
    attrs = fieldmesh.attrs.copy()
    attrs['gridGeometry'] = 'cylindrical'
    attrs['axisLabels'] =  ('r', 'theta', 'z')
    attrs['gridOriginOffset']  = (0, 0,  fieldmesh.attrs['gridOriginOffset'][2])
    attrs['gridSize'] = components['electricField/r'].shape
                  
    return {'attrs':attrs, 'components':components}