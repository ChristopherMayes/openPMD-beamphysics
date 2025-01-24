from pmd_beamphysics.fields.corrector_modeling import make_thin_straight_wire_fieldmesh
from pmd_beamphysics.fields.corrector_modeling import bfield_from_thin_straight_wire
from scipy.constants import mu_0 as u0
import numpy as np


def get_bfield_from_thin_straight_wire(r1, r2, x, y, z, current=1):

    """
    Function to compute the cylindrical coordinates systems used to calculate Bfield from wire segment

    Parameters
    ----------

    r1 : array
        The 3D coordinates [x, y, z] of one end of the wire [m].

    r2 : array
        The 3D coordinates [x, y, z] of the other end of the wire [m].

    x: float
        The x coordinate of the observation point in global coordinates

    y: float
        The x coordinate of the observation point in global coordinates

    z: float
        The x coordinate of the observation point in global coordinates

    current : float, optional
        The current in the wire in [A]
    """

    # Form 3 vector of observation point
    P = np.array([x, y, z])

    # Check that wire has nonzero length
    assert np.linalg.norm(r2-r1) > 0, 'Must provide distinct points for wire segment'

    # Length of wire
    L = np.linalg.norm(r2-r1)
    
    # Align wire coordinate system so that z points along wire
    zhatp = (r2-r1)/L

    # Find normal vector to wire z axis that intersect observation point
    tmin = np.dot((P-r1), zhatp)
    r0min = r1 + tmin*zhatp

    rp = P - r0min                         # Radial coordinate of the observation point in wire coordinates
    xhatp = rp/np.linalg.norm(rp)          # x-hat prime points from wire to observation point
    yhatp = np.cross(zhatp, xhatp)         # y-hat prime points in direction of magnetic field

    # Same radial coordinate computed two ways as check
    assert np.isclose(np.dot(P-r1, xhatp), np.dot(P-r2, xhatp))  

    # Observation coordinates in write coordinate system
    Rp = np.dot(P-r1, xhatp)
    zp = np.dot(P-r1, zhatp) - L/2

    cm = zp-L/2
    cp = zp+L/2

    rcm = np.sqrt(Rp**2 + cm**2)
    rcp = np.sqrt(Rp**2 + cp**2)

    return (-u0*current/4/np.pi/Rp) * ( cm/rcm - cp/rcp ) * yhatp


def test_make_thin_straight_wire_fieldmesh(seed=42):

    current = 1

    # Randomly oriented wire with length 1 m
    if isinstance(seed, int):
        np.random.seed(seed)
    
    rands = np.random.rand(3)

    # Cylindrical coordinate angles for wire in global coordinates
    phi = 2*np.pi * rands[0]
    theta = np.pi * rands[1]

    # Corresponding global coordinates
    x0 = np.cos(phi)*np.sin(theta)
    y0 = np.sin(phi)*np.sin(theta)
    z0 = np.cos(theta)

    # Corresponding unit vector
    v = np.array([x0, y0, z0])
    vhat = v/np.linalg.norm(v)

    # Wire coordimates
    r1 = -0.5*vhat    # Start point of the wire
    r2 = +0.5*vhat    # End point of the wire

    # Somewhat arbitrary grid in x, y, z
    xmin, xmax, nx = -0.55, +0.55, 4
    ymin, ymax, ny = -0.65, +0.65, 6
    zmin, zmax, nz = -0.75, +0.75, 8

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Get FieldMesh to compute and check fields
    FM = make_thin_straight_wire_fieldmesh(r1, r2,
                                           xmin=xs.min(), xmax=xs.max(), nx=len(xs),
                                           ymin=ys.min(), ymax=ys.max(), ny=len(ys),
                                           zmin=zs.min(), zmax=zs.max(), nz=len(zs))
    
    Bnorm = np.sqrt(FM.Bx**2 + FM.By**2 + FM.Bz**2)
    
    e3x, e3y, e3z = FM.Bx/Bnorm, FM.By/Bnorm, FM.Bz/Bnorm

    for ii, x in enumerate(xs):
        for jj, y in enumerate(ys):
            for kk, z in enumerate(zs):  # Check everywhere in z
    
                ix = np.argwhere(FM.coord_vec('x')==x)[0][0]
                iy = np.argwhere(FM.coord_vec('y')==y)[0][0]
                iz = np.argwhere(FM.coord_vec('z')==z)[0][0]
                
                # Get full field components from a separate standalone test function
                Bvec = get_bfield_from_thin_straight_wire(r1, r2, x, y, z)

                # Get full field components from main underlying function in module
                Bx, By, Bz = bfield_from_thin_straight_wire(X, Y, Z, r1, r2, current)
                
                # Compare with FieldMesh and function underlying FieldMesh
                assert np.isclose(Bvec[0], FM.Bx[ix, iy, iz] )
                assert np.isclose(Bvec[0], Bx[ix, iy, iz]) 
                
                assert np.isclose(Bvec[1], FM.By[ix, iy, iz] )
                assert np.isclose(Bvec[1], By[ix, iy, iz]) 
                
                assert np.isclose(Bvec[2], FM.Bz[ix, iy, iz] )
                assert np.isclose(Bvec[2], Bz[ix, iy, iz]) 


def test_make_thin_straight_wire_fieldmesh_in_inifite_limit():
    
    #*****************************************************************
    # Infinite wire test
    #*****************************************************************
    current = 1
    R = 1
    L = 1000

    p1 = np.array([-L/2, 0, 0])
    p2 = np.array([+L/2, 0, 0])

    x = np.linspace(-L/2, L/2, 11)
    y = np.linspace(0.1, R, 10)  # Makes sure not to evluate on the wire
    z = np.linspace(0.0, R, 10)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    B = get_bfield_from_thin_straight_wire(p1, p2, 0, R, 0, current)

    FM = make_thin_straight_wire_fieldmesh(p1, p2,
                                           xmin=x.min(), xmax=x.max(), nx=len(x),
                                           ymin=y.min(), ymax=y.max(), ny=len(y),
                                           zmin=z.min(), zmax=z.max(), nz=len(z))
    
    B0 = FM.Bz[(X == 0) & (Y == R) & (Z == 0)]
    assert np.isclose(u0 * current / 2 / np.pi / R, B0), "Wire expression does not reproduce infinite limit"
    assert np.isclose(B[2], B0), "Wire expression does not reproduce infinite limit"
    #*****************************************************************
    
