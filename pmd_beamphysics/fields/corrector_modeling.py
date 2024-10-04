from scipy.constants import mu_0 as u0
from scipy.constants import pi

from matplotlib import pyplot as plt
import numpy as np

from pmd_beamphysics import FieldMesh

def set_axes_equal(ax):
    """
    Set 3D plot axes to have equal scale.
    This makes the aspect ratio of the plot equal so that spheres appear as spheres,
    cubes as cubes, etc.
    
    Parameters:
    ax (Axes3D): A matplotlib 3D axis object.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    
    # Find the max range for all axes
    max_range = np.abs(limits[:, 1] - limits[:, 0]).max() / 2.0
    
    # Calculate midpoints for all axes
    mid_x = np.mean(limits[0])
    mid_y = np.mean(limits[1])
    mid_z = np.mean(limits[2])
    
    # Set limits to be centered and equal in range
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def plot_3d_vector(v, 
                   origin=np.array([0,0,0]), 
                   plot_arrow=True,
                   plot_line=False,
                   ax=None, 
                   color='b', 
                   elev=45, 
                   azim=-45):
    """
    Plot a 3D vector as an arrow using matplotlib.
    
    Parameters:
    v (array-like): Direction of the vector (dx, dy, dz).
    origin (array-like): Starting point of the vector (x, y, z).
    ax (Axes3D): Optional. The Axes3D object to plot on. If not provided, a new figure will be created.
    """
    # Create a new figure and 3D axes if none are provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    if plot_arrow and not plot_line:
        
        # Plot the vector as an arrow
        ax.quiver(
            origin[0], origin[1], origin[2],  # Starting point of the vector
            v[0], v[1], v[2],  # Vector components (dx, dy, dz)
            arrow_length_ratio=0.2,  # Controls the size of the arrowhead
            color=color,  # Color of the vector (blue)
            linewidth=2  # Line width for the vector
        )
        
    elif not plot_arrow and plot_line:
        ax.plot(origin, v, color=color)
        
    else:

        r = v + origin
        ax.scatter(r[0], r[1], r[2])
    
    ax.view_init(elev=elev, azim=azim)

    return ax


def bfield_from_thin_straight_wire(x, y, z, p1, p2, I, plot_wire=False, elev=45, azim=-45, ax=None):
    """
    Vectorized calculation of magnetic field from a thin straight wire
    over a grid of points specified by x, y, z arrays.
    
    Parameters:
    x, y, z : ndarray, [m]
        Arrays of coordinates of points in space where the magnetic field is calculated.
    p1, p2 : ndarray, [m]
        Arrays specifying the start and end points of the wire.
    I : float, [A]
        Current through the wire (in Amperes).
    plot_wire: boolean
        Plot the wire in 3D space.
    elev: float, [deg]
        elev option to 3d plot in matplotlib.
    azim: float, [deg]
        azim option to 3d plot in matplotlib.
    ax: axes handle
        axes handle for plotting the wire.
    
    
    Returns:
    B : ndarray, [T]
        Magnetic field vector at each point (x, y, z) with shape (Nx, Ny, Nz, 3).
    """
    # Convert input points to numpy arrays
    p1 = np.array(p1)     # three vector defining beginning of current element
    p2 = np.array(p2)     # three vector defining end of current element
    
    # Ensure the wire is specified by two distinct points
    assert np.linalg.norm(p2 - p1) > 0, 'Line must be specified by 2 distinct points'

    # Create a grid of observation points
    P = np.stack((x, y, z), axis=-1)  # Shape (Nx, Ny, Nz, 3)

    # Vector from p1 to p2 (the wire direction)
    L = p2 - p1
    Lhat = L / np.linalg.norm(L)  # Unit vector along the wire

    # Project P onto the line p1p2 to find the nearest point on the line to P
    tmin = np.dot(P - p1, Lhat)  # Shape (Nx, Ny, Nz)
    lmin = p1 + tmin[..., np.newaxis] * Lhat  # Shape (Nx, Ny, Nz, 3)

    # Calculate the vectors e1, e2, e3
    e1 = Lhat  # Shape (3,)
    e2 = P - lmin
    e2_norm = np.linalg.norm(e2, axis=-1, keepdims=True)
    e2 = e2 / e2_norm  # Normalize e2
    e3 = np.cross(e1, e2)  # Cross product to find e3, shape (Nx, Ny, Nz, 3)

    # Calculate x1 and x2
    x1 = np.dot(p1 - lmin, e1)  # Shape (Nx, Ny, Nz)
    x2 = np.dot(p2 - lmin, e1)  # Shape (Nx, Ny, Nz)

    # Distance R from the line to the point P
    R = e2_norm[..., 0]  # Shape (Nx, Ny, Nz)

    # Calculate the magnetic field magnitude B0
    B0 = (u0 * I / (4 * pi * R)) * (x2 / np.sqrt(x2**2 + R**2) - x1 / np.sqrt(x1**2 + R**2))  # Shape (Nx, Ny, Nz)

    # Final magnetic field vector at each point
    B = B0[..., np.newaxis] * e3  # Shape (Nx, Ny, Nz, 3)

    if plot_wire:
        ax = plot_3d_vector(p2-p1, p1, plot_arrow=True, color='k', elev=45, azim=-45, ax=ax)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

    return B[:,:,:,0], B[:,:,:,1], B[:,:,:,2]


def bfield_from_thin_rectangular_coil(X, Y, Z, a, b, y0, I, plot_wire=False, elev=45, azim=-45, ax=None):

    """
    Compute the fields from a thin rectangular coil of size a (in x) and b (in z).

    Parameters:
    x, y, z : ndarray, [m]
        Arrays of coordinates of points in space where the magnetic field is calculated.
    a, b : float, [m]
        Horizontal (a) and longitudinal (b) size of the coil in x and z.
    y0: float, [m]
        Vertical offset of the coil [m].
    I : float, [A]
        Current through the wire (in Amperes).
    plot_wire: boolean
        Plot the wire in 3D space.
    elev: float, [deg]
        elev option to 3d plot in matplotlib.
    azim: float, [deg]
        azim option to 3d plot in matplotlib.
    ax: axes handle
        axes handle for plotting the wire.

    Returns:
    B : ndarray, [T]
        Magnetic field vector at each point (x, y, z) with shape (Nx, Ny, Nz, 3).
    """
    
    p1 = np.array([-a/2, y0, -b/2])
    p2 = np.array([+a/2, y0, -b/2])
    p3 = np.array([+a/2, y0, +b/2])
    p4 = np.array([-a/2, y0, +b/2])
    
    Bx1, By1, Bz1 = bfield_from_thin_straight_wire(X, Y, Z, p1, p2, I, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)

    if plot_wire:
        ax = plt.gca()
    
    Bx2, By2, Bz2 = bfield_from_thin_straight_wire(X, Y, Z, p2, p3, I, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)
    Bx3, By3, Bz3 = bfield_from_thin_straight_wire(X, Y, Z, p3, p4, I, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)
    Bx4, By4, Bz4 = bfield_from_thin_straight_wire(X, Y, Z, p4, p1, I, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)

    return (Bx1+Bx2+Bx3+Bx4, By1+By2+By3+By4, Bz1+Bz2+Bz3+Bz4)


def bfield_from_thin_rectangular_corrector(X, Y, Z, a, b, h, I, plot_wire=False, elev=45, azim=-45, ax=None):

    """
    Compute the fields from a thin rectangular corrector.  
    Coils are orientated so that corrector steer in x-direction

    Parameters:
    x, y, z : ndarray, [m]
        Arrays of coordinates of points in space where the magnetic field is calculated.
    a, b : float, [m]
        Horizontal (a) and longitudinal (b) size of the coil in x and z.
    h: float, [m]
        Vertical (y) distance between rectangular coils
    I : float, [A]
        Current through the wire (in Amperes).
    plot_wire: boolean
        Plot the wire in 3D space.
    elev: float, [deg]
        elev option to 3d plot in matplotlib.
    azim: float, [deg]
        azim option to 3d plot in matplotlib.
    ax: axes handle
        axes handle for plotting the wire.

    Returns:
    B : ndarray, [T]
        Magnetic field vector at each point (x, y, z) with shape (Nx, Ny, Nz, 3).
    """

    Bx1, By1, Bz1 = bfield_from_thin_rectangular_coil(X, Y, Z, a, b, -h/2, I, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)

    if plot_wire:
        ax = plt.gca()
    
    Bx2, By2, Bz2 = bfield_from_thin_rectangular_coil(X, Y, Z, a, b, +h/2, I, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)
    
    return (Bx1+Bx2, By1+By2, Bz1+Bz2)


def rotate_around_e3(theta):

    """
    Rotation matrix around z-axis

    Parameters:

    theta: float, [rad]
        Rotation angle.

    Returns:
    3D Rotation Matrix for rotation by theta [rad] around z-axis
    """
    
    C, S = np.cos(theta),np.sin(theta)

    return np.array( [[C, -S, 0],[+S, C, 0], [0,0,0]] )


def get_arc_vectors(h, R, theta, 
                    npts=100, 
                    arc_e3=np.array([0,0,1]) ):

    """
    Function to generate points of an arc with radius R in the plane y=h.
    The points subtend an angle of theta [rad] in the xz plane.

    Parameters:
    h: float, [m]
        height offset of the arc, defines the plane the arc lives in: y=h.
    R: float, [m]
        Radius of the arc.
    npts, int 
        Number of points to sample on the arc, arc is made of npts-1 line segments

    Returns:
        ndarray of size = (npts, 3) storing the points on the arc
    """

    phi = (np.pi - theta)/2

    arc_e1 = np.matmul(rotate_around_e3(phi), np.array([1,0,0]))

    assert np.isclose(np.dot(arc_e1, arc_e3), 0)
    
    arc_e2 = np.cross(arc_e3, arc_e1)

    ths = np.linspace(0, theta, npts)

    ps = np.zeros( (len(ths), 3) ) 

    for ii, th in enumerate(ths):
        ps[ii, :] = np.array([0,0,h])+ R*np.matmul(rotate_around_e3(th), arc_e1)

    return ps
    
        
def plot_arc_vectors(ps, color='k', elev=45, azim=-45, ax=None):

    for ii in range(ps.shape[0]-1):

        p1 = ps[ii,:]
        p2 = ps[ii+1,:]

        if ax is None:
            ax = plot_3d_vector(p2-p1, 
                                origin=p1, 
                                color='k', 
                                elev=elev, azim=azim, 
                                plot_arrow=True, plot_line=False)
        else:
            ax = plot_3d_vector(p2-p1, 
                                origin=p1, 
                                color='k', 
                                elev=elev, azim=azim, 
                                plot_arrow=True, plot_line=False, 
                                ax=ax)

    return ax
    

def bfield_from_thin_wire_arc(X, Y, Z, h, R, theta, npts=100, I=1, plot_wire=False, elev=45, azim=-45, ax=None):

    ps = get_arc_vectors(h, R, theta, npts=npts)

    Bx = np.zeros(X.shape)
    By = np.zeros(Y.shape)
    Bz = np.zeros(Z.shape)

    for ii in range(ps.shape[0]-1):

        p1 = ps[ii,:]
        p2 = ps[ii+1,:]

        if ii == 1 and plot_wire:
            ax = plt.gca()

        Bxii, Byii, Bzii = bfield_from_thin_straight_wire(X, Y, Z, p1, p2, I, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)

        Bx = Bx + Bxii
        By = By + Byii
        Bz = Bz + Bzii

    return Bx, By, Bz


def bfield_from_thin_saddle_coil(X, Y, Z, L, R, theta, I, npts=10, plot_wire=False, elev=45, azim=-45, ax=None):

    phi = (np.pi - theta)/2

    Bx = np.zeros(X.shape)
    By = np.zeros(Y.shape)
    Bz = np.zeros(Z.shape)

    BxA1, ByA1, BzA1 = bfield_from_thin_wire_arc(X, Y, Z, -L/2, R, +theta, npts=npts, I=I, plot_wire=plot_wire, ax=ax, elev=elev, azim=azim)

    if plot_wire:
        ax = plt.gca()
        
    BxA2, ByA2, BzA2 = bfield_from_thin_wire_arc(X, Y, Z, +L/2, R, -theta, npts=npts, I=I, plot_wire=plot_wire, ax=ax, elev=elev, azim=azim)

    Bx += BxA1 + BxA2
    By += ByA1 + ByA2
    Bz += BzA1 + BzA2

    # Straight section 1
    p11 = np.array([R*np.cos(phi), R*np.sin(phi), +L/2])
    p21 = np.array([R*np.cos(phi), R*np.sin(phi), -L/2])

    BxS1, ByS1, BzS1 = bfield_from_thin_straight_wire(X, Y, Z, p11, p21, I=I, plot_wire=plot_wire, ax=ax, elev=elev, azim=azim)

    # Straight section 2
    p12 = np.array([-R*np.cos(phi), R*np.sin(phi), -L/2])
    p22 = np.array([-R*np.cos(phi), R*np.sin(phi), +L/2])

    BxS2, ByS2, BzS2 = bfield_from_thin_straight_wire(X, Y, Z, p12, p22, I=I, plot_wire=plot_wire, ax=ax, elev=elev, azim=azim)

    Bx += BxS1 + BxS2
    By += ByS1 + ByS2
    Bz += BzS1 + BzS2

    return (Bx, By, Bz)


def bfield_from_thin_saddle_corrector(X, Y, Z, L, R, theta, I, npts=10, plot_wire=False, elev=45, azim=-45, ax=None):

    Bx1, By1, Bz1 = bfield_from_thin_saddle_coil(X, Y, Z, +L, +R, theta, I, npts=npts, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)

    if plot_wire:
        ax = plt.gca()
    Bx2, By2, Bz2 = bfield_from_thin_saddle_coil(X, Y, Z, -L, -R, theta, I, npts=npts, plot_wire=plot_wire, elev=elev, azim=azim, ax=ax)
    
    return Bx1+Bx2, By1+By2, Bz1+Bz2


def make_rectangular_dipole_corrector_fieldmesh(a, b, h, I, 
                                                xmin, xmax, nx,
                                                ymin, ymax, ny,
                                                zmin, zmax, nz,
                                                *,
                                                plot_wire=False):

    """
    Generate a FieldMesh object representing a rectangular aircore corrector magnet.

    Parameters:
    a, b : float, [m]
        Horizontal (a) and longitudinal (b) size of the coil in x and z.
    h: float, [m]
        Vertical (y) distance between rectangular coils
    I: float, [A]
        Current through the wire (in Amperes).
    xs: ndarray, [m]
        Grid coordinates in x
    ys: ndarray, [m]
        Grid coordinates in y
    zs: ndarray, [m]
        Grid coordinates in z
    plot_wire: boolean
        Plot the wire in 3D space.

    Returns:
    B : ndarray, [T]
        Magnetic field vector at each point (x, y, z) with shape (Nx, Ny, Nz, 3).
    """

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    Bx, By, Bz = bfield_from_thin_rectangular_corrector(X, Y, Z, a, b, h, I, plot_wire=True)
    
    dx = np.diff(xs)[0]
    dy = np.diff(ys)[0]
    dz = np.diff(zs)[0]
    
    attrs = {}
    attrs['gridOriginOffset'] = (xs[0], ys[0], zs[0])
    attrs['gridSpacing'] = (dx, dy, dz)
    attrs['gridSize'] = Bx.shape
    attrs['eleAnchorPt'] = 'center'
    attrs['gridGeometry'] = 'rectangular'
    attrs['axisLabels'] = ('x', 'y', 'z')
    attrs['gridLowerBound'] = (0, 0, 0)
    attrs['harmonic'] = 1
    attrs['fundamentalFrequency'] = 0

    components = {}
    components['magneticField/x'] = Bx
    components['magneticField/y'] = By
    components['magneticField/z'] = Bz

    data = dict(attrs=attrs, components=components)

    return FieldMesh(data=data)


def make_saddle_dipole_corrector_fieldmesh(R, L, theta, I, 
                                           xmin, xmax, nx,
                                           ymin, ymax, ny,
                                           zmin, zmax, nz,
                                           *, 
                                           npts=20, plot_wire=False):

    """
    Generate a FieldMesh object representing a rectangular airccore corrector magnet.

    Parameters:
    R: float, [m]
        Radius of the curved portion of the saddle coils.
    L : float, [m]
        Longitudinal size of the coil in z.
    theta: float, [rad]
        Opening angle of the saddles.
    I: float, [A]
        Current through the wire (in Amperes).
    xmin, xmax, nxs : float, float, int
        Defines a unifomly spaced array of x points for evaluating corrector fields on.
    ymin, ymax, nys : float, float, int
        Defines a unifomly spaced array of y points for evaluating corrector fields on.
    zmin, zmax, nzs : float, float, int
        Defines a unifomly spaced array of z points for evaluating corrector fields on.
    plot_wire: boolean
        Plot the wire in 3D space.

    Returns:
    B : ndarray, [T]
        Magnetic field vector at each point (x, y, z) with shape (Nx, Ny, Nz, 3).
    """

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    
    Bx, By, Bz = bfield_from_thin_saddle_corrector(X, Y, Z, L, R, theta, npts=npts, I=I, plot_wire=plot_wire)
    
    dx = np.diff(xs)[0]
    dy = np.diff(ys)[0]
    dz = np.diff(zs)[0]
    
    attrs = {}
    attrs['gridOriginOffset'] = (xs[0], ys[0], zs[0])
    attrs['gridSpacing'] = (dx, dy, dz)
    attrs['gridSize'] = Bx.shape
    attrs['eleAnchorPt'] = 'center'
    attrs['gridGeometry'] = 'rectangular'
    attrs['axisLabels'] = ('x', 'y', 'z')
    attrs['gridLowerBound'] = (0, 0, 0)
    attrs['harmonic'] = 1
    attrs['fundamentalFrequency'] = 0

    components = {}
    components['magneticField/x'] = Bx
    components['magneticField/y'] = By
    components['magneticField/z'] = Bz

    data = dict(attrs=attrs, components=components)

    return FieldMesh(data=data)


def make_dipole_corrector_fieldmesh(I,
                                    xmin, xmax, nx,
                                    ymin, ymax, ny,
                                    zmin, zmax, nz, 
                                    *, 
                                    mode='rectangular',
                                    a=None, b=None, h=None,                   # Parameters for rectangular dipole corrector
                                    R=None, L=None, theta=None, npts=None,    # Parameters for saddle dipole corrector
                                    plot_wire=False):

    """
    Generates a field mesh for either a saddle or rectangular dipole corrector.

    Parameters:
    xmin, xmax, nxs : float, float, int
        Defines a unifomly spaced array of x points for evaluating corrector fields on.
    ymin, ymax, nys : float, float, int
        Defines a unifomly spaced array of y points for evaluating corrector fields on.
    zmin, zmax, nzs : float, float, int
        Defines a unifomly spaced array of z points for evaluating corrector fields on.
    I : float
        Current through the corrector.
    mode : str, optional
        Type of dipole corrector ('rectangular' or 'saddle'). Default is 'rectangular'.
    a, b, h : float, optional
        Parameters for the rectangular dipole corrector.
        a - width of the rectangular coil.
        b - height of the rectangular coil.
        h - distance between the coils.
    R, L, theta : float, optional
        Parameters for the saddle dipole corrector.
        R - radius of the saddle coil.
        L - length of the saddle coil.
        theta - tilt angle of the coil.
    npts : int, optional
        Number of points for the saddle dipole discretization. Default is 20.
    plot_wire : bool, optional
        If True, plots the wire configuration. Default is False.

    Returns:
    fieldmesh : ndarray
        Magnetic field mesh corresponding to the selected dipole corrector type.
    """
    #print(mode)

    if mode == 'rectangular':
        if a is None or b is None or h is None:
            raise ValueError("Parameters 'a', 'b', and 'h' must be provided for rectangular mode.")
        # Call the rectangular dipole corrector function
        return make_rectangular_dipole_corrector_fieldmesh(a, b, h, I, 
                                                           xmin, xmax, nx,
                                                           ymin, ymax, ny,
                                                           zmin, zmax, nz,
                                                           plot_wire=plot_wire)

    elif mode == 'saddle':
        # Check that necessary parameters are provided
        if R is None or L is None or theta is None:
            raise ValueError("Parameters 'R', 'L', and 'theta' must be provided for saddle mode.")
        # Call the saddle dipole corrector function
        return make_saddle_dipole_corrector_fieldmesh(R, L, theta, I, 
                                                      xmin, xmax, nx,
                                                      ymin, ymax, ny,
                                                      zmin, zmax, nz,
                                                      npts=npts, plot_wire=plot_wire)
    
    else:
        raise ValueError("Invalid mode. Choose either 'rectangular' or 'saddle'.")
    
    

