import numpy as np


def Ry(angle):
    """
    The 3D rotation matrix around the y axis.

    Parameters
    ----------
    angle : float
        Angle of rotation, radians

    Returns
    -------
    np.ndarray
        The 3x3 rotation matrix
    """

    C = np.cos(angle)
    S = np.sin(angle)

    # Fast, manual construction of the matrix
    return np.array([[C, 0, S], [0, 1, 0], [-S, 0, C]])


def Rx(angle):
    """
    The 3D rotation matrix around the x axis.

    Parameters
    ----------
    angle : float
        Angle of rotation, radians

    Returns
    -------
    np.ndarray
        The 3x3 rotation matrix
    """

    C = np.cos(angle)
    S = np.sin(angle)

    return np.array([[1, 0, 0], [0, C, -S], [0, +S, C]])


def Rz(angle):
    """
    The 3D rotation matrix around the z axis.

    Parameters
    ----------
    angle : float
        Angle of rotation, radians

    Returns
    -------
    np.ndarray
        The 3x3 rotation matrix
    """

    C = np.cos(angle)
    S = np.sin(angle)

    return np.array([[C, -S, 0], [S, C, 0], [0, 0, 1]])


def get_rotation_matrix(
    y_rot: float = 0.0, x_rot: float = 0.0, z_rot: float = 0.0
) -> np.ndarray:
    """
    Returns a general rotation matrix by performing a rotation around the z axis, then around the x axis, and finally around y.

    Parameters
    ----------
    y_rot : float, optional
        Rotation around the y axis, radians
    x_rot : float, optional
        Rotation around the x axis, radians
    z_rot : float, optional
        Rotation around the z axis, radians

    Returns
    -------
    np.ndarray
        The 3x3 rotation matrix
    """
    ry = Ry(y_rot)
    rx = Rx(x_rot)
    rz = Rz(z_rot)
    return np.matmul(ry, np.matmul(rx, rz))
