import numpy as np


def Ry(angle):
    """Defines a roation in around +y direction"""

    C = np.cos(angle)
    S = np.sin(angle)

    # Fast, manual construction of the matrix
    return np.array([[C, 0, S], [0, 1, 0], [-S, 0, C]])


def Rx(angle):
    """Defines a roation around +x direction"""

    C = np.cos(angle)
    S = np.sin(angle)

    return np.array([[1, 0, 0], [0, C, -S], [0, +S, C]])


def Rz(angle):
    """Defines a roation arounx +z direction"""

    C = np.cos(angle)
    S = np.sin(angle)

    return np.array([[C, -S, 0], [S, C, 0], [0, 0, 1]])


def get_rotation_matrix(pitch, yaw, tilt):
    """Defines a general 3d rotation in terms of the orientation angles theta, phi, psi"""

    if pitch is None:
        pitch = 0
    if yaw is None:
        yaw = 0
    if tilt is None:
        tilt = 0

    ry = Ry(pitch)
    rx = Rx(yaw)
    rz = Rz(tilt)

    return np.matmul(ry, np.matmul(rx, rz))
