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


_ROTATION_FUNCS = {
    "x": Rx,
    "y": Ry,
    "z": Rz,
}


def get_rotation_matrix(
    *, x_rot: float = 0.0, y_rot: float = 0.0, z_rot: float = 0.0, order: str = "zxy"
) -> np.ndarray:
    """
    Constructs a 3D rotation matrix by applying intrinsic rotations in the specified order.

    Parameters
    ----------
    x_rot : float
        Rotation angle around the x-axis (radians)
    y_rot : float
        Rotation angle around the y-axis (radians)
    z_rot : float
        Rotation angle around the z-axis (radians)
    order : str
        A 3-character string specifying the rotation order (e.g., 'zxy', 'zyx').
        Each character must be one of 'x', 'y', or 'z'.

    Returns
    -------
    np.ndarray
        The resulting 3×3 rotation matrix
    """
    # Use lowercase internally
    order = order.lower()

    if sorted(order) != ["x", "y", "z"]:
        raise ValueError(
            f"Invalid rotation order '{order}'. Must contain 'x', 'y', and 'z' once each."
        )

    angles = {"x": x_rot, "y": y_rot, "z": z_rot}
    R = np.eye(3)
    for axis in order:
        R = _ROTATION_FUNCS[axis](angles[axis]) @ R
    return R
