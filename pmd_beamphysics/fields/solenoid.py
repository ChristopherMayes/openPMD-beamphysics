from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import ellipe, ellipk

from pmd_beamphysics import FieldMesh
from pmd_beamphysics.units import mu_0


def C_full(kc: float, p: float, c: float, s: float) -> float:
    r"""
    Generalized complete elliptic integral.

    Computes the generalized elliptic integral:

    .. math::
        C(k_c, p, c, s) = \int_0^{\pi/2} \frac{c \cos^2\phi + s \sin^2\phi}
        {\left(\cos^2\phi + p \sin^2\phi\right) \sqrt{\cos^2\phi + k_c^2 \sin^2\phi}} d\phi

    This function evaluates the integral numerically.

    Reference
    ---------
    Derby, N., & Olbert, S. (2010). Cylindrical magnets and ideal solenoids.
    American Journal of Physics, 78(3), 229–235. https://doi.org/10.1119/1.3256157

    Parameters
    ----------
    kc : float
        Modulus of the elliptic integral.
    p : float
        Parameter for the denominator.
    c : float
        Parameter for the numerator.
    s : float
        Scaling factor for the numerator's trigonometric weight.

    Returns
    -------
    float
        Value of the generalized complete elliptic integral.
    """

    # Special case: p = 0, s = 0
    if p == 0 and s == 0:
        return c * ellipe(kc**2)

    # Special case: p = 1, c = 1, s = -1
    if p == 1 and c == 1 and s == -1:
        # Calculate m = 1 - 1/kc^2
        m = 1 - 1 / kc**2

        # Elliptic integrals in terms of m
        E = ellipe(m)
        K = ellipk(m)

        # Simplified expression
        result = (-2 * kc**2 * E + (1 + kc**2) * K) / (kc * (-1 + kc**2))

        return result

    def integrand(phi):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos2_phi = cos_phi**2
        sin2_phi = sin_phi**2
        numerator = c * cos2_phi + s * sin2_phi
        denominator = (cos2_phi + p * sin2_phi) * np.sqrt(cos2_phi + kc**2 * sin2_phi)
        return numerator / denominator

    result = quad(integrand, 0, np.pi / 2, epsabs=1e-6)[0]

    return result


def cel(kc: float, p: float, c: float, s: float) -> float:
    """
    Translated `cel` function from BASIC for evaluating the generalized elliptic integral.

    Reference
    ---------
    Derby, N., & Olbert, S. (2010). Cylindrical magnets and ideal solenoids.
    American Journal of Physics, 78(3), 229–235. https://doi.org/10.1119/1.3256157

    Parameters
    ----------
    kc : float
        Modulus of the elliptic integral.
    p : float
        Parameter for the denominator.
    c : float
        Parameter for the numerator.
    s : float
        Scaling factor for the numerator's trigonometric weight.

    Returns
    -------
    float
        Value of the generalized elliptic integral.
    """
    if kc == 0:
        return c / p if p != 0 else float("nan")  # Handle k_c = 0 case explicitly
    # if kc == 0:
    #    return float('nan')  # Handle invalid input

    errtol = 1e-6
    k = abs(kc)
    pp = p
    cc = c
    ss = s
    em = 1.0

    if p > 0:
        pp = np.sqrt(p)
        ss /= pp
    else:
        f = kc * kc
        q = 1.0 - f
        g = 1.0 - pp
        f -= pp
        q *= ss - c * pp
        pp = np.sqrt(f / g)
        cc = (c - ss) / g
        ss = -q / (g * g * pp) + cc * pp

    f = cc
    cc += ss / pp
    g = k / pp
    ss = 2 * (ss + f * g)
    pp += g
    g = em
    em += k
    kk = k

    while abs(g - k) > g * errtol:
        k = 2 * np.sqrt(kk)
        kk = k * em
        f = cc
        cc += ss / pp
        g = kk / pp
        ss = 2 * (ss + f * g)
        pp += g
        g = em
        em += k

    return (np.pi / 2) * (ss + cc * em) / (em * (em + pp))


def Bz_on_axis(a: float, z: float, b: float, nI: float) -> float:
    r"""
    Computes the axial component (Bz) of the magnetic field for on-axis points (r = 0).

    Reference
    ---------
    Derby, N., & Olbert, S. (2010). Cylindrical magnets and ideal solenoids.
    American Journal of Physics, 78(3), 229–235. https://doi.org/10.1119/1.3256157

    Parameters
    ----------
    a : float
        Radius of the solenoid (meters).
    z : float
        Axial position of the point where Bz is calculated (meters).
    b : float
        Half-length of the solenoid (meters).
    nI : float
        Total azimuthal current per unit length (n * I) in amperes per meter.

    Returns
    -------
    float
        Axial component of the magnetic field (Tesla) for on-axis points.
    """

    term1 = (z + b) / np.hypot(z + b, a)
    term2 = (z - b) / np.hypot(z - b, a)
    Bz = 0.5 * mu_0 * nI * (term1 - term2)

    return Bz


def compute_Br_Bz(
    a: float, z: float, r: float, b: float, nI: float
) -> tuple[float, float]:
    r"""
    Computes the radial (Br) and axial (Bz) components of the magnetic field of an ideal solenoid.

    Reference
    ---------
    Derby, N., & Olbert, S. (2010). Cylindrical magnets and ideal solenoids.
    American Journal of Physics, 78(3), 229–235. https://doi.org/10.1119/1.3256157

    Parameters
    ----------
    a : float
        Radius of the solenoid (meters).
    z : float
        Axial position of the point where Br and Bz are calculated (meters).
    r : float
        Radial position of the point where Br and Bz are calculated (meters).
    b : float
        Half-length of the solenoid (meters).
    nI : float
        Total azimuthal current per unit length (n * I) in amperes per meter.

    Returns
    -------
    tuple[float, float]
        Br : float
            Radial component of the magnetic field (Tesla).
        Bz : float
            Axial component of the magnetic field (Tesla).
    """

    if r == 0:
        Bz = Bz_on_axis(a, z, b, nI)
        return 0.0, Bz

    # Eq. 5
    B0 = mu_0 * nI / np.pi

    # Eq. 6
    z_plus = z + b
    z_minus = z - b

    # Eq. 7
    alpha_plus = a / np.hypot(z_plus, r + a)
    alpha_minus = a / np.hypot(z_minus, r + a)

    # Eq. 8
    beta_plus = z_plus / np.hypot(z_plus, r + a)
    beta_minus = z_minus / np.hypot(z_minus, r + a)

    # Eq. 9
    gamma = (a - r) / (a + r)

    # Eq. 10
    k_plus = np.hypot(z_plus, a - r) / np.hypot(z_plus, a + r)
    k_minus = np.hypot(z_minus, a - r) / np.hypot(z_minus, a + r)

    # Compute Br
    Br = B0 * (
        alpha_plus * C_full(k_plus, 1, 1, -1) - alpha_minus * C_full(k_minus, 1, 1, -1)
    )

    # Compute Bz
    Bz = (
        B0
        * a
        / (a + r)
        * (
            beta_plus * C_full(k_plus, gamma**2, 1, gamma)
            - beta_minus * C_full(k_minus, gamma**2, 1, gamma)
        )
    )

    return Br, Bz


def make_solenoid_fieldmesh(
    *,
    L: float,
    rmin: float = 0,
    rmax: float = 0.01,
    zmin: float = -0.2,
    zmax: float = 0.2,
    nr: int = 20,
    nz: int = 40,
    radius: float = 0.1,
    nI: Optional[float] = None,
    B0: Optional[float] = None,
):
    """
    Generates a 2D cylindrically symmetric ideal solenoid FieldMesh.

    This function computes the magnetic field of an ideal solenoid with a given
    radius, current density, and geometry, producing a 2D cylindrical mesh
    suitable for simulations or analytical studies. The model follows the
    formulation described in Derby & Olbert (2010).

    Reference
    ---------
    Derby, N., & Olbert, S. (2010). Cylindrical magnets and ideal solenoids.
    American Journal of Physics, 78(3), 229–235. https://doi.org/10.1119/1.3256157

    Parameters
    ----------
    L : float
        Length of the solenoid (in meters).
    rmin : float, optional
        The minimum radial coordinate of the field mesh (in meters). Default is 0.
    rmax : float, optional
        The maximum radial coordinate of the field mesh (in meters). Default is 0.01.
    zmin : float, optional
        The minimum axial coordinate of the field mesh (in meters). Default is -0.2.
    zmax : float, optional
        The maximum axial coordinate of the field mesh (in meters). Default is 0.2.
    nr : int, optional
        The number of points in the radial direction for the mesh. Default is 20.
    nz : int, optional
        The number of points in the axial direction for the mesh. Default is 40.
    radius : float, optional
        Radius of the solenoid windings (in meters) at which the current loops reside.
        Default is 0.1.
    nI : float or None, optional
        The product of the number of turns per unit length (n) and the current (I), (in amperes per meter).
        Must be specified if `B0` is not. Exactly one of `nI` or `B0` must be provided.
    B0 : float or None, optional
        Peak on-axis magnetic field (in Tesla). Must be specified if `nI` is not.
        Exactly one of `nI` or `B0` must be provided.

    Returns
    -------
    FieldMesh
        A FieldMesh with cylindrical geometry representing the solenoid.

    Examples
    --------
    >>> field_mesh = make_solenoid_fieldmesh(L=.1, B0=1, zmin=-.2, zmax=.2, rmax=.01, radius=0.1)
    >>> print(field_mesh)
    """

    if (nI is None and B0 is None) or (nI is not None and B0 is not None):
        raise ValueError(
            "Must specify exactly one of `nI` or `B0`, not both or neither."
        )

    if nI is None:
        nI = B0 * np.hypot(radius, L / 2) / (mu_0 * L / 2)

    # Form coordinate mesh
    rs = np.linspace(rmin, rmax, nr)
    zs = np.linspace(zmin, zmax, nz)

    R, Z = np.meshgrid(rs, zs, indexing="ij")

    # Initialize field arrays
    Br = np.zeros_like(R)
    Bz = np.zeros_like(R)

    # Calculate field components at each point
    for i in range(nr):
        for j in range(nz):
            z = Z[i, j]
            r = R[i, j]
            Br[i, j], Bz[i, j] = compute_Br_Bz(radius, z, r, L / 2, nI)

    dr = (rmax - rmin) / (nr - 1)
    dz = (zmax - zmin) / (nz - 1)

    attrs = {}
    attrs["gridOriginOffset"] = (rs[0], 0, zs[0])
    attrs["gridSpacing"] = (dr, 0, dz)
    attrs["gridSize"] = (nr, 1, nz)
    attrs["eleAnchorPt"] = "center"
    attrs["gridGeometry"] = "cylindrical"
    attrs["axisLabels"] = ("r", "theta", "z")
    attrs["gridLowerBound"] = (0, 0, 0)
    attrs["harmonic"] = 0
    attrs["fundamentalFrequency"] = 0

    components = {}
    components["magneticField/r"] = np.expand_dims(Br, axis=1)
    # components["magneticField/theta"] = 0 # omit, this is zero.
    components["magneticField/z"] = np.expand_dims(Bz, axis=1)

    data = dict(attrs=attrs, components=components)

    return FieldMesh(data=data)


def ideal_solenoid_onaxis_field(z: np.ndarray, radius: float, L: float) -> np.ndarray:
    """
    Computes the normalized magnetic field on the solenoid's axis.

    This function calculates the on-axis magnetic field of an ideal solenoid
    normalized to a maximum value of 1.

    Parameters
    ----------
    z : np.ndarray
        Array of axial positions (z) where the magnetic field is evaluated.
    radius : float
        Radius of the solenoid (meters).
    L : float
        Total length of the solenoid (meters).

    Returns
    -------
    np.ndarray
        Normalized magnetic field values at the specified z positions.
    """
    a = radius
    b = L / 2

    A = np.hypot(a, b) / (2 * b)
    term1 = (z + b) / np.hypot(z + b, a)
    term2 = (z - b) / np.hypot(z - b, a)
    return A * (term1 - term2)


def fit_ideal_solenoid(
    z0: np.ndarray, Bz0: np.ndarray, radius0: float = 0.1, L0: float = 0.1
) -> dict:
    """
    Fits the ideal solenoid model to experimental or simulated data.

    This function fits the normalized on-axis magnetic field of an ideal solenoid
    to the provided data using a non-linear least squares optimization. It returns
    the fitted solenoid parameters: radius, length, and the normalization factor (B0).

    Parameters
    ----------
    z0 : np.ndarray
        Array of axial positions (z) corresponding to the measured magnetic field data.
    Bz0 : np.ndarray
        Array of measured magnetic field values (Bz) at the positions in `z0`.
    radius0 : float, optional
        Initial guess for the radius of the solenoid (meters). Default is 0.1 m.
    L0 : float, optional
        Initial guess for the total length of the solenoid (meters). Default is 0.1 m.

    Returns
    -------
    dict
        Dictionary containing the fitted parameters:
        - 'B0' : float
            Normalization factor for the magnetic field (peak field at z = 0).
        - 'radius' : float
            Fitted solenoid radius (meters).
        - 'L' : float
            Fitted solenoid length (meters).
    """

    B0 = Bz0.max()
    Bz_data = Bz0 / B0
    z_data = z0 - z0[np.argmax(Bz_data)]  # Shift z so maximum is at z=0

    # Fit the data to the model
    popt, pcov = curve_fit(
        ideal_solenoid_onaxis_field, z_data, Bz_data, p0=[radius0, L0]
    )

    # Extract fitted parameters
    radius_fit, L_fit = popt

    d = {}
    d["B0"] = B0
    d["radius"] = float(abs(radius_fit))
    d["L"] = float(abs(L_fit))

    return d
