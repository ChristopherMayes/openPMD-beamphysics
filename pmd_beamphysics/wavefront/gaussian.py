from collections.abc import Sequence

from math import pi

import numpy as np

from pmd_beamphysics.wavefront.wavefront import Wavefront

from pmd_beamphysics.units import Z0, c_light


def add_gaussian(
    w: Wavefront, z=0, w0=None, zR=None, sigma_z=3e-6, x0=0, y0=0, energy=1, phase=0
):
    """
    Adds a Gaussian beam in-place


    Parameters
    ----------


    z: float
        distance from waist (m)

    zR: float
        Rayleigh length (m)
        This or w0 must be defined.

    w0: float
        Waist size (m)
        Convenience method, related to zR with:
        zR = n π w0^2 / λ
        Here we set n=1.



    References
    ----------
    https://en.wikipedia.org/wiki/Gaussian_beam

    https://en.wikipedia.org/wiki/Complex_beam_parameter

    Siegman "Lasers" 1986, ISBN 0-935702-11-3
    Chapter 16.3 GAUSSIAN SPHERICAL WAVES

    """

    if zR is None and w0 is None:
        raise ValueError("please define w0 or zR")

    if zR is None:
        zR = pi * w0**2 / w.wavelength

    assert w.in_rspace

    dx, dy, dz = w.dx, w.dy, w.dz
    X, Y, Z = np.meshgrid(w.xvec, w.yvec, w.zvec, indexing="ij")

    k = 2 * pi / w.wavelength

    q = z + 1j * zR

    uxy = (1 / q) * np.exp(-0.5j * k * ((X - x0) ** 2 + (Y - y0) ** 2) / q)

    integral_uxy_squared = pi / (k * zR)  # = \int |u_{xy}|^2 dx dy

    uz = np.sqrt(1 / np.sqrt(2 * pi) / sigma_z) * np.exp(
        -(Z**2) / (4 * sigma_z**2)
    )  # Note this is the sqrt of a Gaussian

    u = uxy * uz / np.sqrt(integral_uxy_squared)

    integral2 = np.sum(np.abs(u) ** 2) * dx * dy * dz
    # if not np.isclose(integral2, 1):
    #    raise ValueError(f'The domain is too small for this pulse {integral2}')

    # Make normalization exact for simple sum
    u = u / np.sqrt(integral2)

    # Scale for desired energy
    u = u * np.sqrt(energy * 2 * Z0 * c_light)

    w.Ex = w.Ex + u * np.exp(1j * phase)

    return w


# OLD
def create_gaussian_mesh(
    *,
    wavelength: float,
    zR: float,
    sigma_z: float,
    mins: Sequence[float],
    maxs: Sequence[float],
    shape: Sequence[int],
    energy=1.0,
    dtype=np.complex64,
):
    """

    This creates a complex electric field mesh representing a Gaussian pulse at a waist.


    Parameters
    ----------
    wavelength : float
        Wavelength in meters
    zR : float
        Rayleigh range in meters
    sigma_z : float
        rms pulse length in meters

    mins : tuple of floats
        Per-axis grid spacing.
    shape : tuple of ints
        3D mesh shape
    dtype : np.dtype, default=np.complex64

    Returns
    -------
    mesh: np.ndarray
        3D mesh representing the complex electric field in V/m
        The integrated field energy is normalized to 1 J

    References
    ----------
    https://en.wikipedia.org/wiki/Gaussian_beam



    """
    ranges = [np.linspace(min_, max_, n) for (min_, max_, n) in zip(mins, maxs, shape)]
    dx, dy, dz = [(max_ - min_) / (n - 1) for (min_, max_, n) in zip(mins, maxs, shape)]
    x, y, z = np.meshgrid(*ranges, indexing="ij")

    k = 2 * pi / wavelength

    # From https://en.wikipedia.org/wiki/Gaussian_beam
    # z0 = 0
    # qx = z0 + 1j*zR
    # qy = z0 + 1j*zR
    # ux = 1/np.sqrt(qx) * np.exp(-1j * k * x**2 / 2 / qx)
    # uy = 1/np.sqrt(qy) * np.exp(-1j * k * y**2 / 2 / qy)
    # uxy = ux*uy

    # Simplified
    uxy = -1j / zR * np.exp(-k * (x**2 + y**2) / (2 * zR))

    integral_uxy_squared = pi / (k * zR)  # = \int |u_{xy}|^2 dx dy

    # Longitudinal sqrt Gaussian pulse
    uz = np.sqrt(1 / np.sqrt(2 * pi) / sigma_z) * np.exp(
        -(z**2) / (4 * sigma_z**2)
    )  # Note this is the sqrt of a Gaussian

    u = uxy * uz / np.sqrt(integral_uxy_squared)

    integral2 = np.sum(np.abs(u) ** 2) * dx * dy * dz
    if not np.isclose(integral2, 1):
        raise ValueError("The domain is too small for this pulse")

    # Make normalization exact for simple sum
    u = u / np.sqrt(integral2)

    # Scale for desired energy
    u = u * np.sqrt(energy * 2 * Z0 * c_light)

    return u
