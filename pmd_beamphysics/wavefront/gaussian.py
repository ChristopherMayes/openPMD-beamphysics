from math import pi

import numpy as np

from pmd_beamphysics.wavefront.wavefront import Wavefront

from pmd_beamphysics.units import Z0, c_light


def add_gaussian(
    w: Wavefront,
    z=0,
    w0=None,
    zR=None,
    sigma_z=3e-6,
    x0=0,
    y0=0,
    energy=1,
    phase=0,
    polarization="x",
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

    if polarization == "x":
        w.Ex = w.Ex + u * np.exp(1j * phase)
    elif polarization == "y":
        w.Ey = w.Ey + u * np.exp(1j * phase)
    else:
        raise ValueError(f"polarization must be 'x' or 'y' only. Got: {polarization}")

    return w
