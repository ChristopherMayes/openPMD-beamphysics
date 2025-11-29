from pmd_beamphysics.wavefront.wavefront import Wavefront

from pmd_beamphysics.wavefront.gaussian import add_gaussian

from pmd_beamphysics.wavefront.propagators import (
    drift_wavefront,
    drift_wavefront_advanced,
)

import numpy as np


def test_gaussian_statistics():
    energy = 1.2345
    wavelength = 1e-9
    w0 = 100e-6
    sigma_z0 = 50e-6
    x0 = 100e-6
    y0 = -100e-6

    W = Wavefront(
        Ex=np.zeros((101, 101, 51)),
        dx=10e-6,
        dy=10e-6,
        dz=10e-6,
        wavelength=wavelength,
    )
    add_gaussian(W, z=0, x0=x0, y0=y0, w0=w0, energy=energy, sigma_z=sigma_z0)

    assert np.isclose(energy, W.energy)

    assert np.isclose(energy, np.sum(W.fluence) * W.dx * W.dy)

    assert np.isclose(x0, W.mean_x)

    assert np.isclose(y0, W.mean_y)

    assert np.isclose(w0, W.sigma_x * 2)

    assert np.isclose(w0, W.sigma_y * 2)

    assert np.isclose(sigma_z0, W.sigma_z)

    Wk = W.to_kspace()

    assert np.isclose(energy, Wk.energy)

    assert np.isclose(energy, np.sum(Wk.spectral_fluence) * W.dkx * W.dky)

    assert np.isclose(
        energy, np.sum(Wk.spectral_fluence) * W.dthetax * W.dthetay * W.k0**2
    )

    assert np.isclose(
        energy,
        Wk.spectral_fluence.max()
        * Wk.k0**2
        * Wk.sigma_thetax
        * Wk.sigma_thetay
        * 2
        * np.pi,
    )

    W2 = Wk.to_rspace()
    assert np.isclose(energy, W2.energy)


def test_gaussian_propagation():
    energy = 1.2345
    wavelength = 1e-9
    sigma_z0 = 50e-6
    x0 = 100e-6
    y0 = -100e-6
    zR = 10

    W00 = Wavefront(
        Ex=np.zeros((101, 101, 51)),
        dx=10e-6,
        dy=10e-6,
        dz=10e-6,
        wavelength=wavelength,
    )

    W0 = W00.copy()
    add_gaussian(W0, z=0, zR=zR, x0=x0, y0=y0, energy=energy, sigma_z=sigma_z0)

    W1 = W00.copy()
    add_gaussian(W1, z=zR, zR=zR, x0=x0, y0=y0, energy=energy, sigma_z=sigma_z0)

    W2 = drift_wavefront(W0, zR)

    W3 = drift_wavefront_advanced(W0, zR, curvature=1 / 10)

    assert np.isclose(W1.sigma_x, np.sqrt(2) * W0.sigma_x)

    assert np.isclose(W1.sigma_x, W2.sigma_x)

    assert np.isclose(W1.sigma_x, W3.sigma_x)
