import matplotlib.pyplot as plt
import pytest
from pmd_beamphysics.wavefront.wavefront import Wavefront

from pmd_beamphysics.wavefront.propagators import (
    drift_wavefront,
    drift_wavefront_advanced,
)

import numpy as np


def make_gaussian(wavelength: float = 1e-9):
    return Wavefront.from_gaussian(
        shape=(101, 101, 51),
        dx=10e-6,
        dy=10e-6,
        dz=10e-6,
        wavelength=wavelength,
        sigma0=50e-6,
        energy=1.0,
    )


def test_gaussian_statistics():
    energy = 1.2345
    wavelength = 1e-9
    sigma0 = 50e-6  # w0 = 2 * sigma0 = 100e-6
    sigma_z0 = 50e-6
    x0 = 100e-6
    y0 = -100e-6

    W = Wavefront.from_gaussian(
        shape=(101, 101, 51),
        dx=10e-6,
        dy=10e-6,
        dz=10e-6,
        wavelength=wavelength,
        sigma0=sigma0,
        x0=x0,
        y0=y0,
        sigma_z=sigma_z0,
        energy=energy,
    )
    assert W.in_rspace

    assert np.isclose(energy, W.energy)

    assert np.isclose(energy, np.sum(W.fluence) * W.dx * W.dy)

    assert np.isclose(x0, W.mean_x)

    assert np.isclose(y0, W.mean_y)

    assert np.isclose(sigma0, W.sigma_x, rtol=0.01)

    assert np.isclose(sigma0, W.sigma_y, rtol=0.01)

    assert np.isclose(sigma_z0, W.sigma_z, rtol=0.01)

    Wk = W.to_kspace()
    assert Wk.in_kspace

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
    assert W2.in_rspace
    assert np.isclose(energy, W2.energy)

    # TODO: what should these be approximately?
    W.kzmin
    W.kzmax
    Wk.mean_kx
    Wk.mean_ky
    Wk.mean_kz
    Wk.sigma_kx
    Wk.sigma_ky
    Wk.sigma_kz
    Wk.mean_thetax
    Wk.mean_thetay


def test_gaussian_propagation():
    energy = 1.2345
    wavelength = 1e-9
    sigma_z0 = 50e-6
    x0 = 100e-6
    y0 = -100e-6
    sigma0 = 50e-6
    # Rayleigh length: zR = 4π·σ₀²/λ
    zR = 4 * np.pi * sigma0**2 / wavelength

    W0 = Wavefront.from_gaussian(
        shape=(101, 101, 51),
        dx=10e-6,
        dy=10e-6,
        dz=10e-6,
        wavelength=wavelength,
        sigma0=sigma0,
        z0=0,
        x0=x0,
        y0=y0,
        sigma_z=sigma_z0,
        energy=energy,
    )

    W1 = Wavefront.from_gaussian(
        shape=(101, 101, 51),
        dx=10e-6,
        dy=10e-6,
        dz=10e-6,
        wavelength=wavelength,
        sigma0=sigma0,
        z0=zR,
        x0=x0,
        y0=y0,
        sigma_z=sigma_z0,
        energy=energy,
    )

    W2 = drift_wavefront(W0, zR)

    W3 = drift_wavefront_advanced(W0, zR, curvature=1 / zR)

    # At z = zR, beam size should be sqrt(2) times the waist size
    assert np.isclose(W1.sigma_x, np.sqrt(2) * W0.sigma_x, rtol=0.01)

    assert np.isclose(W1.sigma_x, W2.sigma_x, rtol=0.01)

    assert np.isclose(W1.sigma_x, W3.sigma_x, rtol=0.01)


def test_gaussian_smoke():
    W = make_gaussian()
    assert W.in_rspace
    W.fluence_profile_x
    W.fluence_profile_y
    W.power


def test_gaussian_repr():
    W = make_gaussian()
    print("HTML repr:", W._repr_html_())

    class Repr:
        def text(self, lines):
            return lines

    print("Pretty repr:", W._repr_pretty_(Repr(), False))
    print("Pretty repr (cycle):", W._repr_pretty_(Repr(), True))


@pytest.mark.parametrize(
    "logscale",
    [
        pytest.param(False, id="linear"),
        pytest.param(
            True, id="logscale", marks=[pytest.mark.xfail(reason="invalid vmin")]
        ),
    ],
)
def test_gaussian_plot_fluence(logscale: bool):
    W = make_gaussian()
    W.plot_fluence(logscale=logscale)
    plt.show()


@pytest.mark.filterwarnings("ignore:.*identical low and high.*:UserWarning")
def test_gaussian_plot_power():
    W = make_gaussian()
    W.plot_power()
    plt.show()


def test_gaussian_plot2():
    W = make_gaussian()
    W.plot2()
    plt.show()


def test_gaussian_plot_spectral_intensity():
    W = make_gaussian()
    Wk = W.to_kspace()
    Wk.plot_spectral_intensity()
    plt.show()


def test_gaussian_plot_photon_energy_spectrum():
    W = make_gaussian()
    Wk = W.to_kspace()
    Wk.plot_photon_energy_spectrum()
    plt.show()


def test_gaussian_pad_scalar():
    W = make_gaussian()

    nx, ny, nz = 5, 5, 5
    W1 = W.pad(nx, ny, nz)

    assert W1.shape == (W.shape[0] + nx * 2, W.shape[1] + ny * 2, W.shape[2] + nz * 2)


def test_gaussian_pad_asymmetric():
    W = make_gaussian()

    nx, ny, nz = (0, 5), (0, 5), (0, 5)
    W1 = W.pad(nx, ny, nz)

    assert W1.shape == (W.shape[0] + 5, W.shape[1] + 5, W.shape[2] + 5)


def test_axis_index():
    W = make_gaussian()

    for ax in W.axis_labels:
        W.axis_index(ax)


def test_bad_init():
    with pytest.raises(ValueError):
        Wavefront(Ex=None, Ey=None)
    with pytest.raises(ValueError):
        Wavefront(Ex=np.arange(10), Ey=np.arange(11))
    with pytest.raises(ValueError):
        Wavefront(Ex=np.arange(10), dx=-1.0)
    with pytest.raises(ValueError):
        Wavefront(Ex=np.arange(10), dy=-1.0)
    with pytest.raises(ValueError):
        Wavefront(Ex=np.arange(10), dz=-1.0)
    with pytest.raises(ValueError):
        Wavefront(Ex=np.arange(10), wavelength=-1.0)


def test_from_gaussian_validation():
    """Test validation in from_gaussian"""
    with pytest.raises(ValueError, match="sigma0 must be specified"):
        Wavefront.from_gaussian(shape=(10, 10, 10))

    with pytest.raises(ValueError, match="polarization must be"):
        Wavefront.from_gaussian(shape=(10, 10, 10), sigma0=1e-6, polarization="z")

    with pytest.raises(ValueError, match="sigma_z must be non-negative"):
        Wavefront.from_gaussian(shape=(10, 10, 10), sigma0=1e-6, sigma_z=-1.0)


def test_from_gaussian_polarization():
    """Test that polarization parameter works correctly"""
    Wx = Wavefront.from_gaussian(shape=(10, 10, 10), sigma0=1e-6, polarization="x")
    assert Wx.Ex is not None
    assert Wx.Ey is None

    Wy = Wavefront.from_gaussian(shape=(10, 10, 10), sigma0=1e-6, polarization="y")
    assert Wy.Ex is None
    assert Wy.Ey is not None
