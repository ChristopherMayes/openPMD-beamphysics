from dataclasses import replace

import h5py
import matplotlib.pyplot as plt
import pytest
from scipy.constants import h as h_planck

from beamphysics.units import c_light, dimension, e_charge
from beamphysics.wavefront.openpmd import WavefrontAttrs
from beamphysics.wavefront.wavefront import Wavefront

from beamphysics.wavefront.propagators import (
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


# -----------------------------------------------------------------------------
# openPMD EXT_Wavefront I/O
# -----------------------------------------------------------------------------


def make_small(shape=(9, 11, 7), dtype=None, polarization="x"):
    """
    Small non-cubic Wavefront for I/O tests.

    Parameters
    ----------
    shape : tuple of int, default=(9, 11, 7)
        Grid shape (nx, ny, nz). Deliberately non-cubic and with all three axes
        distinct, so that an axis-order mistake cannot pass silently.
    dtype : numpy dtype, optional
        If given, the field arrays are cast to this dtype.
    polarization : {'x', 'y', 'xy'}, default='x'
        Which components to populate.

    Returns
    -------
    Wavefront
    """
    kwargs = {
        "shape": shape,
        "dx": 1e-6,
        "dy": 2e-6,
        "dz": 3e-6,
        "wavelength": 1.5e-9,
        "sigma0": 5e-6,
        "energy": 1.0,
    }
    if polarization == "xy":
        Wx = Wavefront.from_gaussian(polarization="x", **kwargs)
        Wy = Wavefront.from_gaussian(polarization="y", **kwargs)
        W = replace(Wx, Ey=2.0 * Wy.Ey)
    else:
        W = Wavefront.from_gaussian(polarization=polarization, **kwargs)

    if dtype is not None:
        W = replace(
            W,
            Ex=None if W.Ex is None else W.Ex.astype(dtype),
            Ey=None if W.Ey is None else W.Ey.astype(dtype),
        )
    return W


@pytest.mark.parametrize("polarization", ["x", "y", "xy"])
def test_openpmd_round_trip(tmp_path, polarization):
    """Arrays and grid come back exactly, for each polarization case."""
    W = make_small(polarization=polarization)
    path = tmp_path / "wavefront.h5"
    W.write_openpmd(path)
    W2 = Wavefront.from_openpmd(path)

    for original, restored in ((W.Ex, W2.Ex), (W.Ey, W2.Ey)):
        if original is None:
            assert restored is None
        else:
            assert np.array_equal(original, restored)

    assert W2.shape == W.shape
    assert (W2.dx, W2.dy, W2.dz) == (W.dx, W.dy, W.dz)
    # Wavelength goes out as a photon energy and comes back through the same
    # SI-exact constants, so it is bit-identical.
    assert W2.wavelength == W.wavelength


def test_openpmd_round_trip_preserves_dtype(tmp_path):
    """complex64 stays complex64; it is not silently widened."""
    W = make_small(dtype=np.complex64)
    assert W.Ex.dtype == np.complex64

    path = tmp_path / "wavefront.h5"
    W.write_openpmd(path)

    with h5py.File(path) as h5:
        dataset = h5["data/1/meshes/electricField/x"]
        # h5py maps the {r, i} compound back to a numpy complex dtype on read, so
        # inspect the HDF5 type itself to see what is actually on disk.
        hdf5_type = dataset.id.get_type()
        assert hdf5_type.get_nmembers() == 2
        assert [hdf5_type.get_member_name(i).decode() for i in range(2)] == [
            "r",
            "i",
        ], "real part first, then imaginary, per FORMAT_HDF5"
        assert (
            hdf5_type.get_size() == 8
        ), "complex64 must stay 32+32 bits, not be widened to 64+64"

    W2 = Wavefront.from_openpmd(path)
    assert W2.Ex.dtype == np.complex64
    assert np.array_equal(W.Ex, W2.Ex)


def test_openpmd_layout(tmp_path):
    """
    The written layout is a contract with other codes; assert it directly.

    This test is deliberately about bytes on disk, not about round tripping.
    """
    W = replace(make_small(polarization="xy"), s_position=12.5)
    path = tmp_path / "wavefront.h5"
    W.write_openpmd(path)

    nx, ny, nz = W.shape

    with h5py.File(path) as h5:
        assert h5.attrs["openPMD"].decode() == "2.0.0"
        assert h5.attrs["openPMDextension"].decode() == "Wavefront"
        assert h5.attrs["basePath"].decode() == "/data/%T/"
        assert h5.attrs["meshesPath"].decode() == "meshes/"
        assert h5.attrs["iterationEncoding"].decode() == "groupBased"

        iteration = h5["data/1"]
        assert iteration.attrs["timeUnitSI"] == 1.0

        mesh = iteration["meshes/electricField"]

        # Stored slowest-varying first: (z, y, x).
        assert [label.decode() for label in mesh.attrs["axisLabels"]] == ["z", "y", "x"]
        assert mesh["x"].shape == (nz, ny, nx)
        assert mesh["y"].shape == (nz, ny, nx)

        # gridSpacing follows axisLabels, so it is (dz, dy, dx).
        assert np.array_equal(mesh.attrs["gridSpacing"], [W.dz, W.dy, W.dx])

        # gridGlobalOffset is the first cell of each axis, in the same order.
        # Components declare position = 0, so that is the first sample.
        assert np.allclose(
            mesh.attrs["gridGlobalOffset"], [W.zmin, W.ymin, W.xmin], rtol=1e-15
        )

        # openPMD 2.0 makes gridUnitSI one value per axis.
        assert mesh.attrs["gridUnitSI"].shape == (3,)
        assert np.array_equal(mesh.attrs["gridUnitSI"], np.ones(3))
        assert mesh.attrs["gridUnitDimension"].shape == (21,)

        # electricField is V/m, the package's own "electric_field" dimension.
        assert np.array_equal(mesh.attrs["unitDimension"], dimension("electric_field"))
        assert np.array_equal(mesh.attrs["gridUnitDimension"], dimension("length") * 3)

        assert mesh.attrs["geometry"].decode() == "cartesian"
        assert mesh.attrs["temporalDomain"].decode() == "time"
        assert mesh.attrs["spatialDomain"].decode() == "r"
        assert mesh.attrs["zCoordinate"] == 12.5

        # photonEnergy is in joules, not eV.
        expected = h_planck * c_light / W.wavelength
        assert mesh.attrs["photonEnergy"] == expected
        assert not np.isclose(mesh.attrs["photonEnergy"], expected / e_charge)

        for component in ("x", "y"):
            assert mesh[component].attrs["unitSI"] == 1.0
            assert np.array_equal(mesh[component].attrs["position"], np.zeros(3))

        # The class has no longitudinal field, so no z component is written.
        assert "z" not in mesh


def test_openpmd_transposes_rather_than_reshapes(tmp_path):
    """
    The stored array is a real transpose, not a reshape of the same buffer.

    A reshape would produce the right dataset shape while scrambling the data, so
    check an individual element against its transposed index.
    """
    W = make_small()
    path = tmp_path / "wavefront.h5"
    W.write_openpmd(path)

    with h5py.File(path) as h5:
        stored = h5["data/1/meshes/electricField/x"][()]

    assert np.array_equal(stored, W.Ex.transpose(2, 1, 0))
    assert stored[3, 2, 1] == W.Ex[1, 2, 3]


def _write_foreign_file(path, W, labels=("z", "y", "x")):
    """
    Write a file by hand the way a foreign code plausibly would.

    Scalars are stored as shape-(1,) arrays and strings as bytes, which is what a
    Fortran HDF5 writer tends to emit. The axis order is configurable so that the
    reader's handling of an arbitrary `axisLabels` permutation can be exercised
    without the writer being involved.

    Parameters
    ----------
    path : pathlib.Path
        File to create.
    W : Wavefront
        Source of the field data and grid.
    labels : tuple of str, default=('z', 'y', 'x')
        Axis order to store.
    """
    spacing = {"x": W.dx, "y": W.dy, "z": W.dz}
    first = {"x": W.xmin, "y": W.ymin, "z": W.zmin}
    permutation = tuple(("x", "y", "z").index(label) for label in labels)

    with h5py.File(path, "w") as h5:
        h5.attrs["openPMD"] = np.bytes_("2.0.0")
        h5.attrs["openPMDextension"] = np.bytes_("Wavefront")
        h5.attrs["basePath"] = np.bytes_("/data/%T/")
        h5.attrs["meshesPath"] = np.bytes_("meshes/")

        mesh = h5.create_group("data/7/meshes/electricField")
        mesh.attrs["geometry"] = np.bytes_("cartesian")
        mesh.attrs["axisLabels"] = np.array([np.bytes_(label) for label in labels])
        mesh.attrs["gridSpacing"] = np.array([spacing[label] for label in labels])
        mesh.attrs["gridGlobalOffset"] = np.array([first[label] for label in labels])
        mesh.attrs["gridUnitSI"] = np.ones(3)

        # Scalars as length-1 arrays, and photonEnergy up at the series root.
        h5.attrs["photonEnergy"] = np.array([h_planck * c_light / W.wavelength])
        mesh.attrs["temporalDomain"] = np.bytes_("time")
        mesh.attrs["spatialDomain"] = np.bytes_("r")
        mesh.attrs["zCoordinate"] = np.array([3.25])
        mesh.attrs["beamline"] = np.bytes_("undulator")

        mesh["x"] = W.Ex.transpose(permutation)
        mesh["x"].attrs["unitSI"] = np.array([1.0])


def test_openpmd_read_foreign_file(tmp_path):
    """Bytes strings, length-1 array scalars and a non-default iteration all read."""
    W = make_small()
    path = tmp_path / "foreign.h5"
    _write_foreign_file(path, W)

    W2 = Wavefront.from_openpmd(path)

    assert np.array_equal(W2.Ex, W.Ex)
    assert W2.Ey is None
    assert (W2.dx, W2.dy, W2.dz) == (W.dx, W.dy, W.dz)
    assert np.isclose(W2.wavelength, W.wavelength)

    # Decoded to str and float, not left as bytes and ndarray.
    assert W2.attrs.beamline == "undulator"
    assert W2.s_position == 3.25


@pytest.mark.parametrize(
    "labels",
    [("z", "y", "x"), ("x", "y", "z"), ("y", "x", "z"), ("z", "x", "y")],
)
def test_openpmd_read_axis_permutations(tmp_path, labels):
    """Any permutation of (x, y, z) in axisLabels is honored on read."""
    W = make_small()
    path = tmp_path / "permuted.h5"
    _write_foreign_file(path, W, labels=labels)

    W2 = Wavefront.from_openpmd(path)

    assert W2.shape == W.shape
    assert (W2.dx, W2.dy, W2.dz) == (W.dx, W.dy, W.dz)
    assert np.array_equal(W2.Ex, W.Ex)

    # gridGlobalOffset is permuted alongside gridSpacing, so a reader that
    # permuted one but not the other would put the grid in the wrong place.
    assert np.allclose([W2.xmin, W2.ymin, W2.zmin], [W.xmin, W.ymin, W.zmin])


@pytest.mark.parametrize("grid_unit_si", [np.full(3, 1e-3), np.array([1e-3])])
def test_openpmd_read_non_si_units(tmp_path, grid_unit_si):
    """
    gridUnitSI and unitSI are applied, so a file in non-SI units reads correctly.

    A reader that ignored them would return a grid that is wrong by a factor of a
    thousand while looking entirely plausible. gridUnitSI is written both as the
    openPMD 2.0 per-axis array and as the 1.x scalar that older EXT_Wavefront
    implementations still emit.
    """
    W = make_small()
    path = tmp_path / "millimeters.h5"
    W.write_openpmd(path)

    # Restate the same grid in mm and the same field in kV/m.
    with h5py.File(path, "r+") as h5:
        mesh = h5["data/1/meshes/electricField"]
        mesh.attrs["gridSpacing"] = mesh.attrs["gridSpacing"] * 1e3
        # gridUnitSI scales gridGlobalOffset too, not just gridSpacing.
        mesh.attrs["gridGlobalOffset"] = mesh.attrs["gridGlobalOffset"] * 1e3
        mesh.attrs["gridUnitSI"] = grid_unit_si
        mesh["x"][...] = mesh["x"][()] / 1e3
        mesh["x"].attrs["unitSI"] = 1e3

    W2 = Wavefront.from_openpmd(path)

    assert np.allclose(W2.Ex, W.Ex, rtol=1e-15, atol=0.0)
    assert np.allclose([W2.dx, W2.dy, W2.dz], [W.dx, W.dy, W.dz], rtol=1e-15)
    assert np.allclose([W2.xmin, W2.ymin, W2.zmin], [W.xmin, W.ymin, W.zmin])


def _spike(W, index=(6, 3, 2)):
    """A wavefront whose only nonzero sample sits at `index`."""
    Ex = np.zeros(W.shape, dtype=complex)
    Ex[index] = 1.0
    return replace(W, Ex=Ex)


def _feature_x(W):
    """x coordinate of the largest |Ex| sample."""
    index = np.unravel_index(np.argmax(np.abs(W.Ex)), W.Ex.shape)
    return W.xvec[index[0]]


def test_offset_shifts_the_grid():
    """The grid midpoint moves the grid without touching its spacing or extent."""
    W = make_small()
    S = replace(W, xmid=1e-5)

    assert S.dx == W.dx
    assert S.xmin == pytest.approx(W.xmin + 1e-5)
    assert S.xmax == pytest.approx(W.xmax + 1e-5)
    assert np.allclose(S.xvec, W.xvec + 1e-5)

    # Translating the grid translates the mean and leaves the width alone.
    assert S.mean_x == pytest.approx(W.mean_x + 1e-5)
    assert S.sigma_x == pytest.approx(W.sigma_x)


def test_crop_and_pad_preserve_physical_coordinates():
    """
    An asymmetric crop or pad must not translate the physics.

    Before the grid carried an origin, cropping two samples off the front of the
    x axis silently moved every remaining sample by one dx, because the grid was
    re-centered on whatever was left.
    """
    W = _spike(make_small())
    x0 = _feature_x(W)

    assert _feature_x(W.crop(nx=(2, 0))) == pytest.approx(x0)
    assert _feature_x(W.crop(nx=(0, 2))) == pytest.approx(x0)
    assert _feature_x(W.pad(nx=(3, 0))) == pytest.approx(x0)
    assert _feature_x(W.pad(nx=(0, 3))) == pytest.approx(x0)

    # A crop and an equal pad put the grid back exactly where it started.
    restored = W.crop(nx=(1, 3)).pad(nx=(1, 3))
    assert restored.xmid == pytest.approx(W.xmid)
    assert np.allclose(restored.xvec, W.xvec)


def test_kspace_carries_grid_midpoint_through_unchanged():
    """
    `WavefrontK` holds the real-space grid midpoint as inert state.

    A real-space shift is a linear phase in k-space, not a shift of the k grid, so
    the transform must neither apply it nor lose it. Cropping in k-space discards
    k samples, which does not move the real-space grid either.
    """
    W = replace(make_small(), xmid=1e-5, ymid=-2e-5, zmid=3e-5)
    mids = (W.xmid, W.ymid, W.zmid)

    K = W.to_kspace()
    assert (K.xmid, K.ymid, K.zmid) == mids
    assert (K.crop(nx=(2, 0)).xmid, K.pad(ny=(1, 0)).ymid) == mids[:2]

    W2 = K.to_rspace()
    assert (W2.xmid, W2.ymid, W2.zmid) == mids
    assert np.allclose(W2.Ex, W.Ex)


def test_drift_preserves_grid_midpoint():
    """Drift is a shift-invariant convolution, so the grid rides along."""
    W = replace(make_small(), xmid=1e-5)
    assert drift_wavefront(W, 1.0).xmid == W.xmid


def test_drift_advances_s_position():
    """
    Propagating records itself, as `ParticleGroup.drift` does by moving `z`.

    The mesh here is co-moving, so `zmid` cannot record the propagation and
    `s_position` is the only place the information can go.
    """
    W = replace(make_small(), s_position=3.0)

    W2 = drift_wavefront(W, 2.0)
    assert W2.s_position == pytest.approx(5.0)
    assert W2.zmid == W.zmid

    # It accumulates, so two steps land where one step of the total would.
    assert drift_wavefront(W2, 4.0).s_position == pytest.approx(9.0)

    # A drift of zero is a no-op, and a backwards drift walks it back.
    assert drift_wavefront(W, 0.0).s_position == pytest.approx(3.0)
    assert drift_wavefront(W2, -2.0).s_position == pytest.approx(3.0)


def test_advanced_drift_advances_s_position_by_the_physical_distance():
    """
    The curved propagator advances `s_position` by `z`, not by its internal `z_eff`.

    It delegates to the basic propagator with a scaled distance `z / (1 + curv*z)`,
    which is a change of variables rather than a distance anything travels. Letting
    that increment stand would silently under-report how far the wavefront went.
    """
    z, curvature = 1.0, 0.5
    W = replace(make_small(), s_position=3.0)

    W2 = drift_wavefront(W, z, curvature=curvature)

    assert W2.s_position == pytest.approx(3.0 + z)
    # The scaled distance is a different number, so this is not a vacuous check.
    assert z / (1 + curvature * z) != pytest.approx(z)


def test_transforms_and_resizes_leave_s_position_alone():
    """
    Only propagation moves the wavefront along the beamline.

    `crop`, `pad` and the domain transforms change the representation, not the
    position, exactly as they would leave `ParticleGroup.z` alone.
    """
    W = replace(make_small(), s_position=3.0)

    assert W.to_kspace().s_position == 3.0
    assert W.to_kspace().to_rspace().s_position == 3.0
    assert W.crop(nx=(1, 1)).s_position == 3.0
    assert W.pad(nz=(2, 2)).s_position == 3.0


def test_derived_wavefronts_do_not_share_attrs():
    """
    Every derivation must deep-enough copy `attrs`.

    `attrs` is a mutable dataclass, and `dataclasses.replace` rebinds the very same
    object onto the new instance. Without an explicit copy, editing metadata on a
    propagated or cropped wavefront silently reaches back into the original.
    """
    W = replace(make_small(), attrs={"beamline": "SXR"})

    derived = [
        W.drift(1.0),
        drift_wavefront(W.to_kspace(), 1.0),
        drift_wavefront(W, 1.0, curvature=0.5),
        W.crop(nx=(1, 1)),
        W.pad(nx=(1, 1)),
        W.to_kspace(),
        W.to_kspace().to_rspace(),
    ]

    for w in derived:
        assert w.attrs is not W.attrs
        assert w.attrs.beamline == "SXR"
        w.attrs.beamline = "HXR"

    assert W.attrs.beamline == "SXR"


def test_advanced_drift_magnifies_the_grid_midpoint():
    """
    The curved propagator rescales the grid about the optical axis.

    It scales dx and dy by 1/M; both of its quadratic phases are referenced to
    x = y = 0, so the map is x -> x/M and the midpoint has to scale by the same
    factor. Leaving it fixed would expand the grid around an off-axis feature while
    the feature kept its old coordinate.
    """
    W = replace(make_small(), xmid=1e-5, ymid=-2e-5)
    z, curvature = 1.0, 0.5

    W2 = drift_wavefront_advanced(W, z, curvature=curvature)

    M = (z / (1 + curvature * z)) / z
    assert W2.dx == pytest.approx(W.dx / M)
    assert W2.xmid == pytest.approx(W.xmid / M)
    assert W2.ymid == pytest.approx(W.ymid / M)

    # The whole grid magnifies as one, so the extent scales by the same factor.
    assert W2.xmin == pytest.approx(W.xmin / M)
    assert W2.xmax == pytest.approx(W.xmax / M)

    # The intra-pulse axis is untouched by a transverse rescaling.
    assert W2.zmid == W.zmid


def test_openpmd_grid_midpoint_round_trip(tmp_path):
    """A grid midpoint survives a write and read, which it previously did not."""
    W = replace(make_small(), xmid=1e-5, ymid=-2e-5, zmid=3e-5)
    path = tmp_path / "offset.h5"
    W.write_openpmd(path)

    W2 = Wavefront.from_openpmd(path)

    assert np.allclose([W2.xmid, W2.ymid, W2.zmid], [W.xmid, W.ymid, W.zmid])
    assert np.allclose(W2.xvec, W.xvec)
    assert np.allclose(W2.yvec, W.yvec)
    assert np.allclose(W2.zvec, W.zvec)
    assert np.array_equal(W2.Ex, W.Ex)


def test_openpmd_read_without_grid_global_offset(tmp_path):
    """
    A file with no gridGlobalOffset reads as a centered grid.

    Defaulting the attribute to zero and treating that as the first sample would
    instead shift the whole grid so that it started at the origin.
    """
    W = replace(make_small(), xmid=1e-5)
    path = tmp_path / "no_offset.h5"
    W.write_openpmd(path)

    with h5py.File(path, "r+") as h5:
        del h5["data/1/meshes/electricField"].attrs["gridGlobalOffset"]

    W2 = Wavefront.from_openpmd(path)

    assert (W2.xmid, W2.ymid, W2.zmid) == (0.0, 0.0, 0.0)
    assert W2.xmin == pytest.approx(-(W.nx - 1) * W.dx / 2)


def test_genesis4_write_refuses_offset_grid(tmp_path):
    """
    Genesis4 stores a point count and a spacing, with nowhere to put an origin.

    Writing anyway would silently re-center the wavefront.
    """
    W = replace(make_small(shape=(8, 8, 4)), dy=1e-6, xmid=1e-5)

    with pytest.raises(ValueError, match="grid origin"):
        W.write_genesis4(tmp_path / "genesis.h5")


def test_genesis4_round_trips_s_position_as_refposition(tmp_path):
    """
    Genesis4's `refposition` is the same quantity as `s_position`.

    It is the position of the dump along the undulator line, so it is written from
    `s_position` and read back into it rather than being dropped at both ends.
    """
    W = replace(make_small(shape=(8, 8, 4)), dy=1e-6, s_position=7.25)
    path = tmp_path / "genesis.h5"

    W.write_genesis4(path)

    with h5py.File(path) as h5:
        assert h5["refposition"][0] == pytest.approx(7.25)

    assert Wavefront.from_genesis4(path).s_position == pytest.approx(7.25)


def test_openpmd_read_constant_component(tmp_path):
    """
    A uniform component stored as a `value`/`shape` group is read, not crashed on.

    openPMD allows this compression, and the package's other readers support it via
    `readers.is_constant_component`.
    """
    W = make_small()
    path = tmp_path / "constant.h5"
    W.write_openpmd(path)

    nx, ny, nz = W.shape
    with h5py.File(path, "r+") as h5:
        mesh = h5["data/1/meshes/electricField"]
        del mesh["x"]
        constant = mesh.create_group("x")
        constant.attrs["value"] = 2.0 + 3.0j
        constant.attrs["shape"] = np.array([nz, ny, nx])
        constant.attrs["unitSI"] = 1.0

    W2 = Wavefront.from_openpmd(path)

    assert W2.shape == (nx, ny, nz)
    assert np.all(W2.Ex == 2.0 + 3.0j)


def test_openpmd_attrs_round_trip(tmp_path):
    """Extension attributes survive a round trip and follow domain transforms."""
    W = replace(make_small(), s_position=4.0)
    path = tmp_path / "attrs.h5"
    W.write_openpmd(
        path,
        beamline="SXR",
        radius_of_curvature_x=2.5,
        radius_of_curvature_y=3.5,
    )
    W2 = Wavefront.from_openpmd(path)

    assert W2.s_position == 4.0
    assert W2.attrs.beamline == "SXR"
    assert W2.attrs.radius_of_curvature_x == 2.5
    assert W2.attrs.radius_of_curvature_y == 3.5
    # Unset optional attributes stay absent rather than defaulting to zero.
    assert W2.attrs.delta_radius_of_curvature_x is None

    # attrs are carried across the k-space transform, and are copies, not aliases.
    Wk = W2.to_kspace()
    assert Wk.attrs == W2.attrs
    assert Wk.attrs is not W2.attrs
    assert Wk.to_rspace().attrs == W2.attrs

    # `s_position` is a coordinate on the wavefront, so it survives a rewrite.
    path2 = tmp_path / "attrs2.h5"
    W2.write_openpmd(path2)
    assert Wavefront.from_openpmd(path2).s_position == 4.0


def test_openpmd_default_attrs():
    """A Wavefront built in memory has default attrs, not None."""
    assert make_small().attrs == WavefrontAttrs()
    assert make_small().to_kspace().attrs == WavefrontAttrs()
    assert make_small().attrs.beamline is None


def test_wavefront_attrs_accepts_mapping():
    """A mapping keyed by either openPMD or field names is coerced."""
    Ex = make_small().Ex
    assert (
        Wavefront(Ex=Ex, attrs={"radiusOfCurvatureX": 2.0}).attrs.radius_of_curvature_x
        == 2.0
    )
    assert (
        Wavefront(
            Ex=Ex, attrs={"radius_of_curvature_x": 2.0}
        ).attrs.radius_of_curvature_x
        == 2.0
    )


def test_wavefront_attrs_assignment_is_coerced():
    """
    Assigning a mapping coerces it, rather than leaving a raw dict on the instance.

    Coercing only at construction meant `w.attrs = {...}` left a dict behind, so
    `w.attrs.beamline` raised `AttributeError` and the mapping went unvalidated
    until write time.
    """
    W = make_small()
    W.attrs = {"radiusOfCurvatureX": 2.0}

    assert isinstance(W.attrs, WavefrontAttrs)
    assert W.attrs.radius_of_curvature_x == 2.0

    with pytest.raises(ValueError, match="radiusOfCurvatureX"):
        W.attrs = {"radiusOfCurvatureX": 1.0, "radius_of_curvature_x": 2.0}


def test_wavefront_attrs_unknown_field_is_type_error():
    """A misspelled attribute fails at construction, not silently at write time."""
    with pytest.raises(TypeError, match="radius_of_curvature_z"):
        WavefrontAttrs(radius_of_curvature_z=1.0)


def test_wavefront_attrs_other_rejects_computed():
    """`other` cannot override values the writer takes from the wavefront."""
    with pytest.raises(ValueError, match="photonEnergy"):
        WavefrontAttrs(other={"photonEnergy": 1.0})
    with pytest.raises(ValueError, match="gridSpacing"):
        WavefrontAttrs(other={"gridSpacing": [1.0, 2.0, 3.0]})

    # `zCoordinate` is `Wavefront.s_position`, so it is not settable through attrs
    # either: a value here would conflict with the one the propagators maintain.
    with pytest.raises(ValueError, match="zCoordinate"):
        WavefrontAttrs(other={"zCoordinate": 1.0})
    with pytest.raises(ValueError, match="zCoordinate"):
        Wavefront(Ex=make_small().Ex, attrs={"zCoordinate": 1.0})


def test_wavefront_attrs_rejects_conflicting_spellings():
    """
    Giving an attribute under both spellings is an error, not a silent preference.

    Preferring the openPMD name would drop the field-name value into `other`, where
    the writer would emit it as a nonstandard attribute holding a conflicting value.
    """
    with pytest.raises(ValueError, match="radiusOfCurvatureX"):
        WavefrontAttrs.from_pmd(
            {"radiusOfCurvatureX": 1.0, "radius_of_curvature_x": 2.0}
        )

    # A field whose two spellings are identical is not a conflict.
    assert WavefrontAttrs.from_pmd({"beamline": "SXR"}).beamline == "SXR"


def test_wavefront_attrs_unwraps_nested_other():
    """
    An `other` key in a mapping is merged, not nested.

    Nesting it produced `other={'other': {...}}`, which the writer would then try to
    store as a dict-valued HDF5 attribute.
    """
    attrs = WavefrontAttrs.from_pmd(
        {"beamline": "SXR", "other": {"someFutureAttr": 3.0}}
    )

    assert attrs.beamline == "SXR"
    assert attrs.other == {"someFutureAttr": 3.0}

    # The nested form is still validated against the computed names.
    with pytest.raises(ValueError, match="photonEnergy"):
        WavefrontAttrs.from_pmd({"other": {"photonEnergy": 1.0}})


def test_wavefront_attrs_is_importable_from_package():
    """`WavefrontAttrs` is part of the public API, so it must be reachable there."""
    import beamphysics

    assert beamphysics.WavefrontAttrs is WavefrontAttrs
    assert "WavefrontAttrs" in beamphysics.__all__


def test_wavefront_attrs_other_rejects_known_field():
    """`other` cannot shadow an attribute that has a field of its own."""
    with pytest.raises(ValueError, match="beamline"):
        WavefrontAttrs(other={"beamline": "SXR"})


def test_openpmd_unknown_attr_round_trips_via_other(tmp_path):
    """
    A record attribute this module does not know is preserved across a round trip.

    EXT_Wavefront is still in flux, so a file written against a newer revision must
    not silently lose attributes by passing through this class.
    """
    W = make_small()
    path = tmp_path / "future.h5"
    W.write_openpmd(path)

    with h5py.File(path, "r+") as h5:
        h5["data/1/meshes/electricField"].attrs["someFutureAttr"] = 7.0

    W2 = Wavefront.from_openpmd(path)
    assert W2.attrs.other == {"someFutureAttr": 7.0}

    path2 = tmp_path / "future2.h5"
    W2.write_openpmd(path2)
    with h5py.File(path2, "r") as h5:
        assert h5["data/1/meshes/electricField"].attrs["someFutureAttr"] == 7.0


def test_openpmd_iteration_selection(tmp_path):
    """Multiple iterations must be disambiguated explicitly."""
    W1 = make_small()
    W2 = replace(W1, Ex=2.0 * W1.Ex)

    path = tmp_path / "multi.h5"
    with h5py.File(path, "w") as h5:
        W1.write_openpmd(h5, iteration=1)
        W2.write_openpmd(h5, iteration=2)

    with pytest.raises(ValueError, match="2 iterations"):
        Wavefront.from_openpmd(path)

    assert np.array_equal(Wavefront.from_openpmd(path, iteration=2).Ex, W2.Ex)
    assert np.array_equal(Wavefront.from_openpmd(path, iteration=1).Ex, W1.Ex)

    with pytest.raises(ValueError, match="iteration 3 not in the file"):
        Wavefront.from_openpmd(path, iteration=3)


def test_openpmd_group_and_path_agree(tmp_path):
    """Passing an open group is equivalent to passing a path."""
    W = make_small()
    by_path = tmp_path / "by_path.h5"
    by_group = tmp_path / "by_group.h5"

    W.write_openpmd(by_path)
    with h5py.File(by_group, "w") as h5:
        W.write_openpmd(h5)

    with h5py.File(by_group) as h5:
        from_group = Wavefront.from_openpmd(h5)

    assert np.array_equal(from_group.Ex, Wavefront.from_openpmd(by_path).Ex)


def test_openpmd_bad_file_argument():
    """Only str, Path and h5py.Group are accepted."""
    W = make_small()
    with pytest.raises(ValueError, match="h5py.Group"):
        W.write_openpmd(42)
    with pytest.raises(ValueError, match="h5py.Group"):
        Wavefront.from_openpmd(42)


def test_openpmd_rejects_unknown_extension_attr(tmp_path):
    """An unrecognized attribute is refused rather than silently written."""
    W = make_small()
    with pytest.raises(TypeError, match="radius_of_curvature_z"):
        W.write_openpmd(tmp_path / "bad.h5", radius_of_curvature_z=1.0)


def _corrupt(path, mutate):
    """
    Apply `mutate` to the mesh record of an existing file.

    Parameters
    ----------
    path : pathlib.Path
        File to modify in place.
    mutate : callable
        Called with the `h5py.Group` of the mesh record.
    """
    with h5py.File(path, "r+") as h5:
        mutate(h5["data/1/meshes/electricField"])


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda mesh: mesh.attrs.__setitem__(
                "temporalDomain", np.bytes_("frequency")
            ),
            "temporalDomain",
        ),
        (
            lambda mesh: mesh.attrs.__setitem__("spatialDomain", np.bytes_("k")),
            "spatialDomain",
        ),
        (
            lambda mesh: mesh.attrs.__setitem__(
                "axisLabels", np.array([np.bytes_(s) for s in ("z", "y", "r")])
            ),
            "axisLabels",
        ),
        (lambda mesh: mesh.attrs.__delitem__("zCoordinate"), "zCoordinate"),
        (lambda mesh: mesh.attrs.__delitem__("photonEnergy"), "photonEnergy"),
        (lambda mesh: mesh.attrs.__delitem__("gridSpacing"), "gridSpacing"),
    ],
)
def test_openpmd_refusals(tmp_path, mutate, match):
    """
    Unsupported or incomplete files are refused with a message naming the problem.

    A silent wrong answer here would be worse than a crash: the numbers would look
    plausible.
    """
    path = tmp_path / "wavefront.h5"
    make_small().write_openpmd(path)
    _corrupt(path, mutate)

    with pytest.raises(ValueError, match=match):
        Wavefront.from_openpmd(path)


def test_openpmd_refuses_non_wavefront_file(tmp_path):
    """A file without the mesh record is named as such, not left to KeyError."""
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("data/1/meshes")

    with pytest.raises(ValueError, match="not an EXT_Wavefront file"):
        Wavefront.from_openpmd(path)


def test_openpmd_energy_agrees_with_genesis4(tmp_path):
    """
    The two formats must not disagree about how much energy is in the pulse.

    Measured on this grid: the openPMD round trip is exact, and Genesis4 differs
    from it by 4.4e-16 relative, i.e. two ulp of float64, from its own unit
    conversions. The assertions below are set just above those measured levels
    rather than at a round number.
    """
    # Genesis4 requires a square transverse grid with equal spacing.
    W = Wavefront.from_gaussian(
        shape=(33, 33, 17),
        dx=10e-6,
        dy=10e-6,
        dz=3e-6,
        wavelength=1.5e-9,
        sigma0=50e-6,
        energy=1.0,
    )

    openpmd_path = tmp_path / "wavefront.h5"
    genesis_path = tmp_path / "wavefront.genesis.h5"
    W.write_openpmd(openpmd_path)
    W.write_genesis4(genesis_path)

    from_openpmd = Wavefront.from_openpmd(openpmd_path)
    from_genesis = Wavefront.from_genesis4(genesis_path)

    assert from_openpmd.energy == W.energy
    assert abs(from_genesis.energy - W.energy) / W.energy < 1e-15
    assert abs(from_openpmd.energy - from_genesis.energy) / W.energy < 1e-15
