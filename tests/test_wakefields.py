"""
Tests for wakefield tabulation and IMPACT-Z interoperability.

The reference table ``data/rfdata41.in`` holds the first 200 rows of the IMPACT-Z
distribution file ``examples/Example3/rfdata41.in``
(https://github.com/ChristopherMayes/IMPACT-Z, commit
18a3fc1d8ae3ef94ae4ce094b8681450410760b2). The rows are unmodified, so the fixture
exercises the Fortran-formatted columns that IMPACT-Z actually writes; only the
length was reduced, since the trailing 4800 rows add no coverage.
"""

import pathlib

import numpy as np
import pytest

from beamphysics.interfaces.impact import (
    IMPACT_Z_MAX_WAKEFIELD_ROWS,
    create_impact_z_wakefield_rfdata,
    parse_impact_z_wakefield,
    write_impact_z_wakefield,
)
from beamphysics.wakefields import (
    Pseudomode,
    PseudomodeWakefield,
    ResistiveWallPseudomode,
    ResistiveWallWakefield,
    TabularWakefield,
    WakefieldBase,
)
from beamphysics.wakefields.base import SAMPLES_PER_WAVELENGTH

test_data = pathlib.Path(__file__).resolve().parent / "data"

MATERIAL = "copper-slac-pub-10707"
RADIUS = 2.5e-3


@pytest.fixture(scope="module")
def impedance_model() -> ResistiveWallWakefield:
    """Impedance-based resistive wall model for a copper pipe."""
    return ResistiveWallWakefield.from_material(MATERIAL, radius=RADIUS)


@pytest.fixture(scope="module")
def pseudomode_model() -> ResistiveWallPseudomode:
    """Pseudomode resistive wall model for the same copper pipe."""
    return ResistiveWallPseudomode.from_material(MATERIAL, radius=RADIUS)


# -----------------------------------------------------------------------------
# TabularWakefield.from_wakefield
# -----------------------------------------------------------------------------


def test_from_wakefield_reproduces_samples(pseudomode_model):
    """The table must agree with the source model exactly at the sample points."""
    zmax = 100 * pseudomode_model.s0
    table = TabularWakefield.from_wakefield(pseudomode_model, zmax=zmax, n=500)

    z = table.z_data
    assert z[0] == pytest.approx(-zmax)
    assert z[-1] == pytest.approx(0.0, abs=1e-18)
    assert np.all(np.diff(z) > 0)

    np.testing.assert_allclose(table.wake(z), pseudomode_model.wake(z), rtol=1e-12)


def test_from_wakefield_interpolates_between_samples(pseudomode_model):
    """Between the sample points the table must hold to interpolation accuracy."""
    zmax = 100 * pseudomode_model.s0
    table = TabularWakefield.from_wakefield(pseudomode_model, zmax=zmax, n=2000)

    # Offset from the sample points to exercise the cubic interpolant.
    z = -np.linspace(0.01 * zmax, 0.99 * zmax, 977)
    error = np.abs(table.wake(z) - pseudomode_model.wake(z))

    assert error.max() < 1e-6 * pseudomode_model.W0


def test_from_wakefield_avoids_transform_wraparound():
    """
    A wide range must not alias the tail of an impedance-based model.

    The array branch of ImpedanceWakefield.wake inverts the impedance on a finite
    grid. A trailing distance beyond that grid would wrap around and return the wake
    near the source particle instead of the decayed tail.
    """
    wakefield = ResistiveWallWakefield.from_material(
        "aluminum-alloy-6061-t6-20C", radius=10e-3
    )
    table = wakefield.to_tabular()

    # The far tail must be small compared with the amplitude at the source particle.
    assert np.abs(table.W_data[:100]).max() < 0.01 * wakefield.W0


@pytest.mark.parametrize("zmax", [0.0, -1e-3])
def test_from_wakefield_rejects_nonpositive_zmax(pseudomode_model, zmax):
    with pytest.raises(ValueError, match="positive trailing distance"):
        TabularWakefield.from_wakefield(pseudomode_model, zmax=zmax)


def test_from_wakefield_rejects_too_few_points(pseudomode_model):
    with pytest.raises(ValueError, match="at least 4 points"):
        TabularWakefield.from_wakefield(pseudomode_model, zmax=1e-3, n=3)


# -----------------------------------------------------------------------------
# ResistiveWallWakefieldBase.to_tabular
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", ["round", "flat"])
@pytest.mark.parametrize(
    "cls", [ResistiveWallWakefield, ResistiveWallPseudomode], ids=lambda c: c.__name__
)
def test_to_tabular_agrees_with_parent_model(cls, geometry):
    """Both resistive wall models must tabulate to their own wake."""
    wakefield = cls.from_material(MATERIAL, radius=RADIUS, geometry=geometry)
    table = wakefield.to_tabular()

    zmax = -table.z_data[0]
    assert zmax == pytest.approx(wakefield.default_zmax)

    z = -np.linspace(0.01 * zmax, 0.99 * zmax, 401)
    error = np.abs(table.wake(z) - wakefield.wake(z))

    assert error.max() < 1e-4 * wakefield.W0


@pytest.mark.parametrize(
    "cls", [ResistiveWallWakefield, ResistiveWallPseudomode], ids=lambda c: c.__name__
)
def test_to_tabular_default_range_covers_the_decay(cls):
    """The default range must extend to where the wake has decayed."""
    wakefield = cls.from_material(MATERIAL, radius=RADIUS)
    table = wakefield.to_tabular()

    # Residual amplitude over the outermost tenth of the tabulated range.
    tail = table.W_data[: len(table.W_data) // 10]

    assert np.abs(tail).max() < 0.01 * wakefield.W0


def test_to_tabular_honors_explicit_range(pseudomode_model):
    table = pseudomode_model.to_tabular(zmax=1e-4, n=64)

    assert len(table.z_data) == 64
    assert table.z_data[0] == pytest.approx(-1e-4)


# -----------------------------------------------------------------------------
# default_zmax
# -----------------------------------------------------------------------------


def test_pseudomode_decay_length_is_the_slowest_mode():
    """The envelope of the sum decays no faster than its most weakly damped mode."""
    model = PseudomodeWakefield(
        [
            Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi / 2),
            Pseudomode(A=5e14, d=2e4, k=2e5, phi=np.pi / 4),
        ]
    )

    assert model.decay_length == pytest.approx(1e-4)
    assert model.default_zmax == pytest.approx(1e-3)


def test_pseudomode_decay_length_rejects_an_undamped_mode():
    """An undamped mode has no finite range, so no default can be offered."""
    model = PseudomodeWakefield([Pseudomode(A=1e15, d=0.0, k=1e5, phi=0.0)])

    with pytest.raises(ValueError, match="positive decay rate"):
        model.decay_length


def test_pseudomode_default_zmax_leaves_a_negligible_tail(pseudomode_model):
    """Ten decay lengths leave exp(-10) of the envelope, about 5e-05 of W0."""
    zmax = pseudomode_model.default_zmax

    assert zmax == pytest.approx(10 / pseudomode_model.modes[0].d)
    assert abs(pseudomode_model.wake(-zmax)) < 1e-4 * pseudomode_model.W0


def test_impedance_model_default_zmax_is_a_multiple_of_s0(impedance_model):
    """The impedance model has no closed-form envelope, so the range follows s0."""
    assert impedance_model.default_zmax == pytest.approx(100 * impedance_model.s0)


def test_tabular_default_zmax_is_its_own_range(pseudomode_model):
    """A table can only be resampled over the range it covers."""
    table = pseudomode_model.to_tabular(zmax=1e-4, n=64)

    assert table.default_zmax == pytest.approx(1e-4)


def test_default_zmax_is_not_offered_without_a_natural_range():
    """A model with no decay length must ask for zmax explicitly."""

    class _Featureless(WakefieldBase):
        def wake(self, z):
            return np.zeros_like(np.atleast_1d(z), dtype=float)

        def impedance(self, k):
            return np.zeros_like(np.atleast_1d(k), dtype=complex)

    with pytest.raises(NotImplementedError, match="Pass zmax explicitly"):
        TabularWakefield.from_wakefield(_Featureless())


# -----------------------------------------------------------------------------
# default_n_samples
# -----------------------------------------------------------------------------


def test_pseudomode_min_wavelength_is_the_fastest_mode():
    """The shortest period among the modes sets the sampling requirement."""
    model = PseudomodeWakefield(
        [
            Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi / 2),
            Pseudomode(A=5e14, d=2e4, k=4e5, phi=np.pi / 4),
        ]
    )

    assert model.min_wavelength == pytest.approx(2 * np.pi / 4e5)


def test_pseudomode_min_wavelength_falls_back_to_the_decay_length():
    """A purely damped mode has no period, so its decay sets the scale."""
    model = PseudomodeWakefield([Pseudomode(A=1e15, d=1e4, k=0.0, phi=np.pi / 2)])

    assert model.min_wavelength == pytest.approx(1e-4)


def test_impedance_model_min_wavelength_follows_the_round_pipe_form(impedance_model):
    """The DC wake of a round pipe oscillates as cos(sqrt(3) z / s0)."""
    expected = 2 * np.pi * impedance_model.s0 / np.sqrt(3)

    assert impedance_model.min_wavelength == pytest.approx(expected)


def test_default_n_samples_scales_with_the_range(pseudomode_model):
    """Sampling density is fixed, so the count is proportional to the range."""
    zmax = pseudomode_model.default_zmax
    n = pseudomode_model.default_n_samples(zmax)

    assert n == pytest.approx(
        pseudomode_model.default_n_samples(2 * zmax) / 2, rel=0.01
    )
    assert n - 1 == pytest.approx(
        SAMPLES_PER_WAVELENGTH * zmax / pseudomode_model.min_wavelength, rel=0.01
    )


@pytest.mark.parametrize(
    "model_class", [ResistiveWallPseudomode, ResistiveWallWakefield]
)
def test_default_n_samples_resolves_the_wake_for_a_linear_consumer(model_class):
    """IMPACT-Z interpolates the table linearly, so test the error of that estimate."""
    model = model_class.from_material(MATERIAL, radius=RADIUS)
    zmax = model.default_zmax
    n = model.default_n_samples(zmax)

    coarse = TabularWakefield.from_wakefield(model, zmax=zmax, n=n)
    fine = TabularWakefield.from_wakefield(model, zmax=zmax, n=2 * n - 1)

    # Midpoints of the coarse grid, where linear interpolation is least accurate.
    z = fine.z_data[1::2]
    linear = np.interp(z, coarse.z_data, coarse.W_data)

    assert np.abs(linear - fine.W_data[1::2]).max() < 1e-3 * model.W0


def test_tabular_default_n_samples_preserves_its_own_spacing(pseudomode_model):
    """Resampling a table more finely than its data recovers only the interpolant."""
    table = pseudomode_model.to_tabular(zmax=1e-4, n=64)

    assert table.default_n_samples(1e-4) == 64
    assert table.default_n_samples(5e-5) == 33


def test_default_n_samples_is_not_offered_without_a_length_scale():
    """A model with no oscillation or decay scale must ask for n explicitly."""

    class _Featureless(WakefieldBase):
        def wake(self, z):
            return np.zeros_like(np.atleast_1d(z), dtype=float)

        def impedance(self, k):
            return np.zeros_like(np.atleast_1d(k), dtype=complex)

    with pytest.raises(NotImplementedError, match="Pass n explicitly"):
        TabularWakefield.from_wakefield(_Featureless(), zmax=1e-4)


def test_impact_z_export_needs_no_range(pseudomode_model):
    """The pseudomode to table to IMPACT-Z chain runs without any chosen parameter."""
    rfdata = create_impact_z_wakefield_rfdata(pseudomode_model)
    zmax = pseudomode_model.default_zmax

    assert rfdata.shape == (pseudomode_model.default_n_samples(zmax), 4)
    assert len(rfdata) <= IMPACT_Z_MAX_WAKEFIELD_ROWS
    assert rfdata[-1, 0] == pytest.approx(zmax)

    table = TabularWakefield.from_impact_z(rfdata)
    z = -np.linspace(0, zmax, 401)
    error = np.abs(table.wake(z) - pseudomode_model.wake(z))

    assert error.max() < 1e-4 * pseudomode_model.W0


def test_impact_z_export_clamps_an_oversized_automatic_table(pseudomode_model):
    """A range too long to resolve is written at the limit, with a warning."""
    zmax = 100 * pseudomode_model.default_zmax

    with pytest.warns(UserWarning, match="under-resolved"):
        rfdata = create_impact_z_wakefield_rfdata(pseudomode_model, zmax=zmax)

    assert rfdata.shape == (IMPACT_Z_MAX_WAKEFIELD_ROWS, 4)


# -----------------------------------------------------------------------------
# IMPACT-Z reader and writer
# -----------------------------------------------------------------------------


def test_impact_z_round_trip(impedance_model, tmp_path):
    """Writing and re-reading must reproduce the wake to interpolation accuracy."""
    zmax = 100 * impedance_model.s0
    filename = tmp_path / "rfdata41.in"

    write_impact_z_wakefield(filename, impedance_model, zmax=zmax, n=2000)
    table = TabularWakefield.from_impact_z(filename)

    z = -np.linspace(0.01 * zmax, 0.99 * zmax, 1301)
    W_model = impedance_model.wake(z)
    error = np.abs(table.wake(z) - W_model)

    assert error.max() < 1e-4 * impedance_model.W0


def test_impact_z_writer_uses_the_impact_z_abscissa(impedance_model, tmp_path):
    """The file must ascend in s = -z from zero, with the wake sign unchanged."""
    zmax = 100 * impedance_model.s0
    filename = tmp_path / "rfdata41.in"

    write_impact_z_wakefield(filename, impedance_model, zmax=zmax, n=256)
    data = parse_impact_z_wakefield(filename)

    assert data["s"][0] == pytest.approx(0.0, abs=1e-18)
    assert data["s"][-1] == pytest.approx(zmax)
    assert np.all(np.diff(data["s"]) > 0)

    # A positive longitudinal wake is energy-losing in both conventions.
    assert data["Wz"][0] > 0
    assert data["Wz"][0] == pytest.approx(impedance_model.wake(np.array([0.0]))[0])

    # The transverse columns are zero when no callables are supplied.
    assert np.all(data["Wx"] == 0)
    assert np.all(data["Wy"] == 0)


def test_impact_z_writer_transverse_columns(impedance_model, tmp_path):
    """Optional callables of z <= 0 fill the two dipole wake columns."""
    zmax = 100 * impedance_model.s0
    filename = tmp_path / "rfdata41.in"

    write_impact_z_wakefield(
        filename,
        impedance_model,
        zmax=zmax,
        n=128,
        wake_x=lambda z: 1e18 * z,
        wake_y=lambda z: np.full_like(z, 2.0),
    )
    data = parse_impact_z_wakefield(filename)

    np.testing.assert_allclose(data["Wx"], -1e18 * data["s"], rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(data["Wy"], 2.0, rtol=1e-12)


def test_impact_z_writer_rejects_too_many_rows(impedance_model, tmp_path):
    """A table longer than Ndataini overruns the IMPACT-Z arrays."""
    with pytest.raises(ValueError, match="5000"):
        write_impact_z_wakefield(
            tmp_path / "rfdata41.in",
            impedance_model,
            zmax=100 * impedance_model.s0,
            n=IMPACT_Z_MAX_WAKEFIELD_ROWS + 1,
        )


def test_create_rfdata_matches_the_written_file(impedance_model, tmp_path):
    """The writer is a thin wrapper: the file must hold the array it is given."""
    zmax = 100 * impedance_model.s0
    filename = tmp_path / "rfdata41.in"

    rfdata = create_impact_z_wakefield_rfdata(impedance_model, zmax, n=512)
    write_impact_z_wakefield(filename, impedance_model, zmax=zmax, n=512)

    assert rfdata.shape == (512, 4)
    np.testing.assert_allclose(np.loadtxt(filename), rfdata, rtol=1e-15, atol=0)


def test_create_rfdata_rejects_too_many_rows(impedance_model):
    """The row limit belongs to the array builder, not to the file writer."""
    with pytest.raises(ValueError, match="5000"):
        create_impact_z_wakefield_rfdata(
            impedance_model,
            zmax=100 * impedance_model.s0,
            n=IMPACT_Z_MAX_WAKEFIELD_ROWS + 1,
        )


def test_parse_impact_z_wakefield_accepts_an_array(impedance_model, tmp_path):
    """A table held in memory must parse identically to the same table on disk."""
    zmax = 100 * impedance_model.s0
    filename = tmp_path / "rfdata41.in"

    rfdata = create_impact_z_wakefield_rfdata(impedance_model, zmax, n=256)
    write_impact_z_wakefield(filename, impedance_model, zmax=zmax, n=256)

    from_array = parse_impact_z_wakefield(rfdata)
    from_file = parse_impact_z_wakefield(filename)

    for key in ("s", "Wz", "Wx", "Wy"):
        np.testing.assert_allclose(from_array[key], from_file[key], rtol=1e-15, atol=0)


def test_from_impact_z_accepts_an_array(impedance_model, tmp_path):
    """An ImpactZInput file_data entry can be turned back into a wakefield model."""
    zmax = 100 * impedance_model.s0
    filename = tmp_path / "rfdata41.in"

    rfdata = create_impact_z_wakefield_rfdata(impedance_model, zmax, n=256)
    write_impact_z_wakefield(filename, impedance_model, zmax=zmax, n=256)

    from_array = TabularWakefield.from_impact_z(rfdata)
    from_file = TabularWakefield.from_impact_z(filename)

    z = -np.linspace(0.0, zmax, 401)
    np.testing.assert_allclose(from_array.wake(z), from_file.wake(z), rtol=1e-15)


def test_from_wakefield_rejects_range_beyond_a_source_table(impedance_model):
    """Resampling past the end of a table would return the fill value, not the wake."""
    zmax = 100 * impedance_model.s0
    table = TabularWakefield.from_wakefield(impedance_model, zmax=zmax, n=256)

    with pytest.raises(ValueError, match="zero"):
        TabularWakefield.from_wakefield(table, zmax=2 * zmax, n=256)

    with pytest.raises(ValueError, match="zero"):
        create_impact_z_wakefield_rfdata(table, zmax=2 * zmax, n=256)

    # The full range of the source table remains available.
    resampled = TabularWakefield.from_wakefield(table, zmax=zmax, n=128)
    assert resampled.z_data[0] == pytest.approx(-zmax)


def test_parse_impact_z_wakefield_rejects_nonuniform_grid(tmp_path):
    """IMPACT-Z takes the step from the first two rows, so the grid must be uniform."""
    s = np.array([0.0, 1e-6, 3e-6, 6e-6, 1e-5])
    dat = np.column_stack([s, np.ones_like(s), np.zeros_like(s), np.zeros_like(s)])

    filename = tmp_path / "nonuniform.in"
    np.savetxt(filename, dat)

    with pytest.raises(ValueError, match="not uniformly spaced"):
        parse_impact_z_wakefield(filename)


def test_parse_impact_z_wakefield_rejects_wrong_column_count(tmp_path):
    filename = tmp_path / "three_columns.in"
    np.savetxt(filename, np.zeros((10, 3)))

    with pytest.raises(ValueError, match="4 columns"):
        parse_impact_z_wakefield(filename)


def test_parse_impact_z_wakefield_shifts_the_origin(tmp_path):
    """read1wk_Data subtracts the first abscissa from the whole column."""
    s = 5e-6 + np.arange(10) * 2e-6
    dat = np.column_stack([s, np.ones_like(s), np.zeros_like(s), np.zeros_like(s)])

    filename = tmp_path / "offset.in"
    np.savetxt(filename, dat)

    data = parse_impact_z_wakefield(filename)

    assert data["s"][0] == 0.0
    np.testing.assert_allclose(data["s"], s - s[0], rtol=1e-12, atol=0)


# -----------------------------------------------------------------------------
# IMPACT-Z reference table
# -----------------------------------------------------------------------------


def test_reference_table_is_a_decaying_wake():
    """The IMPACT-Z Example3 table is a monotonically decaying longitudinal wake."""
    data = parse_impact_z_wakefield(test_data / "rfdata41.in")

    assert len(data["s"]) <= IMPACT_Z_MAX_WAKEFIELD_ROWS
    assert data["s"][1] - data["s"][0] == pytest.approx(2e-6)
    assert data["Wz"][0] > 0
    assert np.all(np.diff(data["Wz"]) < 0)
    assert np.all(data["Wx"] == 0)
    assert np.all(data["Wy"] == 0)


def test_reference_table_as_tabular_wakefield():
    """Reading the reference table gives a causal wake in the z <= 0 convention."""
    table = TabularWakefield.from_impact_z(test_data / "rfdata41.in")

    data = parse_impact_z_wakefield(test_data / "rfdata41.in")

    assert table.z_data[-1] == pytest.approx(0.0, abs=1e-18)
    assert table.z_data[0] == pytest.approx(-data["s"][-1])

    # The wake is largest at the source particle and falls off behind it.
    assert table.wake(0.0) == pytest.approx(data["Wz"][0])
    smax = data["s"][-1]
    assert table.wake(-smax) < table.wake(-smax / 2) < table.wake(0.0)

    # Causality: nothing ahead of the source particle.
    assert table.wake(1e-6) == 0.0
