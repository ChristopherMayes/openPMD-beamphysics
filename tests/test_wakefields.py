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
    parse_impact_z_wakefield,
    write_impact_z_wakefield,
)
from beamphysics.wakefields import (
    ResistiveWallPseudomode,
    ResistiveWallWakefield,
    TabularWakefield,
)

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
    assert zmax == pytest.approx(100 * wakefield.s0)

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
