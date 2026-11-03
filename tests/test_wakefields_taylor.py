"""
Tests for the Taylor-expanded 3D wakefield model (beamphysics.wakefields.taylor).

The implementation was benchmarked against ocelot's Wake physics process
(ocelot.cpbd.wake3D): per-particle kicks agree to machine precision for
file-based wake tables and to ~1.5e-10 for the analytic generators (the
residual comes from ocelot hardcoding a slightly different value of the
free-space impedance). The regression values in this file encode that
agreement.
"""

import numpy as np
import pytest

from beamphysics.testing import pg_from_random_normal
from beamphysics.units import c_light
from beamphysics.wakefields import TaylorWakeComponent, TaylorWakefield


@pytest.fixture
def bunch():
    """Deterministic Gaussian bunch: x, y, z [m] and per-particle charge [C]."""
    rng = np.random.default_rng(123)
    n = 2000
    x = rng.normal(10e-6, 20e-6, n)
    y = rng.normal(20e-6, 20e-6, n)
    z = rng.normal(0, 10e-6, n)
    q = np.full(n, 250e-12 / n)
    return x, y, z, q


@pytest.fixture
def parallel_plate_wake():
    return TaylorWakefield.parallel_plate(
        plate_distance=250e-6,
        half_gap=500e-6,
        corrugation_gap=250e-6,
        corrugation_period=500e-6,
        length=2.0,
        sigma=10e-6,
        orientation="horizontal",
    )


# -----------------------------------------------------------------------------
# Component construction and validation
# -----------------------------------------------------------------------------


def test_component_index_ordering():
    s = np.linspace(0, 1e-4, 10)
    w = np.ones(10)
    comp = TaylorWakeComponent(a=4, b=0, s0=s, w0=w)
    assert comp.key == (0, 4)


def test_component_index_range():
    with pytest.raises(ValueError):
        TaylorWakeComponent(a=0, b=5)


def test_component_mismatched_arrays():
    s = np.linspace(0, 1e-4, 10)
    with pytest.raises(ValueError):
        TaylorWakeComponent(a=0, b=0, s0=s, w0=np.ones(5))
    with pytest.raises(ValueError):
        TaylorWakeComponent(a=0, b=0, s0=s)


def test_duplicate_component_raises():
    s = np.linspace(0, 1e-4, 10)
    w = np.ones(10)
    comps = [
        TaylorWakeComponent(a=0, b=0, s0=s, w0=w),
        TaylorWakeComponent(a=0, b=0, s0=s, w0=2 * w),
    ]
    with pytest.raises(ValueError):
        TaylorWakefield(comps)


def test_missing_component_raises(parallel_plate_wake):
    with pytest.raises(KeyError):
        parallel_plate_wake[(3, 4)]
    assert (0, 0) in parallel_plate_wake
    assert (3, 4) not in parallel_plate_wake


# -----------------------------------------------------------------------------
# File I/O
# -----------------------------------------------------------------------------


def test_file_roundtrip(tmp_path, parallel_plate_wake, bunch):
    filename = tmp_path / "wake_table.dat"
    parallel_plate_wake.to_file(filename)
    wake2 = TaylorWakefield.from_file(filename)

    assert sorted(wake2.components) == sorted(parallel_plate_wake.components)

    x, y, z, q = bunch
    kicks1 = parallel_plate_wake.particle_kicks_3d(x, y, z, q)
    kicks2 = wake2.particle_kicks_3d(x, y, z, q)
    for k1, k2 in zip(kicks1, kicks2):
        np.testing.assert_allclose(k1, k2, rtol=1e-12, atol=1e-30)


def test_rlc_component_roundtrip(tmp_path, bunch):
    """R, L, Cinv lumped terms survive a file round trip and produce kicks."""
    s = np.linspace(0, 500e-6, 100)
    wake = TaylorWakefield(
        [
            TaylorWakeComponent(
                a=0, b=0, s0=s, w0=1e13 * np.exp(-s / 100e-6), R=100.0, L=1e-9, Cinv=1e3
            )
        ]
    )
    filename = tmp_path / "rlc_table.dat"
    wake.to_file(filename)
    wake2 = TaylorWakefield.from_file(filename)
    comp = wake2[(0, 0)]
    assert comp.R == pytest.approx(100.0)
    assert comp.L == pytest.approx(1e-9)
    assert comp.Cinv == pytest.approx(1e3)

    x, y, z, q = bunch
    kicks1 = wake.particle_kicks_3d(x, y, z, q)
    kicks2 = wake2.particle_kicks_3d(x, y, z, q)
    np.testing.assert_allclose(kicks1[2], kicks2[2], rtol=1e-12)


# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------


def test_energy_loss_and_causality(parallel_plate_wake, bunch):
    x, y, z, q = bunch
    dpx, dpy, dpz = parallel_plate_wake.particle_kicks_3d(x, y, z, q)

    # The bunch as a whole loses energy
    assert np.sum(dpz) < 0

    # Causality: the particle closest to the head is barely kicked
    # compared to the tail
    head = np.argmax(z)
    tail = np.argmin(z)
    assert abs(dpz[head]) < 0.01 * abs(dpz[tail])


def test_offset_beam_dipole_kick(parallel_plate_wake, bunch):
    """A beam offset toward the +y plate is kicked further toward it."""
    x, y, z, q = bunch
    _, dpy, _ = parallel_plate_wake.particle_kicks_3d(x, y, z, q)
    assert np.mean(dpy) > 0


def test_centered_beam_no_dipole_kick(bunch):
    """A centered table (plate_distance = half_gap) has no dipole component."""
    wake = TaylorWakefield.parallel_plate(
        plate_distance=500e-6, half_gap=500e-6, sigma=10e-6
    )
    assert (0, 4) not in wake
    assert (0, 2) not in wake

    # Quadrupole-like kicks are antisymmetric in the witness offset
    x, y, z, q = bunch
    x = x - np.average(x, weights=q)
    y = y - np.average(y, weights=q)
    dpx_p, dpy_p, _ = wake.particle_kicks_3d(x, y, z, q)
    dpx_m, dpy_m, _ = wake.particle_kicks_3d(x, -y, z, q)
    np.testing.assert_allclose(dpy_p, -dpy_m, rtol=1e-10)
    np.testing.assert_allclose(dpx_p, dpx_m, rtol=1e-10)


def test_orientation_symmetry(bunch):
    """Vertical plates with (x, y) swapped give the horizontal-plate kicks."""
    kwargs = dict(
        plate_distance=250e-6,
        half_gap=500e-6,
        corrugation_gap=250e-6,
        corrugation_period=500e-6,
        length=2.0,
        sigma=10e-6,
    )
    wake_h = TaylorWakefield.parallel_plate(orientation="horizontal", **kwargs)
    wake_v = TaylorWakefield.parallel_plate(orientation="vertical", **kwargs)

    x, y, z, q = bunch
    dpx_h, dpy_h, dpz_h = wake_h.particle_kicks_3d(x, y, z, q)
    dpx_v, dpy_v, dpz_v = wake_v.particle_kicks_3d(y, x, z, q)

    np.testing.assert_allclose(dpz_v, dpz_h, rtol=1e-10)
    np.testing.assert_allclose(dpx_v, dpy_h, rtol=1e-10)
    np.testing.assert_allclose(dpy_v, dpx_h, rtol=1e-10)


def test_kicks_scale_with_charge(parallel_plate_wake, bunch):
    x, y, z, q = bunch
    _, _, dpz1 = parallel_plate_wake.particle_kicks_3d(x, y, z, q)
    _, _, dpz2 = parallel_plate_wake.particle_kicks_3d(x, y, z, 2 * q)
    np.testing.assert_allclose(dpz2, 2 * dpz1, rtol=1e-12)


def test_dechirper_off_axis(bunch):
    """Mode-sum dechirper table: energy loss and kick away from the plate."""
    wake = TaylorWakefield.dechirper_off_axis(
        plate_distance=500e-6, half_gap=0.01, width=0.02, sigma=10e-6
    )
    x, y, z, q = bunch
    _, dpy, dpz = wake.particle_kicks_3d(x, y, z, q)
    assert np.sum(dpz) < 0
    # The beam is close to the +y plate; the wake pulls it further toward it
    assert np.mean(dpy) > 0


def test_wake_potential(parallel_plate_wake):
    """Longitudinal and dipole wake potentials for a Gaussian profile."""
    z = np.linspace(-300e-6, 300e-6, 1000)
    current = 100 * np.exp(-0.5 * (z / 50e-6) ** 2)
    profile = np.column_stack([z, current])

    z_out, W = parallel_plate_wake.wake_potential(profile, key=(0, 0))
    np.testing.assert_allclose(z_out, z)
    # Energy loss over most of the bunch
    assert np.sum(W * current) < 0
    # Head of the bunch (largest z) is unaffected
    assert abs(W[-1]) < 1e-6 * np.max(np.abs(W))

    _, Wd = parallel_plate_wake.wake_potential(profile, key=(0, 4))
    assert np.max(np.abs(Wd)) > 0
    assert abs(Wd[-1]) < 1e-6 * np.max(np.abs(Wd))


# -----------------------------------------------------------------------------
# Regression against ocelot-benchmarked values
# -----------------------------------------------------------------------------


def test_regression_kicks(parallel_plate_wake, bunch):
    """
    Statistical regression on the kicks for a fixed bunch.

    Reference values were generated with this implementation after
    verifying agreement with ocelot's Wake process to ~1.5e-10 relative
    (see module docstring).
    """
    x, y, z, q = bunch
    dpx, dpy, dpz = parallel_plate_wake.particle_kicks_3d(x, y, z, q)

    assert np.mean(dpx) == pytest.approx(-2.768023240525e03, rel=1e-8)
    assert np.std(dpx) == pytest.approx(1.645980911564e05, rel=1e-8)
    assert np.min(dpx) == pytest.approx(-1.376514584029e06, rel=1e-8)

    assert np.mean(dpy) == pytest.approx(1.277240322088e06, rel=1e-8)
    assert np.std(dpy) == pytest.approx(1.158322651949e06, rel=1e-8)
    assert np.min(dpy) == pytest.approx(6.696566556224e01, rel=1e-8)

    assert np.mean(dpz) == pytest.approx(-3.697528625691e07, rel=1e-8)
    assert np.std(dpz) == pytest.approx(1.962372130059e07, rel=1e-8)
    assert np.min(dpz) == pytest.approx(-7.542893605532e07, rel=1e-8)


# -----------------------------------------------------------------------------
# ParticleGroup integration
# -----------------------------------------------------------------------------


def test_apply_wakefield_3d():
    P = pg_from_random_normal(3000)
    wake = TaylorWakefield.parallel_plate(
        plate_distance=250e-6, half_gap=500e-6, sigma=P["sigma_z"]
    )

    P2 = P.apply_wakefield(wake)

    # z coordinate used internally
    z = np.asarray(P.z) if P.in_t_coordinates else -np.asarray(P.t) * c_light
    dpx, dpy, dpz = wake.particle_kicks_3d(P.x, P.y, z, P.weight)
    np.testing.assert_allclose(P2.px - P.px, dpx, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(P2.py - P.py, dpy, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(P2.pz - P.pz, dpz, rtol=1e-10, atol=1e-6)

    # Original untouched with inplace=False
    assert P2 is not P

    # inplace=True modifies self
    P3 = P.copy()
    assert P3.apply_wakefield(wake, inplace=True) is None
    np.testing.assert_allclose(P3.pz, P2.pz, rtol=1e-12)


def test_apply_wakefield_3d_kwargs():
    P = pg_from_random_normal(1000)
    wake = TaylorWakefield.parallel_plate(
        plate_distance=250e-6, half_gap=500e-6, sigma=P["sigma_z"]
    )
    P2 = P.apply_wakefield(wake, n_points=200, filter_order=10)
    assert not np.array_equal(P2.pz, P.pz)


def test_apply_wakefield_length_validation():
    P = pg_from_random_normal(100)
    wake = TaylorWakefield.parallel_plate(
        plate_distance=250e-6, half_gap=500e-6, sigma=P["sigma_z"]
    )
    with pytest.raises(ValueError, match="length must be None"):
        P.apply_wakefield(wake, length=1.0)


def test_apply_wakefield_1d_requires_length():
    from beamphysics.wakefields import Pseudomode, PseudomodeWakefield

    P = pg_from_random_normal(100)
    wake = PseudomodeWakefield([Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi / 2)])
    with pytest.raises(ValueError, match="length is required"):
        P.apply_wakefield(wake)


def test_plot(parallel_plate_wake):
    ax = parallel_plate_wake.plot()
    assert ax is not None


# -----------------------------------------------------------------------------
# Validation and safety (added after code review)
# -----------------------------------------------------------------------------


def test_unsupported_component_key_raises():
    """(2,2) and (4,4) are not part of the 13-term expansion and must be rejected."""
    s = np.linspace(0, 1e-4, 10)
    w = np.ones(10)
    for a, b in [(2, 2), (4, 4)]:
        comp = TaylorWakeComponent(a=a, b=b, s0=s, w0=w)
        with pytest.raises(ValueError, match="not part of the"):
            TaylorWakefield([comp])


def test_components_do_not_alias(parallel_plate_wake):
    """Components must own their arrays: mutating one never affects another."""
    assert parallel_plate_wake[(0, 2)].w0 is not parallel_plate_wake[(0, 4)].w0
    assert parallel_plate_wake[(0, 0)].s0 is not parallel_plate_wake[(1, 1)].s0

    before = parallel_plate_wake[(0, 2)].w0.copy()
    parallel_plate_wake[(0, 4)].w0 *= 2
    np.testing.assert_array_equal(parallel_plate_wake[(0, 2)].w0, before)


def test_n_points_validation(parallel_plate_wake, bunch):
    x, y, z, q = bunch
    for bad in (2, 1, 0, -5):
        with pytest.raises(ValueError, match="n_points"):
            parallel_plate_wake.particle_kicks_3d(x, y, z, q, n_points=bad)
    with pytest.raises(ValueError, match="filter_order"):
        parallel_plate_wake.particle_kicks_3d(x, y, z, q, filter_order=-1)


def test_empty_distribution_raises(parallel_plate_wake):
    empty = np.array([])
    with pytest.raises(ValueError, match="empty"):
        parallel_plate_wake.particle_kicks_3d(empty, empty, empty, empty)


def test_factor_scales_kicks(parallel_plate_wake, bunch):
    x, y, z, q = bunch
    kicks1 = parallel_plate_wake.particle_kicks_3d(x, y, z, q)
    kicks3 = parallel_plate_wake.particle_kicks_3d(x, y, z, q, factor=3.0)
    for k1, k3 in zip(kicks1, kicks3):
        np.testing.assert_allclose(k3, 3 * k1, rtol=1e-15)


def test_wake_potential_grid_validation(parallel_plate_wake):
    current = np.ones(10)

    z_nonuniform = np.cumsum(np.linspace(1e-6, 2e-6, 10))
    with pytest.raises(ValueError, match="uniform"):
        parallel_plate_wake.wake_potential(np.column_stack([z_nonuniform, current]))

    z_descending = np.linspace(1e-4, -1e-4, 10)
    with pytest.raises(ValueError, match="increasing"):
        parallel_plate_wake.wake_potential(np.column_stack([z_descending, current]))

    with pytest.raises(ValueError, match="shape"):
        parallel_plate_wake.wake_potential(np.array([[0.0, 1.0]]))


def test_apply_wakefield_rejects_kwargs_for_1d():
    from beamphysics.wakefields import Pseudomode, PseudomodeWakefield

    P = pg_from_random_normal(100)
    wake = PseudomodeWakefield([Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi / 2)])
    with pytest.raises(TypeError, match="Unexpected keyword"):
        P.apply_wakefield(wake, length=1.0, n_points=200)


def test_apply_wakefield_rejects_self_kick_flag_for_3d():
    P = pg_from_random_normal(100)
    wake = TaylorWakefield.parallel_plate(
        plate_distance=250e-6, half_gap=500e-6, sigma=P["sigma_z"]
    )
    with pytest.raises(ValueError, match="include_self_kick"):
        P.apply_wakefield(wake, include_self_kick=False)


def test_wakefield_plot_rejects_3d_wakefield(parallel_plate_wake):
    P = pg_from_random_normal(100)
    with pytest.raises(TypeError, match="longitudinal wakefield"):
        P.wakefield_plot(parallel_plate_wake)
