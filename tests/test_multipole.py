import numpy as np
import pytest
from functools import partial

from beamphysics.fields.multipole import decompose_field, scalar_error, synthesize_field


def test_synthesize_pure_quadrupole():
    """A pure normal quadrupole B_1=1 T/m should give B_y = x, B_x = y."""
    multipoles = [(0.0, 0.0), (1.0, 0.0)]
    x = np.linspace(-0.02, 0.02, 10)
    y = np.linspace(-0.02, 0.02, 10)
    X, Y = np.meshgrid(x, y)

    B_x, B_y = synthesize_field(X, Y, multipoles)

    np.testing.assert_allclose(B_y, X, atol=1e-15)
    np.testing.assert_allclose(B_x, Y, atol=1e-15)


def test_synthesize_pure_dipole():
    """A pure normal dipole B_0=1 T should give uniform B_y=1, B_x=0."""
    multipoles = [(1.0, 0.0)]
    B_x, B_y = synthesize_field(0.01, 0.005, multipoles)

    np.testing.assert_allclose(B_y, 1.0, atol=1e-15)
    np.testing.assert_allclose(B_x, 0.0, atol=1e-15)


def test_synthesize_skew_dipole():
    """A pure skew dipole S_0=1 T should give uniform B_x=1, B_y=0."""
    multipoles = [(0.0, 1.0)]
    B_x, B_y = synthesize_field(0.01, 0.005, multipoles)

    np.testing.assert_allclose(B_x, 1.0, atol=1e-15)
    np.testing.assert_allclose(B_y, 0.0, atol=1e-15)


def test_decompose_roundtrip():
    """Decompose should recover the multipoles used to synthesize B_phi."""
    ground_truth = [
        (1.0, 0.5),
        (10.0, -5.0),
        (100.0, 60.0),
    ]
    r0 = 0.01
    phi_samples = np.linspace(0, 2 * np.pi, 72, endpoint=False)

    data = []
    for phi in phi_samples:
        x = r0 * np.cos(phi)
        y = r0 * np.sin(phi)
        Bx, By = synthesize_field(x, y, ground_truth)
        B_phi = -Bx * np.sin(phi) + By * np.cos(phi)
        data.append((phi, B_phi))

    recovered = decompose_field(data, r0, nmax=2)

    for n, (Bn_true, Sn_true) in enumerate(ground_truth):
        Bn_rec, Sn_rec = recovered[n]
        np.testing.assert_allclose(Bn_rec, Bn_true, atol=1e-10)
        np.testing.assert_allclose(Sn_rec, Sn_true, atol=1e-10)


def test_scalar_error_identical_fields():
    """Identical fields should give zero error."""
    multipoles = [(0.0, 0.0), (10.0, 0.0)]
    B = partial(synthesize_field, multipoles=multipoles)

    error = scalar_error(B, B, r0=0.01)
    np.testing.assert_allclose(error, 0.0, atol=1e-14)


def test_scalar_error_with_octupole():
    """Reproduce the notebook's octupole error example."""
    design_multipoles = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    error_multipoles = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (80.0, 0.0)]
    actual_multipoles = [
        (d[0] + e[0], d[1] + e[1])
        for d, e in zip(design_multipoles, error_multipoles)
    ]

    B_design = partial(synthesize_field, multipoles=design_multipoles)
    B_actual = partial(synthesize_field, multipoles=actual_multipoles)

    error = scalar_error(B_actual, B_design, r0=0.010)
    np.testing.assert_allclose(error, 1 / 7500, rtol=1e-10)
