import numpy as np
import pytest
from unittest.mock import Mock

from pmd_beamphysics.utils import get_rotation_matrix
from pmd_beamphysics.testing import assert_pg_close, pg_from_random_normal

# Load the test fixture
pytest_plugins = ("pmd_beamphysics.testing",)


transform_test_cases = [
    # Test case 1: Identity matrix (no change)
    (np.eye(3), "identity"),
    # Test case 2: Scaling in x
    (np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "x_scaling"),
    # Test case 3: Rotation in x-y plane (45 degrees)
    (
        np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "xy_rotation",
    ),
    # Test case 4: Shear in x-z plane
    (np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "xz_shear"),
    # Test case 5: General 3D rotation (around arbitrary axis)
    (
        np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]]),
        "general_rotation",
    ),
]


@pytest.mark.parametrize("transform_matrix,name", transform_test_cases)
def test_point_transform(test_beam, transform_matrix, name):
    """Sanity checks for the linear point transformation method"""
    # Create a copy of the beam to transform
    transformed_beam = test_beam.copy()

    # Apply the transformation
    transformed_beam.linear_point_transform(transform_matrix)

    # Look at how covariance matrices scaled
    np.testing.assert_allclose(
        transformed_beam.cov("x", "y", "z"),
        transform_matrix @ test_beam.cov("x", "y", "z") @ transform_matrix.T,
        err_msg=f"Beam covariance transformation did not match ({name})",
    )

    # Check emittance is conserved
    np.testing.assert_allclose(
        np.linalg.det(transformed_beam.cov("x", "y", "z", "px", "py", "pz")),
        np.linalg.det(test_beam.cov("x", "y", "z", "px", "py", "pz")),
        err_msg=f"Beam covariance transformation did not match ({name})",
    )

    # Apply the inverse transform
    inverse_transform = np.linalg.inv(transform_matrix)
    transformed_beam.linear_point_transform(inverse_transform)

    # Check that we recover the original coordinates
    for coord in ["x", "y", "z", "px", "py", "pz"]:
        np.testing.assert_allclose(
            transformed_beam[coord],
            test_beam[coord],
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"Coordinate {coord} not recovered after inverse transform ({name})",
        )


def test_rotate_x(test_beam, theta=1.2345):
    # Setup beams
    pg_test = test_beam
    pg_ref = pg_test.copy()

    # Transform using method under test and reference method (rotation matrices)
    pg_test.rotate_x(theta)
    pg_ref.linear_point_transform(get_rotation_matrix(x_rot=theta))

    # Check against each other
    assert_pg_close(pg_test, pg_ref)


def test_rotate_y(test_beam, theta=1.2345):
    # Setup beams
    pg_test = test_beam
    pg_ref = pg_test.copy()

    # Transform using method under test and reference method (rotation matrices)
    pg_test.rotate_y(theta)
    pg_ref.linear_point_transform(get_rotation_matrix(y_rot=theta))

    # Check against each other
    assert_pg_close(pg_test, pg_ref)


def test_rotate_z(test_beam, theta=1.2345):
    # Setup beams
    pg_test = test_beam
    pg_ref = pg_test.copy()

    # Transform using method under test and reference method (rotation matrices)
    pg_test.rotate_z(theta)
    pg_ref.linear_point_transform(get_rotation_matrix(z_rot=theta))

    # Check against each other
    assert_pg_close(pg_test, pg_ref)


def test_rotate_method_calls_with_mock(test_beam):
    """Test that the correct single-axis methods are called using vanilla pytest mocks."""
    # Mock the single-axis rotation methods
    test_beam.rotate_x = Mock()
    test_beam.rotate_y = Mock()
    test_beam.rotate_z = Mock()
    test_beam.linear_point_transform = Mock()

    # Test z-only rotation
    test_beam.rotate(x_rot=0.0, y_rot=0.0, z_rot=1.5)
    test_beam.rotate_z.assert_called_once_with(1.5, xc=0.0, yc=0.0)
    test_beam.rotate_x.assert_not_called()
    test_beam.rotate_y.assert_not_called()
    test_beam.linear_point_transform.assert_not_called()

    # Reset mocks
    test_beam.rotate_x.reset_mock()
    test_beam.rotate_y.reset_mock()
    test_beam.rotate_z.reset_mock()
    test_beam.linear_point_transform.reset_mock()

    # Test y-only rotation
    test_beam.rotate(x_rot=0.0, y_rot=1.5, z_rot=0.0)
    test_beam.rotate_y.assert_called_once_with(1.5, xc=0.0, zc=0.0)
    test_beam.rotate_x.assert_not_called()
    test_beam.rotate_z.assert_not_called()
    test_beam.linear_point_transform.assert_not_called()

    # Reset mocks
    test_beam.rotate_x.reset_mock()
    test_beam.rotate_y.reset_mock()
    test_beam.rotate_z.reset_mock()
    test_beam.linear_point_transform.reset_mock()

    # Test x-only rotation
    test_beam.rotate(x_rot=1.5, y_rot=0.0, z_rot=0.0)
    test_beam.rotate_x.assert_called_once_with(1.5, zc=0.0, yc=0.0)
    test_beam.rotate_y.assert_not_called()
    test_beam.rotate_z.assert_not_called()
    test_beam.linear_point_transform.assert_not_called()

    # Reset mocks
    test_beam.rotate_x.reset_mock()
    test_beam.rotate_y.reset_mock()
    test_beam.rotate_z.reset_mock()
    test_beam.linear_point_transform.reset_mock()

    # Test general case (should not call single-axis methods)
    test_beam.rotate(x_rot=1.0, y_rot=1.0, z_rot=1.0)
    test_beam.rotate_x.assert_not_called()
    test_beam.rotate_y.assert_not_called()
    test_beam.rotate_z.assert_not_called()
    test_beam.linear_point_transform.assert_called_once()


@pytest.mark.parametrize(
    "fn",
    ["linear_point_transform"],
)
def test_point_transformation_performance(fn, benchmark):
    pg = pg_from_random_normal(100_000)
    trn = transform_test_cases[-1][0]
    benchmark(getattr(pg, fn), trn)


def test_rotate_with_custom_center_single_axis(test_beam):
    """Test that rotate() properly passes center coordinates to single-axis methods"""
    # Mock the single-axis rotation methods
    test_beam.rotate_x = Mock()
    test_beam.rotate_y = Mock()
    test_beam.rotate_z = Mock()

    # Test z-only rotation with custom center
    test_beam.rotate(x_rot=0.0, y_rot=0.0, z_rot=1.5, xc=2.0, yc=3.0)
    test_beam.rotate_z.assert_called_once_with(1.5, xc=2.0, yc=3.0)

    # Reset mocks
    test_beam.rotate_x.reset_mock()
    test_beam.rotate_y.reset_mock()
    test_beam.rotate_z.reset_mock()

    # Test y-only rotation with custom center
    test_beam.rotate(x_rot=0.0, y_rot=1.5, z_rot=0.0, xc=1.0, zc=-2.0)
    test_beam.rotate_y.assert_called_once_with(1.5, xc=1.0, zc=-2.0)

    # Reset mocks
    test_beam.rotate_x.reset_mock()
    test_beam.rotate_y.reset_mock()
    test_beam.rotate_z.reset_mock()

    # Test x-only rotation with custom center
    test_beam.rotate(x_rot=1.5, y_rot=0.0, z_rot=0.0, yc=-1.0, zc=4.0)
    test_beam.rotate_x.assert_called_once_with(1.5, yc=-1.0, zc=4.0)


def test_rotate_with_custom_center_general_case(test_beam):
    """Test rotate() with custom center for general 3D rotation"""
    x_rot, y_rot, z_rot = 0.5, 0.3, 0.8
    xc, yc, zc = 1.2, -0.8, 2.5

    # Setup beams
    pg_test = test_beam.copy()
    pg_ref = test_beam.copy()

    # Apply rotation with custom center
    pg_test.rotate(
        x_rot=x_rot,
        y_rot=y_rot,
        z_rot=z_rot,
        xc=xc,
        yc=yc,
        zc=zc,
    )

    # Reference: manually shift, rotate, shift back
    pg_ref.x = pg_ref.x - xc
    pg_ref.y = pg_ref.y - yc
    pg_ref.z = pg_ref.z - zc
    pg_ref.linear_point_transform(
        get_rotation_matrix(x_rot=x_rot, y_rot=y_rot, z_rot=z_rot)
    )
    pg_ref.x = pg_ref.x + xc
    pg_ref.y = pg_ref.y + yc
    pg_ref.z = pg_ref.z + zc

    # Check results match
    assert_pg_close(pg_test, pg_ref)
