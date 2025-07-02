import numpy as np
import pytest

from pmd_beamphysics import ParticleGroup


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


@pytest.mark.parametrize(
    "fn", ["linear_point_transform_v1", "linear_point_transform_v2"]
)
def test_point_transformation_performance(fn, benchmark):
    pg = ParticleGroup.from_random_normal(100_000)
    trn = transform_test_cases[-1][0]
    benchmark(getattr(pg, fn), trn)
