import pytest
import numpy as np

from pmd_beamphysics.testing import pg_from_random_normal


@pytest.mark.parametrize("t_or_z", ["t", "z"])
def test_from_random_normal(t_or_z):
    """Test ParticleGroup.from_random_normal with explicit parameters"""
    n_particle = 500_000

    # Define explicit mean vector [x, px, y, py, z/t, pz]
    mean = np.array([1e-3, 5e3, 2e-3, -3e3, 0.5, 1e6])

    # Define explicit covariance matrix (6x6)
    cov = np.array(
        [
            [100e-6**2, 10e-3, 0, 0, 0, 0],
            [10e-3, 10e3**2, 0, 0, 0, 0],
            [0, 0, 150e-6**2, 5e-9, 0, 0],
            [0, 0, 5e-9, 15e3**2, 0, 0],
            [0, 0, 0, 0, 1e-3**2, 0],
            [0, 0, 0, 0, 0, 1e2**2],
        ]
    )

    # Test
    np.random.seed(42)
    pg = pg_from_random_normal(
        n_particle, mean=mean, cov=cov, species="electron", t_or_z=t_or_z
    )

    # Basic properties
    assert len(pg) == n_particle
    assert pg.species == "electron"
    assert pg.n_particle == n_particle

    # Check that all particles have status = 1 and weight = 1
    assert np.all(pg.status == 1)
    assert np.all(pg.weight == 1)

    # Statistical tests - means should be close to specified values
    measured_means = np.array(
        [
            pg.avg("x"),
            pg.avg("px"),
            pg.avg("y"),
            pg.avg("py"),
            pg.avg(t_or_z),
            pg.avg("pz"),
        ]
    )
    np.testing.assert_allclose(measured_means, mean, rtol=1e-2)

    # Test covariance matrix - should be close to specified values
    measured_cov = pg.cov("x", "px", "y", "py", t_or_z, "pz")
    # Diagonal elements
    np.testing.assert_allclose(np.diag(measured_cov), np.diag(cov), rtol=1e-2)
    # Covariances (normalize by std. deviation)
    np.testing.assert_allclose(
        np.diag(measured_cov, 1)
        / np.sqrt(np.diag(measured_cov))[:-1]
        / np.sqrt(np.diag(measured_cov))[1:],
        np.diag(cov, 1) / np.sqrt(np.diag(cov))[:-1] / np.sqrt(np.diag(cov))[1:],
        rtol=3e-1,
        atol=1e-3,
    )


@pytest.mark.parametrize("t_or_z", ["t", "z"])
def test_from_random_normal_default(t_or_z):
    """Test ParticleGroup.from_random_normal with default parameters"""
    n_particle = 500_000

    # Test with z coordinates (default)
    np.random.seed(42)
    pg = pg_from_random_normal(n_particle, t_or_z=t_or_z)

    # Basic properties
    assert len(pg) == n_particle
    assert pg.species == "electron"
    assert pg.n_particle == n_particle

    # Check that all particles have status = 1 and weight = 1
    assert np.all(pg.status == 1)
    assert np.all(pg.weight == 1)

    # Check the mean
    measured_means = np.array(
        [
            pg.avg("x") / pg.std("x"),
            pg.avg("px") / pg.std("px"),
            pg.avg("y") / pg.std("y"),
            pg.avg("py") / pg.std("py"),
            pg.avg(t_or_z) / pg.std(t_or_z),
            pg.avg("pz"),
        ]
    )
    mean = np.zeros_like(measured_means)
    mean[5] = 1e6
    np.testing.assert_allclose(measured_means, mean, rtol=1e-2, atol=1e-2)
