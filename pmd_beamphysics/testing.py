import pytest
import numpy as np

from .particles import ParticleGroup


@pytest.fixture
def test_beam() -> ParticleGroup:
    """Create a test beam with random particles"""
    # Create a simple beam with position and momentum spread
    n_particle = 1000
    np.random.seed(42)  # For reproducibility
    return ParticleGroup.from_random_normal(n_particle=n_particle)


def assert_pg_close(pg_test, pg_ref, atol=1e-7, rtol=0, err_msg=""):
    """
    For use with pytest, confirm two particle groups are the same within tolerances using np.testing.assert_all_close.

    Parameters
    ----------
    pg_test : ParticleGroup
        ParticleGroup under test
    pg_ref : ParticleGroup
        ParticleGroup to compare with
    atol : float, optional
        absolute tolerance, by default 1e-7
    rtol : float, optional
        relative tolerance, by default 0
    err_msg : str, optional
        Error message to emit when failed, by default ""
    """
    assert pg_test.species == pg_ref.species
    for key in ["x", "px", "y", "py", "z", "pz", "t"]:
        np.testing.assert_allclose(
            pg_ref[key], pg_test[key], rtol=rtol, atol=atol, err_msg=err_msg
        )
