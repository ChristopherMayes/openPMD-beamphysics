import pytest
import numpy as np
from typing import Union, Literal
from .particles import ParticleGroup


def pg_from_random_normal(
    n_particle: int,
    mean: Union[np.ndarray, None] = None,
    cov: Union[np.ndarray, None] = None,
    species: str = "electron",
    t_or_z: Literal["t", "z"] = "z",
) -> ParticleGroup:
    """
    Generate beam from normal distribution specified from mean vector and covariance matrix.

    Coordinates are ordered as [x, px, y, py, <see notes>, pz]

    You have a choice of time-like coordinate by specifying `t_or_z`.

    Parameters
    ----------
    n_particle : int
        Number of particles to generate
    mean : np.ndarray | None, optional
        Mean vector (6d vector), by default None
    cov : np.ndarray | None, optional
        Covariance matrix (6x6 numpy array or 6d vector interpreted as the diagonal), by default None
    species : str, optional
        Which particle, by default 'electron'
    t_or_z : Literal[&quot;t&quot;, &quot;z&quot;], optional
        Which time-like coordinate to use, by default "z"

    Returns
    -------
    ParticleGroup
        The ParticleGroup object with generated samples
    """
    # Clean up input; process defaults
    t_or_z = t_or_z.lower()
    if mean is None:
        mean = np.zeros(6)
        mean[5] = 1e6
    if cov is None:
        cov = np.zeros(6)
        cov[0] = cov[2] = 100e-6**2
        cov[1] = cov[3] = 10e3**2
        cov[4] = 1e-3**2
        cov[5] = 1e2**2
    if len(cov.shape) == 1:
        cov = np.diag(cov)

    # Create samples
    samp = np.random.multivariate_normal(mean, cov, n_particle)

    # Pack into object
    data = dict(
        x=samp[:, 0],
        px=samp[:, 1],
        y=samp[:, 2],
        py=samp[:, 3],
        pz=samp[:, 5],
        status=np.ones(n_particle),
        weight=np.ones(n_particle),
        species=species,
    )
    if t_or_z == "t":
        data["t"] = samp[:, 4]
        data["z"] = np.zeros(n_particle)
    elif t_or_z == "z":
        data["t"] = np.zeros(n_particle)
        data["z"] = samp[:, 4]
    else:
        raise ValueError(f"t_or_z must be either `t` or `z`. Got: {t_or_z}")
    return ParticleGroup(data=data)


@pytest.fixture
def test_beam() -> ParticleGroup:
    """Create a test beam with random particles"""
    # Create a simple beam with position and momentum spread
    n_particle = 1000
    np.random.seed(42)  # For reproducibility
    return pg_from_random_normal(n_particle=n_particle)


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
