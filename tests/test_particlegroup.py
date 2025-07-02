import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.particles import single_particle

P = ParticleGroup("docs/examples/data/bmad_particles.h5")


ARRAY_KEYS = """
x y z px py pz t status weight id
z/c
p energy kinetic_energy xp yp higher_order_energy
r theta pr ptheta
Lz
gamma beta beta_x beta_y beta_z
x_bar px_bar Jx Jy
weight

""".split()


SPECIAL_STATS = """
norm_emit_x norm_emit_y norm_emit_4d higher_order_energy_spread
average_current
n_alive
n_dead
""".split()


OPERATORS = ("min_", "max_", "sigma_", "delta_", "ptp_", "mean_")


@pytest.fixture(params=ARRAY_KEYS)
def array_key(request):
    return request.param


array_key2 = array_key


@pytest.fixture(params=OPERATORS)
def operator(request):
    return request.param


def test_operator(operator, array_key):
    key = f"{operator}{array_key}"
    P[key]


def test_cov_(array_key, array_key2):
    key = f"cov_{array_key}__{array_key2}"
    P[key]


@pytest.fixture(params=SPECIAL_STATS)
def special_stat(request):
    return request.param


def test_special_stat(special_stat):
    x = P[special_stat]
    assert np.isscalar(x)


def test_array_units_exist(array_key):
    P.units(array_key)


def test_special_units_exist(special_stat):
    P.units(special_stat)


def test_twiss():
    P.twiss("xy", fraction=0.95)


def test_write_reload(tmp_path):
    h5file = os.path.join(tmp_path, "test.h5")
    P.write(h5file)

    # Equality and inequality
    P2 = ParticleGroup(h5file)
    assert P == P2

    P2.x += 1
    assert P != P2


def test_fractional_split():
    head, tail = P.fractional_split(0.5, "t")
    head, core, tail = P.fractional_split((0.1, 0.9), "t")


def test_plot_vs_z(array_key: str):
    P.plot("z", array_key)
    plt.show()


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
    pg = ParticleGroup.from_random_normal(
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
    pg = ParticleGroup.from_random_normal(n_particle, t_or_z=t_or_z)

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


@pytest.mark.filterwarnings("ignore:.*invalid value encountered in.*")
@pytest.mark.filterwarnings("ignore:.*divide by zero.*")
@pytest.mark.filterwarnings("ignore:.*Degrees of freedom.*")
@pytest.mark.filterwarnings("ignore:.*The fit may be poorly conditioned.*")
def test_plot_single_particle_vs_z(array_key: str):
    # Single particle plots aren't particularly useful, so we're mainly testing
    # for coverage and that this doesn't crash.  Filter out any warnings
    # from this that complain about bad calculated values.
    Ps = single_particle(pz=10e6)
    Ps.plot("z", array_key)
    plt.show()
