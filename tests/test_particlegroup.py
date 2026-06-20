import os
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from beamphysics import ParticleGroup
from beamphysics.particles import single_particle

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


def test_twiss_respects_weights():
    """Twiss must use the weighted covariance: particles with (nearly) zero
    weight should not influence the result."""
    P2 = P.copy()
    keep = np.arange(len(P2)) % 2 == 0
    w = np.array(P2.weight, dtype=float).copy()
    w[~keep] *= 1e-13  # effectively remove the odd-indexed particles
    P2.weight = w

    t_weighted = P2.twiss(plane="x")
    t_subset = P[keep].twiss(plane="x")
    assert t_weighted["beta_x"] == pytest.approx(t_subset["beta_x"], rel=1e-6)
    # And the down-weighted group differs from the unmodified full group.
    t_full = P.twiss(plane="x")
    assert t_weighted["beta_x"] != pytest.approx(t_full["beta_x"], rel=1e-9)


def test_twiss_match_forwards_p0c():
    """twiss_match must forward p0c to matched_particles."""
    from beamphysics.statistics import matched_particles

    via_method = P.twiss_match(beta=10, alpha=0, plane="x", p0c=2 * P["mean_p"])
    direct = matched_particles(P, beta=10, alpha=0, plane="x", p0c=2 * P["mean_p"])
    np.testing.assert_allclose(via_method.px, direct.px)
    # And a different p0c gives a different result.
    other = P.twiss_match(beta=10, alpha=0, plane="x", p0c=P["mean_p"])
    assert not np.allclose(via_method.px, other.px)


def test_slice_statistics_twiss_keys_filled():
    """Requesting twiss keys must return computed per-slice twiss values,
    never an uninitialized array under the raw key."""
    from beamphysics.statistics import slice_statistics

    sdat = slice_statistics(P, keys=["mean_z", "twiss_x"], n_slice=5, slice_key="z")
    assert "twiss_x" not in sdat  # expanded, not returned raw
    assert "twiss_beta_x" in sdat
    assert np.all(np.isfinite(sdat["twiss_beta_x"]))


def test_eq_checks_species_and_does_not_assign_ids():
    """Comparing two groups must not assign ids as a side effect, and
    groups of different species must not compare equal."""
    P1 = P.copy()
    P2 = P.copy()
    P1._data.pop("id", None)
    P2._data.pop("id", None)
    assert P1 == P2
    assert "id" not in P1._data and "id" not in P2._data  # no side effect

    P3 = P.copy()
    P3._data["species"] = "positron"
    assert P != P3


def test_eq_default_ids_match_missing_ids():
    """A group whose explicit ids are the default 1..n must compare equal to an
    otherwise-identical group that has no ids stored."""
    P1 = P.copy()
    P2 = P.copy()
    P1._data.pop("id", None)
    P2.assign_id()  # default ids: 1..n
    assert np.array_equal(P2._data["id"], np.arange(1, len(P2) + 1))
    assert P1 == P2
    assert P2 == P1  # symmetric
    assert "id" not in P1._data  # no side effect


def test_eq_custom_ids_differ_from_missing_ids():
    """A group with explicit non-default ids must not compare equal to an
    otherwise-identical group that has no ids (which would default to 1..n)."""
    P1 = P.copy()
    P2 = P.copy()
    P1._data.pop("id", None)
    P2.id = np.arange(1, len(P2) + 1) + 100  # non-default ids
    assert P1 != P2
    assert P2 != P1  # symmetric
    assert "id" not in P1._data  # no side effect


def test_write_reload(tmp_path):
    h5file = os.path.join(tmp_path, "test.h5")
    P.write(h5file)

    # Equality and inequality
    P2 = ParticleGroup(h5file)
    assert P == P2

    P2.x += 1
    assert P != P2


def test_write_reload_h5(tmp_path: pathlib.Path):
    h5file = tmp_path / "test.h5"
    with h5py.File(h5file, "w") as fp:
        P.write(fp)

    P2 = ParticleGroup(h5file)
    assert P == P2


def test_fractional_split():
    head, tail = P.fractional_split(0.5, "t")
    head, core, tail = P.fractional_split((0.1, 0.9), "t")


def test_plot_vs_z(array_key: str):
    P.plot("z", array_key)
    plt.show()


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
