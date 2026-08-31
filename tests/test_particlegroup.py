import os
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from beamphysics import ParticleGroup, pmd_init
from beamphysics.exceptions import MultipleIterationsError, NotOpenPMDError
from beamphysics.particles import load_bunch_data, single_particle
from beamphysics.readers import (
    get_root_metadata,
    expected_record_unit_dimension,
    load_only_time_offset,
    load_time_offset,
    particle_array,
)

# File with no species subgroup
LEGACY_H5FILE = "docs/examples/data/bmad_particles.h5"

P = ParticleGroup(LEGACY_H5FILE)


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


@pytest.fixture
def simple_pg() -> ParticleGroup:
    """Tiny PG for tests."""
    return ParticleGroup(
        data={
            "x": np.array([0.0, 1e-3]),
            "px": np.array([0.0, 10.0]),
            "y": np.array([0.0, 2e-3]),
            "py": np.array([0.0, 20.0]),
            "z": np.array([0.0, 0.0]),
            "pz": np.array([1e6, 1.1e6]),
            "t": np.array([0.0, 1e-12]),
            "status": np.array([1, 1]),
            "weight": np.array([0.5e-9, 0.5e-9]),
            "species": "electron",
        }
    )


@pytest.fixture
def species_h5file(simple_pg: ParticleGroup, tmp_path: pathlib.Path) -> pathlib.Path:
    """File written by ParticleGroup.write, which nests a species subgroup."""
    h5file = tmp_path / "simple.h5"
    simple_pg.write(h5file)
    return h5file


def test_init_species_path(simple_pg: ParticleGroup, species_h5file: pathlib.Path):
    assert ParticleGroup(str(species_h5file)) == simple_pg
    assert ParticleGroup(species_h5file) == simple_pg


def test_init_species_group(simple_pg: ParticleGroup, species_h5file: pathlib.Path):
    with h5py.File(species_h5file, "r") as fp:
        assert ParticleGroup(fp) == simple_pg


def test_init_legacy_path():
    ParticleGroup(LEGACY_H5FILE)
    ParticleGroup(pathlib.Path(LEGACY_H5FILE))


def test_init_legacy_group():
    with h5py.File(LEGACY_H5FILE, "r") as fp:
        assert ParticleGroup(fp) == P


def test_init_legacy_root_base_path():
    """distgen_particles.h5 uses basePath '/' with the records at the root."""
    P2 = ParticleGroup("docs/examples/data/distgen_particles.h5")
    assert len(P2) > 0
    assert P2.species == "electron"


def test_init_multiple_iterations():
    """astra_particles.h5 holds two iterations: /screen/0 and /screen/1."""
    with pytest.raises(MultipleIterationsError):
        ParticleGroup("docs/examples/data/astra_particles.h5")


@pytest.mark.parametrize("h5", [{"x": [0.0]}, 3, object()])
def test_init_unsupported_type(h5):
    with pytest.raises(TypeError):
        ParticleGroup(h5)


def test_init_nested_openpmd_root(simple_pg: ParticleGroup, tmp_path: pathlib.Path):
    """
    A group initialized as an openPMD root inside a larger file resolves its
    particles path relative to itself, not to the file root.
    """
    h5file = tmp_path / "nested.h5"
    with h5py.File(h5file, "w") as fp:
        run = fp.create_group("run1")
        pmd_init(run, basePath="/", particlesPath="particles")
        simple_pg.write(run.create_group("particles"))

        # Same path from the file root, holding different particles
        decoy = fp.create_group("decoy")
        pmd_init(decoy, basePath="/", particlesPath="particles")
        other = simple_pg.copy()
        other.x += 1.0
        other.write(decoy.create_group("particles"))
        fp["particles"] = h5py.SoftLink("/decoy/particles")

    with h5py.File(h5file, "r") as fp:
        with pytest.warns(FutureWarning):
            assert ParticleGroup(fp["run1"]) == simple_pg
        with pytest.warns(FutureWarning):
            assert ParticleGroup(fp["decoy"]) == other


@pytest.mark.parametrize("subpath", ["data/00001", "data/00001/particles"])
def test_init_legacy_below_root_warns(subpath: str):
    with h5py.File(LEGACY_H5FILE, "r") as fp:
        with pytest.warns(FutureWarning):
            assert ParticleGroup(fp[subpath]) == P


def test_not_openpmd_file_raises():
    """elegant_raw.h5 is plain HDF5, with no openPMD root attributes."""
    with pytest.raises(NotOpenPMDError):
        ParticleGroup.from_hdf5("docs/examples/data/elegant_raw.h5")


def test_get_root_metadata_attrs(species_h5file: pathlib.Path):
    with h5py.File(species_h5file, "r") as fp:
        attrs = get_root_metadata(fp)

    assert attrs["openPMD"] == "2.0.0"


BUNCH_DATA_KEYS = {
    "species",
    "total_charge",
    "x",
    "px",
    "y",
    "py",
    "z",
    "pz",
    "t",
    "status",
    "weight",
}


@pytest.mark.parametrize("subpath", ["particles", "particles/electron"])
def test_load_bunch_data_species(
    simple_pg: ParticleGroup, species_h5file: pathlib.Path, subpath: str
):
    with h5py.File(species_h5file, "r") as fp:
        data = load_bunch_data(fp[subpath])

    assert BUNCH_DATA_KEYS <= set(data)
    assert data["species"] == "electron"
    assert np.allclose(data["x"], simple_pg.x)
    assert np.allclose(data["pz"], simple_pg.pz)


def test_load_bunch_data_legacy():
    with h5py.File(LEGACY_H5FILE, "r") as fp:
        data = load_bunch_data(fp["data/00001/particles"])

    assert BUNCH_DATA_KEYS <= set(data)
    assert data["species"] == "electron"
    assert np.allclose(data["x"], P.x)
    assert np.allclose(data["pz"], P.pz)


def test_write_t_offset(tmp_path: pathlib.Path):
    t_offset = 5e-9
    h5file = tmp_path / "test_offset.h5"
    P.write(h5file, t_offset=t_offset)

    with h5py.File(h5file, "r") as fp:
        g = fp[f"particles/{P.species}"]

        # Constant component: a group with value and shape
        offset = g["timeOffset"]
        assert isinstance(offset, h5py.Group)
        assert offset.attrs["value"] == t_offset
        assert tuple(offset.attrs["shape"]) == (len(P),)
        assert offset.attrs["unitSI"] == 1.0
        assert tuple(offset.attrs["unitDimension"]) == tuple(
            expected_record_unit_dimension["timeOffset"]
        )

        # The time record itself is not shifted
        assert np.allclose(particle_array(g, "t", include_offset=False), P.t)

    # Readers add the offset to t
    P2 = ParticleGroup(h5file)
    assert np.allclose(P2.t, P.t + t_offset)
    assert np.allclose(P2.x, P.x)
    assert np.allclose(P2.pz, P.pz)


def test_write_t_offset_array(tmp_path: pathlib.Path):
    t_offset = np.linspace(0, 1e-9, len(P))
    h5file = tmp_path / "test_offset_array.h5"
    P.write(h5file, t_offset=t_offset)

    with h5py.File(h5file, "r") as fp:
        assert isinstance(fp[f"particles/{P.species}/timeOffset"], h5py.Dataset)

    assert np.allclose(ParticleGroup(h5file).t, P.t + t_offset)


def test_write_t_offset_bad_shape(tmp_path: pathlib.Path):
    h5file = tmp_path / "test_offset_bad.h5"
    with pytest.raises(ValueError):
        P.write(h5file, t_offset=np.zeros(len(P) + 1) + 1e-9)


def test_from_hdf5_matches_init(simple_pg: ParticleGroup, species_h5file: pathlib.Path):
    assert ParticleGroup.from_hdf5(species_h5file) == simple_pg

    with h5py.File(species_h5file, "r") as fp:
        assert ParticleGroup.from_hdf5(fp) == simple_pg


def test_from_hdf5_include_time_offset(tmp_path: pathlib.Path):
    t_offset = 5e-9
    h5file = tmp_path / "test_offset.h5"
    P.write(h5file, t_offset=t_offset)

    assert np.allclose(ParticleGroup.from_hdf5(h5file).t, P.t + t_offset)

    P2 = ParticleGroup.from_hdf5(h5file, include_time_offset=False)
    assert np.allclose(P2.t, P.t)
    assert np.allclose(P2.x, P.x)
    assert np.allclose(P2.pz, P.pz)


def test_from_hdf5_include_time_offset_array(tmp_path: pathlib.Path):
    t_offset = np.linspace(0, 1e-9, len(P))
    h5file = tmp_path / "test_offset_array.h5"
    P.write(h5file, t_offset=t_offset)

    assert np.allclose(ParticleGroup.from_hdf5(h5file).t, P.t + t_offset)
    assert np.allclose(
        ParticleGroup.from_hdf5(h5file, include_time_offset=False).t, P.t
    )


def test_load_only_time_offset_scalar(tmp_path: pathlib.Path):
    """A constant component is returned as a float."""
    t_offset = 5e-9
    h5file = tmp_path / "test_offset.h5"
    P.write(h5file, t_offset=t_offset)

    offset = load_only_time_offset(h5file)
    assert isinstance(offset, float)
    assert offset == t_offset
    assert load_only_time_offset(str(h5file)) == t_offset

    with h5py.File(h5file, "r") as fp:
        assert load_only_time_offset(fp) == t_offset
        assert load_time_offset(fp[f"particles/{P.species}"]) == t_offset


def test_load_only_time_offset_array(tmp_path: pathlib.Path):
    """A per-particle dataset is returned as an array."""
    t_offset = np.linspace(0, 1e-9, len(P))
    h5file = tmp_path / "test_offset_array.h5"
    P.write(h5file, t_offset=t_offset)

    offset = load_only_time_offset(h5file)
    assert isinstance(offset, np.ndarray)
    assert np.allclose(offset, t_offset)


@pytest.mark.parametrize("t_offset", [0.0, 5e-9])
def test_load_only_time_offset_recovers_t(tmp_path: pathlib.Path, t_offset: float):
    """Bare time plus the offset is the shifted time."""
    h5file = tmp_path / "test_recover.h5"
    P.write(h5file, t_offset=t_offset)

    bare = ParticleGroup.from_hdf5(h5file, include_time_offset=False)
    shifted = ParticleGroup.from_hdf5(h5file)
    assert np.allclose(bare.t + load_only_time_offset(h5file), shifted.t)


def test_load_only_time_offset_missing_file(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        load_only_time_offset(tmp_path / "does_not_exist.h5")


@pytest.mark.parametrize("h5", [{"x": [0.0]}, 3])
def test_load_only_time_offset_unsupported_type(h5):
    with pytest.raises(TypeError):
        load_only_time_offset(h5)


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


def test_id_ensure_int():
    data = single_particle().data
    data.pop("status", None)
    data.pop("id", None)

    P = ParticleGroup(data={**data, "status": [0.1], "id": [0.0]})
    np.testing.assert_array_equal(P.status, [0])
    np.testing.assert_array_equal(P.id, [0])
    assert np.issubdtype(P.status.dtype, int)
    assert np.issubdtype(P.id.dtype, int)

    for key in P._settable_array_keys:
        if key not in {"id", "status"}:
            assert np.issubdtype(getattr(P, key).dtype, float), key


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        pytest.param(
            [0.0, 1.0],
            np.asarray([0, 1]),
            id="list-float",
        ),
        pytest.param(
            [0.0, 0.999],
            np.asarray([0, 1]),
            id="list-float-rounded",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.asarray([0, 1]),
            id="ndarray-float",
        ),
        pytest.param(
            np.array([0.0, 0.999]),
            np.asarray([0, 1]),
            id="ndarray-float-rounded",
        ),
    ],
)
def test_coerce_int_array_round(val, expected: np.ndarray):
    from beamphysics.particles import _round_to_int_array

    result = _round_to_int_array(val)
    np.testing.assert_array_equal(actual=result, desired=expected)
    assert np.issubdtype(result.dtype, int)
