from __future__ import annotations

import numpy as np
import pytest

from beamphysics import ParticleGroup


def make_particle_group(pz):
    n = len(pz)
    data = {
        "x": np.zeros(n),
        "px": np.zeros(n),
        "y": np.zeros(n),
        "py": np.zeros(n),
        "z": np.zeros(n),
        "pz": np.array(pz, dtype=float),
        "t": np.zeros(n),
        "status": np.ones(n, dtype=int),
        "weight": np.full(n, 1e-12),
        "species": "electron",
    }
    return ParticleGroup(data=data)


def test_write_impact_does_not_mutate_caller(tmp_path):
    """The cathode-start small-pz shift must act on a copy, not the
    caller's internal pz array."""
    from beamphysics.interfaces.impact import write_impact

    P = make_particle_group([5.0, 1e6, 1e6])
    pz_before = P.pz.copy()
    write_impact(P, tmp_path / "impact.in", cathode_kinetic_energy_ref=1.0)
    np.testing.assert_array_equal(P.pz, pz_before)


def test_write_lucretia_with_stop_ix(tmp_path):
    """A valid stop_ix array must produce a file; a wrong-length one must
    raise instead of silently writing nothing."""
    from beamphysics.interfaces.lucretia import write_lucretia

    P = make_particle_group([1e6, 2e6, 3e6])
    outfile = tmp_path / "beam.mat"
    write_lucretia(P, str(outfile), stop_ix=[0, 0, 0], verbose=False)
    assert outfile.exists()

    with pytest.raises(ValueError, match="stop_ix"):
        write_lucretia(P, str(tmp_path / "bad.mat"), stop_ix=[0, 0], verbose=False)


def test_cst_get_scale_rejects_unknown_units():
    from beamphysics.interfaces.cst import get_scale

    assert get_scale("[mm]") == 1e-3
    assert get_scale("[V/m]") == 1
    with pytest.raises(ValueError, match="Unknown unit"):
        get_scale("[cm]")


BMAD_H5FILE = "docs/examples/data/bmad_particles.h5"


def _bmad_ascii_rows(path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = lines.index("BEGIN_BUNCH") + 5
    stop = lines.index("END_BUNCH")
    return np.array([[float(v) for v in line.split()] for line in lines[start:stop]])


def test_bmad_dict_round_trip_optional_keys():
    P = ParticleGroup(BMAD_H5FILE)[:50]
    P.assign_id()

    d = P.to_bmad()
    assert {"s", "ix_branch", "ix_ele", "location", "spin_x", "ix_user"} <= set(d)
    np.testing.assert_array_equal(d["location"], 2)  # downstream end
    np.testing.assert_allclose(d["p0c"], P.p0c)
    np.testing.assert_array_equal(d["ix_user"], P.id)

    P2 = ParticleGroup.from_bmad(d)
    assert P2 == P
    assert "p0c" in P2
    np.testing.assert_array_equal(P2.location, P.location)


def test_bmad_dict_round_trip_without_optional_keys():
    P = make_particle_group([1e6, 2e6])
    d = P.to_bmad()
    assert "s" not in d
    assert np.ndim(d["p0c"]) == 0

    P2 = ParticleGroup.from_bmad(d)
    assert P2 == P
    assert "p0c" not in P2


@pytest.mark.parametrize(("openpmd", "bmad"), [(-1, 1), (0, 3), (1, 2)])
def test_bmad_location_conversion(openpmd: int, bmad: int):
    from beamphysics.interfaces.bmad import (
        bmad_location_to_openpmd,
        openpmd_location_to_bmad,
    )

    np.testing.assert_array_equal(openpmd_location_to_bmad([openpmd, openpmd]), bmad)
    np.testing.assert_array_equal(bmad_location_to_openpmd([bmad, bmad]), openpmd)


def test_write_bmad_plain_columns(tmp_path):
    P = make_particle_group([1e6, 2e6])
    P.write_bmad(tmp_path / "bmad.txt", p0c=1e6)
    assert _bmad_ascii_rows(tmp_path / "bmad.txt").shape == (2, 8)


def test_write_bmad_spin_and_element_columns(tmp_path):
    P = make_particle_group([1e6, 2e6])
    P.spin_x = [0.1, 0.2]
    P.ix_ele = [5, 6]
    P.location = [-1, 1]
    P.write_bmad(tmp_path / "bmad.txt", p0c=1e6)

    rows = _bmad_ascii_rows(tmp_path / "bmad.txt")
    assert rows.shape == (2, 13)
    np.testing.assert_allclose(rows[:, 8], [0.1, 0.2])
    np.testing.assert_array_equal(rows[:, 9:11], 0)
    np.testing.assert_array_equal(rows[:, 11], [5, 6])
    np.testing.assert_array_equal(rows[:, 12], [1, 2])


@pytest.fixture
def impact_particles():
    P = make_particle_group([1e6, 2e6, 3e6])
    P.x = np.array([1e-3, -2e-3, 3e-3])
    P.px = np.array([10.0, -20.0, 30.0])
    P.z = np.array([0.0, 1e-4, -1e-4])
    P.weight = np.array([1e-12, 2e-12, 3e-12])
    P.id = np.array([1003, 7, 42])
    return P


def test_write_impact_default_is_6_columns(impact_particles, tmp_path):
    outfile = tmp_path / "partcl.data"
    res = impact_particles.write_impact(outfile)
    assert res["Flagdist"] == 16
    assert np.loadtxt(outfile, skiprows=1, ndmin=2).shape == (3, 6)


@pytest.mark.parametrize("cathode_kinetic_energy_ref", [None, 1.0])
def test_write_impact_9_columns(impact_particles, tmp_path, cathode_kinetic_energy_ref):
    P = impact_particles
    outfile = tmp_path / "partcl.data"
    res = P.write_impact(
        outfile, cathode_kinetic_energy_ref=cathode_kinetic_energy_ref, n_columns=9
    )
    assert res["Flagdist"] == 166
    assert res["Np"] == 3

    with open(outfile) as f:
        assert int(f.readline()) == 3
        assert f.readline().split()[-1] == "1003"

    dat = np.loadtxt(outfile, skiprows=1, ndmin=2)
    assert dat.shape == (3, 9)
    np.testing.assert_allclose(dat[:, 6], -1 / P.mass)
    np.testing.assert_allclose(dat[:, 7], -P.weight)
    np.testing.assert_array_equal(dat[:, 8], P.id)


def test_write_impact_9_columns_positive_species(tmp_path):
    P = make_particle_group([1e9, 2e9])
    P.species = "proton"
    outfile = tmp_path / "partcl.data"
    P.write_impact(outfile, n_columns=9)
    dat = np.loadtxt(outfile, skiprows=1, ndmin=2)
    np.testing.assert_allclose(dat[:, 6], 1 / P.mass)
    np.testing.assert_allclose(dat[:, 7], P.weight)


def test_write_impact_invalid_n_columns(impact_particles, tmp_path):
    with pytest.raises(ValueError, match="n_columns"):
        impact_particles.write_impact(tmp_path / "partcl.data", n_columns=7)


def test_impact_9_column_roundtrip(impact_particles, tmp_path):
    from beamphysics.interfaces.impact import (
        IMPACT_PARTICLE_NAMES_9,
        impact_particles_to_particle_data,
        parse_impact_particles,
    )

    P = impact_particles
    outfile = tmp_path / "partcl.data"
    P.write_impact(outfile, n_columns=9)

    tout = parse_impact_particles(outfile, names=IMPACT_PARTICLE_NAMES_9, skiprows=1)
    P2 = ParticleGroup(
        data=impact_particles_to_particle_data(tout, mc2=P.mass, species=P.species)
    )

    np.testing.assert_array_equal(P2.id, P.id)
    for key in ("x", "px", "z", "pz", "weight"):
        np.testing.assert_allclose(P2[key], P[key], rtol=1e-14)


def make_impact_tout(ids, charge_per_macroparticle):
    from beamphysics.interfaces.impact import IMPACT_PARTICLE_NAMES_9

    tout = np.zeros(len(ids), dtype=[(name, float) for name in IMPACT_PARTICLE_NAMES_9])
    tout["GBz"] = 1.0
    tout["charge_per_macroparticle"] = charge_per_macroparticle
    tout["id"] = ids
    return tout


@pytest.mark.parametrize(
    "ids",
    [[1, 2, 2], [0, 1, 2], [1, np.nan, 3]],
    ids=["repeated", "zero", "nan"],
)
def test_impact_particles_unusable_ids_are_dropped(ids):
    from beamphysics.interfaces.impact import impact_particles_to_particle_data

    data = impact_particles_to_particle_data(
        make_impact_tout(ids, -1e-12), mc2=0.511e6, species="electron"
    )
    assert "id" not in data


@pytest.mark.parametrize(("macrocharge", "expected"), [(0, 1 / 3), (-2e-12, 2e-12)])
def test_impact_particles_zero_charge_column_falls_back(macrocharge, expected):
    from beamphysics.interfaces.impact import impact_particles_to_particle_data

    data = impact_particles_to_particle_data(
        make_impact_tout([1, 2, 3], 0.0),
        mc2=0.511e6,
        species="electron",
        macrocharge=macrocharge,
    )
    np.testing.assert_allclose(data["weight"], expected)
    np.testing.assert_array_equal(data["id"], [1, 2, 3])
