"""Tests for the ImpactX interface.

The pure conversion tests need nothing but numpy. The reader tests run against
``docs/examples/data/impactx/*.h5``, written by ImpactX 26.08 with the
``generate.py`` script next to them, and need ``openpmd-api`` but not ImpactX.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from beamphysics import ParticleGroup
from beamphysics.interfaces.impactx import (
    IMPACTX_TO_PMD_SPECIES,
    PARTICLE_STATUS_LOST,
    ImpactXRefPart,
    UnrepresentableParticleData,
    _check_representable,
    particle_id_from_idcpu,
    beam_monitor_iterations,
    impactx_to_particlegroup_data,
    particlegroup_to_impactx,
    pmd_species_of,
    read_beam_monitor,
    refpart_from_openpmd,
)
from beamphysics.species import charge_of, e_charge, mass_of
from beamphysics.status import ParticleStatus
from beamphysics.units import c_light

DATA_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "docs"
    / "examples"
    / "data"
    / "impactx"
)
MONITOR = DATA_DIR / "monitor.h5"
PARTICLES_LOST = DATA_DIR / "particles_lost.h5"

try:
    import openpmd_api  # noqa: F401

    HAVE_OPENPMD_API = True
except ImportError:
    HAVE_OPENPMD_API = False

requires_data = pytest.mark.skipif(
    not (HAVE_OPENPMD_API and MONITOR.exists()),
    reason="needs openpmd-api and the ImpactX test data",
)

# The reference energy of both the fixtures and the committed test data.
KIN_ENERGY_MeV = 2.0e3


def make_ref(species: str = "electron", kin_energy_MeV: float = KIN_ENERGY_MeV):
    """An on-axis reference particle at s = 1.5 m, using beamphysics' own constants."""
    mass_MeV = mass_of(species) / 1.0e6
    gamma = 1.0 + kin_energy_MeV / mass_MeV
    return ImpactXRefPart(
        x=0.0,
        y=0.0,
        z=1.5,
        t=1.5,
        px=0.0,
        py=0.0,
        pz=np.sqrt(gamma**2 - 1.0),
        pt=-gamma,
        mass_MeV=mass_MeV,
        charge_qe=charge_of(species) / e_charge,
        s=1.5,
    )


@pytest.fixture
def electron_ref() -> ImpactXRefPart:
    return make_ref()


@pytest.fixture
def bunch(electron_ref: ImpactXRefPart) -> ParticleGroup:
    """A 2 GeV electron bunch in z-coordinates, matched to `electron_ref`."""
    rng = np.random.default_rng(42)
    n = 500
    mass_eV = electron_ref.mass_eV
    p0 = electron_ref.beta_gamma * mass_eV  # eV/c

    px = p0 * rng.normal(0.0, 1.0e-5, n)
    py = p0 * rng.normal(0.0, 1.0e-5, n)
    pz = p0 * (1.0 + rng.normal(0.0, 2.0e-3, n))

    return ParticleGroup(
        data={
            "x": rng.normal(0.0, 1.0e-4, n),
            "y": rng.normal(0.0, 1.0e-4, n),
            "z": np.full(n, electron_ref.z),
            "px": px,
            "py": py,
            "pz": pz,
            "t": electron_ref.t / c_light + rng.normal(0.0, 3.0e-12, n),
            "weight": np.full(n, 1.0e-9 / n),
            "status": np.full(n, int(ParticleStatus.ALIVE)),
            "species": "electron",
        }
    )


# ---------------------------------------------------------------------------
# Reference particle
# ---------------------------------------------------------------------------


def test_qm_units_relate_by_c_squared(electron_ref):
    assert electron_ref.qm_SI == pytest.approx(electron_ref.qm_eV * c_light**2)
    # -e / m_e in C/kg
    assert electron_ref.qm_SI == pytest.approx(-1.75882e11, rel=1e-5)


def test_gamma_and_beta_gamma_agree(electron_ref):
    assert electron_ref.gamma == pytest.approx(-electron_ref.pt)
    assert electron_ref.beta_gamma == pytest.approx(
        np.sqrt(electron_ref.gamma**2 - 1.0)
    )


@pytest.mark.parametrize("pmd_name", sorted(IMPACTX_TO_PMD_SPECIES.values()))
def test_species_inference(pmd_name):
    assert pmd_species_of(make_ref(pmd_name)) == pmd_name


def test_species_inference_refuses_the_unknown(electron_ref):
    from dataclasses import replace

    with pytest.raises(ValueError, match="Pass species= explicitly"):
        pmd_species_of(replace(electron_ref, mass_MeV=123.456))


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def test_roundtrip_is_exact(bunch, electron_ref):
    data = particlegroup_to_impactx(bunch, electron_ref)
    back = ParticleGroup(data=impactx_to_particlegroup_data(data, electron_ref))

    for key in ("x", "y", "px", "py", "pz", "t", "weight"):
        np.testing.assert_allclose(
            back[key], bunch[key], rtol=1e-11, atol=0, err_msg=key
        )
    assert back.species == bunch.species


def test_roundtrip_preserves_beam_statistics(bunch, electron_ref):
    data = particlegroup_to_impactx(bunch, electron_ref)
    back = ParticleGroup(data=impactx_to_particlegroup_data(data, electron_ref))

    assert back.charge == pytest.approx(bunch.charge, rel=1e-12)
    assert back.norm_emit_x == pytest.approx(bunch.norm_emit_x, rel=1e-9)
    assert back.norm_emit_y == pytest.approx(bunch.norm_emit_y, rel=1e-9)
    assert back.std("t") == pytest.approx(bunch.std("t"), rel=1e-9)
    assert back.avg("energy") == pytest.approx(bunch.avg("energy"), rel=1e-12)


def test_result_is_in_z_coordinates(bunch, electron_ref):
    """The bunch comes back on the reference plane, with the length in t."""
    data = particlegroup_to_impactx(bunch, electron_ref)
    back = ParticleGroup(data=impactx_to_particlegroup_data(data, electron_ref))

    assert back.in_z_coordinates
    np.testing.assert_array_equal(np.unique(back.z), [0.0])
    assert back.std("t") > 0.0


def test_weighting_counts_real_particles(bunch, electron_ref):
    data = particlegroup_to_impactx(bunch, electron_ref)
    assert data["weighting"].sum() * e_charge == pytest.approx(bunch.charge, rel=1e-12)


def test_momentum_t_is_accurate(bunch, electron_ref):
    """momentum_t must resolve gamma - gamma_ref, which is 1e-3 of gamma at 2 GeV.

    Errors are measured against the spread of momentum_t itself: a particle sitting
    exactly on the reference momentum has momentum_t ~ 0, where a per-particle
    relative error is meaningless.
    """
    data = particlegroup_to_impactx(bunch, electron_ref)

    mass_eV = np.longdouble(electron_ref.mass_eV)
    gamma_ref = np.longdouble(electron_ref.gamma)
    beta_gamma = np.longdouble(electron_ref.beta_gamma)
    gamma = np.sqrt(1.0 + (np.longdouble(bunch.p) / mass_eV) ** 2)
    exact = -(gamma - gamma_ref) / beta_gamma

    naive_gamma = np.sqrt(1.0 + (bunch.p / electron_ref.mass_eV) ** 2)
    naive = -(naive_gamma - electron_ref.gamma) / electron_ref.beta_gamma

    scale = float(np.std(exact))
    ours_error = float(np.max(np.abs(data["momentum_t"] - exact))) / scale
    naive_error = float(np.max(np.abs(naive - exact))) / scale
    assert ours_error < 1e-13
    # the algebraic form must never be worse than the plain difference
    assert ours_error <= naive_error


def test_input_particlegroup_is_not_mutated(electron_ref):
    """A t-coordinate bunch is drifted on a copy, never in place."""
    n = 32
    rng = np.random.default_rng(0)
    p0 = electron_ref.beta_gamma * electron_ref.mass_eV
    pg = ParticleGroup(
        data={
            "x": np.zeros(n),
            "y": np.zeros(n),
            "z": rng.normal(0.0, 1.0e-3, n),
            "px": np.zeros(n),
            "py": np.zeros(n),
            "pz": np.full(n, p0),
            "t": np.zeros(n),
            "weight": np.full(n, 1.0e-12),
            "status": np.ones(n, dtype=int),
            "species": "electron",
        }
    )
    assert pg.in_t_coordinates
    z_before, t_before = pg.z.copy(), pg.t.copy()

    particlegroup_to_impactx(pg, electron_ref)

    np.testing.assert_array_equal(pg.z, z_before)
    np.testing.assert_array_equal(pg.t, t_before)


def test_mismatched_reference_is_refused(bunch, electron_ref):
    """A reference particle from another beam makes pz imaginary; say so."""
    data = particlegroup_to_impactx(bunch, electron_ref)
    data["momentum_x"] = np.full(bunch.n_particle, 10.0)

    with pytest.raises(ValueError, match="transverse momentum"):
        impactx_to_particlegroup_data(data, electron_ref)


def test_status_and_id_are_carried_through(bunch, electron_ref):
    data = particlegroup_to_impactx(bunch, electron_ref)
    data["id"] = np.arange(bunch.n_particle)
    data["status"] = np.zeros(bunch.n_particle, dtype=int)

    back = ParticleGroup(data=impactx_to_particlegroup_data(data, electron_ref))
    np.testing.assert_array_equal(back.id, np.arange(bunch.n_particle))
    assert back.n_alive == 0


# ---------------------------------------------------------------------------
# AMReX id packing
# ---------------------------------------------------------------------------


VALID_BIT = np.uint64(1) << np.uint64(63)


def pack_idcpu(ids, cpus, valid=True):
    """Pack ids and ranks the way AMReX does, for the tests below."""
    ids = np.asarray(ids, dtype=np.uint64)
    cpus = np.asarray(cpus, dtype=np.uint64)
    packed = (ids << np.uint64(24)) | cpus
    return (packed | VALID_BIT) if valid else packed


def test_particle_id_keeps_the_whole_idcpu():
    """Only the validity bit comes off; id and rank both stay in the value."""
    packed = pack_idcpu([1, 2, 12345], [0, 3, 7])

    got, valid = particle_id_from_idcpu(packed)
    np.testing.assert_array_equal(valid, [True, True, True])
    np.testing.assert_array_equal(got >> 24, [1, 2, 12345])  # AMReX' own id
    np.testing.assert_array_equal(got & 0xFFFFFF, [0, 3, 7])  # originating rank
    # int64 is what ParticleGroup stores ids in, and the value must fit
    assert got.dtype == np.int64
    assert got.min() >= 0
    # and the original is recoverable
    np.testing.assert_array_equal(got.astype(np.uint64) | VALID_BIT, packed)


def test_particle_id_survives_a_particlegroup(bunch):
    """The raw idcpu overflows the int64 ParticleGroup stores ids in; the id must not.

    Stripping the validity bit is what makes it fit, so this checks the whole path,
    not just the dtype: ParticleGroup coerces ids with `_round_to_int_array`, which
    wraps silently rather than raising.
    """
    n = bunch.n_particle
    packed = pack_idcpu(np.arange(1, n + 1), np.zeros(n, dtype=np.uint64))
    assert packed.max() > np.iinfo(np.int64).max

    ids, _ = particle_id_from_idcpu(packed)
    data = particlegroup_to_impactx(bunch, make_ref())
    data["id"] = ids
    stored = ParticleGroup(data=impactx_to_particlegroup_data(data, make_ref())).id

    np.testing.assert_array_equal(stored, ids)
    assert stored.min() > 0


def test_particle_id_is_unique_across_ranks():
    """AMReX counts ids per rank, so the id alone collides on a parallel run."""
    packed = pack_idcpu([1, 1, 2, 2], [0, 1, 0, 1])

    got, _ = particle_id_from_idcpu(packed)
    assert len(np.unique(got)) == 4
    assert len(np.unique(got >> 24)) == 2  # what the AMReX id alone would give


def test_particle_id_reports_invalid_particles():
    packed = pack_idcpu([1, 2], [0, 0], valid=False)

    got, valid = particle_id_from_idcpu(packed)
    np.testing.assert_array_equal(valid, [False, False])
    np.testing.assert_array_equal(got >> 24, [1, 2])


# ---------------------------------------------------------------------------
# Unrepresentable per-particle data
# ---------------------------------------------------------------------------


def test_zero_spin_is_representable():
    _check_representable({name: np.zeros(4) for name in ("spin_x", "spin_y", "spin_z")})


def test_nonzero_spin_refuses_loudly():
    columns = {name: np.zeros(4) for name in ("spin_x", "spin_y", "spin_z")}
    columns["spin_z"][2] = 1.0
    with pytest.raises(UnrepresentableParticleData, match="spin"):
        _check_representable(columns)


def test_runtime_component_refuses_loudly():
    with pytest.raises(UnrepresentableParticleData, match="s_lost"):
        _check_representable({"s_lost": np.full(4, 0.25)})


# ---------------------------------------------------------------------------
# Reference particle from openPMD attributes
# ---------------------------------------------------------------------------


class _FakeSpecies:
    def __init__(self, attributes: dict):
        self._attributes = attributes

    @property
    def attributes(self):
        return list(self._attributes)

    def get_attribute(self, name):
        return self._attributes[name]


def test_refpart_from_openpmd_rejects_a_foreign_species():
    with pytest.raises(KeyError, match="ImpactX BeamMonitor"):
        refpart_from_openpmd(_FakeSpecies({"x_ref": 0.0}))


# ---------------------------------------------------------------------------
# BeamMonitor reader
# ---------------------------------------------------------------------------


@requires_data
def test_beam_monitor_iterations():
    assert beam_monitor_iterations(MONITOR) == [1, 5]


@requires_data
def test_read_beam_monitor_matches_impactx_moments():
    """ImpactX's own reduced beam characteristics must come back out."""
    import openpmd_api as io

    P = read_beam_monitor(MONITOR)

    series = io.Series(str(MONITOR), io.Access.read_only)
    beam = series.iterations[5].particles["beam"]
    attrs = {name: beam.get_attribute(name) for name in beam.attributes}
    series.close()

    assert P.species == "electron"
    assert P.n_particle == 184
    assert P.n_alive == P.n_particle
    assert P.charge == pytest.approx(abs(attrs["charge_C"]), rel=1e-12)

    assert P.std("x") == pytest.approx(attrs["sigma_x"], rel=1e-12)
    assert P.std("y") == pytest.approx(attrs["sigma_y"], rel=1e-12)
    assert P.std("t") * c_light == pytest.approx(attrs["sigma_t"], rel=1e-12)
    assert P.avg("x") == pytest.approx(attrs["mean_x"], rel=1e-9)
    assert P.avg("y") == pytest.approx(attrs["mean_y"], rel=1e-9)
    assert P.avg("t") * c_light - attrs["t_ref"] == pytest.approx(
        attrs["mean_t"], rel=1e-9
    )
    # beamphysics' covariance is bias-corrected and ImpactX' is not, and with equal
    # weights that is exactly a factor n/(n-1) on the emittance
    n = P.n_particle
    bias = n / (n - 1)
    assert P.norm_emit_x == pytest.approx(attrs["emittance_xn"] * bias, rel=1e-9)
    assert P.norm_emit_y == pytest.approx(attrs["emittance_yn"] * bias, rel=1e-9)


@requires_data
def test_conversion_to_impactx_matches_the_file_it_came_from():
    """Check the write direction against real ImpactX output, not synthetic data.

    Every other conversion test starts from a bunch this module itself built, so a
    paired sign or normalization error present in *both* directions would round-trip
    cleanly and pass. This one converts a bunch read from the file back to ImpactX
    arrays and compares against what ImpactX actually wrote.
    """
    import openpmd_api as io

    series = io.Series(str(MONITOR), io.Access.read_only)
    beam = series.iterations[5].particles["beam"]
    ref = refpart_from_openpmd(beam)
    raw = {
        "position_x": beam["position"]["x"].load_chunk(),
        "position_y": beam["position"]["y"].load_chunk(),
        "position_t": beam["position"]["t"].load_chunk(),
        "momentum_x": beam["momentum"]["x"].load_chunk(),
        "momentum_y": beam["momentum"]["y"].load_chunk(),
        "momentum_t": beam["momentum"]["t"].load_chunk(),
        "weighting": beam["weighting"][io.Record_Component.SCALAR].load_chunk(),
    }
    series.flush()
    series.close()

    data = particlegroup_to_impactx(read_beam_monitor(MONITOR), ref)

    for key, expected in raw.items():
        expected = np.asarray(expected)
        # measure against the spread, or the level for a uniform column like weighting
        scale = np.std(expected) or np.abs(np.mean(expected)) or 1.0
        error = np.max(np.abs(data[key] - expected)) / scale
        assert error < 1e-12, f"{key}: {error:e}"


@requires_data
def test_read_beam_monitor_reference_particle():
    import openpmd_api as io

    series = io.Series(str(MONITOR), io.Access.read_only)
    ref = refpart_from_openpmd(series.iterations[5].particles["beam"])
    series.close()

    assert pmd_species_of(ref) == "electron"
    assert ref.charge_qe == pytest.approx(-1.0)
    assert ref.mass_MeV == pytest.approx(mass_of("electron") / 1e6, rel=1e-8)
    assert ref.gamma == pytest.approx(1.0 + KIN_ENERGY_MeV / ref.mass_MeV, rel=1e-8)
    assert ref.s == pytest.approx(1.25)
    assert ref.t == pytest.approx(ref.s, rel=1e-6)  # ultrarelativistic


@requires_data
def test_read_beam_monitor_carries_ids():
    P = read_beam_monitor(MONITOR)
    assert P.id.dtype.kind == "i"
    assert len(np.unique(P.id)) == P.n_particle
    # the test data is a serial run, so every rank field is 0 and the AMReX ids are
    # 1..200 for the 200 macroparticles the run started with
    np.testing.assert_array_equal(P.id & 0xFFFFFF, 0)
    assert (P.id >> 24).min() >= 1
    assert (P.id >> 24).max() <= 200


@requires_data
def test_read_beam_monitor_iteration_selection():
    first = read_beam_monitor(MONITOR, iteration=1)
    last = read_beam_monitor(MONITOR, iteration=5)

    assert first.n_particle == 200  # before the collimator
    assert last.n_particle == 184
    assert first.std("x") != last.std("x")
    # the default is the last iteration
    assert read_beam_monitor(MONITOR).n_particle == last.n_particle


@requires_data
def test_read_beam_monitor_rejects_bad_arguments():
    with pytest.raises(KeyError, match="Iteration 99"):
        read_beam_monitor(MONITOR, iteration=99)
    with pytest.raises(KeyError, match="always names it 'beam'"):
        read_beam_monitor(MONITOR, species_name="particles_lost")


@requires_data
def test_particlegroup_from_impactx_classmethod():
    from beamphysics.testing import assert_pg_close

    assert_pg_close(ParticleGroup.from_impactx(MONITOR), read_beam_monitor(MONITOR))


@requires_data
@pytest.mark.parametrize("species", ["proton", "positron"])
def test_species_cannot_relabel_across_species(species):
    """The momenta are normalized by the *reference* mass, so a relabel is not free.

    Read as protons, this 2 GeV electron beam would come back at gamma = 2.35 and no
    exception. `positron` is the sharp case: same mass as the reference, opposite
    charge, so only checking the mass would let it through.
    """
    with pytest.raises(ValueError, match="not the species the reference particle"):
        read_beam_monitor(MONITOR, species=species)


def test_species_can_name_what_inference_cannot(bunch):
    """`species=` is for species ImpactX has no name for, not for converting."""
    from dataclasses import replace

    muon_ref = replace(
        make_ref(),
        mass_MeV=mass_of("muon") / 1.0e6,
        charge_qe=charge_of("muon") / e_charge,
    )
    with pytest.raises(ValueError, match="Pass species= explicitly"):
        pmd_species_of(muon_ref)

    data = particlegroup_to_impactx(bunch, make_ref())
    back = ParticleGroup(
        data=impactx_to_particlegroup_data(data, muon_ref, species="muon")
    )
    assert back.species == "muon"


requires_lost_data = pytest.mark.skipif(
    not (HAVE_OPENPMD_API and PARTICLES_LOST.exists()),
    reason="needs openpmd-api and the ImpactX particles_lost test data",
)


@requires_lost_data
def test_particles_lost_has_no_reference_particle():
    """ImpactX writes a default-constructed RefPart into particles_lost output."""
    import openpmd_api as io

    series = io.Series(str(PARTICLES_LOST), io.Access.read_only)
    ref = refpart_from_openpmd(series.iterations[0].particles["beam"])
    series.close()

    assert ref.mass_MeV == 0.0
    assert ref.gamma == 0.0

    with pytest.raises(ValueError, match="zeroed reference particle"):
        read_beam_monitor(PARTICLES_LOST)


@requires_lost_data
def test_particles_lost_carries_an_unrepresentable_component():
    """ImpactX' lost-particle output always has a runtime `s_lost` column."""
    import openpmd_api as io

    series = io.Series(str(MONITOR), io.Access.read_only)
    ref = refpart_from_openpmd(series.iterations[5].particles["beam"])
    series.close()

    with pytest.raises(UnrepresentableParticleData, match=r"s_lost"):
        read_beam_monitor(PARTICLES_LOST, ref=ref)

    with pytest.warns(UserWarning, match=r"s_lost"):
        lost = read_beam_monitor(PARTICLES_LOST, ref=ref, strict=False)
    kept = read_beam_monitor(MONITOR, iteration=5)

    assert lost.n_particle == 16
    assert lost.n_particle + kept.n_particle == 200
    # AMReX marks these valid -- they are live entries of the loss container -- so the
    # validity bit alone would report them alive, which is exactly backwards
    assert lost.n_alive == 0
    assert lost.n_dead == 16
    np.testing.assert_array_equal(lost.status, PARTICLE_STATUS_LOST)
    assert kept.n_alive == kept.n_particle
    # the same species is used for lost particles; ids partition the original beam
    assert set(lost.id).isdisjoint(set(kept.id))
    assert len(set(lost.id) | set(kept.id)) == 200


def test_unphysical_reference_particle_is_refused(bunch):
    zero_ref = ImpactXRefPart(
        x=0.0, y=0.0, z=0.0, t=0.0,
        px=0.0, py=0.0, pz=0.0, pt=0.0,
        mass_MeV=0.0, charge_qe=0.0,
    )  # fmt: skip
    with pytest.raises(ValueError, match="not physical"):
        particlegroup_to_impactx(bunch, zero_ref)
