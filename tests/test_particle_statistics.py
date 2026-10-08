import warnings

import numpy as np
import pytest

from beamphysics import ParticleGroup
from beamphysics.particles import _LEGACY_KEYS
from beamphysics.standards.statistics import (
    OPERATORS,
    get_all_statistics_by_key,
    scalar_statistic_keys,
)
from beamphysics.statistics import StatisticKey, StatisticOperator, TwissParameter

H5FILE = "docs/examples/data/bmad_particles.h5"
EXTRA_KEYS = [
    "mean_mass",
    "sigma_mass",
    "bunching",
    "mean_bogus",
    "cov_x__bogus",
    "bogus",
]


@pytest.fixture(scope="module")
def particle_group() -> ParticleGroup:
    P = ParticleGroup(H5FILE)
    rng = np.random.default_rng(0)
    # Unequal weights so the weight statistics and normalizations are nontrivial
    weight = P.weight * rng.uniform(0.5, 1.5, len(P))
    return ParticleGroup(data={**P.data, "weight": weight})


def per_key_statistics(P: ParticleGroup, keys: list[str]) -> dict:
    stats = {}
    for key in keys:
        try:
            stats[key] = P[key]
        except Exception:
            pass
    return stats


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("cov_x__px", StatisticKey(StatisticOperator.COV, ("x", "px"))),
        ("cov_z/c__beta_x", StatisticKey(StatisticOperator.COV, ("z/c", "beta_x"))),
        ("sigma_x", StatisticKey(StatisticOperator.SIGMA, ("x",))),
        (
            "mean_kinetic_energy",
            StatisticKey(StatisticOperator.MEAN, ("kinetic_energy",)),
        ),
        ("delta_pz", StatisticKey(StatisticOperator.DELTA, ("pz",))),
        ("ptp_t", StatisticKey(StatisticOperator.PTP, ("t",))),
        ("higher_order_energy_spread", None),
        ("norm_emit_x", None),
        ("x", None),
    ],
)
def test_statistic_key_from_string(key: str, expected: StatisticKey | None) -> None:
    assert StatisticKey.from_string(key) == expected


@pytest.mark.parametrize("key", ["max_", "cov_x", "cov_x__", "cov_x__px__py"])
def test_statistic_key_from_string_malformed(key: str) -> None:
    with pytest.raises(ValueError, match="exactly"):
        StatisticKey.from_string(key)


@pytest.mark.parametrize("n_particle", [0, 1, 2, None])
def test_particle_statistics_matches_per_key(
    particle_group: ParticleGroup, n_particle: int | None
) -> None:
    P = particle_group if n_particle is None else particle_group[0:n_particle]
    keys = [*get_all_statistics_by_key(), *EXTRA_KEYS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = per_key_statistics(P, keys)
        stats = P.statistics(*keys, skip_errors=True)

        def magnitude(name: str) -> float:
            return float(np.max(np.abs(P[name]), initial=0))

    assert list(stats) == list(expected)
    for key, value in expected.items():
        assert type(stats[key]) is type(value), key
        split = StatisticKey.from_string(_LEGACY_KEYS.get(key, key))
        if split is None or isinstance(value, str):
            np.testing.assert_equal(stats[key], value, err_msg=key)
            continue
        op, names = split
        if op is StatisticOperator.COV and not np.isfinite(value):
            # Undefined for a single particle: inf or nan, depending on rounding
            assert not np.isfinite(stats[key]), key
            continue
        # Rounding is relative to the arrays' magnitudes, not the (possibly ~0) result
        atol = 1e-12 * np.prod([magnitude(name) for name in names]) * len(names)
        np.testing.assert_allclose(stats[key], value, rtol=1e-9, atol=atol, err_msg=key)


def test_particle_statistics_raises(particle_group: ParticleGroup) -> None:
    with pytest.raises(ValueError, match="bunching"):
        particle_group.statistics("mean_x", "bunching")
    with pytest.raises(AttributeError):
        particle_group.statistics("mean_bogus")


def test_particle_statistics_skip_errors(particle_group: ParticleGroup) -> None:
    stats = particle_group.statistics(
        "mean_x",
        "bunching",
        "cov_x__bogus",
        "sigma_x",
        skip_errors=True,
    )
    assert list(stats) == ["mean_x", "sigma_x"]


def test_operator_prefixes_match_standard() -> None:
    assert {op.prefix for op in StatisticOperator if op.n_arrays == 1} == set(OPERATORS)


@pytest.mark.parametrize(
    ("op", "names", "key"),
    [
        (StatisticOperator.MEAN, ("x",), "mean_x"),
        (StatisticOperator.DELTA, ("z/c",), "delta_z/c"),
        (StatisticOperator.COV, ("x", "px"), "cov_x__px"),
    ],
)
def test_operator_key_round_trip(
    op: StatisticOperator, names: tuple[str, ...], key: str
) -> None:
    assert op.key(*names) == key
    assert op.parse(key) == names


@pytest.mark.parametrize(
    ("op", "names"),
    [
        (StatisticOperator.MEAN, ()),
        (StatisticOperator.MEAN, ("x", "px")),
        (StatisticOperator.MEAN, ("",)),
        (StatisticOperator.COV, ("x",)),
        (StatisticOperator.COV, ("x", "")),
    ],
)
def test_operator_key_invalid(op: StatisticOperator, names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        op.key(*names)


@pytest.mark.parametrize(
    ("op", "key"),
    [
        (StatisticOperator.MEAN, "sigma_x"),
        (StatisticOperator.COV, "mean_x"),
    ],
)
def test_operator_parse_other_prefix(op: StatisticOperator, key: str) -> None:
    assert op.parse(key) is None


@pytest.mark.parametrize(
    ("op", "key"),
    [
        (StatisticOperator.MEAN, "mean_"),
        (StatisticOperator.COV, "cov_x"),
        (StatisticOperator.COV, "cov_x__"),
        (StatisticOperator.COV, "cov_x__px__py"),
    ],
)
def test_operator_parse_malformed(op: StatisticOperator, key: str) -> None:
    with pytest.raises(ValueError, match="exactly"):
        op.parse(key)


@pytest.mark.parametrize("key", ["cov_x", "cov_x__px__py", "mean_"])
def test_getitem_malformed(particle_group: ParticleGroup, key: str) -> None:
    with pytest.raises(ValueError, match="exactly"):
        particle_group[key]


def test_particle_statistics_malformed(particle_group: ParticleGroup) -> None:
    with pytest.raises(ValueError, match="exactly"):
        particle_group.statistics("mean_x", "cov_x")
    stats = particle_group.statistics("mean_x", "cov_x", skip_errors=True)
    assert list(stats) == ["mean_x"]


@pytest.mark.parametrize("include_covariance", [True, False])
@pytest.mark.parametrize("include_twiss", [True, False])
def test_scalar_statistic_keys(include_covariance: bool, include_twiss: bool) -> None:
    keys = scalar_statistic_keys(
        include_covariance=include_covariance, include_twiss=include_twiss
    )
    by_key = get_all_statistics_by_key()
    assert all(by_key[key]["shape"] == [] for key in keys)
    assert "bunching" not in keys
    assert {"mean_x", "sigma_x", "norm_emit_x", "n_particle"} <= set(keys)
    assert any(key.startswith("cov_") for key in keys) == include_covariance
    assert any(key.startswith("twiss_") for key in keys) == include_twiss
    assert keys is scalar_statistic_keys(
        include_covariance=include_covariance, include_twiss=include_twiss
    )


def test_particle_statistics_default_keys(particle_group: ParticleGroup) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = particle_group.statistics()
    assert list(stats) == list(scalar_statistic_keys())
    assert all(np.ndim(value) == 0 for value in stats.values())


def test_particle_statistics_legacy_key(particle_group: ParticleGroup) -> None:
    stats = particle_group.statistics(
        "higher_order_energy_spread", "sigma_higher_order_energy"
    )
    assert stats["higher_order_energy_spread"] == stats["sigma_higher_order_energy"]
    np.testing.assert_allclose(
        stats["higher_order_energy_spread"],
        particle_group.higher_order_energy_spread,
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("twiss_beta_x", (TwissParameter.beta, "x")),
        ("twiss_norm_emit_y", (TwissParameter.norm_emit, "y")),
        ("twiss_bogus_x", None),
        ("twiss_x", None),
        ("twiss_beta_z", None),
        ("sigma_x", None),
    ],
)
def test_twiss_parameter_parse(
    key: str, expected: tuple[TwissParameter, str] | None
) -> None:
    assert TwissParameter.parse(key) == expected
    if expected is not None:
        param, plane = expected
        assert param.key(plane) == key


def test_statistics_twiss_once_per_plane(
    particle_group: ParticleGroup, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    twiss = ParticleGroup.twiss

    def counting_twiss(self, plane="x", **kwargs):
        calls.append(plane)
        return twiss(self, plane, **kwargs)

    monkeypatch.setattr(ParticleGroup, "twiss", counting_twiss)
    stats = particle_group.statistics(
        "twiss_beta_x", "twiss_alpha_x", "twiss_eta_y", "mean_x"
    )
    assert calls == ["x", "y"]
    expected = {**twiss(particle_group, "x"), **twiss(particle_group, "y")}
    assert stats["twiss_beta_x"] == expected["beta_x"]
    assert stats["twiss_alpha_x"] == expected["alpha_x"]
    assert stats["twiss_eta_y"] == expected["eta_y"]
    assert list(stats) == ["twiss_beta_x", "twiss_alpha_x", "twiss_eta_y", "mean_x"]


def test_statistics_twiss_unknown_key(particle_group: ParticleGroup) -> None:
    with pytest.raises(KeyError):
        particle_group.statistics("twiss_bogus_x")
    assert particle_group.statistics("twiss_bogus_x", skip_errors=True) == {}
