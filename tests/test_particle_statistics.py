import warnings

import numpy as np
import pytest

from beamphysics import ParticleGroup
from beamphysics.standards.statistics import (
    OPERATORS,
    get_all_statistics_by_key,
    scalar_statistic_keys,
)
from beamphysics.statistics import (
    _OPERATOR_PREFIXES,
    particle_statistics,
    split_statistic_key,
)

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
        ("cov_x__px", ("cov", ("x", "px"))),
        ("cov_z/c__beta_x", ("cov", ("z/c", "beta_x"))),
        ("sigma_x", ("sigma", ("x",))),
        ("mean_kinetic_energy", ("mean", ("kinetic_energy",))),
        ("delta_pz", ("delta", ("pz",))),
        ("ptp_t", ("ptp", ("t",))),
        ("higher_order_energy_spread", ("sigma", ("higher_order_energy",))),
        ("norm_emit_x", None),
        ("x", None),
        ("max_", None),
        ("cov_x", None),
        ("cov_x__px__py", None),
    ],
)
def test_split_statistic_key(
    key: str, expected: tuple[str, tuple[str, ...]] | None
) -> None:
    assert split_statistic_key(key) == expected


@pytest.mark.parametrize("n_particle", [0, 1, 2, None])
def test_particle_statistics_matches_per_key(
    particle_group: ParticleGroup, n_particle: int | None
) -> None:
    P = particle_group if n_particle is None else particle_group[0:n_particle]
    keys = [*get_all_statistics_by_key(), *EXTRA_KEYS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = per_key_statistics(P, keys)
        stats = particle_statistics(P, keys, skip_errors=True)

        def magnitude(name: str) -> float:
            return float(np.max(np.abs(P[name]), initial=0))

    assert list(stats) == list(expected)
    for key, value in expected.items():
        assert type(stats[key]) is type(value), key
        split = split_statistic_key(key)
        if split is None or isinstance(value, str):
            np.testing.assert_equal(stats[key], value, err_msg=key)
            continue
        op, names = split
        if op == "cov" and not np.isfinite(value):
            # Undefined for a single particle: inf or nan, depending on rounding
            assert not np.isfinite(stats[key]), key
            continue
        # Rounding is relative to the arrays' magnitudes, not the (possibly ~0) result
        atol = 1e-12 * np.prod([magnitude(name) for name in names]) * len(names)
        np.testing.assert_allclose(stats[key], value, rtol=1e-9, atol=atol, err_msg=key)


def test_particle_statistics_raises(particle_group: ParticleGroup) -> None:
    with pytest.raises(ValueError, match="bunching"):
        particle_statistics(particle_group, ["mean_x", "bunching"])
    with pytest.raises(AttributeError):
        particle_statistics(particle_group, ["mean_bogus"])


def test_particle_statistics_skip_errors(particle_group: ParticleGroup) -> None:
    stats = particle_statistics(
        particle_group,
        ["mean_x", "bunching", "cov_x__bogus", "sigma_x"],
        skip_errors=True,
    )
    assert list(stats) == ["mean_x", "sigma_x"]


def test_operator_prefixes_match_standard() -> None:
    assert set(_OPERATOR_PREFIXES) == set(OPERATORS)


def test_getitem_malformed_covariance(particle_group: ParticleGroup) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        particle_group["cov_x__px__py"]


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
        stats = particle_statistics(particle_group)
    assert list(stats) == list(scalar_statistic_keys())
    assert all(np.ndim(value) == 0 for value in stats.values())
