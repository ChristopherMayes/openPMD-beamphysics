from __future__ import annotations
import numpy as np
import pytest
from pmd_beamphysics.units import (
    SHORT_PREFIX_FACTOR,
    dimension,
    nice_array,
    nice_scale_prefix,
    plottable_array,
    pmd_unit,
    unit,
)


def test_smoke_properties() -> None:
    u = unit("K")
    u.unitSymbol
    u.unitSI
    u.unitDimension


def test_equality() -> None:
    assert pmd_unit("T") == pmd_unit("T")
    assert pmd_unit("T") != pmd_unit("eV")
    assert pmd_unit("T") != "foo"


def test_hashability() -> None:
    assert len({pmd_unit("T"), pmd_unit("eV"), pmd_unit("T")}) == 2


def test_divide() -> None:
    kg_over_g = pmd_unit("kg") / pmd_unit("g")
    assert kg_over_g.unitDimension == (0,) * 7
    assert kg_over_g.unitSI == 1000.0


def test_multiply() -> None:
    momentum = pmd_unit("kg") * pmd_unit("m") / pmd_unit("s")
    assert momentum.unitDimension == dimension("momentum")


@pytest.mark.parametrize(
    ("symbol", "expected_unit"),
    [
        pytest.param("A*s", pmd_unit("A") * pmd_unit("s")),
        pytest.param("A/s", pmd_unit("A") / pmd_unit("s")),
        pytest.param("A/s/s", pmd_unit("A") / pmd_unit("s") / pmd_unit("s")),
    ],
)
def test_unit(symbol: str, expected_unit: pmd_unit) -> None:
    assert unit(symbol) == expected_unit


@pytest.mark.parametrize(
    ("prefix", "factor"),
    [
        pytest.param(prefix, factor, id=prefix)
        for prefix, factor in SHORT_PREFIX_FACTOR.items()
    ],
)
def test_smoke_nice_scale_prefix(prefix: str, factor: float) -> None:
    res = nice_scale_prefix(factor)
    print(f"{factor=} raw {prefix=} ->", res)


@pytest.mark.parametrize(
    ("value", "prefix", "factor"),
    [
        (1e-3, "m", 1e-3),
        (1000.0, "k", 1000.0),
        (1e6, "M", 1e6),
        (1e9, "G", 1e9),
        (1e25, "Y", 1e24),
        (1e-25, "y", 1e-24),
        (0.0, "", 1),
    ],
)
def test_nice_scale_prefix(value: float, prefix: str, factor: float) -> None:
    assert nice_scale_prefix(value) == (factor, prefix)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(1.0, (1.0, 1, "")),
        pytest.param([1.0], (1.0, 1, "")),
        pytest.param([1.0, 1.0, 1.0], (1.0, 1, "")),
    ],
)
def test_nice_array(
    value: float | list[float],
    expected: tuple[np.ndarray | float, float, str],
) -> None:
    expected_scaled, expected_scaling, expected_prefix = expected
    res_scaled, res_scaling, res_prefix = nice_array(value)
    np.testing.assert_allclose(actual=res_scaled, desired=expected_scaled)
    assert expected_scaling == res_scaling
    assert res_prefix == expected_prefix


@pytest.mark.parametrize(
    ("value",),
    [
        pytest.param(1.0),
        pytest.param([1.0]),
        pytest.param([1.0, 1.0, 1.0]),
    ],
)
def test_plottable_array_smoke(value: float | list[float]) -> None:
    for lim, nice in [
        (None, True),
        (None, False),
        ((-10, 10), False),
        ((None, 10), False),
        ((None, None), False),
        ((-10, None), False),
    ]:
        res, *_ = plottable_array(value, lim=lim, nice=nice)
        np.testing.assert_allclose(actual=res, desired=value)
