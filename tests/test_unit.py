from __future__ import annotations
import numpy as np
import pytest
import itertools
import math

from beamphysics.units import (
    NAMED_UNITS,
    SHORT_PREFIX_FACTOR,
    dimension,
    nice_array,
    nice_scale_prefix,
    plottable_array,
    pmd_unit,
    unit as deprecated_unit,
)


def test_deprecated_unit():
    with pytest.deprecated_call():
        assert deprecated_unit("T") == pmd_unit("T")


def _round_trips(u: pmd_unit) -> bool:
    """A unit round-trips if re-parsing its symbol reproduces it."""
    try:
        reparsed = pmd_unit(u.unitSymbol)
    except (ValueError, KeyError):
        return False
    return reparsed.unitDimension == u.unitDimension and (
        reparsed.unitSI == u.unitSI
        or math.isclose(reparsed.unitSI, u.unitSI, rel_tol=1e-9)
    )


@pytest.mark.parametrize(
    ("build", "expected_symbol"),
    [
        # Atomic matches (the float-tolerant path).
        pytest.param(lambda: pmd_unit("eV/c") * pmd_unit("c"), "eV", id="eV/c*c->eV"),
        pytest.param(lambda: pmd_unit("J") / pmd_unit("C"), "V", id="J/C->V"),
        pytest.param(lambda: pmd_unit("V") * pmd_unit("A"), "W", id="V*A->W"),
        # Compound matches.
        pytest.param(lambda: pmd_unit("T") * pmd_unit("m"), "T*m", id="T*m"),
        pytest.param(
            lambda: pmd_unit("T") / pmd_unit("m") / pmd_unit("m") * pmd_unit("m"),
            "T/m",
            id="T/m",
        ),
    ],
)
def test_simplify_examples(build, expected_symbol: str) -> None:
    assert build().simplify().unitSymbol == expected_symbol


def test_simplify_electric_potential_volt() -> None:
    # V = J/C = kg*m^2*s^-3*A^-1
    assert pmd_unit("V").unitDimension == (2, 1, -3, -1, 0, 0, 0)


def test_simplify_division_by_compound_round_trips() -> None:
    # Regression: a compound divisor (e.g. "W") must not be emitted as a
    # bare "a/b/c" string that the flat, left-associative parser regroups.
    simplified = (pmd_unit("V/m") / pmd_unit("W")).simplify()
    assert _round_trips(simplified)


@pytest.mark.parametrize("si", [1e-3, 1e3, 1e-6, 2.0])
def test_simplify_dimensionless_never_bare_prefix(si: float) -> None:
    # Regression: an SI prefix on a dimensionless unit produced bogus symbols
    # like "m" (re-parses as metres) or "k" (unparseable). The result must
    # stay dimensionless and never be a bare SI prefix.
    result = pmd_unit("ratio", si, (0, 0, 0, 0, 0, 0, 0)).simplify()
    assert result.unitDimension == (0, 0, 0, 0, 0, 0, 0)
    assert result.unitSymbol not in SHORT_PREFIX_FACTOR
    try:
        reparsed = pmd_unit(result.unitSymbol)
    except (ValueError, KeyError):
        # No simplification found: self returned with its placeholder symbol.
        return
    # If the symbol is parseable, it must not re-parse to a dimensioned unit.
    assert reparsed.unitDimension == (0, 0, 0, 0, 0, 0, 0)


def test_simplify_never_breaks_round_trip() -> None:
    # Across every product/quotient of named units, simplify() must never
    # turn a round-tripping unit into a non-round-tripping one.
    named = [u for u in NAMED_UNITS if u.unitSymbol]
    for a, b in itertools.product(named, named):
        for u in (a * b, a / b):
            if not _round_trips(u):
                continue  # construction already mangled the symbol; not our concern
            assert _round_trips(
                u.simplify()
            ), f"{u.unitSymbol} -> {u.simplify().unitSymbol}"


def test_parsing_compound_does_not_mutate_named_units() -> None:
    # Regression: parsing a compound whose first token is a known unit used to
    # mutate the shared NAMED_UNITS entry in place (e.g. parsing "m*rad" rewrote
    # the global "m" symbol to "m*rad"), corrupting all later simplify() calls.
    before = [(u.unitSymbol, u.unitSI, u.unitDimension) for u in NAMED_UNITS]
    for symbol in ("m*rad", "rad*s", "rad*eV", "m/rad", "T*m", "kg*m/s"):
        pmd_unit(symbol)
    after = [(u.unitSymbol, u.unitSI, u.unitDimension) for u in NAMED_UNITS]
    assert before == after


def test_smoke_properties() -> None:
    u = pmd_unit("K")
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
    assert pmd_unit(symbol) == expected_unit


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
